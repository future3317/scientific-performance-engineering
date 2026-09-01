from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import sys

import pytest

from benchmark.formal import attest, host_agent
from benchmark.harness.skill_view import render_skill_view
from scripts import materialize_host_controller_contract as materializer


FROZEN_BUDGET = {
    "tokens": 81_920,
    "tool_calls": 24,
    "wall_time_s": 900,
    "context_tokens": 32_768,
}


def test_parse_controller_limits_accepts_exact_budget() -> None:
    value = host_agent.parse_controller_limits(json.dumps(FROZEN_BUDGET))

    assert (value.input_tokens, value.output_tokens, value.max_output_tokens) == (65_536, 16_384, 4_096)
    assert (value.context_tokens, value.tool_calls, value.turns) == (32_768, 24, 12)
    assert (value.wall_time_s, value.api_timeout_s) == (900, 120)


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokens", 81_919), ("tool_calls", 25), ("wall_time_s", 901), ("context_tokens", 4_096)],
)
def test_parse_controller_limits_rejects_contract_drift(field: str, value: int) -> None:
    budget = dict(FROZEN_BUDGET)
    budget[field] = value

    with pytest.raises(ValueError, match=field):
        host_agent.parse_controller_limits(json.dumps(budget))


def test_agent_attestation_records_policy_and_runtime_semantics() -> None:
    value = host_agent.build_agent_attestation(
        model_list_digest="a" * 64,
        agent_code_commit="b" * 40,
        container_digest="c" * 64,
        pricing_revision="unpriced-provider-probe-2026-08-30",
    )

    assert value["thinking"] == "disabled"
    assert value["reasoning_effort"] == "none"
    assert value["max_output_tokens"] == 4_096
    assert value["input_token_limit"] == 65_536
    assert value["output_token_limit"] == 16_384
    assert value["turn_limit"] == 12
    assert value["timeout_s"] == 900
    assert value["runtime_attestation_kind"] == "trusted_host_python_runtime_manifest"
    assert value["container_digest_semantics"] == "runtime_manifest_digest"


def test_remaining_output_allowance_shrinks() -> None:
    limits = host_agent.ControllerLimits()
    usage = host_agent.UsageState(output_tokens=15_900)

    assert host_agent._remaining_output_tokens(limits, usage) == 484


def test_provider_input_partition_is_outcome_failure() -> None:
    limits = host_agent.ControllerLimits()
    usage = host_agent.UsageState(input_tokens=65_000)

    with pytest.raises(host_agent.BudgetOutcome, match="input_token_limit"):
        usage.add_provider_usage({"prompt_tokens": 600, "completion_tokens": 1}, limits)

    assert (usage.input_tokens, usage.output_tokens) == (65_600, 1)


def test_provider_output_partition_counts_reasoning_output() -> None:
    limits = host_agent.ControllerLimits()
    usage = host_agent.UsageState(output_tokens=16_300)

    with pytest.raises(host_agent.BudgetOutcome, match="output_token_limit"):
        usage.add_provider_usage({"prompt_tokens": 10, "completion_tokens": 100}, limits)

    assert usage.output_tokens == 16_400


def test_tool_limit_blocks_extra_execution() -> None:
    limits = host_agent.ControllerLimits()
    usage = host_agent.UsageState(tool_calls=24)

    with pytest.raises(host_agent.BudgetOutcome, match="tool_call_limit"):
        usage.reserve_tool_call(limits)

    assert usage.tool_calls == 24


def _serve_responses(responses: list[dict[str, object]]) -> tuple[ThreadingHTTPServer, list[dict[str, object]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(responses.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def test_conversation_disables_thinking_and_preserves_remaining_output(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "choices": [{"message": {"content": None, "reasoning_content": "private", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{\"path\":\"task/public_task.json\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 16_000},
        },
        {"choices": [{"message": {"content": "done"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 100}},
    ]
    server, requests = _serve_responses(responses)
    monkeypatch.setattr(host_agent, "_tool_call", lambda process, name, arguments: {"ok": True})
    try:
        usage, trace, outcome = host_agent._run_conversation(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="api-key", model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "public"}], process=None,
            limits=host_agent.ControllerLimits(), started=0.0, clock=lambda: 1.0,
        )
    finally:
        server.shutdown()

    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[0]
    assert requests[0]["max_tokens"] == 4_096
    assert requests[1]["max_tokens"] == 384
    assert usage.output_tokens == 16_100
    assert trace == ["read_file"]
    assert outcome == "completed"


def test_conversation_returns_tool_error_for_malformed_arguments_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": "malformed-1",
                "function": {"name": "read_file", "arguments": '{"path": "unterminated'},
            }]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "choices": [{"message": {"content": "recovered"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    ]
    server, _ = _serve_responses(responses)
    tool_calls: list[tuple[str, object]] = []

    def unexpected_tool_call(process: object, name: str, arguments: object) -> dict[str, object]:
        tool_calls.append((name, arguments))
        return {"ok": True}

    monkeypatch.setattr(host_agent, "_tool_call", unexpected_tool_call)
    messages = [{"role": "user", "content": "public"}]
    try:
        usage, trace, outcome = host_agent._run_conversation(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="api-key", model="deepseek-v4-flash",
            messages=messages, process=None, limits=host_agent.ControllerLimits(), started=0.0,
            clock=lambda: 1.0,
        )
    finally:
        server.shutdown()

    assert usage.tool_calls == 1
    assert trace == ["read_file"]
    assert outcome == "completed"
    assert tool_calls == []
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "malformed-1",
        "content": json.dumps({"ok": False, "error": "invalid tool arguments JSON"}),
    }


def test_conversation_returns_error_for_non_object_arguments() -> None:
    responses = [
        {
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": "array-1",
                "function": {"name": "read_file", "arguments": "[]"},
            }]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {"choices": [{"message": {"content": "recovered"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    ]
    server, _ = _serve_responses(responses)
    messages = [{"role": "user", "content": "public"}]
    try:
        usage, trace, outcome = host_agent._run_conversation(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="api-key", model="deepseek-v4-flash",
            messages=messages, process=None, limits=host_agent.ControllerLimits(), started=0.0,
            clock=lambda: 1.0,
        )
    finally:
        server.shutdown()

    assert usage.tool_calls == 1
    assert trace == ["read_file"]
    assert outcome == "completed"
    assert messages[2]["content"] == json.dumps({"ok": False, "error": "tool arguments must be a JSON object"})


def test_conversation_rejects_empty_final_as_outcome() -> None:
    server, _ = _serve_responses([
        {"choices": [{"message": {"content": None, "reasoning_content": "private"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    ])
    try:
        with pytest.raises(host_agent.BudgetOutcome, match="incomplete_agent_outcome"):
            host_agent._run_conversation(
                base_url=f"http://127.0.0.1:{server.server_port}", api_key="api-key", model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "public"}], process=None,
                limits=host_agent.ControllerLimits(), started=0.0, clock=lambda: 1.0,
            )
    finally:
        server.shutdown()


def test_controller_receipt_excludes_reasoning_and_credentials(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"network_mode": "none"}), encoding="utf-8")
    usage = host_agent.UsageState(input_tokens=10, output_tokens=20, tool_calls=1)

    host_agent._write_controller_receipt(
        receipt_path=receipt_path, worker_root=tmp_path, usage=usage, wall_time_s=2.5,
        outcome="output_token_limit", model="deepseek-v4-flash", tool_trace=["read_file"],
    )

    serialized = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(serialized)
    assert receipt["host_controller"]["outcome"] == "output_token_limit"
    assert receipt["usage"] == {"input_tokens": 10, "output_tokens": 20, "tool_calls": 1, "wall_time_s": 2.5}
    assert "reasoning_content" not in serialized
    assert "api-key" not in serialized

def test_controller_receipt_binds_present_skill_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    worker_root = tmp_path / "worker"
    for name in (
        "task",
        "solution",
        "retrieved_context",
        "context_state",
        "result",
        "executor_receipt",
    ):
        (worker_root / name).mkdir(parents=True, exist_ok=True)
    (worker_root / "task" / "public_task.json").write_text(
        json.dumps({"allowed_files": ["solution.py"]}), encoding="utf-8"
    )
    (worker_root / "retrieved_context" / "retrieved_context.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (worker_root / "context_state" / "context_state.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (worker_root / "solution" / "solution.py").write_text("", encoding="utf-8")
    render_skill_view(root, worker_root / "skill_view")
    receipt_path = tmp_path / "executor_receipt.json"
    server, _ = _serve_responses([{
        "choices": [{"message": {"content": "done"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }])
    monkeypatch.setenv("SPE_WORKER_ROOT", str(worker_root))
    monkeypatch.setenv("SPE_EXECUTOR_RECEIPT_PATH", str(receipt_path))
    monkeypatch.setenv("SPE_BUDGET_JSON", json.dumps(FROZEN_BUDGET))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    try:
        result = host_agent.run()
    finally:
        server.shutdown()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["returncode"] == 0
    assert receipt["skill_view_digest"] == attest.skill_view_digest(worker_root / "skill_view")


def test_budget_outcome_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        raise host_agent.BudgetOutcome("output_token_limit")

    monkeypatch.setattr(host_agent, "_api_call", fail)
    with pytest.raises(host_agent.BudgetOutcome, match="output_token_limit"):
        host_agent._api_call_with_retry("http://provider", "key", {}, 1)
    assert attempts == 1


def test_provider_contract_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("authentication failed")

    monkeypatch.setattr(host_agent, "_api_call", fail)
    with pytest.raises(RuntimeError, match="authentication failed"):
        host_agent._api_call_with_retry("http://provider", "key", {}, 1)
    assert attempts == 1


def test_transient_provider_failure_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def call(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise host_agent.TransientProviderError("temporary")
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(host_agent, "_api_call", call)
    value = host_agent._api_call_with_retry("http://provider", "key", {}, 1)
    assert value["choices"]
    assert attempts == 2


def test_main_returns_controlled_outcome_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(host_agent, "run", lambda: {"returncode": 2, "usage": {}, "outcome": "turn_limit"})

    assert host_agent.main() == 2
    assert json.loads(capsys.readouterr().out)["outcome"] == "turn_limit"


def test_materialized_contract_binds_runtime_and_frozen_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = materializer.build_runtime_manifest(root)
    probe = {
        "status": "pass",
        "model_id": "deepseek-v4-flash",
        "returned_model": "deepseek-v4-flash",
        "provider_model_list_response_digest": "a" * 64,
    }

    contract = materializer.build_contract(runtime, probe, benchmark_revision="d" * 40)

    assert runtime["runtime_attestation_kind"] == "trusted_host_python_runtime_manifest"
    assert "installed_packages" not in runtime
    assert runtime["python_executable"] == sys.executable
    assert runtime["controller_commit"]
    assert len(runtime["controller_source_digest"]) == 64
    assert contract["container_digest"] == materializer.canonical_digest(runtime)
    assert contract["container_digest_semantics"] == "runtime_manifest_digest"
    assert contract["pricing_revision"] == "unpriced-provider-probe-2026-08-30"
    assert contract["campaign_budget"] == FROZEN_BUDGET
    assert contract["benchmark_revision"] == "d" * 40
    assert contract["agent_config"]["reasoning_effort"] == "none"
    assert materializer.validate_contract(contract) == []
    assert "api_key" not in json.dumps(contract).lower()


def test_materializer_rejects_malformed_provider_digest() -> None:
    runtime = {"runtime_attestation_kind": "trusted_host_python_runtime_manifest"}
    probe = {"status": "pass", "model_id": "deepseek-v4-flash", "returned_model": "deepseek-v4-flash", "provider_model_list_response_digest": "bad"}

    with pytest.raises(ValueError, match="provider_model_list_response_digest"):
        materializer.build_contract(runtime, probe, benchmark_revision="d" * 40)


def test_materializer_rejects_credential_fields() -> None:
    runtime = {"runtime_attestation_kind": "trusted_host_python_runtime_manifest"}
    probe = {"status": "pass", "model_id": "deepseek-v4-flash", "returned_model": "deepseek-v4-flash", "provider_model_list_response_digest": "a" * 64, "api_key": "secret"}

    with pytest.raises(ValueError, match="credential"):
        materializer.build_contract(runtime, probe, benchmark_revision="d" * 40)


def test_contract_validation_uses_explicit_benchmark_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def validate(manifest: dict[str, object]) -> list[str]:
        captured.update(manifest)
        return []

    monkeypatch.setattr(materializer.attest, "validate_experiment", validate)
    contract = {
        "runtime_manifest": {},
        "container_digest": materializer.canonical_digest({}),
        "campaign_budget": dict(FROZEN_BUDGET),
        "benchmark_revision": "e" * 40,
        "agent_config": {},
    }

    assert materializer.validate_contract(contract) == []
    assert captured["benchmark_revision"] == "e" * 40
