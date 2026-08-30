"""Trusted host controller for hosted agents and a network-free tool server.

The controller owns the provider API call.  The tool server is the only process
that edits the worker workspace or runs commands, and it is started through the
same namespace executor used by formal campaigns.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark.formal.reference_executor import ReferenceExecutor


TOOL_NAMES = ("read_file", "write_file", "run_command")
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a public task, skill-view, retrieved-context, or solution file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write one allowed solution file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a public test or workspace command inside the isolated worker namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "number", "minimum": 1, "maximum": 120},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]

FIXED_PREFIX_PARTS = (
    "You are the primary SPE formal agent. Work only through the supplied tools.",
    "SPE policy: preserve correctness, scientific invariants, and fail-closed behavior; never claim hidden evidence.",
    "Interaction rules: inspect public_task and public context, edit only allowed solution files, run public tests when useful, and leave the best executable solution in the workspace.",
    "Output schema: finish with a concise status message; the controller records the authoritative usage and result receipt.",
)
FIXED_PREFIX_TEXT = "\n\n".join(FIXED_PREFIX_PARTS)


@dataclass(frozen=True)
class ControllerLimits:
    input_tokens: int = 65_536
    output_tokens: int = 16_384
    max_output_tokens: int = 4_096
    context_tokens: int = 32_768
    tool_calls: int = 24
    turns: int = 12
    wall_time_s: float = 900.0
    api_timeout_s: float = 120.0


def parse_controller_limits(raw: str) -> ControllerLimits:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("SPE_BUDGET_JSON must be an object")
    limits = ControllerLimits()
    expected = {
        "tokens": limits.input_tokens + limits.output_tokens,
        "tool_calls": limits.tool_calls,
        "wall_time_s": limits.wall_time_s,
        "context_tokens": limits.context_tokens,
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise ValueError(f"SPE_BUDGET_JSON {key} must equal {required}")
    return limits


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixed_prefix_digest() -> str:
    return _digest_bytes(FIXED_PREFIX_TEXT.encode("utf-8"))


def tool_schema_digest() -> str:
    payload = json.dumps(TOOL_DEFINITIONS, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _digest_bytes(payload.encode("utf-8"))


def _safe_environment() -> dict[str, str]:
    blocked = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SPE_") and not any(marker in key.upper() for marker in blocked)
    }


TOOL_SERVER = r'''
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/worker")
ALLOWED_FILES = set(json.loads(sys.argv[1]))

def inside(base, relative):
    value = str(relative)
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError("path must be a non-empty relative POSIX path")
    path = (base / value).resolve()
    if path != base and base not in path.parents:
        raise ValueError("path escapes the worker mount")
    return path

def read_file(relative):
    value = str(relative)
    first = value.split("/", 1)[0]
    if first not in {"task", "solution", "retrieved_context", "skill_view", "context_state"}:
        raise ValueError("read_file path is outside the public worker mounts")
    path = inside(ROOT, value)
    data = path.read_text(encoding="utf-8")
    return {"ok": True, "path": value, "content": data[:200000], "truncated": len(data) > 200000}

def write_file(relative, content):
    value = str(relative)
    if value not in ALLOWED_FILES:
        raise ValueError("write_file path is not an allowed solution file")
    path = inside(ROOT / "solution", value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return {"ok": True, "path": value, "bytes": len(str(content).encode("utf-8"))}

def run_command(command, timeout_s):
    completed = subprocess.run(
        str(command), shell=True, cwd="/worker/solution", capture_output=True,
        text=True, timeout=max(1, min(float(timeout_s), 120.0)), check=False,
    )
    return {"ok": completed.returncode == 0, "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:], "stderr": completed.stderr[-20000:]}

for line in sys.stdin:
    try:
        request = json.loads(line)
        name = request["name"]
        args = request.get("arguments", {})
        if name == "read_file":
            result = read_file(args["path"])
        elif name == "write_file":
            result = write_file(args["path"], args["content"])
        elif name == "run_command":
            result = run_command(args["command"], args.get("timeout_s", 120))
        elif name == "shutdown":
            print(json.dumps({"ok": True}), flush=True)
            break
        else:
            raise ValueError("tool is not allowlisted")
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False), flush=True)
'''


def _api_call(base_url: str, api_key: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list) or not value["choices"]:
        raise RuntimeError("DeepSeek API response has no choices")
    return value


def _api_call_with_retry(base_url: str, api_key: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    for attempt in range(2):
        try:
            return _api_call(base_url, api_key, payload, timeout_s)
        except RuntimeError:
            if attempt == 1:
                raise
            time.sleep(1.0)
    raise RuntimeError("unreachable API retry state")


def _tool_call(process: subprocess.Popen[str], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        return {"ok": False, "error": "tool is not allowlisted"}
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"name": name, "arguments": arguments}) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("sandbox tool server exited before returning a result")
    result = json.loads(line)
    return result if isinstance(result, dict) else {"ok": False, "error": "invalid sandbox tool result"}


def _public_bundle(worker_root: Path) -> dict[str, Any]:
    def read_json(path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

    skill: dict[str, str] = {}
    skill_root = worker_root / "skill_view"
    if skill_root.is_dir():
        for relative in ("SKILL.md", "agents/openai.yaml"):
            path = skill_root / relative
            if path.is_file():
                skill[relative] = path.read_text(encoding="utf-8")
    return {
        "public_task": read_json(worker_root / "task" / "public_task.json", {}),
        "retrieved_context": read_json(worker_root / "retrieved_context" / "retrieved_context.json", {}),
        "skill_view": skill,
    }


def build_agent_attestation(*, model_list_digest: str, agent_code_commit: str, container_digest: str, pricing_revision: str) -> dict[str, Any]:
    return {
        "provider": "deepseek",
        "model_id": "deepseek-v4-flash",
        "provider_model_version": "DeepSeek-V4-Flash",
        "provider_model_list_response_digest": model_list_digest,
        "agent_code_commit": agent_code_commit,
        "system_prompt_digest": fixed_prefix_digest(),
        "tool_schema_digest": tool_schema_digest(),
        "tool_allowlist": list(TOOL_NAMES),
        "retry_policy": {"max_attempts": 2, "backoff_s": 1},
        "input_token_limit": 65_536,
        "output_token_limit": 16_384,
        "max_output_tokens": 4_096,
        "context_token_limit": 32_768,
        "tool_call_limit": 24,
        "turn_limit": 12,
        "timeout_s": 900,
        "container_digest": container_digest,
        "runtime_attestation_kind": "trusted_host_conda_prefix_manifest",
        "container_digest_semantics": "runtime_manifest_digest",
        "pricing_revision": pricing_revision,
        "backend_weight_immutability_claim": False,
        "thinking": "enabled",
        "reasoning_effort": "medium",
        "temperature": 0,
        "top_p": 1,
    }


def run() -> dict[str, Any]:
    worker_root = Path(os.environ["SPE_WORKER_ROOT"]).resolve()
    receipt_path = Path(os.environ["SPE_EXECUTOR_RECEIPT_PATH"]).resolve()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("\r")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip().rstrip("\r")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required by the host controller")
    allowed_files = json.loads((worker_root / "task" / "public_task.json").read_text(encoding="utf-8")).get("allowed_files", [])
    executor = ReferenceExecutor()
    command = executor.command([sys.executable, "-c", TOOL_SERVER, json.dumps(allowed_files)], worker_root)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_safe_environment(), bufsize=1)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": FIXED_PREFIX_TEXT},
        {"role": "user", "content": json.dumps(_public_bundle(worker_root), ensure_ascii=False, sort_keys=True)},
    ]
    input_tokens = output_tokens = tool_calls = 0
    tool_trace: list[str] = []
    started = time.monotonic()
    try:
        for _ in range(16):
            response = _api_call_with_retry(base_url, api_key, {
                "model": model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 8192,
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            }, timeout_s=120)
            usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
            input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            message = response["choices"][0].get("message", {})
            if not isinstance(message, dict):
                raise RuntimeError("DeepSeek response message is invalid")
            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            if message.get("reasoning_content") is not None:
                assistant["reasoning_content"] = message["reasoning_content"]
            calls = message.get("tool_calls") or []
            if not calls:
                messages.append(assistant)
                break
            assistant["tool_calls"] = calls
            messages.append(assistant)
            for call in calls:
                function = call.get("function", {})
                name = str(function.get("name", ""))
                arguments = json.loads(function.get("arguments", "{}"))
                result = _tool_call(process, name, arguments)
                tool_calls += 1
                tool_trace.append(name)
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"tool-{tool_calls}"), "content": json.dumps(result, ensure_ascii=False)})
        else:
            raise RuntimeError("host controller tool-call loop exceeded 16 turns")
    finally:
        try:
            _tool_call(process, "shutdown", {})
        except Exception:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    completed = executor.execute([sys.executable, "-c", "pass"], worker_root, receipt_path=receipt_path, worker_uid="host-deepseek-controller", include_skill=(worker_root / "skill_view").is_dir())
    if completed.returncode != 0:
        raise RuntimeError(f"sandbox receipt canary failed: {completed.returncode}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "tool_calls": tool_calls, "wall_time_s": time.monotonic() - started}
    receipt["usage"] = usage
    receipt["usage_meter_source"] = "trusted-host-controller-provider-observed"
    receipt["host_controller"] = {"provider": "deepseek", "model_id": model, "system_prompt_digest": fixed_prefix_digest(), "tool_schema_digest": tool_schema_digest(), "tool_trace": tool_trace}
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (worker_root / "result" / "agent_usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    (worker_root / "result" / "worker_result.json").write_text(json.dumps({"status": "completed", "controller": "trusted_host_deepseek"}) + "\n", encoding="utf-8")
    return {"returncode": 0, "usage": usage}


def main() -> int:
    try:
        result = run()
    except Exception as exc:
        print(f"host agent controller failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
