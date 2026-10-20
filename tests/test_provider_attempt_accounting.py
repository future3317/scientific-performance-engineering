from __future__ import annotations

import sys
from types import SimpleNamespace

import benchmark.formal.host_agent as host_agent
import experiments.external_real.run_scienceworld as scienceworld_runner
import scripts.run_scienceworld_train_formal as train_formal


def test_retry_reports_each_transient_provider_request_attempt(monkeypatch) -> None:
    request_attempts: list[int] = []
    accounted_attempts: list[int] = []

    def fail_once(base_url, api_key, payload, timeout_s):
        request_attempts.append(len(request_attempts) + 1)
        if len(request_attempts) == 1:
            raise host_agent.TransientProviderError("transient provider failure")
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(host_agent, "_api_call", fail_once)
    monkeypatch.setattr(host_agent.time, "sleep", lambda _seconds: None)
    response = host_agent._api_call_with_retry(
        "https://example.invalid",
        "not-a-real-key",
        {"model": "fixture"},
        timeout_s=1,
        on_attempt=lambda: accounted_attempts.append(len(accounted_attempts) + 1),
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert request_attempts == [1, 2]
    assert accounted_attempts == [1, 2]


def test_scienceworld_backend_counts_each_retry_attempt(monkeypatch, tmp_path) -> None:
    class FakeEnvironment:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self, *_args):
            pass

        def reset(self):
            return "observation", {}

        def get_possible_actions(self):
            return []

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=FakeEnvironment))
    accounted: list[int] = []

    def fake_retry(*_args, on_attempt=None, **_kwargs):
        on_attempt()
        on_attempt()
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(host_agent, "_api_call_with_retry", fake_retry)
    backend = scienceworld_runner.ScienceWorldProviderBackend(
        scienceworld_root=tmp_path,
        model_id="fixture",
        request_counter=lambda: accounted.append(len(accounted) + 1),
    )

    backend.execute({"task_name": "fixture", "variation_idx": 0}, None, seed=0)

    assert backend.provider_calls == 2
    assert accounted == [1, 2]


def test_skill_generation_forwards_attempt_counter(monkeypatch) -> None:
    accounted: list[int] = []

    def fake_retry(*_args, on_attempt=None, **_kwargs):
        on_attempt()
        on_attempt()
        return {"choices": [{"message": {"content": "frozen skill"}}]}

    monkeypatch.setattr(train_formal, "_api_call_with_retry", fake_retry)
    result = train_formal._generate_skill(
        prompt="fixture", model_id="fixture", on_attempt=lambda: accounted.append(len(accounted) + 1)
    )

    assert result == "frozen skill"
    assert accounted == [1, 2]
