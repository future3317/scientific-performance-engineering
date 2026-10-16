import json
from types import SimpleNamespace

import pytest

from core.acre.resource_budget import ProviderBudgetExceeded, ProviderUsageBudget
from experiments.external_real.alfworld_stateful_runner import (
    DeepSeekStatefulClient,
    EpisodeState,
    run_episode,
)


def _client(monkeypatch, create, budget):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
        ),
    )
    return DeepSeekStatefulClient(
        "key", "https://api.deepseek.com", "deepseek-flash", budget=budget
    )


def _response(input_tokens=40, completion_tokens=5, **usage_fields):
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=completion_tokens,
        **usage_fields,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Action: look", reasoning_content=""),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def _choose(client):
    state = EpisodeState.start("Your task is to: look.", 2)
    return client.choose_action(state, ["look"], None, "progress", None)


def test_rejects_first_call_when_full_max_tokens_reservation_does_not_fit(monkeypatch):
    calls = []
    budget = ProviderUsageBudget(max_calls=3, max_total_tokens=8191)
    client = _client(monkeypatch, lambda **request: calls.append(request), budget)

    with pytest.raises(ProviderBudgetExceeded, match="provider_token_cap"):
        _choose(client)

    assert calls == []
    assert budget.calls_reserved == 0


def test_actual_usage_releases_reservation_and_cache_tokens_stay_separate(monkeypatch):
    calls = []
    responses = iter(
        [
            _response(
                prompt_cache_hit_tokens=30,
                prompt_cache_miss_tokens=10,
            ),
            _response(prompt_cache_hit_tokens=30, prompt_cache_miss_tokens=10),
        ]
    )

    def create(**request):
        calls.append(request)
        return next(responses)

    budget = ProviderUsageBudget(max_calls=2, max_total_tokens=100_000)
    client = _client(monkeypatch, create, budget)
    first = _choose(client)
    _choose(client)

    assert len(calls) == 2
    assert all(request["max_tokens"] == 8192 for request in calls)
    assert budget.tokens_used == 90
    assert first["cache_hit_tokens"] == 30
    assert first["cache_miss_tokens"] == 10

    ledger = ProviderUsageBudget(max_calls=2, max_total_tokens=100)
    ledger.reserve(input_token_bound=10, max_completion_tokens=50)
    ledger.record_usage(input_tokens=5, completion_tokens=10)
    ledger.reserve(input_token_bound=20, max_completion_tokens=50)


def test_unknown_usage_stops_before_the_next_provider_call(monkeypatch):
    calls = []

    def create(**request):
        calls.append(request)
        return _response() if len(calls) == 1 else SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Action: look"), finish_reason="stop")],
            usage=None,
        )

    budget = ProviderUsageBudget(max_calls=3, max_total_tokens=100_000)
    client = _client(monkeypatch, create, budget)
    _choose(client)
    # Unknown usage must not release an action to the environment, even for
    # the response that first made accounting unknown.
    with pytest.raises(ProviderBudgetExceeded, match="provider_usage_unknown"):
        _choose(client)
    assert budget.tokens_used == 45
    assert budget.as_dict()["tokens_unknown"] is True
    assert budget.as_dict()["reservations"]["status"] == "usage_unknown"
    with pytest.raises(ProviderBudgetExceeded):
        _choose(client)

    assert len(calls) == 2
    assert budget.stopped


def test_unknown_usage_never_reaches_env_step(monkeypatch):
    calls = []
    def create(**request):
        calls.append(request)
        return SimpleNamespace(usage=None)
    budget = ProviderUsageBudget(max_calls=3, max_total_tokens=100_000)
    client = _client(monkeypatch, create, budget)
    environment = _Environment()
    result = run_episode(environment, client, max_steps=2)
    assert environment.steps == result["env_steps"] == 0
    assert result["success"] is None and budget.tokens_unknown
    assert len(calls) == budget.calls_reserved == 1


def test_provider_exception_consumes_admission_and_freezes_unknown_budget(monkeypatch):
    calls = []

    def fail(**request):
        calls.append(request)
        raise TimeoutError("provider timed out")

    budget = ProviderUsageBudget(max_calls=3, max_total_tokens=100_000)
    client = _client(monkeypatch, fail, budget)
    with pytest.raises(TimeoutError):
        _choose(client)
    with pytest.raises(ProviderBudgetExceeded):
        _choose(client)

    assert len(calls) == 1
    assert budget.calls_reserved == 1
    assert budget.stopped
    snapshot = budget.as_dict()
    json.dumps(snapshot)
    assert snapshot["tokens_unknown"] is True
    assert snapshot["reservations"]["status"] == "usage_unknown"


def test_usage_is_recorded_before_choice_access(monkeypatch):
    class Response:
        usage = SimpleNamespace(prompt_tokens=17, completion_tokens=3)

        @property
        def choices(self):
            raise RuntimeError("malformed choices")

    budget = ProviderUsageBudget(max_calls=1, max_total_tokens=100_000)
    client = _client(monkeypatch, lambda **request: Response(), budget)
    with pytest.raises(RuntimeError, match="malformed choices"):
        _choose(client)

    snapshot = budget.as_dict()
    assert snapshot["tokens_used_known"] == 20
    assert snapshot["tokens_unknown"] is False
    assert snapshot["reservations"] is None


def test_actual_input_over_prompt_reserve_is_recorded_and_stops_even_under_total_cap():
    budget = ProviderUsageBudget(max_calls=2, max_total_tokens=1000)
    budget.reserve(input_token_bound=10, max_completion_tokens=20)
    budget.record_usage(input_tokens=11, completion_tokens=1)

    snapshot = budget.as_dict()
    assert snapshot["tokens_used_known"] == 12
    assert snapshot["bound_violation"] is True
    assert snapshot["stop_reason"] == "input_token_bound_violation"
    with pytest.raises(ProviderBudgetExceeded):
        budget.reserve(input_token_bound=1, max_completion_tokens=1)


class _Environment:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return ["Your task is to: look."], {"admissible_commands": [["look"]]}

    def step(self, actions):
        self.steps += 1
        return ["Still here."], [0.0], [False], {"won": [False], "admissible_commands": [["look"]]}


def test_episode_reports_budget_termination_and_keeps_prior_trajectory():
    class Client:
        def __init__(self):
            self.calls = 0

        def choose_action(self, *args):
            self.calls += 1
            if self.calls == 1:
                return {"raw_output": "Action: look", "input_tokens": 12, "output_tokens": 2}
            raise ProviderBudgetExceeded("provider_token_cap")

    result = run_episode(_Environment(), Client(), max_steps=3)

    assert result["termination_reason"] == "provider_budget_exhausted"
    assert result["success"] is None
    assert len(result["trajectory"]) == 1
    assert result["trajectory"][0]["selected_action"] == "look"
    assert result["env_steps"] == 1
    assert result["progress_events"][-1]["kind"] == "provider_budget_exhausted"


def test_smoke_persists_budget_before_calls_and_does_not_launch_after_exhaustion(monkeypatch, tmp_path):
    import json
    import sys
    from experiments.external_real import alfworld_stateful_runner as runner

    output = tmp_path / "new-run"
    budget = ProviderUsageBudget(0, 0)
    client = SimpleNamespace(budget=budget, model="deepseek-flash", temperature=0.0)
    calls = []

    class Environment:
        def seed(self, seed):
            pass

        def close(self):
            calls.append("closed")

    class Official:
        def __init__(self, config, train_eval):
            self.game_files = ["a", "b", "c"]

        def init_env(self, batch_size):
            return Environment()

    monkeypatch.setitem(sys.modules, "alfworld.agents.environment.alfred_tw_env", SimpleNamespace(AlfredTWEnv=Official))
    monkeypatch.setattr(runner, "_select_game_files", lambda *args: ["a", "b", "c"])

    def episode(*args, **kwargs):
        config = json.loads((output / "config.json").read_text())
        assert config["provider_budget"]["max_calls"] == 0
        calls.append("episode")
        return {"success": None, "invalid_actions": 0, "termination_reason": "provider_budget_exhausted"}

    monkeypatch.setattr(runner, "run_episode", episode)
    result = runner.run_smoke(tmp_path, output, client)
    assert calls == ["episode", "closed"]
    assert result["scheduled_episodes"] == 3
    assert result["nonlaunched_episodes"] == 2
    assert result["provider_budget"]["calls_reserved"] == 0
    assert json.loads((output / "results.json").read_text()) == result
    with pytest.raises(FileExistsError):
        runner.run_smoke(tmp_path, output, client)
