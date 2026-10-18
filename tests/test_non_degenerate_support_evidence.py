from __future__ import annotations

from benchmark.external.non_degenerate_evidence import classify_observation, repeated_observation


def test_acre_decision_uses_context_key_not_observation_id(monkeypatch) -> None:
    from types import SimpleNamespace
    from scripts import run_non_degenerate_support_evidence as driver

    positive = repeated_observation(observation_id="positive", context={"workload": {"x": 1}}, effect=0.9, gate_passed=True, seed=1, repetitions=20, noise_scale=0.01, delta=0.05)
    negative = repeated_observation(observation_id="negative", context={"workload": {"x": 2}}, effect=-0.9, gate_passed=False, seed=1, repetitions=20, noise_scale=0.01, delta=0.05)
    result = SimpleNamespace(predicate={"accepted": True}, to_dict=lambda: {})
    monkeypatch.setattr(driver.StatisticalCEGIS, "synthesize", lambda *args, **kwargs: result)
    monkeypatch.setattr(driver, "match_predicate", lambda predicate, context: context["workload"]["x"] == 1)
    policy, _ = driver._policy_predictions("acre_full", "compile", [positive, negative], [], [positive, negative])
    assert policy(positive.context).action == "reuse"
    assert policy(negative.context).action == "abstain"


def test_repeated_observation_has_a_non_degenerate_interval() -> None:
    observation = repeated_observation(
        observation_id="x",
        context={"workload": {"x": 1}},
        effect=0.9,
        gate_passed=True,
        seed=3,
        repetitions=20,
        noise_scale=0.01,
        delta=0.05,
    )
    assert observation.effect_lower < observation.effect_upper
    assert classify_observation(observation, practical_margin=0.0) == "positive_anchor"


def test_interval_straddling_margin_is_unresolved() -> None:
    observation = repeated_observation(
        observation_id="x",
        context={},
        effect=0.0,
        gate_passed=True,
        seed=3,
        repetitions=5,
        noise_scale=0.01,
        delta=0.05,
    )
    assert classify_observation(observation, practical_margin=0.0) == "unresolved"


def test_scientific_gate_failure_is_counterexample() -> None:
    observation = repeated_observation(
        observation_id="x",
        context={},
        effect=0.4,
        gate_passed=False,
        seed=3,
        repetitions=5,
        noise_scale=0.01,
        delta=0.05,
    )
    assert classify_observation(observation, practical_margin=0.0) == "counterexample"
