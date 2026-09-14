from __future__ import annotations

from benchmark.external.non_degenerate_evidence import classify_observation, repeated_observation


def test_repeated_observation_has_a_non_degenerate_interval() -> None:
    observation = repeated_observation(
        observation_id="x",
        context={"workload": {"x": 1}},
        effect=0.4,
        gate_passed=True,
        seed=3,
        repetitions=5,
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
