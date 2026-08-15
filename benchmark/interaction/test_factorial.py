from __future__ import annotations

import math

import pytest

from core.acre.factorial import FactorialBlock, FactorialEngine, simulate_coverage


def block(values: dict[str, float], block_id: str = "b1") -> FactorialBlock:
    return FactorialBlock(block_id=block_id, outcomes=values)


def test_complete_block_recovers_synergy_and_bounded_interaction_ci() -> None:
    engine = FactorialEngine(delta=0.05, practical_margin=0.05)
    for index in range(24):
        engine.add_block(block({"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}, f"b{index}"))
    estimate = engine.estimate()
    assert estimate.classification == "synergy"
    assert estimate.interaction == pytest.approx(0.175)
    assert estimate.ci_low <= estimate.interaction <= estimate.ci_high


def test_engine_distinguishes_antagonism_independence_and_prerequisite() -> None:
    cases = {
        "antagonism": {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
        "independence": {"00": 0.0, "10": 0.3, "01": 0.4, "11": 0.7},
        "prerequisite": {"00": 0.0, "10": 0.0, "01": 0.4, "11": 0.9},
    }
    for expected, values in cases.items():
        estimate = FactorialEngine(delta=0.05, practical_margin=0.05)
        for index in range(24):
            estimate.add_block(block(values, f"{expected}-{index}"))
        assert estimate.estimate().classification == expected


def test_factorial_rejects_incomplete_or_out_of_range_blocks() -> None:
    with pytest.raises(ValueError, match="complete"):
        block({"00": 0.0, "10": 0.1, "11": 0.2})
    with pytest.raises(ValueError, match="bounded"):
        block({"00": 0.0, "10": 0.1, "01": 0.2, "11": 1.1})


def test_coverage_simulation_is_deterministic_and_meets_contract() -> None:
    result = simulate_coverage(true_interaction=0.2, noise=0.04, blocks=24, repetitions=200, seed=7)
    assert result == simulate_coverage(true_interaction=0.2, noise=0.04, blocks=24, repetitions=200, seed=7)
    assert result.coverage >= 0.93
    assert math.isclose(result.true_interaction, 0.2)
