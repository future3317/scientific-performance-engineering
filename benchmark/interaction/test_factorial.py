from __future__ import annotations

import math

import pytest

from core.acre.factorial import FactorialBlock, FactorialEngine, simulate_coverage
from core.acre.relation import RelationIdentifier
from benchmark.interaction.factorial_bench import generate_interaction_surface


def block(values: dict[str, float], block_id: str = "b1") -> FactorialBlock:
    return FactorialBlock(block_id=block_id, outcomes=values)


def test_complete_block_recovers_synergy_and_bounded_interaction_ci() -> None:
    engine = FactorialEngine(delta=0.05, practical_margin=0.05)
    for index in range(5000):
        engine.add_block(block({"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}, f"b{index}"))
    estimate = engine.estimate()
    assert estimate.decision == "confirmed_synergy"
    assert estimate.gamma == pytest.approx(0.175)
    assert estimate.gamma_lcb > 0.05
    assert estimate.gamma_lcb <= estimate.gamma <= estimate.gamma_ucb


def test_engine_distinguishes_antagonism_independence_and_prerequisite() -> None:
    cases = {
        "antagonism": {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
        "independence": {"00": 0.0, "10": 0.3, "01": 0.4, "11": 0.7},
        "prerequisite": {"00": 0.0, "10": 0.4, "01": 0.0, "11": 0.9},
    }
    for expected, values in cases.items():
        estimate = FactorialEngine(delta=0.05, practical_margin=0.05)
        # The directed prerequisite gate requires the null conditional effect
        # interval itself to fit inside the practical equivalence region.  The
        # formal Hoeffding/alpha-spending contract therefore needs more blocks
        # than the point-estimate smoke cases above.
        blocks = 20000 if expected == "prerequisite" else 5000
        for index in range(blocks):
            estimate.add_block(block(values, f"{expected}-{index}"))
        decision = estimate.estimate().decision
        assert decision == {
            "antagonism": "confirmed_antagonism",
            "independence": "confirmed_independence",
            "prerequisite": "prerequisite_a_to_b",
        }[expected]


def test_factorial_requires_confidence_before_relation_decision() -> None:
    engine = FactorialEngine(delta=0.05, practical_margin=0.05)
    engine.add_block(block({"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}))
    estimate = engine.estimate()
    assert estimate.decision == "unresolved"


def test_semantic_conflict_requires_independent_scientific_gate_pattern() -> None:
    engine = FactorialEngine()
    for index in range(5000):
        engine.add_block(block({"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2}, f"normal-{index}"))
    gated = FactorialEngine()
    for index in range(5000):
        gated.add_block(FactorialBlock(
            f"conflict-{index}",
            {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
            {"00": True, "10": True, "01": True, "11": False},
        ))
    assert engine.estimate().decision == "confirmed_antagonism"
    estimate = gated.estimate()
    assert estimate.decision == "semantic_conflict"
    assert estimate.scientific_11 is False and estimate.scientific_10 is True


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


def test_parameterized_interaction_surface_is_deterministic_and_stratified() -> None:
    first = generate_interaction_surface(count=128, seed=11)
    assert first == generate_interaction_surface(count=128, seed=11)
    assert len(first) == 128
    assert {item["kind"] for item in first} == {
        "synergy", "antagonism", "independence", "prerequisite_a_to_b",
        "prerequisite_b_to_a", "redundancy", "semantic_conflict", "context_dependent_interaction",
    }


def test_relation_identifier_detects_context_dependent_sign() -> None:
    estimates = {}
    for name, value in (("baseline", {"00": -0.3, "10": -0.2, "01": -0.2, "11": 0.9}), ("shifted", {"00": -0.3, "10": -0.2, "01": -0.2, "11": -0.9})):
        engine = FactorialEngine(delta=0.05, practical_margin=0.05)
        for index in range(5000):
            engine.add_block(block(value, f"{name}-{index}"))
        estimates[name] = engine.estimate()
    result = RelationIdentifier(practical_margin=0.05).identify(estimates)
    assert result.decision == "context_dependent_relation"
