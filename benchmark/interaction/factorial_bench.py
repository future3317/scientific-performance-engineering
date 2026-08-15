"""Synthetic factorial interaction pilot, separate from formal benchmark scoring."""

from __future__ import annotations

from core.acre.factorial import FactorialBlock, FactorialEngine, simulate_coverage
import random


_CASES = {
    "synergy": {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9},
    "antagonism": {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
    "independence": {"00": 0.0, "10": 0.3, "01": 0.4, "11": 0.7},
    "prerequisite": {"00": 0.0, "10": 0.4, "01": 0.0, "11": 0.9},
}


def generate_interaction_surface(*, count: int = 128, seed: int = 7) -> list[dict[str, object]]:
    """Generate deterministic relation surfaces for pilot calibration."""
    if count < 8:
        raise ValueError("count must be at least 8")
    rng = random.Random(seed)
    kinds = ("synergy", "antagonism", "independence", "prerequisite_a_to_b", "prerequisite_b_to_a", "redundancy", "semantic_conflict", "context_dependent_interaction")
    surface: list[dict[str, object]] = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        a, b, effect = (0.1 + rng.random() * 0.2 for _ in range(3))
        if kind == "synergy":
            outcomes = {"00": 0.0, "10": a, "01": b, "11": min(0.95, a + b + effect)}
        elif kind == "antagonism":
            outcomes = {"00": 0.0, "10": a, "01": b, "11": max(-0.95, a + b - effect)}
        elif kind == "independence":
            outcomes = {"00": 0.0, "10": a, "01": b, "11": min(0.95, a + b)}
        elif kind == "prerequisite_a_to_b":
            outcomes = {"00": 0.0, "10": a, "01": 0.0, "11": min(0.95, a + effect)}
        elif kind == "prerequisite_b_to_a":
            outcomes = {"00": 0.0, "10": 0.0, "01": b, "11": min(0.95, b + effect)}
        elif kind == "redundancy":
            outcomes = {"00": 0.0, "10": a, "01": b, "11": max(a, b)}
        else:
            outcomes = {"00": 0.0, "10": a, "01": b, "11": min(0.95, a + b + effect)}
        gates = {arm: not (kind == "semantic_conflict" and arm == "11") for arm in ("00", "10", "01", "11")}
        surface.append({"surface_id": f"interaction-{index:04d}", "kind": kind, "outcomes": outcomes, "scientific_gates": gates, "cost": 1.0 + rng.random() * 4.0})
    return surface


def run_factorial_benchmark(*, blocks: int = 5000, seed: int = 7) -> dict[str, object]:
    classifications: dict[str, str] = {}
    estimates: dict[str, dict[str, float]] = {}
    for name, values in _CASES.items():
        engine = FactorialEngine()
        for index in range(blocks):
            engine.add_block(FactorialBlock(f"{name}-{index}", values))
        estimate = engine.estimate()
        classifications[name] = estimate.decision
        estimates[name] = {
            "gamma": estimate.gamma,
            "gamma_lcb": estimate.gamma_lcb,
            "gamma_ucb": estimate.gamma_ucb,
            "delta_a_given_b0": estimate.delta_a_given_b0,
            "delta_a_given_b1": estimate.delta_a_given_b1,
            "delta_b_given_a0": estimate.delta_b_given_a0,
            "delta_b_given_a1": estimate.delta_b_given_a1,
            "scientific_00": estimate.scientific_00,
            "scientific_10": estimate.scientific_10,
            "scientific_01": estimate.scientific_01,
            "scientific_11": estimate.scientific_11,
        }
    coverage = simulate_coverage(true_interaction=0.2, noise=0.04, blocks=blocks, repetitions=200, seed=seed)
    return {
        "classifications": classifications,
        "estimates": estimates,
        "coverage": coverage.coverage,
        "coverage_repetitions": coverage.repetitions,
    }
