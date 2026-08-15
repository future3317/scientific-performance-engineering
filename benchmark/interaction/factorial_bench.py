"""Synthetic factorial interaction pilot, separate from formal benchmark scoring."""

from __future__ import annotations

from core.acre.factorial import FactorialBlock, FactorialEngine, simulate_coverage
import random
from benchmark.families import family_instances, resolve_family_id
from benchmark.families.catalog import FAMILY_SPECS


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


def generate_family_interaction_surface(
    family_ids: tuple[str, ...] = ("h2d_pipeline", "compile"),
    *,
    count: int = 32,
    seed: int = 7,
) -> list[dict[str, object]]:
    """Build factorial surfaces from composable canonical family interventions.

    The old ``generate_interaction_surface`` remains a calibration fixture.
    This path attaches every surface to family instances and derives the four
    arms from intervention parameters, so hidden interaction views can share
    the same workload generator as atomic and boundary tasks.
    """
    if len(family_ids) != 2 or count < 1:
        raise ValueError("exactly two families and a positive count are required")
    resolved = tuple(resolve_family_id(item) for item in family_ids)
    left = family_instances(resolved[0], count=count, seed=seed)
    right = family_instances(resolved[1], count=count, seed=seed + 1)
    surfaces: list[dict[str, object]] = []
    for index, (a, b) in enumerate(zip(left, right)):
        # The mechanism score is deliberately bounded and deterministic.  It
        # is an environment generator, not a relation label supplied to the
        # estimator; the hidden verifier can recompute the four outcomes.
        a_effect = 0.08 + (sum(sum(ord(char) for char in str(v)) for v in a.parameters.values()) % 12) / 100.0
        b_effect = 0.08 + (sum(sum(ord(char) for char in str(v)) for v in b.parameters.values()) % 12) / 100.0
        interaction = ((index + seed) % 7 - 3) / 100.0
        outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": min(0.95, a_effect + b_effect + interaction)}
        surfaces.append({
            "surface_id": f"{resolved[0]}__{resolved[1]}-{seed:04d}-{index:04d}",
            "family_ids": list(resolved),
            "instance_ids": [a.instance_id, b.instance_id],
            "interventions": [list(FAMILY_SPECS[resolved[0]].interventions), list(FAMILY_SPECS[resolved[1]].interventions)],
            "outcomes": outcomes,
            "cost": 1.0 + ((index + seed) % 9) / 2.0,
        })
    return surfaces


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
