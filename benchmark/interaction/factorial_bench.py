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
        mode = (index + seed) % 8
        if mode == 0:  # compositional synergy
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": min(0.95, a_effect + b_effect + 0.85)}
        elif mode == 1:  # destructive interference
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": max(-0.95, a_effect + b_effect - 0.75)}
        elif mode == 2:  # independent additive effects
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": min(0.95, a_effect + b_effect)}
        elif mode == 3:  # B is useful only when A is present
            outcomes = {"00": 0.0, "10": a_effect, "01": 0.0, "11": min(0.95, a_effect + 0.9)}
        elif mode == 4:  # A is useful only when B is present
            outcomes = {"00": 0.0, "10": 0.0, "01": b_effect, "11": min(0.95, b_effect + 0.9)}
        elif mode == 5:  # redundant interventions
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": max(a_effect, b_effect)}
        elif mode == 6:  # semantic conflict: the joint arm is invalid
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": 0.0}
        else:  # a sign flip is represented by a hidden context variant
            outcomes = {"00": 0.0, "10": a_effect, "01": b_effect, "11": min(0.95, a_effect + b_effect + 0.45)}
        relation = (
            ("synergy", "antagonism", "independence", "prerequisite_a_to_b",
             "prerequisite_b_to_a", "redundancy", "semantic_conflict",
             "context_dependent_sign_flip")[mode]
        )
        surfaces.append({
            "surface_id": f"{resolved[0]}__{resolved[1]}-{seed:04d}-{index:04d}",
            "family_ids": list(resolved),
            "instance_ids": [a.instance_id, b.instance_id],
            "interventions": [list(FAMILY_SPECS[resolved[0]].interventions), list(FAMILY_SPECS[resolved[1]].interventions)],
            "outcomes": outcomes,
            "hidden_relation": relation,
            "context_variant": {arm: -value for arm, value in outcomes.items()} if mode == 7 else None,
            "higher_order_residual": 0.18 if mode == 7 else 0.0,
            "cost": 1.0 + ((index + seed) % 9) / 2.0,
        })
    return surfaces


def run_family_factorial_benchmark(*, count: int = 100, seed: int = 7, blocks: tuple[int, ...] = (8, 16, 32, 64, 128)) -> dict[str, object]:
    """Run sequential confirmation on family-derived hidden surfaces."""
    if count < 1 or not blocks or any(item < 2 for item in blocks):
        raise ValueError("count must be positive and blocks must contain values >= 2")
    surfaces = generate_family_interaction_surface(("h2d_pipeline", "compile"), count=count, seed=seed)
    classifications: dict[str, int] = {}
    hidden_labels: dict[str, str] = {}
    details: list[dict[str, object]] = []
    confusion: dict[str, dict[str, int]] = {}
    for surface in surfaces:
        values = surface["outcomes"]
        hidden = str(surface["hidden_relation"])
        hidden_labels[str(surface["surface_id"])] = hidden
        rng = random.Random(seed * 1009 + len(details))
        max_blocks = max(blocks)
        generated: list[FactorialBlock] = []
        for block_index in range(max_blocks):
            outcomes = {arm: max(-1.0, min(1.0, float(value) + rng.uniform(-0.02, 0.02))) for arm, value in values.items()}
            gates = {arm: not (hidden == "semantic_conflict" and arm == "11") for arm in ("00", "10", "01", "11")}
            generated.append(FactorialBlock(f"{surface['surface_id']}-{block_index}", outcomes, gates))
        chosen = None
        estimates: list[dict[str, object]] = []
        for block_count in blocks:
            engine = FactorialEngine(delta=0.2, practical_margin=0.1)
            for block in generated[:block_count]:
                engine.add_block(block)
            estimate = engine.estimate()
            row = {"blocks": block_count, "decision": estimate.decision, "gamma": estimate.gamma, "gamma_lcb": estimate.gamma_lcb, "gamma_ucb": estimate.gamma_ucb}
            estimates.append(row)
            if chosen is None and estimate.decision != "unresolved":
                chosen = (block_count, estimate)
        if chosen is None:
            stopping_blocks, estimate = None, engine.estimate()
        else:
            stopping_blocks, estimate = chosen
        predicted = estimate.decision
        classifications[predicted] = classifications.get(predicted, 0) + 1
        confusion.setdefault(hidden, {})[predicted] = confusion.setdefault(hidden, {}).get(predicted, 0) + 1
        details.append({
            "surface_id": surface["surface_id"],
            "hidden_relation": hidden,
            "predicted_relation": predicted,
            "gamma": estimate.gamma,
            "gamma_lcb": estimate.gamma_lcb,
            "gamma_ucb": estimate.gamma_ucb,
            "stopping_blocks": stopping_blocks,
            "experiment_cost": float(surface["cost"]) * float(stopping_blocks or max(blocks)),
            "estimates": estimates,
        })
    return {
        "surface_count": count,
        "block_schedule": list(blocks),
        "classifications": classifications,
        "hidden_relation_counts": {label: list(hidden_labels.values()).count(label) for label in sorted(set(hidden_labels.values()))},
        "confusion_matrix": confusion,
        "surface_results": details,
        "false_relation_rate": sum(item["predicted_relation"] not in {item["hidden_relation"], "confirmed_independence"} for item in details) / len(details),
        "unresolved_rate": sum(item["predicted_relation"] == "unresolved" for item in details) / len(details),
        "median_blocks_to_confirmation": sorted(item["stopping_blocks"] for item in details if item["stopping_blocks"] is not None)[len([item for item in details if item["stopping_blocks"] is not None]) // 2] if any(item["stopping_blocks"] is not None for item in details) else None,
        "experiment_cost": sum(float(item["experiment_cost"]) for item in details),
    }


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
