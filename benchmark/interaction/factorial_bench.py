"""Synthetic factorial interaction pilot, separate from formal benchmark scoring."""

from __future__ import annotations

from core.acre.factorial import FactorialBlock, FactorialEngine, ThreeWayBlock, estimate_higher_order, simulate_coverage
from core.acre.relation import RelationIdentifier
import random
from benchmark.families import family_instances, resolve_family_id
from benchmark.families.catalog import FAMILY_SPECS, CompositionSpec, InteractionOracle
from core.acre.factorial import CANONICAL_RELATIONS, canonical_relation_label


_CASES = {
    "synergy": {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9},
    "antagonism": {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
    "independence": {"00": 0.0, "10": 0.3, "01": 0.4, "11": 0.7},
    "prerequisite": {"00": 0.0, "10": 0.4, "01": 0.0, "11": 0.9},
}


def build_three_way_oracle(*, residual: float, baseline: float = -0.1) -> dict[str, float]:
    """Construct 2^3 cells with an exact raw inclusion-exclusion residual."""
    if not -1.0 <= residual / 8.0 <= 1.0:
        raise ValueError("residual must produce a normalized value in [-1, 1]")
    # Choose the two cells that carry the contrast so every requested target
    # in the normalized [-1, 1] domain remains inside the utility bounds.
    sign = 1.0 if residual >= 0.0 else -1.0
    anchor = sign if residual else float(baseline)
    cells = {arm: 0.0 for arm in ("000", "001", "010", "011", "100", "101", "110", "111")}
    cells["111"] = anchor
    cells["000"] = anchor - residual
    if not -1.0 <= cells["111"] <= 1.0:
        raise ValueError("requested residual produces an out-of-range u111")
    return cells


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
    oracle = InteractionOracle(CompositionSpec(resolved[0], resolved[1]))
    for index, (a, b) in enumerate(zip(left, right)):
        sign_flip = a.parameters.get("worker_count", 0) > 6 and b.parameters.get("dynamic_shape_rate", 0) > 0.5
        semantic_conflict = a.parameters.get("worker_count", 0) > 6 and b.parameters.get("dynamic_shape_rate", 0) <= 0.5
        redundancy = a.parameters.get("worker_count", 0) <= 4 and b.parameters.get("dynamic_shape_rate", 0) <= 0.2
        context = {"sign_flip": sign_flip, "semantic_conflict": semantic_conflict, "redundancy": redundancy}
        base_context = dict(context)
        base_context["sign_flip"] = False
        base_context["force_synergy"] = sign_flip
        result = oracle.evaluate(a, b, base_context)
        outcomes = result["outcomes"]
        relation = result["hidden_relation"]
        contexts = [{"name": "baseline", "outcomes": outcomes}]
        if context["sign_flip"]:
            shifted = oracle.evaluate(a, b, {"sign_flip": False, "force_antagonism": True})
            contexts.append({"name": "shifted", "outcomes": shifted["outcomes"], "hidden_relation": shifted["hidden_relation"]})
            relation = "context_dependent_relation"
        surfaces.append({
            "surface_id": f"{resolved[0]}__{resolved[1]}-{seed:04d}-{index:04d}",
            "family_ids": list(resolved),
            "instance_ids": [a.instance_id, b.instance_id],
            "interventions": [list(FAMILY_SPECS[resolved[0]].interventions), list(FAMILY_SPECS[resolved[1]].interventions)],
            "outcomes": outcomes,
            "hidden_relation": relation,
            "scientific_gates": result["scientific_gates"],
            "contexts": contexts,
            "context_variant": contexts[1]["outcomes"] if len(contexts) > 1 else None,
            "higher_order_residual": None,
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
        context_rows = surface.get("contexts") or [{"name": "baseline", "outcomes": values}]
        generated_by_context: dict[str, list[FactorialBlock]] = {}
        for context_index, context_row in enumerate(context_rows):
            context_name = str(context_row.get("name", f"context-{context_index}"))
            context_values = context_row["outcomes"]
            generated: list[FactorialBlock] = []
            for block_index in range(max_blocks):
                outcomes = {arm: max(-1.0, min(1.0, float(value) + rng.uniform(-0.02, 0.02))) for arm, value in context_values.items()}
                gates = dict(surface.get("scientific_gates") or {arm: True for arm in ("00", "10", "01", "11")})
                generated.append(FactorialBlock(f"{surface['surface_id']}-{context_name}-{block_index}", outcomes, gates))
            generated_by_context[context_name] = generated
        chosen = None
        estimates: list[dict[str, object]] = []
        for block_count in blocks:
            context_estimates = {}
            for context_name, generated in generated_by_context.items():
                engine = FactorialEngine(delta=0.05, practical_margin=0.15, look_count=len(blocks))
                for block in generated[:block_count]:
                    engine.add_block(block)
                context_estimates[context_name] = engine.estimate()
            estimate = context_estimates["baseline"]
            identified = RelationIdentifier(practical_margin=0.08, equivalence_margin=0.35).identify(context_estimates)
            row = {"blocks": block_count, "decision": estimate.decision, "gamma": estimate.gamma, "gamma_lcb": estimate.gamma_lcb, "gamma_ucb": estimate.gamma_ucb}
            row["contrast_intervals"] = {key: list(value) for key, value in estimate.contrast_intervals.items()}
            row["relation_identifier"] = identified.decision
            row["context_decisions"] = dict(identified.context_decisions)
            estimates.append(row)
            if chosen is None and identified.decision != "unresolved":
                chosen = (block_count, estimate, identified)
        if chosen is None:
            stopping_blocks, estimate = None, context_estimates["baseline"]
            identified = RelationIdentifier(practical_margin=0.08, equivalence_margin=0.35).identify(context_estimates)
        else:
            stopping_blocks, estimate, identified = chosen
        predicted = identified.decision
        classifications[predicted] = classifications.get(predicted, 0) + 1
        hidden_canonical = canonical_relation_label(hidden)
        confusion.setdefault(hidden_canonical, {})[predicted] = confusion.setdefault(hidden_canonical, {}).get(predicted, 0) + 1
        details.append({
            "surface_id": surface["surface_id"],
            "hidden_relation": hidden,
            "predicted_relation": predicted,
            "gamma": estimate.gamma,
            "gamma_lcb": estimate.gamma_lcb,
            "gamma_ucb": estimate.gamma_ucb,
            "contrast_intervals": {key: list(value) for key, value in estimate.contrast_intervals.items()},
            "context_decisions": dict(identified.context_decisions),
            "applicability_predicate": identified.applicability_predicate,
            "stopping_blocks": stopping_blocks,
            "experiment_cost": float(surface["cost"]) * float(stopping_blocks or max(blocks)),
            "contexts": surface.get("contexts", []),
            "higher_order_residual": surface.get("higher_order_residual", 0.0),
            "estimates": estimates,
        })
    false_by_relation: dict[str, list[bool]] = {}
    unresolved_by_relation: dict[str, list[bool]] = {}
    for item in details:
        canonical_hidden = canonical_relation_label(str(item["hidden_relation"]))
        false_by_relation.setdefault(str(canonical_hidden), []).append(item["predicted_relation"] != canonical_hidden)
        unresolved_by_relation.setdefault(str(canonical_hidden), []).append(item["predicted_relation"] == "unresolved")
    return {
        "surface_count": count,
        "block_schedule": list(blocks),
        "classifications": classifications,
        "hidden_relation_counts": {label: list(hidden_labels.values()).count(label) for label in sorted(set(hidden_labels.values()))},
        "confusion_matrix": confusion,
        "surface_results": details,
        "false_relation_rate": sum(item["predicted_relation"] != canonical_relation_label(str(item["hidden_relation"])) for item in details) / len(details),
        "false_relation_rate_by_relation": {key: sum(values) / len(values) for key, values in false_by_relation.items()},
        "unresolved_rate_by_relation": {key: sum(values) / len(values) for key, values in unresolved_by_relation.items()},
        "unresolved_rate": sum(item["predicted_relation"] == "unresolved" for item in details) / len(details),
        "median_blocks_to_confirmation": sorted(item["stopping_blocks"] for item in details if item["stopping_blocks"] is not None)[len([item for item in details if item["stopping_blocks"] is not None]) // 2] if any(item["stopping_blocks"] is not None for item in details) else None,
        "experiment_cost": sum(float(item["experiment_cost"]) for item in details),
    }


def run_interaction_power_curve(
    *,
    blocks: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024),
    repetitions: int = 12,
    seed: int = 7,
    practical_margin: float = 0.05,
) -> dict[str, object]:
    """Calibrate confirmation power across effect strength and noise."""
    # ``effect`` is the target normalized factorial contrast gamma.  Construct
    # the latent cells from that target first; no clipping is applied to the
    # latent surface, so the realized contrast remains auditable.
    strengths = {"near-null": 0.0, "near-margin": 0.06, "moderate": 0.12, "strong": 0.20}
    noises = {"low": 0.005, "medium": 0.03, "high": 0.08}
    rows: list[dict[str, object]] = []
    for strength_name, effect in strengths.items():
        for noise_name, noise in noises.items():
            confirmations: list[int] = []
            stopping: list[int] = []
            for repetition in range(repetitions):
                rng = random.Random(seed * 1009 + repetition * 97 + len(rows))
                chosen: int | None = None
                for block_count in blocks:
                    engine = FactorialEngine(delta=0.05, practical_margin=practical_margin, look_count=len(blocks))
                    for block_index in range(block_count):
                        outcomes = {"00": -0.20, "10": -0.10, "01": -0.10, "11": 4.0 * effect}
                        outcomes = {arm: max(-1.0, min(1.0, value + rng.uniform(-noise, noise))) for arm, value in outcomes.items()}
                        engine.add_block(FactorialBlock(f"{strength_name}-{noise_name}-{repetition}-{block_index}", outcomes))
                    if engine.estimate().decision == "confirmed_synergy":
                        chosen = block_count
                        break
                confirmations.append(int(chosen is not None))
                if chosen is not None:
                    stopping.append(chosen)
            rows.append({
                "effect_strength": strength_name,
                "target_gamma": effect,
                "realized_gamma": (4.0 * effect - (-0.10) - (-0.10) + (-0.20)) / 4.0,
                "absolute_gamma_minus_margin": abs(effect) - practical_margin,
                "noise": noise_name,
                "noise_scale": noise,
                "correct_confirmation_probability": sum(confirmations) / len(confirmations),
                "expected_blocks_to_confirmation": sum(stopping) / len(stopping) if stopping else None,
                "repetitions": repetitions,
            })
    return {"blocks": list(blocks), "practical_margin": practical_margin, "results": rows}


def run_higher_order_benchmark(*, count: int = 20, seed: int = 7, blocks: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)) -> dict[str, object]:
    """Measure a genuine 2^3 residual over three composable interventions."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    details: list[dict[str, object]] = []
    for index in range(count):
        # Sweep normalized residual magnitudes around the practical margin;
        # ``build_three_way_oracle`` receives the corresponding raw value.
        normalized_target = (0.0, 0.03, 0.07, 0.12, 0.20)[index % 5]
        residual = normalized_target * 8.0
        if index % 2:
            residual = -residual
        latent = build_three_way_oracle(residual=residual)
        chosen = None
        estimates = []
        for block_count in blocks:
            sample = [ThreeWayBlock(f"three-{index}-{j}", {arm: max(-1.0, min(1.0, value + rng.uniform(-0.01, 0.01))) for arm, value in latent.items()}) for j in range(block_count)]
            estimate = estimate_higher_order(sample, delta=0.05, look_count=len(blocks), practical_margin=0.05)
            estimates.append({"blocks": block_count, "raw_residual": estimate.raw_residual, "normalized_residual": estimate.normalized_residual, "residual_lcb": estimate.residual_lcb, "residual_ucb": estimate.residual_ucb, "status": estimate.status})
            if chosen is None and estimate.status != "unresolved":
                chosen = estimate
        final = chosen or estimate
        details.append({"surface_id": f"three-way-{index:04d}", "hidden_raw_residual": residual, "hidden_normalized_residual": residual / 8.0, "raw_residual": final.raw_residual, "normalized_residual": final.normalized_residual, "predicted_raw_residual": final.raw_residual, "predicted_normalized_residual": final.normalized_residual, "residual_lcb": final.residual_lcb, "residual_ucb": final.residual_ucb, "bundle_certificate": {"residual_lcb": final.residual_lcb, "residual_ucb": final.residual_ucb, "status": final.status}, "stopping_blocks": final.blocks if chosen else None, "estimates": estimates})
    return {"surface_count": count, "block_schedule": list(blocks), "results": details, "confirmed_rate": sum(item["stopping_blocks"] is not None for item in details) / count}


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
