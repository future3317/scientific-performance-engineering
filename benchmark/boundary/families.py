"""Deterministic BoundaryBench views over the canonical family catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.acre.predicates import PredicateGrammar
from benchmark.families import family_views
from .evaluator import sealed_errors


@dataclass(frozen=True)
class BoundaryCase(BoundaryObservation):
    expected_applicable: bool = False


def family_cases(family: str, *, surface_count: int | None = None, seed: int = 0) -> dict[str, list[BoundaryCase]]:
    # Canonical family views are generated from benchmark/families.  The
    # historical names below remain calibration aliases for ACRE-v0 tests.
    aliases = {"graph_cache_geometry_motion": "graph_cache", "compile_horizon": "compile"}
    if family in aliases:
        views = family_cases(aliases[family], surface_count=surface_count or 24, seed=seed)
        return {
            "representative_pool": views["representative_pool"],
            "query_pool": views["active_query_pool"],
            "sealed_test_pool": views["sealed_boundary_pool"],
        }
    canonical = {"compile", "graph_cache", "h2d_pipeline", "checkpoint", "scalar_sync"}
    if family in canonical and surface_count is None:
        surface_count = 24
    if family in canonical and surface_count is not None:
        views = family_views(family, count=surface_count, seed=seed)
        def convert(item: Any) -> BoundaryCase:
            params = dict(item.parameters)
            if family == "compile":
                mechanism, path = "compile", "logical_steps"
            elif family == "graph_cache":
                mechanism, path = "graph_cache", "geometry_displacement"
            elif family == "h2d_pipeline":
                mechanism, path = "h2d_pipeline", "worker_count"
            elif family == "checkpoint":
                mechanism, path = "checkpoint", "memory_pressure"
            else:
                mechanism, path = "scalar_sync", "scalar_syncs_per_step"
            positive = bool(item.applicable)
            effect = 0.2 if positive else -0.1
            context = {"workload": {"mechanism": mechanism, **params}}
            return BoundaryCase(item.instance_id, context, effect, True, effect - 0.02, effect + 0.02, positive)
        converted = [convert(item) for values in views.values() for item in values]
        positives = [item for item in converted if item.positive_anchor()]
        negatives = [item for item in converted if item.certified_counterexample()]
        representative_count = max(1, min(len(positives), surface_count // 3))
        query_count = max(1, min(len(negatives), surface_count // 3))
        if representative_count == 1:
            representative = positives[:1]
        else:
            representative = [positives[round(index * (len(positives) - 1) / (representative_count - 1))] for index in range(representative_count)]
        query = negatives[:query_count]
        used = {item.observation_id for item in representative + query}
        sealed = [item for item in converted if item.observation_id not in used]
        return {"representative_pool": representative, "active_query_pool": query, "sealed_boundary_pool": sealed}
    raise ValueError(f"unknown BoundaryBench family: {family}")


def _grammar_for(family: str) -> PredicateGrammar:
    canonical = {"graph_cache_geometry_motion": "graph_cache", "compile_horizon": "compile"}.get(family, family)
    from benchmark.families import family_predicate_grammar
    grammar = family_predicate_grammar(canonical)
    if not grammar:
        raise ValueError(f"family {canonical} has no registered predicate grammar")
    grammar["max_literals"] = min(3, int(grammar.get("max_literals", 3)))
    return PredicateGrammar.from_dict(grammar)


def run_boundary_family(family: str, *, surface_count: int = 24, seed: int = 0) -> dict[str, Any]:
    canonical_family = {"graph_cache_geometry_motion": "graph_cache", "compile_horizon": "compile"}.get(family, family)
    if canonical_family in {"compile", "graph_cache", "h2d_pipeline", "checkpoint", "scalar_sync"}:
        pools = family_cases(canonical_family, surface_count=surface_count, seed=seed)
        path = {
            "compile": "workload.logical_steps",
            "graph_cache": "workload.geometry_displacement",
            "h2d_pipeline": "workload.worker_count",
            "checkpoint": "workload.memory_pressure",
            "scalar_sync": "workload.scalar_syncs_per_step",
        }[canonical_family]
        mechanism = canonical_family
        grammar_path = path
    else:
        pools = family_cases(family)
        grammar_path = "workload.geometry_displacement" if family == "graph_cache_geometry_motion" else "workload.logical_steps"
        mechanism = "graph_cache" if family.startswith("graph") else "compile"
    representative = [item for item in pools["representative_pool"] if item.positive_anchor()]
    query = pools.get("active_query_pool", pools.get("query_pool", []))
    counterexamples = [item for item in query if item.certified_counterexample()]
    # Fit on a bounded evidence slice; all generated surfaces remain in the
    # sealed pool.  This keeps grammar enumeration finite as context count
    # grows from the 24-case calibration to the 100--500 pilot.
    def fit_slice(items: list[BoundaryCase], limit: int = 16) -> list[BoundaryCase]:
        if len(items) <= limit:
            return items
        return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]

    representative = fit_slice(representative)
    counterexamples = fit_slice(counterexamples)
    parent = {"equals": {"workload.mechanism": mechanism}}
    grammar = _grammar_for(canonical_family)
    result = StatisticalCEGIS(grammar).synthesize(
        positive=representative,
        counterexamples=counterexamples,
        parent_predicate=parent,
    )
    sealed = pools.get("sealed_boundary_pool", pools.get("sealed_test_pool", []))
    errors = sealed_errors(result.predicate, sealed)
    return {
        "family": family,
        "status": result.status,
        "state": result.status,
        "predicate": result.predicate,
        "sealed_errors": errors,
        "result": result.to_dict(),
        "pool_sizes": {name: len(items) for name, items in pools.items()},
    }
