"""Two disjoint, deterministic BoundaryBench families for ACRE-v0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.acre.predicates import PredicateGrammar
from .evaluator import sealed_errors


@dataclass(frozen=True)
class BoundaryCase(BoundaryObservation):
    expected_applicable: bool = False


def _case(case_id: str, value: float, path: str, mechanism: str, positive: bool) -> BoundaryCase:
    context = {"workload": {"mechanism": mechanism, path: value}}
    effect = 0.2 if positive else -0.1
    return BoundaryCase(case_id, context, effect, True, effect - 0.02, effect + 0.02, positive)


def family_cases(family: str) -> dict[str, list[BoundaryCase]]:
    if family == "graph_cache_geometry_motion":
        return {
            "representative_pool": [_case("G-REP-01", 0.01, "geometry_displacement", "graph_cache", True), _case("G-REP-02", 0.03, "geometry_displacement", "graph_cache", True)],
            "query_pool": [_case("G-QUERY-01", 0.08, "geometry_displacement", "graph_cache", False)],
            "sealed_test_pool": [_case("G-SEALED-01", 0.04, "geometry_displacement", "graph_cache", True), _case("G-SEALED-02", 0.07, "geometry_displacement", "graph_cache", False)],
        }
    if family == "compile_horizon":
        return {
            "representative_pool": [_case("C-REP-01", 128, "logical_steps", "compile", True), _case("C-REP-02", 256, "logical_steps", "compile", True)],
            "query_pool": [_case("C-QUERY-01", 64, "logical_steps", "compile", False)],
            "sealed_test_pool": [_case("C-SEALED-01", 112, "logical_steps", "compile", True), _case("C-SEALED-02", 80, "logical_steps", "compile", False)],
        }
    raise ValueError(f"unknown BoundaryBench family: {family}")


def _grammar_for(family: str) -> PredicateGrammar:
    path = "workload.geometry_displacement" if family == "graph_cache_geometry_motion" else "workload.logical_steps"
    root = Path(__file__).resolve().parents[2]
    grammar = json.loads((root / "assets" / "predicate_grammar.json").read_text(encoding="utf-8"))
    grammar["features"] = [feature for feature in grammar["features"] if feature["path"] in {path, "workload.mechanism"}]
    grammar["max_literals"] = 2
    return PredicateGrammar.from_dict(grammar)


def run_boundary_family(family: str) -> dict[str, Any]:
    pools = family_cases(family)
    representative = [item for item in pools["representative_pool"] if item.positive_anchor()]
    query = pools["query_pool"]
    counterexamples = [item for item in query if item.certified_counterexample()]
    parent = {"equals": {"workload.mechanism": "graph_cache" if family.startswith("graph") else "compile"}}
    result = StatisticalCEGIS(_grammar_for(family)).synthesize(
        positive=representative,
        counterexamples=counterexamples,
        parent_predicate=parent,
    )
    errors = sealed_errors(result.predicate, pools["sealed_test_pool"])
    return {
        "family": family,
        "status": result.status,
        "predicate": result.predicate,
        "sealed_errors": errors,
        "result": result.to_dict(),
        "pool_sizes": {name: len(items) for name, items in pools.items()},
    }
