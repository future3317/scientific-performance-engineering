#!/usr/bin/env python3
"""Focused ACRE-v0 grammar, CEGIS, and BoundaryBench checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.boundary.cegis import BoundaryObservation, StatisticalCEGIS
from benchmark.boundary.families import family_cases, run_boundary_family
from core.acre.predicate_synthesis import PredicateGrammar
from core.predicates import match_predicate


def main() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [
            {"path": "workload.threshold", "type": "numeric"},
            {"path": "workload.kind", "type": "categorical"},
        ],
        "max_depth": 2,
        "max_literals": 2,
    })
    contexts = [
        {"workload": {"threshold": 1, "kind": "a"}},
        {"workload": {"threshold": 3, "kind": "b"}},
    ]
    candidates = grammar.candidates(contexts)
    assert {2.0} <= {
        atom["compare"]["workload.threshold"]["lte"]
        for predicate in candidates
        for atom in predicate.get("all", [predicate])
        if "compare" in atom and "lte" in atom["compare"].get("workload.threshold", {})
    }
    assert any("not" in predicate for predicate in candidates)
    assert any("any" in predicate for predicate in candidates)

    synthesizer = StatisticalCEGIS(grammar)
    result = synthesizer.synthesize(
        positive=[BoundaryObservation("p", contexts[0], 0.2, True, 0.1, 0.3)],
        counterexamples=[BoundaryObservation("c", contexts[1], -0.1, True, -0.3, -0.05)],
        parent_predicate=None,
    )
    assert result.status == "accepted", result
    assert result.predicate is not None
    assert match_predicate(result.predicate, contexts[0])
    assert not match_predicate(result.predicate, contexts[1])
    assert result.synthesizer_version

    uncertain = synthesizer.synthesize(
        positive=[BoundaryObservation("p", contexts[0], 0.2, True, 0.1, 0.3)],
        counterexamples=[BoundaryObservation("u", contexts[1], -0.01, True, -0.1, 0.1)],
        parent_predicate=None,
    )
    assert uncertain.status == "no_consistent_hypothesis"
    assert uncertain.predicate is None
    assert uncertain.certified_counterexamples == ()

    impossible = synthesizer.synthesize(
        positive=[BoundaryObservation("p", contexts[0], 0.2, True, 0.1, 0.3)],
        counterexamples=[BoundaryObservation("c", contexts[0], -0.1, False, -0.2, -0.05)],
        parent_predicate=None,
    )
    assert impossible.status == "no_consistent_hypothesis"

    for family in ("graph_cache_geometry_motion", "compile_horizon"):
        report = run_boundary_family(family)
        assert report["status"] == "accepted", report
        assert report["sealed_errors"] == 0, report
        pools = family_cases(family)
        ids = [item.observation_id for pool in pools.values() for item in pool]
        assert len(ids) == len(set(ids))

    print("ACRE CEGIS boundary fixtures: ok")


if __name__ == "__main__":
    main()
