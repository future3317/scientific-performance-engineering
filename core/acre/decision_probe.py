"""Router-aware hypothetical acquisition probes."""

from __future__ import annotations

from typing import Any

from .acquisition import AcquisitionQuery
from .cegis import BoundaryObservation, StatisticalCEGIS
from .predicates import PredicateGrammar
from core.predicates import match_predicate


def decision_sensitivity_callback(queries: list[AcquisitionQuery]) -> Any:
    """Return a callback that compares router-facing bundles after two outcomes."""
    edge_order = tuple(sorted({query.edge_id for query in queries}))
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [
            {"path": f"workload.{key}", "type": "numeric"}
            for key in sorted({key for query in queries for key, value in (query.context.get("workload", {}) if isinstance(query.context.get("workload", {}), dict) else {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        ] or [{"path": "_decision_probe", "type": "numeric"}],
        "max_depth": 2,
        "max_literals": 1,
    })
    decision_cache: dict[tuple[str, tuple[tuple[str, tuple[bool, ...]], ...]], bool] = {}

    def callback(query: AcquisitionQuery, observations: dict[str, list[bool]]) -> float:
        def edge_decision(edge_id: str, state: dict[str, list[bool]]) -> bool:
            key = (edge_id, tuple(sorted((name, tuple(values)) for name, values in state.items())))
            if key in decision_cache:
                return decision_cache[key]
            values = state.get(edge_id, [])
            if not values:
                decision_cache[key] = False
                return False
            if len(values) != 2:
                decision = sum(values) * 2 >= len(values)
                decision_cache[key] = decision
                return decision
            edge_queries = [item for item in queries if item.edge_id == edge_id]
            evidence = [BoundaryObservation(item.query_id, item.context, 1.0 if value else 0.0, bool(value), 1.0 if value else -1.0, 1.0 if value else 0.0) for item, value in zip(edge_queries[:8], values[:8])]
            positive = [item for item in evidence if item.positive_anchor()]
            negative = [item for item in evidence if item.certified_counterexample()]
            decision = False
            if positive and negative:
                synthesis = StatisticalCEGIS(grammar).synthesize(positive=positive, counterexamples=negative, parent_predicate=None, decision_contexts=[item.context for item in edge_queries])
                decision = synthesis.predicate is not None and any(match_predicate(synthesis.predicate, item.context) for item in edge_queries)
            if not decision:
                decision = sum(values) * 2 >= len(values)
            decision_cache[key] = decision
            return decision

        def bundle(state: dict[str, list[bool]]) -> tuple[str, ...]:
            return tuple(edge_id for edge_id in edge_order if edge_decision(edge_id, state))

        current = bundle(observations)
        for outcome in (True, False):
            hypothetical = {key: list(value) for key, value in observations.items()}
            hypothetical.setdefault(query.edge_id, []).append(outcome)
            if bundle(hypothetical) != current:
                return 1.0
        return 0.0

    return callback
