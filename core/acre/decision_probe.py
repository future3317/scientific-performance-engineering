"""Router-aware hypothetical acquisition probes."""

from __future__ import annotations

from typing import Any

from .acquisition import AcquisitionQuery
from .cegis import BoundaryObservation, StatisticalCEGIS
from .predicates import PredicateGrammar
from core.models import RuleSpec, RuleState
from .engine import AcreEngine


def decision_sensitivity_callback(queries: list[AcquisitionQuery]) -> Any:
    """Return a callback that compares router-facing bundles after two outcomes."""
    edge_order = tuple(sorted({query.edge_id for query in queries}))
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [
            {"path": f"workload.{key}", "type": "numeric"}
            for key in sorted({key for query in queries for key, value in (query.context.get("workload", {}) if isinstance(query.context.get("workload", {}), dict) else {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        ][:1] or [{"path": "_decision_probe", "type": "numeric"}],
        "max_depth": 2,
        "max_literals": 1,
    })
    learned_predicates: dict[str, dict[str, object]] = {}
    def predicate_for_edge(edge_id: str, state: dict[str, list[bool]]) -> dict[str, object]:
        values = state.get(edge_id, [])
        edge_queries = [item for item in queries if item.edge_id == edge_id]
        if not values or len(values) < 2:
            return {"all": []}
        if len(values) % 4 and edge_id in learned_predicates:
            return learned_predicates[edge_id]
        evidence = [BoundaryObservation(item.query_id, item.context, 1.0 if value else 0.0, bool(value), 1.0 if value else -1.0, 1.0 if value else 0.0) for item, value in zip(edge_queries[:8], values[:8])]
        positive = [item for item in evidence if item.positive_anchor()]
        negative = [item for item in evidence if item.certified_counterexample()]
        if not positive or not negative:
            learned_predicates[edge_id] = {"all": []}
            return learned_predicates[edge_id]
        synthesis = StatisticalCEGIS(grammar).synthesize(
            positive=positive,
            counterexamples=negative,
            parent_predicate=None,
            decision_contexts=[item.context for item in edge_queries],
        )
        learned_predicates[edge_id] = dict(synthesis.predicate or {"all": []})
        return learned_predicates[edge_id]

    def route_bundle(state: dict[str, list[bool]], context: dict[str, object], focus_edge: str | None = None) -> tuple[str, ...]:
        route_edges = (focus_edge,) if focus_edge is not None else edge_order
        specs = [RuleSpec(
            rule_id=edge_id, version=1, parent=None,
            applicability=predicate_for_edge(edge_id, state),
            intervention={"action": edge_id}, expected_mechanism="boundary",
            evidence_requirements=["boundary"], scientific_invariants=[],
            abstain_conditions={}, relations={}, runtime_cost={"tokens": 1},
            provenance_policy={"required": True},
        ) for edge_id in route_edges]
        states = {spec.rule_id: RuleState(
            rule_id=spec.rule_id, version=1, status="canonical",
            effect={"utility": 0.1}, confidence_sequence={"lcb": 0.1},
        ) for spec in specs}
        # The probe asks whether the next observation can change the
        # decision-relevant bundle; a small fixed bundle budget keeps this
        # hypothetical replay bounded while still using the production router.
        return AcreEngine(rule_specs=specs, rule_states=states).route(context, token_budget=min(2, len(specs))).selected_rule_ids

    def callback(query: AcquisitionQuery, observations: dict[str, list[bool]]) -> float:
        current = route_bundle(observations, dict(query.context), query.edge_id)
        for outcome in (True, False):
            hypothetical = {key: list(value) for key, value in observations.items()}
            hypothetical.setdefault(query.edge_id, []).append(outcome)
            if route_bundle(hypothetical, dict(query.context), query.edge_id) != current:
                return 1.0
        return 0.0

    return callback
