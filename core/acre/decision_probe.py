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
    predicate_cache: dict[tuple[str, tuple[bool, ...]], dict[str, object]] = {}
    route_cache: dict[tuple[tuple[tuple[str, str], ...], str], tuple[str, ...]] = {}
    def predicate_for_edge(edge_id: str, state: dict[str, list[bool]]) -> dict[str, object]:
        values = state.get(edge_id, [])
        # The CEGIS probe uses a bounded evidence window.  Once that window is
        # fixed, additional observations cannot change the hypothetical
        # predicate and should not trigger a full grammar enumeration.
        cache_key = (edge_id, tuple(values[:8]))
        if cache_key in predicate_cache:
            return predicate_cache[cache_key]
        edge_queries = [item for item in queries if item.edge_id == edge_id]
        if not values or len(values) < 2:
            return {"all": []}
        evidence = [BoundaryObservation(item.query_id, item.context, 1.0 if value else 0.0, bool(value), 1.0 if value else -1.0, 1.0 if value else 0.0) for item, value in zip(edge_queries[:8], values[:8])]
        positive = [item for item in evidence if item.positive_anchor()]
        negative = [item for item in evidence if item.certified_counterexample()]
        if not positive or not negative:
            predicate_cache[cache_key] = {"all": []}
            return predicate_cache[cache_key]
        synthesis = StatisticalCEGIS(grammar).synthesize(
            positive=positive,
            counterexamples=negative,
            parent_predicate=None,
            decision_contexts=[item.context for item in edge_queries],
        )
        predicate_cache[cache_key] = dict(synthesis.predicate or {"all": []})
        return predicate_cache[cache_key]

    def route_bundle(state: dict[str, list[bool]], context: dict[str, object], focus_edge: str | None = None) -> tuple[str, ...]:
        # Route the complete decision-relevant pool.  Restricting the
        # hypothetical to the queried edge would miss bundle changes caused
        # by prerequisites, conflicts, or token competition with other rules.
        predicates = {edge_id: predicate_for_edge(edge_id, state) for edge_id in edge_order}
        state_key = tuple(sorted((key, repr(value)) for key, value in predicates.items()))
        context_key = repr(sorted(context.items()))
        cache_key = (state_key, context_key)
        if cache_key in route_cache:
            return route_cache[cache_key]
        specs = [RuleSpec(
            rule_id=edge_id, version=1, parent=None,
            applicability=predicates[edge_id],
            intervention={"action": edge_id}, expected_mechanism="boundary",
            evidence_requirements=["boundary"], scientific_invariants=[],
            abstain_conditions={}, relations={}, runtime_cost={"tokens": 1},
            provenance_policy={"required": True},
        ) for edge_id in edge_order]
        states = {spec.rule_id: RuleState(
            rule_id=spec.rule_id, version=1, status="canonical",
            effect={"utility": 0.1}, confidence_sequence={"lcb": 0.1},
        ) for spec in specs}
        # The probe asks whether the next observation can change the
        # decision-relevant bundle; a small fixed bundle budget keeps this
        # hypothetical replay bounded while still using the production router.
        selected = AcreEngine(rule_specs=specs, rule_states=states).route(context, token_budget=min(2, len(specs))).selected_rule_ids
        route_cache[cache_key] = selected
        return selected

    def callback(query: AcquisitionQuery, observations: dict[str, list[bool]]) -> float:
        current = route_bundle(observations, dict(query.context), query.edge_id)
        for outcome in (True, False):
            hypothetical = {key: list(value) for key, value in observations.items()}
            hypothetical.setdefault(query.edge_id, []).append(outcome)
            if route_bundle(hypothetical, dict(query.context), query.edge_id) != current:
                return 1.0
        return 0.0

    return callback
