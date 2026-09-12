from __future__ import annotations

from benchmark.tests.test_acre_v04 import _rule, _state
from core.acre.router import ConservativeCausalRouter
from core.models import RelationSpec, RelationState, TaskContext


def _scientific_rule(rule_id: str):
    spec = _rule(rule_id, rule_id)
    return spec.__class__(
        rule_id=spec.rule_id, version=spec.version, parent=spec.parent,
        applicability=spec.applicability, intervention=spec.intervention,
        expected_mechanism=spec.expected_mechanism, evidence_requirements=spec.evidence_requirements,
        scientific_invariants=["preserved"], abstain_conditions=spec.abstain_conditions,
        relations=spec.relations, runtime_cost=spec.runtime_cost, provenance_policy=spec.provenance_policy,
    )


def test_unresolved_pairwise_screen_rejects_three_rule_superset() -> None:
    specs = [_scientific_rule(name) for name in ("a", "b", "c")]
    states = {spec.rule_id: _state(spec.rule_id) for spec in specs}
    decision = ConservativeCausalRouter(token_budget=8).route(
        specs, states, [], {}, TaskContext("x", {}, {}, {}, {}),
        require_higher_order_certificate=True,
    )
    assert len(decision.selected_rule_ids) < 3
    assert any("unknown_scientific_interaction" in reasons for reasons in decision.rejected_reasons.values())


def test_pairwise_conflict_rejects_superset_even_with_higher_order_flag() -> None:
    specs = [_scientific_rule(name) for name in ("a", "b", "c")]
    states = {spec.rule_id: _state(spec.rule_id) for spec in specs}
    relation = RelationSpec(
        "ab-conflict", 1, None, {"left": "a", "right": "b"}, "symmetric", "semantic_conflict",
        {"all": []}, {"source": "test"}, 0.1, [], {"required": True},
    )
    relation_state = RelationState("ab-conflict", 1, -0.2, {"lcb": -0.2}, "canonical")
    decision = ConservativeCausalRouter(token_budget=8).route(
        specs, states, [relation], {"ab-conflict": relation_state}, TaskContext("x", {}, {}, {}, {}),
        require_higher_order_certificate=True,
    )
    assert len(decision.selected_rule_ids) < 3
    assert any("hard_conflict" in reasons for reasons in decision.rejected_reasons.values())

