from __future__ import annotations

from core.acre.router import ConservativeCausalRouter
from core.models import RelationSpec, RelationState, RuleSpec, RuleState, TaskContext


def rule(rule_id: str, utility: float, *, sensitive: bool = False, applicability=None) -> RuleSpec:
    return RuleSpec(
        rule_id, 1, None, applicability or {"equals": {}}, {"action": rule_id}, "mechanism", ["evidence"],
        ["physics"] if sensitive else [], {}, {}, {"tokens": 1.0}, {"required": True},
    )


def states(values: dict[str, float]) -> dict[str, RuleState]:
    return {rule_id: RuleState(rule_id, 1, status="canonical", effect={"lower_utility": value}) for rule_id, value in values.items()}


def test_router_enforces_prerequisites_conflicts_and_unknown_sensitive_edges() -> None:
    specs = [rule("base", 0.55, sensitive=True), rule("specialized", 0.75, sensitive=True), rule("conflict", 0.2)]
    relation = RelationSpec("base-specialized", 1, None, ["base", "specialized"], "prerequisite", {"equals": {}}, {"direction": "a_to_b"}, 0.05, [], {"required": True})
    conflict = RelationSpec("base-conflict", 1, None, ["base", "conflict"], "semantic_conflict", {"equals": {}}, {"contrast": "hard"}, 0.05, [], {"required": True})
    router = ConservativeCausalRouter(token_budget=8, lambda_tokens=0.01)
    decision = router.route(specs, states({"base": 0.55, "specialized": 0.75, "conflict": 0.2}), [relation, conflict], {"base-specialized": RelationState("base-specialized", 1, 0.2, {"lcb": 0.2}, status="canonical"), "base-conflict": RelationState("base-conflict", 1, 0.0, {}, status="canonical")}, {})
    assert decision.selected_rule_ids == ("base", "specialized")
    assert "conflict" not in decision.selected_rule_ids

    unknown_specs = [rule("base", 0.55, sensitive=True), rule("specialized", 0.4, sensitive=True)]
    unknown = router.route(
        unknown_specs,
        states({"base": 0.55, "specialized": 0.4}),
        [],
        {},
        {},
    )
    assert unknown.selected_rule_ids == ("base",)
    assert "unknown_scientific_interaction" in unknown.rejected_reasons["specialized"]


def test_pairwise_lower_bound_changes_conservative_bundle_score() -> None:
    specs = [rule("a", 0.5), rule("b", 0.5)]
    state = states({"a": 0.5, "b": 0.5})
    router = ConservativeCausalRouter(token_budget=4, lambda_tokens=0.0, zeta=0.01)
    independent = router.route(specs, state, [], {}, {})
    relation = RelationSpec("a-b", 1, None, ["a", "b"], "synergy", {"equals": {}}, {"contrast": "gamma"}, 0.05, [], {"required": True})
    relation_state = RelationState("a-b", 1, 0.4, {"lcb": 0.4}, status="canonical")
    synergy = router.route(specs, state, [relation], {"a-b": relation_state}, {})
    assert synergy.objective > independent.objective
    assert synergy.selected_rule_ids == ("a", "b")


def test_router_applies_typed_applicability_predicate() -> None:
    specs = [
        rule("graph", 0.8, applicability={"equals": {"workload": "graph"}}),
        rule("compile", 0.7, applicability={"equals": {"workload": "compile"}}),
    ]
    decision = ConservativeCausalRouter(token_budget=3).route(specs, states({"graph": 0.8, "compile": 0.7}), [], {}, TaskContext("runtime", {"workload": "graph"}, {}, {}, {}))
    assert decision.selected_rule_ids == ("graph",)


def test_router_rejects_dependent_without_directed_prerequisite() -> None:
    specs = [rule("base", 0.1, sensitive=True), rule("specialized", 0.9, sensitive=True)]
    relation = RelationSpec(
        "base-specialized", 1, None, ["base", "specialized"], "prerequisite",
        {"equals": {}}, {"direction": "a_to_b"}, 0.05, [], {"required": True},
    )
    decision = ConservativeCausalRouter(token_budget=1, lambda_tokens=0.0).route(
        specs,
        states({"base": 0.1, "specialized": 0.9}),
        [relation],
        {"base-specialized": RelationState("base-specialized", 1, 0.2, {"lcb": 0.2}, status="canonical")},
        {},
    )
    assert decision.selected_rule_ids == ("base",)
    assert "missing_prerequisite:base" in decision.rejected_reasons["specialized"]
