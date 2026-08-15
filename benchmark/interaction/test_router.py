from __future__ import annotations

from core.acre.router import (
    ConservativeCausalRouter,
    InteractionEvidence,
    RuleCandidate,
)


def test_router_enforces_prerequisites_conflicts_and_unknown_sensitive_edges() -> None:
    rules = [
        RuleCandidate("base", 0.55, 2),
        RuleCandidate("specialized", 0.75, 2, requires=("base",), scientific_sensitive=True),
        RuleCandidate("conflict", 0.95, 1, conflicts=("base",)),
    ]
    router = ConservativeCausalRouter(token_budget=8, lambda_tokens=0.01)
    decision = router.route(rules, relations=(), context={})
    assert decision.selected_rule_ids == ("base", "specialized")
    assert "conflict" not in decision.selected_rule_ids

    blocked = router.route(
        rules[:2],
        relations=(InteractionEvidence("base", "specialized", 0.3, status="unknown", scientific_sensitive=True),),
        context={},
    )
    assert blocked.selected_rule_ids == ("base",)
    assert "unknown_scientific_interaction" in blocked.rejected_reasons["specialized"]


def test_pairwise_lower_bound_changes_conservative_bundle_score() -> None:
    rules = [RuleCandidate("a", 0.5, 1), RuleCandidate("b", 0.5, 1)]
    router = ConservativeCausalRouter(token_budget=4, lambda_tokens=0.0, zeta=0.01)
    independent = router.route(rules, relations=(), context={})
    synergy = router.route(
        rules,
        relations=(InteractionEvidence("a", "b", 0.4, kind="synergy"),),
        context={},
    )
    assert synergy.objective > independent.objective
    assert synergy.selected_rule_ids == ("a", "b")


def test_router_applies_typed_applicability_predicate() -> None:
    rules = [
        RuleCandidate("graph", 0.8, 1, applicability={"equals": {"workload": "graph"}}),
        RuleCandidate("compile", 0.7, 1, applicability={"equals": {"workload": "compile"}}),
    ]
    decision = ConservativeCausalRouter(token_budget=3).route(rules, relations=(), context={"workload": "graph"})
    assert decision.selected_rule_ids == ("graph",)
