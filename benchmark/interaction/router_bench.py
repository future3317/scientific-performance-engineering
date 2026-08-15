"""Synthetic routing ablation using canonical Rule/Relation contracts."""

from __future__ import annotations

from core.acre.router import ConservativeCausalRouter
from core.models import RelationSpec, RelationState, RuleSpec, RuleState


def _rule(rule_id: str, utility: float, *, sensitive: bool = False) -> RuleSpec:
    return RuleSpec(rule_id, 1, None, {"equals": {}}, {"action": rule_id}, "mechanism", ["evidence"], ["physics"] if sensitive else [], {}, {}, {"tokens": 2.0 if rule_id != "auxiliary" else 1.0}, {"required": True})


def _states(values: dict[str, float]) -> dict[str, RuleState]:
    return {rule_id: RuleState(rule_id, 1, status="canonical", effect={"lower_utility": value}) for rule_id, value in values.items()}


def _relation() -> tuple[list[RelationSpec], dict[str, RelationState]]:
    spec = RelationSpec("base-specialized", 1, None, ["base", "specialized"], "synergy", {"equals": {}}, {"contrast": "gamma"}, 0.05, [], {"required": True})
    state = RelationState("base-specialized", 1, 0.30, {"lcb": 0.30}, status="canonical")
    return [spec], {"base-specialized": state}


def run_router_benchmark() -> dict[str, object]:
    rules = [_rule("base", 0.45, sensitive=True), _rule("specialized", 0.55, sensitive=True), _rule("auxiliary", 0.45)]
    relation_specs, relation_states = _relation()
    variants = {
        "current_governed_D": (rules, _states({"base": 0.45, "specialized": 0.55, "auxiliary": 0.45}), [], {}),
        "D_plus_CEGIS": ([_rule("base", 0.45, sensitive=True), _rule("specialized", 0.80, sensitive=True), rules[2]], _states({"base": 0.45, "specialized": 0.80, "auxiliary": 0.45}), [], {}),
        "D_plus_causal_interaction": (rules, _states({"base": 0.45, "specialized": 0.55, "auxiliary": 0.45}), relation_specs, relation_states),
        "full_ACRE": ([_rule("base", 0.45, sensitive=True), _rule("specialized", 0.80, sensitive=True), rules[2]], _states({"base": 0.45, "specialized": 0.80, "auxiliary": 0.45}), relation_specs, relation_states),
    }
    output: dict[str, object] = {}
    for name, (variant_rules, rule_states, relations, relation_states_for_variant) in variants.items():
        decision = ConservativeCausalRouter(token_budget=8, zeta=0.05, lambda_tokens=0.01).route(
            variant_rules, rule_states, relations, relation_states_for_variant, {}
        )
        output[name] = {"selected_rule_ids": list(decision.selected_rule_ids), "objective": decision.objective}
    return output
