"""Synthetic routing ablation for the ACRE pilot."""

from __future__ import annotations

from core.acre.router import ConservativeCausalRouter, InteractionEvidence, RuleCandidate


def run_router_benchmark() -> dict[str, object]:
    context = {"workload": "graph"}
    rules = [
        RuleCandidate("base", 0.45, 2, applicability={"equals": {"workload": "graph"}}, scientific_sensitive=True),
        RuleCandidate("specialized", 0.55, 2, requires=("base",), scientific_sensitive=True),
        RuleCandidate("auxiliary", 0.45, 1),
    ]
    interaction = InteractionEvidence("base", "specialized", 0.30, kind="synergy", scientific_sensitive=True)
    variants = {
        "current_governed_D": (rules, ()),
        "D_plus_CEGIS": ([rules[0], RuleCandidate("specialized", 0.80, 2, requires=("base",), scientific_sensitive=True), rules[2]], ()),
        "D_plus_causal_interaction": (rules, (interaction,)),
        "full_ACRE": ([rules[0], RuleCandidate("specialized", 0.80, 2, requires=("base",), scientific_sensitive=True), rules[2]], (interaction,)),
    }
    output: dict[str, object] = {}
    for name, (variant_rules, relations) in variants.items():
        decision = ConservativeCausalRouter(token_budget=8, zeta=0.05, lambda_tokens=0.01).route(
            variant_rules, relations=relations, context=context
        )
        output[name] = {"selected_rule_ids": list(decision.selected_rule_ids), "objective": decision.objective}
    return output
