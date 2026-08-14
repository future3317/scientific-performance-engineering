"""Conflict-aware, budgeted rule retrieval with a greedy submodular objective."""

from __future__ import annotations

from typing import Any

from .models import RuleSpec, TaskContext
from .predicates import match_predicate


def retrieve_candidates(specs: list[RuleSpec], context: TaskContext) -> list[RuleSpec]:
    return [spec for spec in specs if spec.domain == context.domain and match_predicate(spec.applicability, context.to_dict()) and not (bool(spec.abstain_conditions) and match_predicate(spec.abstain_conditions, context.to_dict()))]


def select_rules(specs: list[RuleSpec], context: TaskContext) -> list[dict[str, Any]]:
    """Greedily maximize coverage + utility - redundancy under token budget."""
    selected: list[RuleSpec] = []
    used = 0
    covered: set[str] = set()
    remaining = list(specs)
    conflicts = {spec.rule_id: set(spec.relations.get("conflicts", [])) for spec in specs}
    while remaining:
        feasible = [spec for spec in remaining if used + int(spec.runtime_cost.get("tokens", 0)) <= context.token_budget and not any(spec.rule_id in conflicts.get(item.rule_id, set()) or item.rule_id in conflicts.get(spec.rule_id, set()) for item in selected)]
        if not feasible:
            break
        def gain(spec: RuleSpec) -> float:
            features = set(spec.evidence_requirements) | set(spec.relations.get("requires", []))
            novelty = len(features - covered)
            return novelty + float(spec.runtime_cost.get("expected_utility", 0.0)) - 0.25 * len(features & covered)
        best = max(feasible, key=gain)
        selected.append(best)
        covered.update(best.evidence_requirements)
        used += int(best.runtime_cost.get("tokens", 0))
        remaining.remove(best)
    return [{"rule_id": spec.rule_id, "version": spec.version, "token_cost": int(spec.runtime_cost.get("tokens", 0)), "marginal_gain": round(float(spec.runtime_cost.get("expected_utility", 0.0)), 8)} for spec in selected]
