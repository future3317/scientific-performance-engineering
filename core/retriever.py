"""Conflict-aware, budgeted rule retrieval with a greedy submodular objective."""

from __future__ import annotations

from typing import Any

from .models import RuleSpec, RuleState, TaskContext
from .predicates import match_predicate


def retrieve_candidates(specs: list[RuleSpec], context: TaskContext) -> list[RuleSpec]:
    return [spec for spec in specs if spec.domain == context.domain and match_predicate(spec.applicability, context.to_dict()) and not (bool(spec.abstain_conditions) and match_predicate(spec.abstain_conditions, context.to_dict()))]


def select_rules(
    specs: list[RuleSpec],
    states_or_context: list[RuleState] | dict[str, RuleState] | TaskContext,
    context: TaskContext | None = None,
) -> list[dict[str, Any]]:
    """Greedily maximize coverage + utility - redundancy under token budget."""
    # The two-argument form remains the deterministic cold-start selector.  A
    # three-argument call enables the feedback loop from RuleState telemetry.
    if context is None:
        context = states_or_context  # type: ignore[assignment]
        state_map: dict[str, RuleState] = {}
    else:
        state_map = (
            {state.rule_id: state for state in states_or_context}
            if isinstance(states_or_context, list)
            else dict(states_or_context)
        )
        superseded = {
            target
            for spec in specs
            if state_map.get(spec.rule_id) is not None
            and state_map[spec.rule_id].status == "canonical"
            for target in spec.relations.get("supersedes", [])
        }
        specs = [spec for spec in specs if spec.rule_id not in superseded]
    selected: list[RuleSpec] = []
    selected_gains: dict[tuple[str, int], float] = {}
    used = 0
    covered: set[str] = set()
    remaining = list(specs)
    conflicts = {spec.rule_id: set(spec.relations.get("conflicts", [])) for spec in specs}
    while remaining:
        feasible = [
            spec for spec in remaining
            if used + int(spec.runtime_cost.get("tokens", 0)) <= context.token_budget
            and not any(spec.rule_id in conflicts.get(item.rule_id, set()) or item.rule_id in conflicts.get(spec.rule_id, set()) for item in selected)
            and not (state_map.get(spec.rule_id) and state_map[spec.rule_id].drift_state in {"stale", "revalidating"})
        ]
        if not feasible:
            break
        def gain(spec: RuleSpec) -> float:
            features = set(spec.evidence_requirements) | set(spec.relations.get("requires", []))
            novelty = len(features - covered)
            utility = float(spec.runtime_cost.get("expected_utility", 0.0))
            state = state_map.get(spec.rule_id)
            if state is not None:
                utility = float(state.confidence_sequence.get("lcb", state.retrieval_utility))
                utility *= max(0.0, 1.0 - state.override_rate)
                if state.drift_state == "suspected_drift":
                    utility *= 0.5
                elif state.drift_state in {"stale", "revalidating"}:
                    utility = -float("inf")
            return novelty + utility - 0.25 * len(features & covered)
        best = max(feasible, key=gain)
        if state_map:
            # Explicit dependency closure is part of the state-aware path.  A
            # rule with a missing dependency is not eligible for selection.
            by_id = {item.rule_id: item for item in remaining + selected}
            dependencies = [by_id[item] for item in best.relations.get("requires", []) if item in by_id]
            if len(dependencies) != len(best.relations.get("requires", [])):
                remaining.remove(best)
                continue
            dependency_blocked = False
            for dependency in dependencies:
                if dependency not in selected:
                    dependency_cost = int(dependency.runtime_cost.get("tokens", 0))
                    if used + dependency_cost > context.token_budget:
                        dependency_blocked = True
                        break
                    selected.append(dependency)
                    selected_gains[(dependency.rule_id, dependency.version)] = gain(dependency)
                    covered.update(set(dependency.evidence_requirements) | set(dependency.relations.get("requires", [])))
                    used += dependency_cost
                    remaining.remove(dependency)
            if dependency_blocked:
                remaining.remove(best)
                continue
        best_gain = gain(best)
        selected.append(best)
        selected_gains[(best.rule_id, best.version)] = best_gain
        covered.update(set(best.evidence_requirements) | set(best.relations.get("requires", [])))
        used += int(best.runtime_cost.get("tokens", 0))
        remaining.remove(best)
    return [{"rule_id": spec.rule_id, "version": spec.version, "token_cost": int(spec.runtime_cost.get("tokens", 0)), "marginal_gain": round(selected_gains[(spec.rule_id, spec.version)], 8)} for spec in selected]
