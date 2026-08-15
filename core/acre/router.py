"""Conservative routing over the canonical rule/relation contracts."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.models import RelationSpec, RelationState, RuleSpec, RuleState, TaskContext
from core.predicates import match_predicate


@dataclass(frozen=True)
class BundleCertificate:
    bundle_ids: tuple[str, ...]
    context_predicate: Mapping[str, Any]
    residual_lcb: float
    residual_ucb: float
    status: str

    @property
    def bounded_auto_allowed(self) -> bool:
        # A non-zero higher-order residual is a hyperedge, not evidence that
        # the pairwise graph is complete.  It must not silently pass the
        # bounded-auto gate.
        return self.status in {"pairwise_certified", "not_applicable"}


@dataclass(frozen=True)
class RoutingDecision:
    selected_rule_ids: tuple[str, ...]
    objective: float
    rejected_reasons: Mapping[str, tuple[str, ...]]
    bundle_certificate: BundleCertificate | None = None


def validate_relation_nonoverlap(
    relation_specs: Sequence[RelationSpec], contexts: Sequence[Mapping[str, Any]] = ()
) -> list[str]:
    """Return active relation overlap errors for a registered relation set."""
    errors: list[str] = []
    grouped: dict[frozenset[str], list[RelationSpec]] = {}
    for spec in relation_specs:
        grouped.setdefault(frozenset(spec.endpoints.values()), []).append(spec)
    for pair, specs in grouped.items():
        for index, left in enumerate(specs):
            for right in specs[index + 1 :]:
                if left.relation_id == right.relation_id:
                    continue
                overlap = left.applicability == right.applicability
                if contexts:
                    overlap = overlap or any(
                        match_predicate(left.applicability, context)
                        and match_predicate(right.applicability, context)
                        for context in contexts
                    )
                if overlap:
                    errors.append(
                        f"relation applicability overlap for {left.relation_id} and {right.relation_id} on {sorted(pair)}"
                    )
    return errors


class ConservativeCausalRouter:
    def __init__(self, *, token_budget: int, zeta: float = 0.05, lambda_tokens: float = 0.01) -> None:
        if token_budget < 1 or zeta < 0.0 or lambda_tokens < 0.0:
            raise ValueError("token budget must be positive and penalties non-negative")
        self.token_budget = token_budget
        self.zeta = zeta
        self.lambda_tokens = lambda_tokens

    @staticmethod
    def _context(value: TaskContext | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(value, TaskContext):
            return value.workload
        return value

    @staticmethod
    def _utility(state: RuleState) -> float:
        value = state.confidence_sequence.get("lcb")
        if value is None:
            value = state.effect.get("lower_utility", state.effect.get("utility", state.retrieval_utility))
        if not math.isfinite(float(value)):
            raise ValueError("rule state utility must be finite")
        return float(value)

    @staticmethod
    def _tokens(spec: RuleSpec) -> int:
        value = spec.runtime_cost.get("tokens", spec.runtime_cost.get("token_cost", 1.0))
        if float(value) < 1:
            raise ValueError(f"rule {spec.rule_id} token cost must be positive")
        return int(math.ceil(float(value)))

    @staticmethod
    def _relation_map(specs: Sequence[RelationSpec]) -> dict[frozenset[str], tuple[RelationSpec, ...]]:
        pairs: dict[frozenset[str], list[RelationSpec]] = {}
        for spec in specs:
            key = frozenset(spec.endpoints.values())
            if len(key) != 2:
                continue
            pairs.setdefault(key, []).append(spec)
        return {key: tuple(value) for key, value in pairs.items()}

    @staticmethod
    def _active(spec: RelationSpec, states: Mapping[str, RelationState], context: Mapping[str, Any]) -> bool:
        state = states.get(spec.relation_id)
        return state is not None and state.status == "canonical" and state.drift_state == "stable" and match_predicate(spec.applicability, context)

    @classmethod
    def _matching_relations(
        cls,
        pair: frozenset[str],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
    ) -> tuple[tuple[RelationSpec, ...], bool]:
        active = [spec for spec in relation_map.get(pair, ()) if cls._active(spec, relation_states, context)]
        # A single canonical semantic relation is the only safe router state.
        # Multiple versions of the same relation resolve to the newest one;
        # distinct active relations are an overlap conflict regardless of kind
        # (independence and redundancy are not interchangeable).
        newest: dict[str, RelationSpec] = {}
        for spec in active:
            previous = newest.get(spec.relation_id)
            if previous is None or spec.version > previous.version:
                newest[spec.relation_id] = spec
        matches = tuple(sorted(newest.values(), key=lambda item: (item.relation_id, item.version)))
        return matches, len(matches) > 1

    @staticmethod
    def _lower_bound(spec: RelationSpec, state: RelationState | None) -> float:
        if spec.kind == "independence" or state is None:
            return 0.0
        return float(state.confidence_sequence.get("lcb", state.estimate))

    def _invalid_reasons(
        self,
        bundle: tuple[RuleSpec, ...],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
    ) -> tuple[str, ...]:
        ids = {spec.rule_id for spec in bundle}
        reasons: set[str] = set()
        if sum(self._tokens(spec) for spec in bundle) > self.token_budget:
            reasons.add("token_budget")
        # A directed prerequisite is a closure constraint, not merely a
        # pairwise bonus.  Reject a dependent rule when its prerequisite is
        # absent from the candidate bundle, even if the pair is not selected.
        for pair, candidates in relation_map.items():
            for relation in candidates:
                if relation.kind != "prerequisite" or not self._active(relation, relation_states, context):
                    continue
                left, right = relation.endpoints["left"], relation.endpoints["right"]
                prerequisite, dependent = (left, right) if relation.orientation == "left_to_right" else (right, left) if relation.orientation == "right_to_left" else (None, None)
                if prerequisite is not None and dependent in ids and prerequisite not in ids:
                    reasons.add("missing_prerequisite:" + prerequisite)
        for left, right in itertools.combinations(bundle, 2):
            matches, overlap_conflict = self._matching_relations(frozenset((left.rule_id, right.rule_id)), relation_map, relation_states, context)
            if overlap_conflict:
                reasons.add("relation_overlap_conflict")
            relation = matches[0] if len(matches) == 1 else None
            if relation is None:
                if left.scientific_invariants and right.scientific_invariants:
                    reasons.add("unknown_scientific_interaction")
                continue
            if relation.kind == "semantic_conflict":
                reasons.add("hard_conflict")
        return tuple(sorted(reasons))

    def _objective(
        self,
        bundle: tuple[RuleSpec, ...],
        states: Mapping[str, RuleState],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
    ) -> float:
        score = sum(self._utility(states[spec.rule_id]) for spec in bundle)
        for left, right in itertools.combinations(bundle, 2):
            matches, _ = self._matching_relations(frozenset((left.rule_id, right.rule_id)), relation_map, relation_states, context)
            relation = matches[0] if len(matches) == 1 else None
            if relation is not None:
                score += self._lower_bound(relation, relation_states.get(relation.relation_id))
        if len(bundle) > 1:
            score -= self.zeta
        score -= self.lambda_tokens * sum(self._tokens(spec) for spec in bundle)
        return score

    def route(
        self,
        rule_specs: Sequence[RuleSpec],
        rule_states: Mapping[str, RuleState],
        relation_specs: Sequence[RelationSpec],
        relation_states: Mapping[str, RelationState],
        context: TaskContext | Mapping[str, Any],
        higher_order_evidence: Mapping[str, float] | None = None,
    ) -> RoutingDecision:
        by_id = {spec.rule_id: spec for spec in rule_specs}
        if len(by_id) != len(rule_specs):
            raise ValueError("rule spec ids must be unique")
        if any(spec.rule_id not in rule_states for spec in rule_specs):
            raise ValueError("rule states must cover every rule spec")
        relation_map = self._relation_map(relation_specs)
        context_map = self._context(context)
        applicable = [spec for spec in rule_specs if match_predicate(spec.applicability, context_map)]
        rejected: dict[str, set[str]] = {spec.rule_id: set() for spec in rule_specs}
        valid: list[tuple[tuple[RuleSpec, ...], float]] = []
        for width in range(0, len(applicable) + 1):
            for selected in itertools.combinations(applicable, width):
                bundle = tuple(sorted(selected, key=lambda spec: spec.rule_id))
                reasons = self._invalid_reasons(bundle, relation_map, relation_states, context_map)
                if reasons:
                    for spec in bundle:
                        rejected[spec.rule_id].update(reasons)
                    continue
                valid.append((bundle, self._objective(bundle, rule_states, relation_map, relation_states, context_map)))
        if not valid:
            raise ValueError("no valid rule bundle under the token budget")
        bundle, objective = max(valid, key=lambda item: (item[1], tuple(spec.rule_id for spec in item[0])))
        selected_ids = {spec.rule_id for spec in bundle}
        for spec in rule_specs:
            if spec.rule_id not in selected_ids and not match_predicate(spec.applicability, context_map):
                rejected[spec.rule_id].add("not_applicable")
        evidence = dict(higher_order_evidence or {})
        if len(bundle) < 3:
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, 0.0, 0.0, "not_applicable")
        elif "lcb" in evidence and "ucb" in evidence:
            lcb, ucb = float(evidence["lcb"]), float(evidence["ucb"])
            eta = float(evidence.get("practical_margin", 0.05))
            status = "pairwise_certified" if lcb >= -eta and ucb <= eta else "hyperedge_required"
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, lcb, ucb, status)
        else:
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, -1.0, 1.0, "higher_order_suspected")
        return RoutingDecision(
            selected_rule_ids=tuple(spec.rule_id for spec in bundle),
            objective=objective,
            rejected_reasons={rule_id: tuple(sorted(reasons)) for rule_id, reasons in rejected.items() if reasons},
            bundle_certificate=certificate,
        )
