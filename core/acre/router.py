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
        return self.status in {"certified", "not_applicable"}


@dataclass(frozen=True)
class RoutingDecision:
    selected_rule_ids: tuple[str, ...]
    objective: float
    rejected_reasons: Mapping[str, tuple[str, ...]]
    bundle_certificate: BundleCertificate | None = None


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
    def _relation_map(specs: Sequence[RelationSpec]) -> dict[frozenset[str], RelationSpec]:
        pairs: dict[frozenset[str], RelationSpec] = {}
        for spec in specs:
            key = frozenset(spec.endpoints.values())
            if len(key) != 2:
                continue
            if key in pairs:
                raise ValueError("duplicate canonical relation for rule pair")
            pairs[key] = spec
        return pairs

    @staticmethod
    def _active(spec: RelationSpec, states: Mapping[str, RelationState]) -> bool:
        state = states.get(spec.relation_id)
        return state is not None and state.status == "canonical" and state.drift_state == "stable"

    @staticmethod
    def _lower_bound(spec: RelationSpec, state: RelationState | None) -> float:
        if spec.kind == "independence" or state is None:
            return 0.0
        return float(state.confidence_sequence.get("lcb", state.estimate))

    def _invalid_reasons(
        self,
        bundle: tuple[RuleSpec, ...],
        relation_map: Mapping[frozenset[str], RelationSpec],
        relation_states: Mapping[str, RelationState],
    ) -> tuple[str, ...]:
        ids = {spec.rule_id for spec in bundle}
        reasons: set[str] = set()
        if sum(self._tokens(spec) for spec in bundle) > self.token_budget:
            reasons.add("token_budget")
        # A directed prerequisite is a closure constraint, not merely a
        # pairwise bonus.  Reject a dependent rule when its prerequisite is
        # absent from the candidate bundle, even if the pair is not selected.
        for relation in relation_map.values():
            if relation.kind != "prerequisite" or not self._active(relation, relation_states):
                continue
            if len(relation.endpoints) != 2:
                continue
            left, right = relation.endpoints["left"], relation.endpoints["right"]
            prerequisite, dependent = (left, right) if relation.orientation == "left_to_right" else (right, left) if relation.orientation == "right_to_left" else (None, None)
            if prerequisite is not None and dependent in ids and prerequisite not in ids:
                reasons.add("missing_prerequisite:" + prerequisite)
        for left, right in itertools.combinations(bundle, 2):
            relation = relation_map.get(frozenset((left.rule_id, right.rule_id)))
            if relation is None or not self._active(relation, relation_states):
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
        relation_map: Mapping[frozenset[str], RelationSpec],
        relation_states: Mapping[str, RelationState],
    ) -> float:
        score = sum(self._utility(states[spec.rule_id]) for spec in bundle)
        for left, right in itertools.combinations(bundle, 2):
            relation = relation_map.get(frozenset((left.rule_id, right.rule_id)))
            if relation is not None and self._active(relation, relation_states):
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
                reasons = self._invalid_reasons(bundle, relation_map, relation_states)
                if reasons:
                    for spec in bundle:
                        rejected[spec.rule_id].update(reasons)
                    continue
                valid.append((bundle, self._objective(bundle, rule_states, relation_map, relation_states)))
        if not valid:
            raise ValueError("no valid rule bundle under the token budget")
        bundle, objective = max(valid, key=lambda item: (item[1], tuple(spec.rule_id for spec in item[0])))
        selected_ids = {spec.rule_id for spec in bundle}
        for spec in rule_specs:
            if spec.rule_id not in selected_ids and not match_predicate(spec.applicability, context_map):
                rejected[spec.rule_id].add("not_applicable")
        return RoutingDecision(
            selected_rule_ids=tuple(spec.rule_id for spec in bundle),
            objective=objective,
            rejected_reasons={rule_id: tuple(sorted(reasons)) for rule_id, reasons in rejected.items() if reasons},
            bundle_certificate=BundleCertificate(
                bundle_ids=tuple(spec.rule_id for spec in bundle),
                context_predicate={"context": dict(context_map)},
                residual_lcb=0.0 if len(bundle) < 3 else -1.0,
                residual_ucb=0.0 if len(bundle) < 3 else 1.0,
                status="not_applicable" if len(bundle) < 3 else "higher_order_suspected",
            ),
        )
