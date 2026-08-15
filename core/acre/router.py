"""Conservative bounded rule-bundle routing for causal interactions."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.predicates import match_predicate


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    lower_utility: float
    tokens: int
    applicability: Mapping[str, Any] | None = None
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    scientific_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.rule_id or self.tokens < 1:
            raise ValueError("rule_id must be non-empty and tokens must be positive")
        if not math.isfinite(self.lower_utility) or not -1.0 <= self.lower_utility <= 1.0:
            raise ValueError("lower_utility must be finite and in [-1, 1]")


@dataclass(frozen=True)
class InteractionEvidence:
    left_rule_id: str
    right_rule_id: str
    lower_bound: float
    kind: str = "context_dependent_interaction"
    status: str = "known"
    scientific_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.left_rule_id or not self.right_rule_id or self.left_rule_id == self.right_rule_id:
            raise ValueError("interaction endpoints must be distinct and non-empty")
        if not -1.0 <= self.lower_bound <= 1.0:
            raise ValueError("interaction lower_bound must be in [-1, 1]")
        if self.kind not in {"synergy", "antagonism", "independence", "prerequisite", "context_dependent_interaction"}:
            raise ValueError("invalid interaction kind")
        if self.status not in {"known", "unknown"}:
            raise ValueError("interaction status must be known or unknown")

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.left_rule_id, self.right_rule_id))


@dataclass(frozen=True)
class RoutingDecision:
    selected_rule_ids: tuple[str, ...]
    objective: float
    rejected_reasons: Mapping[str, tuple[str, ...]]


class ConservativeCausalRouter:
    def __init__(self, *, token_budget: int, zeta: float = 0.05, lambda_tokens: float = 0.01) -> None:
        if token_budget < 1 or zeta < 0.0 or lambda_tokens < 0.0:
            raise ValueError("token budget must be positive and penalties non-negative")
        self.token_budget = token_budget
        self.zeta = zeta
        self.lambda_tokens = lambda_tokens

    @staticmethod
    def _pair_map(relations: Sequence[InteractionEvidence]) -> dict[frozenset[str], InteractionEvidence]:
        result: dict[frozenset[str], InteractionEvidence] = {}
        for relation in relations:
            if relation.key in result:
                raise ValueError("duplicate interaction relation")
            result[relation.key] = relation
        return result

    def _invalid_reasons(
        self,
        bundle: tuple[RuleCandidate, ...],
        by_id: Mapping[str, RuleCandidate],
        pair_map: Mapping[frozenset[str], InteractionEvidence],
    ) -> tuple[str, ...]:
        ids = {candidate.rule_id for candidate in bundle}
        reasons: list[str] = []
        if sum(candidate.tokens for candidate in bundle) > self.token_budget:
            reasons.append("token_budget")
        for candidate in bundle:
            missing = [required for required in candidate.requires if required not in ids]
            if missing:
                reasons.append("missing_prerequisite:" + ",".join(sorted(missing)))
            if any(conflict in ids for conflict in candidate.conflicts):
                reasons.append("hard_conflict")
        for left, right in itertools.combinations(bundle, 2):
            relation = pair_map.get(frozenset((left.rule_id, right.rule_id)))
            if relation is None or relation.status == "unknown":
                if (relation is not None and relation.scientific_sensitive) or (left.scientific_sensitive and right.scientific_sensitive):
                    reasons.append("unknown_scientific_interaction")
        return tuple(sorted(set(reasons)))

    def _objective(self, bundle: tuple[RuleCandidate, ...], pair_map: Mapping[frozenset[str], InteractionEvidence]) -> float:
        score = sum(candidate.lower_utility for candidate in bundle)
        for left, right in itertools.combinations(bundle, 2):
            relation = pair_map.get(frozenset((left.rule_id, right.rule_id)))
            if relation is not None and relation.status == "known":
                score += relation.lower_bound
        if len(bundle) > 1:
            score -= self.zeta
        score -= self.lambda_tokens * sum(candidate.tokens for candidate in bundle)
        return score

    def route(
        self,
        candidates: Sequence[RuleCandidate],
        *,
        relations: Sequence[InteractionEvidence],
        context: Mapping[str, Any],
    ) -> RoutingDecision:
        by_id = {candidate.rule_id: candidate for candidate in candidates}
        if len(by_id) != len(candidates):
            raise ValueError("rule candidate ids must be unique")
        pair_map = self._pair_map(relations)
        applicable = [candidate for candidate in candidates if match_predicate(candidate.applicability, context)]
        rejected: dict[str, set[str]] = {candidate.rule_id: set() for candidate in candidates}
        valid: list[tuple[tuple[RuleCandidate, ...], float]] = []
        for width in range(0, len(applicable) + 1):
            for selected in itertools.combinations(applicable, width):
                bundle = tuple(sorted(selected, key=lambda candidate: candidate.rule_id))
                reasons = self._invalid_reasons(bundle, by_id, pair_map)
                if reasons:
                    for candidate in bundle:
                        rejected[candidate.rule_id].update(reasons)
                    continue
                valid.append((bundle, self._objective(bundle, pair_map)))
        if not valid:
            raise ValueError("no valid rule bundle under the token budget")
        bundle, objective = max(valid, key=lambda item: (item[1], tuple(candidate.rule_id for candidate in item[0])))
        selected_ids = {candidate.rule_id for candidate in bundle}
        for candidate in candidates:
            if candidate.rule_id not in selected_ids and not match_predicate(candidate.applicability, context):
                rejected[candidate.rule_id].add("not_applicable")
        return RoutingDecision(
            selected_rule_ids=tuple(candidate.rule_id for candidate in bundle),
            objective=objective,
            rejected_reasons={rule_id: tuple(sorted(reasons)) for rule_id, reasons in rejected.items() if reasons},
        )
