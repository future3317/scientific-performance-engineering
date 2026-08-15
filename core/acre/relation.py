"""Cross-context causal relation identification.

``FactorialEngine`` estimates one context.  ``RelationIdentifier`` owns the
semantic step that combines those local estimates and therefore keeps
redundancy and context-dependent relations out of the local estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .factorial import CANONICAL_RELATIONS, FactorialEstimate
from .cegis import BoundaryObservation, StatisticalCEGIS
from .predicates import PredicateGrammar
from core.models import RelationSpec


@dataclass(frozen=True)
class RelationIdentification:
    decision: str
    context_decisions: Mapping[str, str]
    applicability_predicate: Mapping[str, object] | None = None
    confidence: Mapping[str, object] = field(default_factory=dict)


class RelationIdentifier:
    """Identify a canonical relation from one or more context estimates."""

    def __init__(self, *, practical_margin: float = 0.05, equivalence_margin: float | None = None) -> None:
        if not 0.0 <= practical_margin <= 1.0:
            raise ValueError("practical_margin must be in [0, 1]")
        self.practical_margin = practical_margin
        self.equivalence_margin = practical_margin if equivalence_margin is None else equivalence_margin

    def identify(self, estimates: Mapping[str, FactorialEstimate]) -> RelationIdentification:
        if not estimates:
            raise ValueError("at least one context estimate is required")
        decisions = {name: self._identify_local(estimate) for name, estimate in estimates.items()}
        if len(estimates) >= 2:
            positive = any(estimate.gamma_lcb > self.practical_margin for estimate in estimates.values())
            negative = any(estimate.gamma_ucb < -self.practical_margin for estimate in estimates.values())
            if positive and negative:
                return RelationIdentification(
                    "context_dependent_relation", decisions,
                    applicability_predicate={"contexts": sorted(estimates), "condition": "context-dependent sign"},
                    confidence={"contexts": {name: {"gamma_lcb": value.gamma_lcb, "gamma_ucb": value.gamma_ucb} for name, value in estimates.items()}},
                )
        return RelationIdentification(next(iter(decisions.values())), decisions)

    def to_spec(self, relation_id: str, left_rule_id: str, right_rule_id: str, identification: RelationIdentification) -> RelationSpec:
        kind = identification.decision
        if kind == "unresolved":
            raise ValueError("cannot materialize an unresolved relation")
        if kind == "context_dependent_relation":
            kind = "context_dependent_interaction"
        if kind.startswith("confirmed_"):
            kind = kind.removeprefix("confirmed_")
        if kind.startswith("prerequisite_"):
            orientation = "left_to_right" if kind.endswith("a_to_b") else "right_to_left"
            kind = "prerequisite"
        else:
            orientation = "symmetric"
        return RelationSpec(
            relation_id=relation_id, version=1, parent=None,
            endpoints={"left": left_rule_id, "right": right_rule_id}, orientation=orientation,
            kind=kind, applicability=dict(identification.applicability_predicate or {"all": []}),
            contrast_definition={"contexts": list(identification.context_decisions)},
            practical_margin=self.practical_margin, scientific_invariants=[], provenance_policy={"required": True},
        )

    def _identify_local(self, estimate: FactorialEstimate) -> str:
        margin = self.practical_margin
        if estimate.decision == "semantic_conflict":
            return estimate.decision
        if self._redundant(estimate):
            return "confirmed_redundancy"
        return estimate.decision

    def _redundant(self, estimate: FactorialEstimate) -> bool:
        """Require CI-gated useful singles and practical-equivalent joint arm."""
        if estimate.contrast_intervals:
            a_lcb = estimate.contrast_intervals["delta_a_given_b0"][0]
            b_lcb = estimate.contrast_intervals["delta_b_given_a0"][0]
            joint_lower, joint_upper = estimate.contrast_intervals["redundancy"]
            return a_lcb > self.practical_margin and b_lcb > self.practical_margin and joint_lower >= -self.equivalence_margin and joint_upper <= self.equivalence_margin
        intervals = estimate.utility_intervals
        if not intervals:
            return False
        margin = self.practical_margin
        a_lcb = intervals["10"][0] - intervals["00"][1]
        b_lcb = intervals["01"][0] - intervals["00"][1]
        joint_lower = intervals["11"][0] - max(intervals["10"][1], intervals["01"][1])
        joint_upper = intervals["11"][1] - max(intervals["10"][0], intervals["01"][0])
        return a_lcb > margin and b_lcb > margin and joint_lower >= -self.equivalence_margin and joint_upper <= self.equivalence_margin


def _relation_kind(decision: str) -> tuple[str, str]:
    if decision.startswith("confirmed_"):
        return decision.removeprefix("confirmed_"), "symmetric"
    if decision.startswith("prerequisite_"):
        orientation = "left_to_right" if decision.endswith("a_to_b") else "right_to_left"
        return "prerequisite", orientation
    if decision == "context_dependent_relation":
        return "context_dependent_interaction", "symmetric"
    return decision, "symmetric"


def relational_cegis(
    identifier: RelationIdentifier,
    contexts: Mapping[str, Mapping[str, object]],
    identification: RelationIdentification,
    grammar: PredicateGrammar,
    *,
    relation_id: str = "REL",
    left_rule_id: str = "a",
    right_rule_id: str = "b",
) -> tuple[RelationSpec, ...]:
    """Specialize a cross-context sign flip into typed predicate children.

    The same finite grammar and CEGIS implementation used for boundary rules
    is applied to relation applicability.  Only local decisions and their
    confidence bounds are used; sealed truth is not part of synthesis.
    """
    if identification.decision != "context_dependent_relation":
        raise ValueError("relational CEGIS requires a context-dependent identification")
    children: list[RelationSpec] = []
    decisions = dict(identification.context_decisions)
    for child_index, (target_context, target_decision) in enumerate(sorted(decisions.items())):
        if target_decision == "unresolved":
            continue
        positive = [
            BoundaryObservation(target_context, contexts[target_context], 1.0, True, 1.0, 1.0)
        ]
        negative = [
            BoundaryObservation(name, contexts[name], 0.0, False, -1.0, 0.0)
            for name, decision in decisions.items() if name != target_context and decision != target_decision
        ]
        if not negative:
            continue
        synthesis = StatisticalCEGIS(grammar).synthesize(
            positive=positive, counterexamples=negative, parent_predicate=None,
            decision_contexts=list(contexts.values()),
        )
        if synthesis.predicate is None:
            continue
        kind, orientation = _relation_kind(target_decision)
        children.append(RelationSpec(
            relation_id=f"{relation_id}-{child_index + 1}", version=1, parent=relation_id,
            endpoints={"left": left_rule_id, "right": right_rule_id}, orientation=orientation,
            kind=kind, applicability=synthesis.predicate,
            contrast_definition={"contexts": [target_context], "cegis": synthesis.to_dict()},
            practical_margin=identifier.practical_margin,
            scientific_invariants=[], provenance_policy={"required": True},
        ))
    return tuple(children)


__all__ = ["RelationIdentifier", "RelationIdentification", "relational_cegis", "CANONICAL_RELATIONS"]
