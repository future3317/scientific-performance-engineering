"""Statistical CEGIS harness for deterministic BoundaryBench families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.acre.predicate_synthesis import PredicateGrammar, SynthesisResult
from core.predicates import match_predicate


@dataclass(frozen=True)
class BoundaryObservation:
    observation_id: str
    context: Mapping[str, Any]
    effect: float
    gate_passed: bool
    effect_lower: float
    effect_upper: float

    def certified_counterexample(self, epsilon_false: float = 0.0) -> bool:
        return not self.gate_passed or self.effect_upper <= epsilon_false

    def positive_anchor(self, epsilon_true: float = 0.0) -> bool:
        return self.gate_passed and self.effect_lower > epsilon_true


class StatisticalCEGIS:
    """Shrink a finite predicate version space using certified observations."""

    def __init__(self, grammar: PredicateGrammar, *, epsilon_true: float = 0.0, epsilon_false: float = 0.0) -> None:
        self.grammar = grammar
        self.epsilon_true = epsilon_true
        self.epsilon_false = epsilon_false

    def synthesize(
        self,
        *,
        positive: list[BoundaryObservation],
        counterexamples: list[BoundaryObservation],
        parent_predicate: dict[str, Any] | None,
    ) -> SynthesisResult:
        anchors = [item for item in positive if item.positive_anchor(self.epsilon_true)]
        certified = [item for item in counterexamples if item.certified_counterexample(self.epsilon_false)]
        anchor_ids = tuple(item.observation_id for item in anchors)
        counterexample_ids = tuple(item.observation_id for item in certified)
        if not anchors:
            return SynthesisResult("no_consistent_hypothesis", None, counterexample_ids, anchor_ids, provenance={"parent_predicate": parent_predicate})
        if not certified:
            # An uncertain slowdown is not a falsifier.  Without a certified
            # counterexample there is no evidence that justifies specializing
            # the parent predicate.
            return SynthesisResult("no_consistent_hypothesis", parent_predicate, counterexample_ids, anchor_ids, provenance={"parent_predicate": parent_predicate, "reason": "awaiting_certified_counterexample"})
        anchor_keys = {repr(sorted(item.context.items())) for item in anchors}
        if any(repr(sorted(item.context.items())) in anchor_keys for item in certified):
            return SynthesisResult("no_consistent_hypothesis", None, counterexample_ids, anchor_ids, provenance={"parent_predicate": parent_predicate})
        contexts = [item.context for item in anchors] + [item.context for item in certified]
        candidates = self.grammar.candidates(contexts, parent_predicate=parent_predicate)
        consistent = [
            predicate
            for predicate in candidates
            if all(match_predicate(predicate, item.context) for item in anchors)
            and all(not match_predicate(predicate, item.context) for item in certified)
        ]
        if not consistent:
            return SynthesisResult(
                "unsynthesizable_boundary",
                None,
                counterexample_ids,
                anchor_ids,
                provenance={"parent_predicate": parent_predicate, "grammar_version": 1},
            )
        predicate = min(consistent, key=lambda value: (len(value.get("all", [value])), str(value)))
        return SynthesisResult(
            "accepted",
            predicate,
            counterexample_ids,
            anchor_ids,
            provenance={
                "parent_predicate": parent_predicate,
                "certified_evidence": list(counterexample_ids),
                "positive_anchors": list(anchor_ids),
                "grammar_version": 1,
            },
        )
