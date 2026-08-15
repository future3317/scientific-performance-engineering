"""Core-owned statistical CEGIS over the finite predicate grammar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.predicates import match_predicate

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, _key, predicate_complexity


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


@dataclass(frozen=True)
class SynthesisResult:
    status: str
    predicate: dict[str, Any] | None
    certified_counterexamples: tuple[str, ...]
    positive_anchors: tuple[str, ...]
    synthesizer_version: str = SYNTHESIZER_VERSION
    provenance: dict[str, Any] | None = None
    version_space: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicate": self.predicate,
            "certified_counterexamples": list(self.certified_counterexamples),
            "positive_anchors": list(self.positive_anchors),
            "synthesizer_version": self.synthesizer_version,
            "provenance": self.provenance,
            "version_space_size": len(self.version_space),
            "version_space": list(self.version_space),
        }


class StatisticalCEGIS:
    def __init__(self, grammar: PredicateGrammar, *, epsilon_true: float = 0.0, epsilon_false: float = 0.0) -> None:
        self.grammar = grammar
        self.epsilon_true = epsilon_true
        self.epsilon_false = epsilon_false
        self._hypothesis_space: tuple[dict[str, Any], ...] | None = None

    def synthesize(
        self,
        *,
        positive: list[BoundaryObservation],
        counterexamples: list[BoundaryObservation],
        parent_predicate: dict[str, Any] | None,
        decision_contexts: list[Mapping[str, Any]] | None = None,
    ) -> SynthesisResult:
        anchors = [item for item in positive if item.positive_anchor(self.epsilon_true)]
        certified = [item for item in counterexamples if item.certified_counterexample(self.epsilon_false)]
        anchor_ids = tuple(item.observation_id for item in anchors)
        counterexample_ids = tuple(item.observation_id for item in certified)
        provenance = {"parent_predicate": parent_predicate, "grammar_version": 2}
        if not anchors:
            return SynthesisResult("insufficient_evidence", None, counterexample_ids, anchor_ids, provenance=provenance)
        if not certified:
            provenance["reason"] = "awaiting_certified_counterexample"
            return SynthesisResult("insufficient_evidence", None, counterexample_ids, anchor_ids, provenance=provenance)
        anchor_keys = {_key(item.context) for item in anchors}
        if any(_key(item.context) in anchor_keys for item in certified):
            return SynthesisResult("unsynthesizable_boundary", None, counterexample_ids, anchor_ids, provenance=provenance)
        contexts = [item.context for item in anchors] + [item.context for item in certified]
        if self._hypothesis_space is None:
            self._hypothesis_space = tuple(self.grammar.candidates(contexts, parent_predicate=parent_predicate))
        consistent = [
            predicate
            for predicate in self._hypothesis_space
            if all(match_predicate(predicate, item.context) for item in anchors)
            and all(not match_predicate(predicate, item.context) for item in certified)
        ]
        if not consistent:
            return SynthesisResult("unsynthesizable_boundary", None, counterexample_ids, anchor_ids, provenance=provenance)
        predicate = min(
            consistent,
            key=lambda value: (
                predicate_complexity(value)["description_length"],
                predicate_complexity(value)["depth"],
                predicate_complexity(value)["literals"],
                _key(value),
            ),
        )
        provenance["certified_evidence"] = list(counterexample_ids)
        provenance["positive_anchors"] = list(anchor_ids)
        provenance["complexity"] = predicate_complexity(predicate)
        # A consistent predicate is identified only when every remaining
        # hypothesis makes the same deploy/no-deploy decision on the
        # observable decision contexts.  The sealed pool is deliberately not
        # accepted here; it is an offline score only.
        if decision_contexts is None:
            status = "identified" if len(consistent) == 1 else "underidentified"
            provenance["decision_context_count"] = 0
            provenance["decision_equivalence_classes"] = len(consistent)
        else:
            signatures = {
                tuple(bool(match_predicate(candidate, context)) for context in decision_contexts)
                for candidate in consistent
            }
            status = "identified" if len(signatures) == 1 else "underidentified"
            provenance["decision_context_count"] = len(decision_contexts)
            provenance["decision_equivalence_classes"] = len(signatures)
        return SynthesisResult(status, predicate, counterexample_ids, anchor_ids, provenance=provenance, version_space=tuple(consistent))
