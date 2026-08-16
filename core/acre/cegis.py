"""Core-owned statistical CEGIS over the finite predicate grammar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json

from core.predicates import match_predicate
from core.sequential_stats import paired_repetition_interval
from core.utility import UTILITY_LOG_SCALE, utility_effect

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, _key, predicate_complexity
from .budget import StatisticalBudget


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
    certificate: "SynthesisCertificate | None" = None

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
            "certificate": self.certificate.to_dict() if self.certificate else None,
        }

    @property
    def decision_context_count(self) -> int:
        return int((self.provenance or {}).get("decision_context_count", 0))


@dataclass(frozen=True)
class SynthesisCertificate:
    """Immutable evidence boundary for a materialized predicate."""

    status: str
    predicate: Mapping[str, Any] | None
    positive_anchor_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]
    grammar_digest: str
    decision_lattice_digest: str
    version_space_digest: str
    alpha_budget: float
    practical_threshold: float
    synthesizer_version: str = SYNTHESIZER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicate": dict(self.predicate) if isinstance(self.predicate, Mapping) else None,
            "positive_anchor_ids": list(self.positive_anchor_ids),
            "counterexample_ids": list(self.counterexample_ids),
            "unresolved_ids": list(self.unresolved_ids),
            "grammar_digest": self.grammar_digest,
            "decision_lattice_digest": self.decision_lattice_digest,
            "version_space_digest": self.version_space_digest,
            "alpha_budget": self.alpha_budget,
            "practical_threshold": self.practical_threshold,
            "synthesizer_version": self.synthesizer_version,
        }

class StatisticalCEGIS:
    def __init__(self, grammar: PredicateGrammar, *, epsilon_true: float = 0.0, epsilon_false: float = 0.0, delta: float = 0.05) -> None:
        self.grammar = grammar
        self.epsilon_true = epsilon_true
        self.epsilon_false = epsilon_false
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self._last_version_space: tuple[dict[str, Any], ...] | None = None

    def _certificate(
        self,
        *,
        status: str,
        predicate: dict[str, Any] | None,
        anchors: tuple[str, ...],
        counterexamples: tuple[str, ...],
        observation_ids: tuple[str, ...],
        decision_contexts: list[Mapping[str, Any]] | None,
        version_space: tuple[dict[str, Any], ...] = (),
    ) -> SynthesisCertificate:
        grammar_payload = {
            "schema_version": self.grammar.schema_version,
            "features": list(self.grammar.features),
            "max_depth": self.grammar.max_depth,
            "max_literals": self.grammar.max_literals,
            "threshold_universe": self.grammar.threshold_universe or {},
        }
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        return SynthesisCertificate(
            status=status,
            predicate=predicate,
            positive_anchor_ids=anchors,
            counterexample_ids=counterexamples,
            unresolved_ids=tuple(
                item for item in observation_ids
                if item not in set(anchors) and item not in set(counterexamples)
            ),
            grammar_digest=digest(grammar_payload),
            decision_lattice_digest=digest(list(decision_contexts or [])),
            version_space_digest=digest(list(version_space)),
            alpha_budget=self.delta,
            practical_threshold=max(float(self.epsilon_true), float(self.epsilon_false)),
        )

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
        provenance = {"source": "harness-cegis", "method_owner": "core", "parent_predicate": parent_predicate, "grammar_version": 2}
        if not anchors:
            cert = self._certificate(status="insufficient_evidence", predicate=None, anchors=anchor_ids, counterexamples=counterexample_ids, observation_ids=tuple(item.observation_id for item in positive + counterexamples), decision_contexts=decision_contexts)
            provenance["certificate"] = cert.to_dict()
            return SynthesisResult("insufficient_evidence", None, counterexample_ids, anchor_ids, provenance=provenance, certificate=cert)
        if not certified:
            provenance["reason"] = "awaiting_certified_counterexample"
            cert = self._certificate(status="insufficient_evidence", predicate=None, anchors=anchor_ids, counterexamples=counterexample_ids, observation_ids=tuple(item.observation_id for item in positive + counterexamples), decision_contexts=decision_contexts)
            provenance["certificate"] = cert.to_dict()
            return SynthesisResult("insufficient_evidence", None, counterexample_ids, anchor_ids, provenance=provenance, certificate=cert)
        anchor_keys = {_key(item.context) for item in anchors}
        if any(_key(item.context) in anchor_keys for item in certified):
            cert = self._certificate(status="unsynthesizable_boundary", predicate=None, anchors=anchor_ids, counterexamples=counterexample_ids, observation_ids=tuple(item.observation_id for item in positive + counterexamples), decision_contexts=decision_contexts)
            provenance["certificate"] = cert.to_dict()
            return SynthesisResult("unsynthesizable_boundary", None, counterexample_ids, anchor_ids, provenance=provenance, certificate=cert)
        contexts = [item.context for item in anchors] + [item.context for item in certified]
        # The finite vocabulary is rebuilt from the complete observed
        # decision context set on every call.  This is an explicit sequential
        # expansion policy: a newly observed boundary value can introduce a
        # new threshold instead of being filtered against a stale first-round
        # hypothesis space.
        vocabulary_contexts = contexts + list(decision_contexts or [])
        hypothesis_space = tuple(self.grammar.candidates(vocabulary_contexts, parent_predicate=parent_predicate))
        if decision_contexts is None and self._last_version_space is not None:
            hypothesis_space = self._last_version_space
        consistent = [
            predicate
            for predicate in hypothesis_space
            if all(match_predicate(predicate, item.context) for item in anchors)
            and all(not match_predicate(predicate, item.context) for item in certified)
        ]
        if not consistent:
            cert = self._certificate(status="unsynthesizable_boundary", predicate=None, anchors=anchor_ids, counterexamples=counterexample_ids, observation_ids=tuple(item.observation_id for item in positive + counterexamples), decision_contexts=decision_contexts)
            provenance["certificate"] = cert.to_dict()
            return SynthesisResult("unsynthesizable_boundary", None, counterexample_ids, anchor_ids, provenance=provenance, certificate=cert)
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
        provenance["hypothesis_vocabulary_context_count"] = len(vocabulary_contexts)
        provenance["version_space_digest"] = hashlib.sha256(json.dumps(list(consistent), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        provenance["version_space"] = list(consistent)
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
        certificate = self._certificate(
            status=status, predicate=predicate, anchors=anchor_ids,
            counterexamples=counterexample_ids,
            observation_ids=tuple(item.observation_id for item in positive + counterexamples),
            decision_contexts=decision_contexts, version_space=tuple(consistent),
        )
        provenance["certificate"] = certificate.to_dict()
        self._last_version_space = tuple(consistent)
        return SynthesisResult(status, predicate, counterexample_ids, anchor_ids, provenance=provenance, version_space=tuple(consistent), certificate=certificate)


def synthesize_boundary(
    observations: list[BoundaryObservation],
    grammar_payload: Mapping[str, Any],
    *,
    decision_contexts: list[Mapping[str, Any]] | None = None,
    delta: float = 0.05,
    statistical_budget: StatisticalBudget | None = None,
    epsilon_true: float = 0.0,
    epsilon_false: float = 0.0,
) -> SynthesisResult:
    """Single core-owned entry point for harness boundary synthesis."""
    grammar = PredicateGrammar.from_dict(dict(grammar_payload))
    positives = [item for item in observations if item.positive_anchor(epsilon_true)]
    negatives = [item for item in observations if item.certified_counterexample(epsilon_false)]
    alpha = statistical_budget.synth if statistical_budget is not None else float(delta)
    return StatisticalCEGIS(
        grammar, epsilon_true=epsilon_true, epsilon_false=epsilon_false, delta=alpha,
    ).synthesize(
        positive=positives,
        counterexamples=negatives,
        parent_predicate=None,
        decision_contexts=decision_contexts,
    )


def _case_effect_interval(case: Mapping[str, Any], *, delta: float = 0.05) -> tuple[float, float, float] | None:
    intervention = case.get("intervention_measurements")
    baseline = case.get("baseline_measurements")
    higher_is_better = bool(case.get("higher_is_better", True))
    log_scale = float(case.get("utility_scale", UTILITY_LOG_SCALE))
    if isinstance(intervention, list) and isinstance(baseline, list):
        if not intervention or len(intervention) != len(baseline):
            return None
        effects = [utility_effect(float(on), float(off), higher_is_better=higher_is_better, log_scale=log_scale) for on, off in zip(intervention, baseline)]
        lower, upper = paired_repetition_interval(effects, delta)
        return sum(effects) / len(effects), lower, upper
    # A point effect without paired repetitions is descriptive only and
    # cannot certify either a positive anchor or a counterexample.
    return None


def synthesize_applicability(
    cases: list[dict[str, Any]],
    *,
    family_id: str | None = None,
    decision_contexts: list[dict[str, Any]] | None = None,
    delta: float = 0.05,
    statistical_budget: StatisticalBudget | None = None,
    epsilon_true: float = 0.0,
    epsilon_false: float = 0.0,
    require_identified: bool = False,
) -> SynthesisResult:
    """Core-owned CEGIS adapter over verifier-produced paired cases."""
    if family_id is None:
        family_id = "compile"
    from benchmark.families import family_decision_lattice, family_predicate_grammar, family_surface
    try:
        surface, _instances = family_surface(family_id)
        lattice = list(decision_contexts or family_decision_lattice(family_id, count=len(surface.synthesis_ids) + len(surface.promotion_ids) + len(surface.validation_ids)))
    except (KeyError, ValueError):
        return SynthesisResult("unavailable", None, (), (), provenance={"reason": "unknown_family"})
    if require_identified and not lattice:
        return SynthesisResult("unavailable", None, (), (), provenance={"reason": "empty_decision_lattice"})
    grammar_payload = family_predicate_grammar(family_id)
    if not grammar_payload:
        return SynthesisResult("unavailable", None, (), (), provenance={"reason": "family_has_no_predicate_grammar"})
    observations: list[BoundaryObservation] = []
    # Callers with a campaign ledger pass it through unchanged.  Constructing
    # a second four-way budget here would silently spend only delta/4 of the
    # already allocated synthesis component.
    budget = statistical_budget or StatisticalBudget(delta_total=float(delta))
    context_delta = budget.lattice_delta(max(1, len(lattice)))
    for index, case in enumerate(cases):
        context = case.get("context") if isinstance(case.get("context"), dict) else {}
        if not context:
            continue
        interval = _case_effect_interval(case, delta=context_delta)
        if interval is None:
            continue
        effect, lower, upper = interval
        observations.append(BoundaryObservation(
            str(case.get("case_id", f"case-{index}")), context, effect,
            bool(case.get("scientific_ok", False)) and bool(case.get("quality_ok", True)),
            lower, upper,
        ))
    if not observations:
        return SynthesisResult("insufficient_evidence", None, (), (), provenance={"reason": "no_certified_observations"})
    observed_paths: set[str] = set()
    for observation in observations:
        for feature in grammar_payload["features"]:
            value: Any = observation.context
            for part in str(feature["path"]).split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value is not None:
                observed_paths.add(str(feature["path"]))
    grammar_payload["features"] = [feature for feature in grammar_payload["features"] if feature["path"] in observed_paths]
    grammar_payload["threshold_universe"] = {path: values for path, values in grammar_payload.get("threshold_universe", {}).items() if path in observed_paths}
    if not grammar_payload["features"]:
        return SynthesisResult("insufficient_evidence", None, (), (), provenance={"reason": "no_observed_public_features"})
    result = synthesize_boundary(
        observations, grammar_payload, decision_contexts=lattice,
        delta=budget.synth, statistical_budget=budget,
        epsilon_true=epsilon_true, epsilon_false=epsilon_false,
    )
    return result
