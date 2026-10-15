"""Core-owned statistical CEGIS over the finite predicate grammar."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
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
    search_exhausted: bool = False
    candidates_examined: int = 0

    def to_dict(self, *, include_version_space: bool = False) -> dict[str, Any]:
        provenance = dict(self.provenance or {})
        provenance.pop("version_space", None)
        version_digest = (
            (self.certificate.version_space_digest if self.certificate else None)
            or provenance.get("version_space_digest")
        )
        if version_digest is None:
            version_digest = hashlib.sha256(
                json.dumps(list(self.version_space), sort_keys=True, separators=(",", ":"), default=str).encode()
            ).hexdigest()
        complexities = [predicate_complexity(item) for item in self.version_space]
        profile = {
            "depth_counts": dict(sorted(Counter(item["depth"] for item in complexities).items())),
            "literal_counts": dict(sorted(Counter(item["literals"] for item in complexities).items())),
            "description_length_min": min((item["description_length"] for item in complexities), default=0),
            "description_length_max": max((item["description_length"] for item in complexities), default=0),
        }
        value = {
            "status": self.status,
            "predicate": self.predicate,
            "certified_counterexamples": list(self.certified_counterexamples),
            "positive_anchors": list(self.positive_anchors),
            "synthesizer_version": self.synthesizer_version,
            "provenance": provenance or None,
            "version_space_size": len(self.version_space),
            "version_space_digest": version_digest,
            "decision_equivalence_classes": int(provenance.get("decision_equivalence_classes", 0)),
            "complexity_profile": profile,
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "search_exhausted": self.search_exhausted,
            "candidates_examined": self.candidates_examined,
        }
        if include_version_space:
            value["version_space"] = list(self.version_space)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SynthesisResult":
        """Read legacy full-version-space records and compact current records."""
        provenance = dict(value.get("provenance") or {})
        raw_space = value.get("version_space", provenance.get("version_space", ()))
        version_space = tuple(dict(item) for item in raw_space) if isinstance(raw_space, (list, tuple)) else ()
        raw_certificate = value.get("certificate")
        certificate = None
        if isinstance(raw_certificate, Mapping):
            certificate = SynthesisCertificate(
                status=str(raw_certificate.get("status", "")),
                predicate=raw_certificate.get("predicate"),
                positive_anchor_ids=tuple(raw_certificate.get("positive_anchor_ids", ())),
                counterexample_ids=tuple(raw_certificate.get("counterexample_ids", ())),
                unresolved_ids=tuple(raw_certificate.get("unresolved_ids", ())),
                grammar_digest=str(raw_certificate.get("grammar_digest", "")),
                decision_lattice_digest=str(raw_certificate.get("decision_lattice_digest", "")),
                version_space_digest=str(raw_certificate.get("version_space_digest", "")),
                alpha_budget=float(raw_certificate.get("alpha_budget", 0.05)),
                practical_threshold=float(raw_certificate.get("practical_threshold", 0.0)),
                synthesizer_version=str(raw_certificate.get("synthesizer_version", value.get("synthesizer_version", "acre-predicate-synth-v2"))),
                support_scope_digest=raw_certificate.get("support_scope_digest"),
                evidence_contexts=tuple(dict(item) for item in raw_certificate.get("evidence_contexts", ())),
            )
        return cls(
            status=str(value.get("status", "unavailable")),
            predicate=dict(value["predicate"]) if isinstance(value.get("predicate"), Mapping) else None,
            certified_counterexamples=tuple(value.get("certified_counterexamples", ())),
            positive_anchors=tuple(value.get("positive_anchors", ())),
            synthesizer_version=str(value.get("synthesizer_version", "acre-predicate-synth-v2")),
            provenance=provenance or None,
            version_space=version_space,
            certificate=certificate,
            search_exhausted=bool(value.get("search_exhausted", provenance.get("search_exhausted", False))),
            candidates_examined=int(value.get("candidates_examined", provenance.get("candidates_examined", 0))),
        )

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
    support_scope_digest: str | None = None
    evidence_contexts: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
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
        if self.support_scope_digest is not None:
            value["support_scope_digest"] = self.support_scope_digest
            value["evidence_contexts"] = [dict(context) for context in self.evidence_contexts]
        return value


def synthesis_certificate_valid(result: SynthesisResult) -> bool:
    certificate = result.certificate
    if not (
        result.status == "identified"
        and not result.search_exhausted
        and result.predicate is not None
        and result.version_space
        and certificate is not None
        and certificate.status == "identified"
        and certificate.predicate == result.predicate
        and certificate.positive_anchor_ids
        and certificate.counterexample_ids
        and certificate.grammar_digest
        and certificate.decision_lattice_digest
        and certificate.version_space_digest
    ):
        return False
    digest = hashlib.sha256(
        json.dumps(list(result.version_space), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return digest == certificate.version_space_digest

class StatisticalCEGIS:
    def __init__(
        self,
        grammar: PredicateGrammar,
        *,
        epsilon_true: float = 0.0,
        epsilon_false: float = 0.0,
        delta: float = 0.05,
        max_candidates: int = 100_000,
        max_wall_time: float = 5.0,
    ) -> None:
        self.grammar = grammar
        self.epsilon_true = epsilon_true
        self.epsilon_false = epsilon_false
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if max_candidates < 1 or max_wall_time <= 0:
            raise ValueError("search limits must be positive")
        self.delta = float(delta)
        self.max_candidates = int(max_candidates)
        self.max_wall_time = float(max_wall_time)

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
        certificate_scope: Mapping[str, Any] | None = None,
        evidence_contexts: tuple[Mapping[str, Any], ...] = (),
        vocabulary_contexts: list[Mapping[str, Any]] | None = None,
    ) -> SynthesisCertificate:
        grammar_payload = {
            "schema_version": self.grammar.schema_version,
            "features": list(self.grammar.features),
            "max_depth": self.grammar.max_depth,
            "max_literals": self.grammar.max_literals,
            "threshold_universe": self.grammar.threshold_universe or {},
            "feature_vocabulary_contexts": list(vocabulary_contexts or []),
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
            support_scope_digest=(
                digest(certificate_scope) if certificate_scope is not None else None
            ),
            evidence_contexts=evidence_contexts if certificate_scope is not None else (),
        )

    def synthesize(
        self,
        *,
        positive: list[BoundaryObservation],
        counterexamples: list[BoundaryObservation],
        parent_predicate: dict[str, Any] | None,
        decision_contexts: list[Mapping[str, Any]] | None = None,
        vocabulary_contexts: list[Mapping[str, Any]] | None = None,
        certificate_scope: Mapping[str, Any] | None = None,
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
        # Rebuild the vocabulary on every call so new evidence can introduce
        # thresholds without reusing a stale first-round hypothesis space.
        # Evaluation/decision contexts define where hypotheses are compared,
        # not which thresholds are available. Transductive callers must opt
        # in by passing those contexts explicitly as vocabulary_contexts.
        hypothesis_contexts = contexts + list(vocabulary_contexts or [])
        search = self.grammar.search_candidates(
            hypothesis_contexts,
            parent_predicate=parent_predicate,
            max_candidates=self.max_candidates,
            max_wall_time=self.max_wall_time,
            comparison_contexts=decision_contexts,
            preserve_distinct=decision_contexts is None,
        )
        provenance["search_exhausted"] = search.search_exhausted
        provenance["candidates_examined"] = search.candidates_examined
        provenance["search_elapsed_seconds"] = search.elapsed_seconds
        provenance["hypothesis_vocabulary_digest"] = search.vocabulary_digest
        if search.search_exhausted:
            provenance["reason"] = "predicate_search_budget_exhausted"
            certificate = self._certificate(
                status="search_exhausted", predicate=None, anchors=anchor_ids,
                counterexamples=counterexample_ids,
                observation_ids=tuple(item.observation_id for item in positive + counterexamples),
                decision_contexts=decision_contexts,
            )
            provenance["certificate"] = certificate.to_dict()
            return SynthesisResult(
                "search_exhausted", None, counterexample_ids, anchor_ids,
                provenance=provenance, certificate=certificate,
                search_exhausted=True, candidates_examined=search.candidates_examined,
            )
        hypothesis_space = search.candidates
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
        provenance["hypothesis_vocabulary_context_count"] = len(hypothesis_contexts)
        provenance["decision_contexts_in_vocabulary"] = bool(vocabulary_contexts)
        provenance["version_space_digest"] = hashlib.sha256(json.dumps(list(consistent), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        provenance["version_space"] = list(consistent)
        # A consistent predicate is identified only when every remaining
        # hypothesis makes the same deploy/no-deploy decision on the
        # observable decision contexts.  The sealed pool is deliberately not
        # accepted here; it is an offline score only.
        if decision_contexts is None:
            status = "underidentified"
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
            certificate_scope=certificate_scope,
            evidence_contexts=tuple(item.context for item in anchors + certified),
            vocabulary_contexts=hypothesis_contexts,
        )
        provenance["certificate"] = certificate.to_dict()
        return SynthesisResult(
            status, predicate, counterexample_ids, anchor_ids,
            provenance=provenance, version_space=tuple(consistent), certificate=certificate,
            candidates_examined=search.candidates_examined,
        )


def synthesize_boundary(
    observations: list[BoundaryObservation],
    grammar_payload: Mapping[str, Any],
    *,
    decision_contexts: list[Mapping[str, Any]] | None = None,
    delta: float = 0.05,
    statistical_budget: StatisticalBudget | None = None,
    epsilon_true: float = 0.0,
    epsilon_false: float = 0.0,
    max_candidates: int = 100_000,
    max_wall_time: float = 5.0,
    vocabulary_contexts: list[Mapping[str, Any]] | None = None,
    certificate_scope: Mapping[str, Any] | None = None,
) -> SynthesisResult:
    """Single core-owned entry point for harness boundary synthesis."""
    grammar = PredicateGrammar.from_dict(dict(grammar_payload))
    positives = [item for item in observations if item.positive_anchor(epsilon_true)]
    negatives = [item for item in observations if item.certified_counterexample(epsilon_false)]
    alpha = statistical_budget.synth if statistical_budget is not None else float(delta)
    return StatisticalCEGIS(
        grammar, epsilon_true=epsilon_true, epsilon_false=epsilon_false, delta=alpha,
        max_candidates=max_candidates, max_wall_time=max_wall_time,
    ).synthesize(
        positive=positives,
        counterexamples=negatives,
        parent_predicate=None,
        decision_contexts=decision_contexts,
        vocabulary_contexts=vocabulary_contexts,
        certificate_scope=certificate_scope,
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
    max_candidates: int = 100_000,
    max_wall_time: float = 5.0,
    certificate_scope: Mapping[str, Any] | None = None,
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
        max_candidates=max_candidates, max_wall_time=max_wall_time,
        certificate_scope=certificate_scope,
    )
    return result
