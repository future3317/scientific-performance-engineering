"""Conservative global/selective ACRE support classification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from core.predicates import match_predicate
from core.sequential_stats import bounded_mean_interval, mixture_lower_bound

from .cegis import SynthesisResult, synthesis_certificate_valid


class SupportRegime(str, Enum):
    GLOBAL_VERIFIED = "GLOBAL_VERIFIED"
    SELECTIVE_IDENTIFIED = "SELECTIVE_IDENTIFIED"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class SupportScope:
    population_id: str
    population_predicate: Mapping[str, Any]
    versions: Mapping[str, str]
    required_strata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_id": self.population_id,
            "population_predicate": dict(self.population_predicate),
            "versions": dict(self.versions),
            "required_strata": list(self.required_strata),
        }

    def covers(self, context: Mapping[str, Any], versions: Mapping[str, str]) -> bool:
        return (
            bool(self.population_id)
            and match_predicate(self.population_predicate, context)
            and dict(versions) == dict(self.versions)
        )


@dataclass(frozen=True)
class SupportGroup:
    group_id: str
    context: Mapping[str, Any]
    effect_lower: float
    effect_upper: float
    hard_gates_passed: bool
    stratum: str | None = None
    versions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("independence group_id is required")
        if not (-1.0 <= self.effect_lower <= self.effect_upper <= 1.0):
            raise ValueError("group effect interval must be ordered within [-1, 1]")


@dataclass(frozen=True)
class SupportRegimeDecision:
    regime: SupportRegime
    action: str
    reason_code: str
    scope: SupportScope
    predicate: Mapping[str, Any] | None = None
    independent_groups: int = 0
    net_benefit_lcb: float | None = None
    harm_ucb: float | None = None
    stratum_bounds: Mapping[str, Mapping[str, float]] | None = None
    source_certificate_digest: str | None = None
    certificate_scope_digest: str | None = None
    certificate_digest: str | None = None

    def certificate_is_scope_bound(self) -> bool:
        if not (self.source_certificate_digest and self.certificate_scope_digest and self.certificate_digest):
            return False
        scope_digest = _scope_digest(self.scope)
        expected = _decision_certificate_digest(
            self.source_certificate_digest, scope_digest, self.regime, self.action,
            self.reason_code, self.predicate, self.independent_groups,
            self.net_benefit_lcb, self.harm_ucb, self.stratum_bounds,
        )
        return self.certificate_scope_digest == scope_digest and self.certificate_digest == expected

    def allows_reuse(self, context: Mapping[str, Any], versions: Mapping[str, str]) -> bool:
        if not self.certificate_is_scope_bound() or not self.scope.covers(context, versions):
            return False
        if self.regime == SupportRegime.GLOBAL_VERIFIED:
            return self.action == "reuse"
        return (
            self.regime == SupportRegime.SELECTIVE_IDENTIFIED
            and self.action == "reuse"
            and self.predicate is not None
            and match_predicate(self.predicate, context)
        )


def _identified_synthesis(result: SynthesisResult | None) -> bool:
    return result is not None and synthesis_certificate_valid(result)


def _scope_digest(scope: SupportScope) -> str:
    return hashlib.sha256(
        json.dumps(scope.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _decision_certificate_digest(
    source_digest: str,
    scope_digest: str,
    regime: SupportRegime,
    action: str,
    reason_code: str,
    predicate: Mapping[str, Any] | None,
    independent_groups: int,
    net_benefit_lcb: float | None,
    harm_ucb: float | None,
    stratum_bounds: Mapping[str, Mapping[str, float]] | None,
) -> str:
    payload = {
        "source_digest": source_digest,
        "scope_digest": scope_digest,
        "regime": regime.value,
        "action": action,
        "reason_code": reason_code,
        "predicate": predicate,
        "independent_groups": independent_groups,
        "net_benefit_lcb": net_benefit_lcb,
        "harm_ucb": harm_ucb,
        "stratum_bounds": stratum_bounds,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _bound_decision(
    regime: SupportRegime,
    action: str,
    reason_code: str,
    scope: SupportScope,
    source_certificate_digest: str,
    **kwargs: Any,
) -> SupportRegimeDecision:
    scope_digest = _scope_digest(scope)
    certificate_digest = _decision_certificate_digest(
        source_certificate_digest, scope_digest, regime, action, reason_code,
        kwargs.get("predicate"), kwargs.get("independent_groups", 0),
        kwargs.get("net_benefit_lcb"), kwargs.get("harm_ucb"), kwargs.get("stratum_bounds"),
    )
    return SupportRegimeDecision(
        regime, action, reason_code, scope,
        source_certificate_digest=source_certificate_digest,
        certificate_scope_digest=scope_digest,
        certificate_digest=certificate_digest,
        **kwargs,
    )


def _group_bounds(groups: Sequence[SupportGroup], delta: float) -> tuple[float, float] | None:
    if not groups or any(not (group.effect_upper < 0.0 or group.effect_lower >= 0.0) for group in groups):
        return None
    net_lcb, _ = bounded_mean_interval([group.effect_lower for group in groups], delta)
    definite_harms = sum(group.effect_upper < 0.0 for group in groups)
    no_harm_lcb = mixture_lower_bound(len(groups) - definite_harms, len(groups), delta)
    return net_lcb, 1.0 - no_harm_lcb


def classify_support_regime(
    groups: Sequence[SupportGroup],
    *,
    scope: SupportScope,
    target_context: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    selective: SynthesisResult | None = None,
    fixed_time_complete: bool = False,
    delta: float = 0.05,
    min_groups: int = 20,
    min_groups_per_stratum: int = 20,
    min_net_benefit: float = 0.0,
    max_harm_probability: float = 0.1,
) -> SupportRegimeDecision:
    """Classify support using independent, paired group intervals at a fixed look."""
    if not 0.0 < delta < 1.0 or min_groups < 1 or min_groups_per_stratum < 1:
        raise ValueError("delta and min_groups are out of range")
    if not -1.0 <= min_net_benefit <= 1.0 or not 0.0 <= max_harm_probability <= 1.0:
        raise ValueError("support thresholds are out of range")
    if not scope.covers(target_context, runtime_versions):
        return SupportRegimeDecision(SupportRegime.UNCERTAIN, "abstain", "out_of_scope_or_version", scope)

    if any(not group.hard_gates_passed for group in groups):
        return SupportRegimeDecision(SupportRegime.REJECTED, "disable", "hard_gate_failed", scope, independent_groups=len(groups))
    if any(
        dict(group.versions or {}) != dict(scope.versions)
        or not match_predicate(scope.population_predicate, group.context)
        for group in groups
    ):
        return SupportRegimeDecision(SupportRegime.UNCERTAIN, "abstain", "evidence_out_of_scope", scope, independent_groups=len(groups))
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        return SupportRegimeDecision(SupportRegime.UNCERTAIN, "abstain", "duplicate_independence_group", scope, independent_groups=len(set(group_ids)))

    net_lcb: float | None = None
    harm_ucb: float | None = None
    global_ready = fixed_time_complete and len(groups) >= min_groups and bool(groups)
    strata_groups = {
        stratum: [group for group in groups if group.stratum == stratum]
        for stratum in scope.required_strata
    }
    if scope.required_strata:
        global_ready = global_ready and all(group.stratum in scope.required_strata for group in groups)
    global_ready = global_ready and all(len(items) >= min_groups_per_stratum for items in strata_groups.values())
    confidence_comparisons = 1 + len(scope.required_strata)
    component_delta = delta / (2.0 * confidence_comparisons)
    global_ready = global_ready and all(
        group.effect_upper < 0.0 or group.effect_lower >= 0.0
        for group in groups
    )
    if global_ready:
        stratum_bounds: dict[str, dict[str, float]] = {}
        overall_bounds = _group_bounds(groups, component_delta)
        per_stratum = {
            stratum: _group_bounds(items, component_delta)
            for stratum, items in strata_groups.items()
        }
        if overall_bounds is None or any(bounds is None for bounds in per_stratum.values()):
            global_ready = False
        else:
            net_lcb, harm_ucb = overall_bounds
            stratum_bounds = {
                stratum: {"net_benefit_lcb": bounds[0], "harm_ucb": bounds[1]}
                for stratum, bounds in per_stratum.items()
                if bounds is not None
            }
        if global_ready and net_lcb > min_net_benefit and harm_ucb <= max_harm_probability and all(
            bounds["net_benefit_lcb"] > min_net_benefit and bounds["harm_ucb"] <= max_harm_probability
            for bounds in stratum_bounds.values()
        ):
            source_digest = hashlib.sha256(json.dumps({
                "group_ids": sorted(group_ids),
                "scope": scope.population_id,
                "delta": delta,
                "min_groups": min_groups,
                "min_groups_per_stratum": min_groups_per_stratum,
                "overall": {"net_benefit_lcb": net_lcb, "harm_ucb": harm_ucb},
                "strata": stratum_bounds,
            }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            return _bound_decision(
                SupportRegime.GLOBAL_VERIFIED, "reuse", "global_support_certified", scope,
                source_digest,
                independent_groups=len(groups), net_benefit_lcb=net_lcb,
                harm_ucb=harm_ucb, stratum_bounds=stratum_bounds,
            )

    if (
        _identified_synthesis(selective)
        and selective.certificate.support_scope_digest == _scope_digest(scope)
        and selective.certificate.evidence_contexts
        and all(match_predicate(scope.population_predicate, context) for context in selective.certificate.evidence_contexts)
    ):
        source_digest = hashlib.sha256(json.dumps(
            selective.certificate.to_dict(), sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        return _bound_decision(
            SupportRegime.SELECTIVE_IDENTIFIED, "reuse", "identified_selective_support", scope,
            source_digest, predicate=selective.predicate, independent_groups=len(groups),
            net_benefit_lcb=net_lcb, harm_ucb=harm_ucb,
        )
    reason = "global_support_not_certified" if groups else "insufficient_evidence"
    return SupportRegimeDecision(
        SupportRegime.UNCERTAIN, "abstain", reason, scope,
        independent_groups=len(groups), net_benefit_lcb=net_lcb, harm_ucb=harm_ucb,
    )
