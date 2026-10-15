"""One interpretation of ACRE synthesis and regime decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.predicates import match_predicate

from .cegis import SynthesisResult, synthesis_certificate_valid
from .regimes import SupportRegime, SupportRegimeDecision


@dataclass(frozen=True)
class GateDecision:
    action: str
    reason_code: str
    deployable: bool
    matched: bool | None = None
    mode: str = "operational"


def decision_from_synthesis(
    result: SynthesisResult,
    context: Mapping[str, Any],
    *,
    mode: str = "operational",
) -> GateDecision:
    if mode not in {"operational", "diagnostic"}:
        raise ValueError("mode must be 'operational' or 'diagnostic'")
    if result.predicate is None:
        return GateDecision("abstain", "no_predicate", False, mode=mode)
    matched = bool(match_predicate(result.predicate, context))
    if mode == "diagnostic":
        return GateDecision(
            "hypothesis_match" if matched else "hypothesis_nonmatch",
            "diagnostic_only_underidentified" if result.status != "identified" else "diagnostic_only",
            False,
            matched,
            mode,
        )
    if not synthesis_certificate_valid(result):
        return GateDecision("abstain", "underidentified_or_uncertified", False, matched, mode)
    return GateDecision("reuse" if matched else "abstain", "identified_match" if matched else "outside_predicate", matched, matched, mode)


def decision_from_regime(
    decision: SupportRegimeDecision,
    context: Mapping[str, Any],
    versions: Mapping[str, str],
    *,
    mode: str = "operational",
) -> GateDecision:
    if mode not in {"operational", "diagnostic"}:
        raise ValueError("mode must be 'operational' or 'diagnostic'")
    in_scope = decision.scope.covers(context, versions)
    matched = (
        decision.regime == SupportRegime.GLOBAL_VERIFIED
        or (decision.predicate is not None and match_predicate(decision.predicate, context))
    )
    allowed = in_scope and decision.allows_reuse(context, versions)
    if mode == "operational":
        return GateDecision("reuse" if allowed else "abstain", decision.reason_code if allowed else "regime_not_deployable_or_out_of_scope", allowed, bool(matched), mode)
    return GateDecision(
        "hypothesis_match" if in_scope and matched else "hypothesis_nonmatch",
        "diagnostic_only",
        False,
        bool(in_scope and matched),
        mode,
    )
