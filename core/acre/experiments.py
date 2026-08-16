"""Core execution loop for active node and relation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.sequential_stats import paired_repetition_interval
from core.models import EvidenceEvent
from core.utility import utility_effect


class ExperimentExecutor(Protocol):
    def execute(self, context: Mapping[str, Any], *, arm: str = "on") -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ExperimentPlan:
    subject_id: str
    contexts: tuple[Mapping[str, Any], ...]
    design: str = "paired"
    max_groups: int = 29


@dataclass(frozen=True)
class ExperimentExecution:
    cases: tuple[Mapping[str, Any], ...]
    stop_reason: str
    groups_executed: int
    certificate: Mapping[str, Any] = field(default_factory=dict)
    evidence_events: tuple[EvidenceEvent, ...] = ()


@dataclass(frozen=True)
class PairedContrastEvidence:
    """One causal contrast per independence group.

    Raw arm observations remain in the case artifact.  This envelope is the
    only evidence object consumed by Core promotion assessment.
    """

    independence_group: str
    paired_effect: float
    lcb: float
    ucb: float
    source_on_event_ids: tuple[str, ...]
    source_off_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "independence_group": self.independence_group,
            "effect": self.paired_effect,
            "paired_effect": self.paired_effect,
            "lcb": self.lcb,
            "ucb": self.ucb,
            "source_on_event_ids": list(self.source_on_event_ids),
            "source_off_event_ids": list(self.source_off_event_ids),
        }


def _paired_effect(on: float, off: float, *, higher_is_better: bool, scale: float) -> float:
    """Return the canonical dimensionless effect, including zero-valued controls."""
    try:
        return utility_effect(on, off, higher_is_better=higher_is_better, log_scale=scale)
    except ValueError:
        if not all(math.isfinite(value) for value in (on, off)):
            raise
        direction = 1.0 if higher_is_better else -1.0
        return max(-1.0, min(1.0, direction * (on - off)))


class ReplaySequentialCertificate:
    """Group-level sequential certificate for paired replay."""

    def __init__(self, *, minimum_groups: int, max_groups: int | None = None, epsilon: float = 0.0, delta: float = 0.05, p_min: float = 0.8) -> None:
        self.minimum_groups = int(minimum_groups)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        if not 0.0 < float(p_min) <= 1.0:
            raise ValueError("p_min must be in (0, 1]")
        self.p_min = float(p_min)
        self.max_groups = max(int(max_groups or self.minimum_groups * 3), self.minimum_groups)

    def update(self, cases: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        grouped: dict[str, Mapping[str, Any]] = {}
        for case in cases:
            if case.get("query_type", "representative") != "representative":
                continue
            group = str(case.get("independence_group") or case.get("case_id"))
            grouped.setdefault(group, case)
        representative = list(grouped.values())
        intervals: list[tuple[float, float]] = []
        for case in representative:
            on = case.get("intervention_measurements")
            off = case.get("baseline_measurements")
            if not isinstance(on, list) or not isinstance(off, list) or len(on) != len(off) or not on:
                continue
            effects = [utility_effect(float(a), float(b), higher_is_better=bool(case.get("higher_is_better", True)), log_scale=float(case.get("utility_scale", 0.5))) for a, b in zip(on, off)]
            intervals.append(paired_repetition_interval(effects, self.delta))
        successes = 0
        failures = 0
        probability_lcb = None
        if len(intervals) >= self.minimum_groups:
            # The success process is a bounded Bernoulli mixture over
            # independent replay groups, not an all-success count.
            successes = sum(lower > self.epsilon for lower, _ in intervals)
            failures = len(intervals) - successes
            from core.sequential_stats import mixture_lower_bound
            probability_lcb = mixture_lower_bound(successes, len(intervals), self.delta)
            if probability_lcb >= self.p_min:
                return {"status": "passed", "stop": True, "groups": len(intervals), "successes": successes, "failures": failures, "promotion_probability_lcb": probability_lcb}
            remaining = max(0, self.max_groups - len(intervals))
            # A failed prefix is futile only when even an all-success future
            # at the preregistered maximum cannot clear p_min.
            best_possible = max(
                mixture_lower_bound(successes + k, len(intervals) + k, self.delta)
                for k in range(remaining + 1)
            )
            if best_possible < self.p_min:
                return {"status": "futility", "stop": True, "groups": len(intervals), "successes": successes, "failures": failures, "promotion_probability_lcb": probability_lcb}
        return {"status": "collecting", "groups": len(intervals), "successes": successes, "failures": failures, "promotion_probability_lcb": probability_lcb}


def execute_paired_plan(
    plan: ExperimentPlan,
    executor: ExperimentExecutor,
    *,
    record_case: Callable[[Mapping[str, Any]], None],
    update_certificate: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> ExperimentExecution:
    """Run each planned context, persist immutable cases, and stop by CS."""
    if plan.design != "paired":
        raise ValueError("only paired experiment plans are supported")
    cases: list[Mapping[str, Any]] = []
    evidence_events: list[EvidenceEvent] = []
    for context in plan.contexts[: plan.max_groups]:
        on = dict(executor.execute(context, arm="on"))
        off = dict(executor.execute(context, arm="off"))
        group_id = str(context.get("independence_group", context.get("context_id", len(cases))))
        case = {
            "case_id": f"{plan.subject_id}:{group_id}",
            "context_id": str(context.get("context_id", group_id)),
            "context": dict(context.get("context", context)),
            "on": on,
            "off": off,
            "independence_group": group_id,
            "paired_replay": True,
            "query_type": context.get("query_type", "representative"),
            "experiment_cost": float(context.get("experiment_cost", 1.0)),
            "higher_is_better": bool(on.get("higher_is_better", True)),
            "utility_scale": float(on.get("utility_scale", 0.5)),
        }
        if isinstance(on.get("measurements"), list) and isinstance(off.get("measurements"), list):
            if len(on["measurements"]) != len(off["measurements"]) or not on["measurements"]:
                raise ValueError("paired executor measurements must be non-empty and equal length")
            case["intervention_measurements"] = list(on["measurements"])
            case["baseline_measurements"] = list(off["measurements"])
            case["control_measured"] = True
            case["scientific_ok"] = bool(on.get("scientific_ok", False) and off.get("scientific_ok", False))
            versions = dict(case["context"].get("rule_versions", {})) if isinstance(case["context"], Mapping) else {}
            if plan.subject_id not in versions:
                versions[plan.subject_id] = int(context.get("version", 1))
                case_context = dict(case["context"])
                case_context["rule_versions"] = versions
                case["context"] = case_context
            effects = [_paired_effect(float(a), float(b), higher_is_better=bool(case.get("higher_is_better", True)), scale=float(case.get("utility_scale", 0.5))) for a, b in zip(on["measurements"], off["measurements"])]
            lcb, ucb = paired_repetition_interval(effects, 0.05)
            contrast = PairedContrastEvidence(
                independence_group=group_id,
                paired_effect=sum(effects) / len(effects),
                lcb=lcb,
                ucb=ucb,
                source_on_event_ids=tuple(f"{plan.subject_id}:{group_id}:on:{index}" for index in range(len(on["measurements"]))),
                source_off_event_ids=tuple(f"{plan.subject_id}:{group_id}:off:{index}" for index in range(len(off["measurements"]))),
            )
            evidence_events.append(EvidenceEvent(
                schema_version=2,
                event_id=f"{plan.subject_id}:{group_id}:paired-contrast",
                context=dict(case["context"]),
                assignment={"interventions": {plan.subject_id: 1}, "propensity": 0.5, "design_id": plan.design},
                outcome_vector={"utility": contrast.paired_effect, "paired_effect": contrast.paired_effect, "contrast": "on-minus-off"},
                scientific_gates={"paired": bool(case["scientific_ok"])},
                artifacts={"case_id": case["case_id"], "paired_contrast": contrast.to_dict()},
                versions={str(key): str(value) for key, value in versions.items()},
                source_id="core-experiment",
                independence_group=group_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                evidence_stream="adversarial" if case["query_type"] == "adversarial" else "representative",
                query_id=str(case["context_id"]),
                trust_zone="harness",
                attacker_controlled_fields=[],
            ))
        record_case(case)
        cases.append(case)
        certificate = dict(update_certificate(tuple(cases)))
        if certificate.get("stop") is True or certificate.get("status") in {"passed", "failed", "futility"}:
            return ExperimentExecution(tuple(cases), str(certificate.get("status", "certificate")), len(cases), certificate, tuple(evidence_events))
    certificate = dict(update_certificate(tuple(cases)))
    return ExperimentExecution(tuple(cases), "plan_exhausted", len(cases), certificate, tuple(evidence_events))


__all__ = ["ExperimentExecutor", "ExperimentPlan", "ExperimentExecution", "PairedContrastEvidence", "ReplaySequentialCertificate", "execute_paired_plan"]
