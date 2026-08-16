"""Core execution loop for active node and relation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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


class ReplaySequentialCertificate:
    """Group-level sequential certificate for paired replay."""

    def __init__(self, *, minimum_groups: int, epsilon: float = 0.0, delta: float = 0.05, p_min: float = 0.8) -> None:
        self.minimum_groups = int(minimum_groups)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        if not 0.0 < float(p_min) <= 1.0:
            raise ValueError("p_min must be in (0, 1]")
        self.p_min = float(p_min)

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
            if failures > len(intervals) - self.minimum_groups and len(intervals) >= self.minimum_groups:
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
            for arm_name, measurements in (("on", on["measurements"]), ("off", off["measurements"])):
                for repetition, value in enumerate(measurements):
                    evidence_events.append(EvidenceEvent(
                        schema_version=2,
                        event_id=f"{plan.subject_id}:{group_id}:{arm_name}:{repetition}",
                        context=dict(case["context"]),
                        assignment={"interventions": {plan.subject_id: 1 if arm_name == "on" else 0}, "propensity": 0.5, "design_id": plan.design},
                        outcome_vector={"utility": float(value)},
                        scientific_gates={"paired": bool(case["scientific_ok"])},
                        artifacts={"case_id": case["case_id"], "repetition": repetition},
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


__all__ = ["ExperimentExecutor", "ExperimentPlan", "ExperimentExecution", "ReplaySequentialCertificate", "execute_paired_plan"]
