"""Core execution loop for active node and relation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.sequential_stats import paired_repetition_interval
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


class ReplaySequentialCertificate:
    """Group-level sequential certificate for paired replay."""

    def __init__(self, *, minimum_groups: int, epsilon: float = 0.0, delta: float = 0.05) -> None:
        self.minimum_groups = int(minimum_groups)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

    def update(self, cases: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        representative = [case for case in cases if case.get("query_type", "representative") == "representative"]
        intervals: list[tuple[float, float]] = []
        for case in representative:
            on = case.get("intervention_measurements")
            off = case.get("baseline_measurements")
            if not isinstance(on, list) or not isinstance(off, list) or len(on) != len(off) or not on:
                continue
            effects = [utility_effect(float(a), float(b), higher_is_better=bool(case.get("higher_is_better", True)), log_scale=float(case.get("utility_scale", 0.5))) for a, b in zip(on, off)]
            intervals.append(paired_repetition_interval(effects, self.delta))
        if len(intervals) >= self.minimum_groups and all(lower > self.epsilon for lower, _ in intervals):
            return {"status": "passed", "stop": True, "groups": len(intervals)}
        return {"status": "collecting", "groups": len(intervals)}


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
        }
        if isinstance(on.get("measurements"), list) and isinstance(off.get("measurements"), list):
            if len(on["measurements"]) != len(off["measurements"]) or not on["measurements"]:
                raise ValueError("paired executor measurements must be non-empty and equal length")
            case["intervention_measurements"] = list(on["measurements"])
            case["baseline_measurements"] = list(off["measurements"])
            case["control_measured"] = True
            case["scientific_ok"] = bool(on.get("scientific_ok", False) and off.get("scientific_ok", False))
        record_case(case)
        cases.append(case)
        certificate = dict(update_certificate(tuple(cases)))
        if certificate.get("stop") is True or certificate.get("status") in {"passed", "failed", "futility"}:
            return ExperimentExecution(tuple(cases), str(certificate.get("status", "certificate")), len(cases), certificate)
    certificate = dict(update_certificate(tuple(cases)))
    return ExperimentExecution(tuple(cases), "plan_exhausted", len(cases), certificate)


__all__ = ["ExperimentExecutor", "ExperimentPlan", "ExperimentExecution", "ReplaySequentialCertificate", "execute_paired_plan"]
