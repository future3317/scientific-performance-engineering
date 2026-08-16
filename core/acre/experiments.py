"""Core execution loop for active node and relation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


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
            "context": dict(context.get("context", context)),
            "on": on,
            "off": off,
            "independence_group": group_id,
            "paired_replay": True,
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


__all__ = ["ExperimentExecutor", "ExperimentPlan", "ExperimentExecution", "execute_paired_plan"]
