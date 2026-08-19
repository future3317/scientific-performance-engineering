"""Frozen per-trial budgets used by the formal campaign driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Budget:
    tokens: int
    tool_calls: int
    wall_time_s: float
    context_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.tokens < 1 or self.tool_calls < 1 or self.wall_time_s <= 0 or self.context_tokens < 1:
            raise ValueError("budget values must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check_cost(self, cost: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("tokens", "tool_calls"):
            value = cost.get(key)
            if value is not None and int(value) > int(getattr(self, key)):
                errors.append(f"{key} exceeded: {value} > {getattr(self, key)}")
        wall = cost.get("wall_time_s")
        if isinstance(wall, (int, float)) and float(wall) > self.wall_time_s:
            errors.append(f"wall_time_s exceeded: {wall} > {self.wall_time_s}")
        return errors

    def validate_usage(self, usage: dict[str, Any]) -> list[str]:
        """Validate the required machine-readable usage receipt for a trial."""
        if not isinstance(usage, dict):
            return ["agent_usage must be an object"]
        required = ("input_tokens", "output_tokens", "tool_calls", "wall_time_s")
        errors = [f"agent_usage missing {key}" for key in required if key not in usage]
        if errors:
            return errors
        for key in ("input_tokens", "output_tokens", "tool_calls"):
            value = usage[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"agent_usage {key} must be a non-negative integer")
        wall = usage["wall_time_s"]
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or float(wall) < 0:
            errors.append("agent_usage wall_time_s must be a non-negative number")
        if errors:
            return errors
        total_tokens = int(usage["input_tokens"]) + int(usage["output_tokens"])
        if total_tokens > self.tokens:
            errors.append(f"tokens exceeded: {total_tokens} > {self.tokens}")
        if int(usage["tool_calls"]) > self.tool_calls:
            errors.append(f"tool_calls exceeded: {usage['tool_calls']} > {self.tool_calls}")
        if float(usage["wall_time_s"]) > self.wall_time_s:
            errors.append(f"wall_time_s exceeded: {usage['wall_time_s']} > {self.wall_time_s}")
        return errors


@dataclass(frozen=True)
class EvolutionComputeBudget:
    """Separate accounting envelope for D-side replay/maintenance compute."""

    replay_executions: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    tokens: int = 0
    wall_time_s: float = 0.0
    usd: float = 0.0

    LIMITS = {
        "replay_executions": 256,
        "cpu_seconds": 3600.0,
        "gpu_seconds": 1800.0,
        "tokens": 500_000,
        "wall_time_s": 7200.0,
        "usd": 100.0,
    }

    def add(self, **delta: float | int) -> "EvolutionComputeBudget":
        values = asdict(self)
        for key, value in delta.items():
            if key in values:
                values[key] += value
        return EvolutionComputeBudget(**values)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "limits": dict(self.LIMITS)}

    def exceeds_limits(self) -> list[str]:
        return [key for key, limit in self.LIMITS.items() if float(getattr(self, key)) > float(limit)]


def classify_failure(*, failure_stage: str | None, protocol_failure: bool = False, agent_crashed: bool = False, budget_exhausted: bool = False) -> dict[str, Any]:
    """Canonical missingness policy used by formal scheduling and reruns."""
    if protocol_failure or failure_stage in {"executor", "hardware", "protocol"}:
        return {"class": "infrastructure_invalid", "efficacy_eligible": False, "rerunnable": True}
    if agent_crashed or budget_exhausted or failure_stage in {"agent", "worker"}:
        return {"class": "outcome_failure", "efficacy_eligible": True, "score": 0.0, "rerunnable": False}
    return {"class": "outcome_failure", "efficacy_eligible": True, "score": 0.0, "rerunnable": False}


def parse_budget(value: dict[str, Any] | None) -> Budget:
    value = value or {}
    return Budget(
        tokens=int(value.get("tokens", 12000)),
        tool_calls=int(value.get("tool_calls", 80)),
        wall_time_s=float(value.get("wall_time_s", 900.0)),
        context_tokens=int(value.get("context_tokens", 4096)),
    )
