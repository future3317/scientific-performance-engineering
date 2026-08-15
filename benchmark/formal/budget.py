"""Frozen per-trial budgets used by the formal campaign driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Budget:
    tokens: int
    tool_calls: int
    wall_time_s: float

    def __post_init__(self) -> None:
        if self.tokens < 1 or self.tool_calls < 1 or self.wall_time_s <= 0:
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


def parse_budget(value: dict[str, Any] | None) -> Budget:
    value = value or {}
    return Budget(
        tokens=int(value.get("tokens", 12000)),
        tool_calls=int(value.get("tool_calls", 80)),
        wall_time_s=float(value.get("wall_time_s", 900.0)),
    )
