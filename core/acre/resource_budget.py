"""Resource accounting that reserves already scheduled execution work.

This ledger is deliberately independent from the statistical alpha budget.
It answers one operational question: may an optional evolution query consume
the portion of compute reserved for future scheduled cells?
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EvolutionAdmission:
    accepted: bool
    reason: str
    query_id: str
    cost: float
    evolution_available_before: float
    evolution_available_after: float


@dataclass(frozen=True)
class ExecutionReserveBudget:
    """Immutable, replayable ledger for optional evolution compute."""

    total_budget: float
    scheduled_minimums: tuple[float, ...]
    evolution_spent: float = 0.0
    task_index: int = 0
    evolution_frozen: bool = False

    def __post_init__(self) -> None:
        if self.total_budget < 0.0:
            raise ValueError("total_budget must be non-negative")
        if any(float(value) < 0.0 for value in self.scheduled_minimums):
            raise ValueError("scheduled execution minima must be non-negative")
        if not 0 <= self.task_index <= len(self.scheduled_minimums):
            raise ValueError("task_index must be within the scheduled task range")
        if self.evolution_spent < 0.0:
            raise ValueError("evolution_spent must be non-negative")
        if self.evolution_spent + self.execution_reserve > self.total_budget + 1e-12:
            raise ValueError("scheduled execution reserve exceeds total budget")

    @property
    def execution_reserve(self) -> float:
        return float(sum(self.scheduled_minimums[self.task_index:]))

    @property
    def evolution_available(self) -> float:
        return max(0.0, float(self.total_budget - self.evolution_spent - self.execution_reserve))

    @property
    def scheduled_execution_allowed(self) -> bool:
        """A frozen evolution ledger never blocks the scheduled task path."""
        return self.execution_reserve <= self.total_budget - self.evolution_spent + 1e-12

    def admit_evolution(self, cost: float, *, query_id: str) -> tuple["ExecutionReserveBudget", EvolutionAdmission]:
        if cost < 0.0 or not query_id:
            raise ValueError("evolution cost must be non-negative and query_id must be non-empty")
        before = self.evolution_available
        if self.evolution_frozen:
            return self, EvolutionAdmission(False, "evolution_frozen", query_id, float(cost), before, before)
        if float(cost) > before + 1e-12:
            frozen = replace(self, evolution_frozen=True)
            return frozen, EvolutionAdmission(False, "execution_reserve", query_id, float(cost), before, before)
        updated = replace(self, evolution_spent=self.evolution_spent + float(cost))
        if updated.evolution_available <= 1e-12:
            updated = replace(updated, evolution_frozen=True)
        return updated, EvolutionAdmission(True, "admitted", query_id, float(cost), before, updated.evolution_available)

    def advance_task(self, minimum_cost: float | None = None) -> "ExecutionReserveBudget":
        """Consume the current scheduled minimum and release its reservation."""
        if self.task_index >= len(self.scheduled_minimums):
            raise IndexError("no scheduled task remains")
        expected = self.scheduled_minimums[self.task_index]
        if minimum_cost is not None and abs(float(minimum_cost) - expected) > 1e-12:
            raise ValueError("minimum_cost does not match the frozen schedule")
        return replace(self, task_index=self.task_index + 1)

    def freeze_evolution(self) -> "ExecutionReserveBudget":
        return replace(self, evolution_frozen=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_budget": self.total_budget,
            "scheduled_minimums": list(self.scheduled_minimums),
            "evolution_spent": self.evolution_spent,
            "task_index": self.task_index,
            "execution_reserve": self.execution_reserve,
            "evolution_available": self.evolution_available,
            "evolution_frozen": self.evolution_frozen,
            "scheduled_execution_allowed": self.scheduled_execution_allowed,
        }


__all__ = ["EvolutionAdmission", "ExecutionReserveBudget"]
