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


@dataclass(frozen=True)
class PairReservation:
    """Separate task/evolution caps with an all-or-none paired-task reserve.

    The task and evolution caps use their respective caller-defined units.
    ``measured_pair_costs`` records actual task usage when known; ``None``
    remains explicitly unknown.
    """

    task_execution_budget: float
    evolution_search_budget: float
    pair_costs: tuple[tuple[str, float], ...]
    reserved: bool
    reason: str
    evolution_spent: float = 0.0
    completed_pairs: tuple[str, ...] = ()
    measured_pair_costs: tuple[tuple[str, float | None], ...] = ()

    def __post_init__(self) -> None:
        if self.task_execution_budget < 0.0 or self.evolution_search_budget < 0.0:
            raise ValueError("task and evolution budgets must be non-negative")
        ids = [pair_id for pair_id, _ in self.pair_costs]
        id_set = set(ids)
        if len(ids) != len(id_set) or any(not pair_id for pair_id in ids):
            raise ValueError("pair ids must be unique and non-empty")
        if any(cost < 0.0 for _, cost in self.pair_costs):
            raise ValueError("pair cost reservations must be non-negative")
        if self.reserved and sum(cost for _, cost in self.pair_costs) > self.task_execution_budget + 1e-12:
            raise ValueError("reserved pairs exceed task_execution_budget")
        if self.evolution_spent < 0.0 or self.evolution_spent > self.evolution_search_budget + 1e-12:
            raise ValueError("evolution spending exceeds its independent budget")
        if any(pair_id not in id_set for pair_id in self.completed_pairs):
            raise ValueError("completed pair is not in the frozen reservation")
        measured_ids = [pair_id for pair_id, _ in self.measured_pair_costs]
        if len(measured_ids) != len(set(measured_ids)) or any(pair_id not in id_set for pair_id in measured_ids):
            raise ValueError("measured pair costs must reference unique reserved pairs")
        if any(cost is not None and cost < 0.0 for _, cost in self.measured_pair_costs):
            raise ValueError("measured pair costs must be non-negative or unknown")

    @classmethod
    def reserve_all_pairs(
        cls,
        pair_costs: tuple[tuple[str, float], ...],
        *,
        task_execution_budget: float,
        evolution_search_budget: float,
    ) -> "PairReservation":
        """Reserve every matched pair, or reserve none of them."""
        if task_execution_budget < 0.0 or evolution_search_budget < 0.0:
            raise ValueError("task and evolution budgets must be non-negative")
        required = sum(cost for _, cost in pair_costs)
        fits = required <= task_execution_budget + 1e-12
        return cls(
            task_execution_budget=float(task_execution_budget),
            evolution_search_budget=float(evolution_search_budget),
            pair_costs=tuple((pair_id, float(cost)) for pair_id, cost in pair_costs),
            reserved=fits,
            reason="reserved" if fits else "task_execution_budget_insufficient",
        )

    @property
    def evolution_available(self) -> float:
        return max(0.0, self.evolution_search_budget - self.evolution_spent)

    @property
    def launch_allowed(self) -> bool:
        return self.reserved

    def record_evolution(self, cost: float) -> "PairReservation":
        if cost < 0.0:
            raise ValueError("evolution cost must be non-negative")
        if cost > self.evolution_available + 1e-12:
            raise ValueError("evolution_search_budget_insufficient")
        return replace(self, evolution_spent=self.evolution_spent + float(cost))

    def complete_pair(self, pair_id: str, measured_cost: float | None) -> "PairReservation":
        if not self.reserved:
            raise ValueError("cannot complete a pair before all pairs are reserved")
        pair_ids = {item for item, _ in self.pair_costs}
        if pair_id not in pair_ids:
            raise ValueError("pair_id is not in the frozen reservation")
        if pair_id in self.completed_pairs:
            raise ValueError("pair is already complete")
        if measured_cost is not None and measured_cost < 0.0:
            raise ValueError("measured_cost must be non-negative or unknown")
        return replace(
            self,
            completed_pairs=self.completed_pairs + (pair_id,),
            measured_pair_costs=self.measured_pair_costs + ((pair_id, measured_cost),),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "task_execution_budget": self.task_execution_budget,
            "evolution_search_budget": self.evolution_search_budget,
            "pair_costs": dict(self.pair_costs),
            "reserved": self.reserved,
            "reason": self.reason,
            "evolution_spent": self.evolution_spent,
            "evolution_available": self.evolution_available,
            "completed_pairs": list(self.completed_pairs),
            "measured_pair_costs": dict(self.measured_pair_costs),
        }


__all__ = ["EvolutionAdmission", "ExecutionReserveBudget", "PairReservation"]
