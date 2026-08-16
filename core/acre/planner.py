"""Decision-aware experiment planning over observable ACRE state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .acquisition import AcquisitionQuery


@dataclass(frozen=True)
class PlannedExperiment:
    query: AcquisitionQuery
    score: float
    information_gain: float
    decision_sensitivity: float
    risk: float
    cost: float


class ExperimentPlanner:
    """Score queries by version-space reduction and possible routing change."""

    def __init__(self, *, cost_floor: float = 0.1) -> None:
        self.cost_floor = float(cost_floor)

    @staticmethod
    def decision_sensitivity(
        query: AcquisitionQuery,
        observations: Mapping[str, list[bool]],
        simulate: Callable[[AcquisitionQuery, bool, Mapping[str, list[bool]]], Any] | None = None,
    ) -> float:
        if simulate is None:
            return 1.0 if not observations.get(query.edge_id) else 0.0
        outcomes = [simulate(query, value, observations) for value in (True, False)]
        signatures = [repr(value) for value in outcomes]
        return 1.0 if len(set(signatures)) > 1 else 0.0

    def rank(
        self,
        queries: Sequence[AcquisitionQuery],
        observations: Mapping[str, list[bool]],
        *,
        information_gain: Callable[[AcquisitionQuery, Mapping[str, list[bool]]], float],
        simulate: Callable[[AcquisitionQuery, bool, Mapping[str, list[bool]]], Any] | None = None,
    ) -> tuple[PlannedExperiment, ...]:
        planned = []
        for query in queries:
            ig = float(information_gain(query, observations))
            sensitivity = self.decision_sensitivity(query, observations, simulate)
            risk = float(query.risk)
            score = (ig * (1.0 + sensitivity) + risk * sensitivity) / (query.cost + self.cost_floor)
            planned.append(PlannedExperiment(query, score, ig, sensitivity, risk, query.cost))
        return tuple(sorted(planned, key=lambda item: (-item.score, item.query.query_id)))


__all__ = ["ExperimentPlanner", "PlannedExperiment"]
