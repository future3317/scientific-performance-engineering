"""Adaptive query acquisition with sealed, offline-only evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence


class AcquisitionPolicy(str, Enum):
    RANDOM = "random"
    UNCERTAINTY_ONLY = "uncertainty-only"
    DECISION_AWARE = "decision-aware"


@dataclass(frozen=True)
class AcquisitionQuery:
    query_id: str
    edge_id: str
    uncertainty: float
    decision_value: float
    cost: float

    def __post_init__(self) -> None:
        if not self.query_id or not self.edge_id:
            raise ValueError("query_id and edge_id must be non-empty")
        if not 0.0 <= self.uncertainty <= 1.0 or not 0.0 <= self.decision_value <= 1.0:
            raise ValueError("uncertainty and decision_value must be in [0, 1]")
        if self.cost <= 0.0:
            raise ValueError("cost must be positive")


@dataclass(frozen=True)
class AcquisitionResult:
    policy: str
    selected_query_ids: tuple[str, ...]
    cumulative_cost: tuple[float, ...]
    stopped_by_policy: bool
    posterior: Mapping[str, Mapping[str, float]]
    selection_trace: tuple[dict[str, object], ...]

    @property
    def total_cost(self) -> float:
        return self.cumulative_cost[-1] if self.cumulative_cost else 0.0


@dataclass(frozen=True)
class OfflineEvaluation:
    cost_to_target: float | None
    final_error: float
    target_reached: bool
    error_trajectory: tuple[float, ...]


def _posterior(observations: Mapping[str, list[bool]], edge_id: str) -> tuple[float, float]:
    values = observations.get(edge_id, [])
    if not values:
        return 0.5, 0.0
    # Beta(1, 1) posterior mean plus an observable agreement confidence.  The
    # latter is deliberately derived only from revealed observations; hidden
    # truth is never used as a stopping signal.
    mean = (1.0 + sum(values)) / (2.0 + len(values))
    confidence = max(sum(values), len(values) - sum(values)) / len(values)
    return mean, confidence


def _dynamic_uncertainty(query: AcquisitionQuery, observations: Mapping[str, list[bool]]) -> float:
    values = observations.get(query.edge_id, [])
    if not values:
        return query.uncertainty
    mean, _ = _posterior(observations, query.edge_id)
    return min(query.uncertainty, 4.0 * mean * (1.0 - mean))


def _choose(
    available: Sequence[AcquisitionQuery],
    policy: AcquisitionPolicy,
    observations: Mapping[str, list[bool]],
    rng: random.Random,
    decision_value_fn: Callable[[AcquisitionQuery, Mapping[str, list[bool]]], float] | None,
) -> tuple[AcquisitionQuery, float, float]:
    if not available:
        raise ValueError("no queries available")
    if policy is AcquisitionPolicy.RANDOM:
        query = available[rng.randrange(len(available))]
        uncertainty = _dynamic_uncertainty(query, observations)
        return query, uncertainty, query.decision_value * (0.5 + uncertainty)
    scored: list[tuple[tuple[float, str], AcquisitionQuery, float, float]] = []
    for query in available:
        uncertainty = _dynamic_uncertainty(query, observations)
        decision = decision_value_fn(query, observations) if decision_value_fn else query.decision_value * (0.5 + uncertainty)
        decision = min(1.0, max(0.0, decision))
        if not 0.0 <= decision <= 1.0:
            raise ValueError("decision_value_fn must return a value in [0, 1]")
        score = uncertainty if policy is AcquisitionPolicy.UNCERTAINTY_ONLY else uncertainty * decision / query.cost
        scored.append(((score, query.query_id), query, uncertainty, decision))
    _, query, uncertainty, decision = max(scored, key=lambda item: item[0])
    return query, uncertainty, decision


def run_acquisition(
    queries: Sequence[AcquisitionQuery],
    sealed_labels: Mapping[str, bool],
    policy: AcquisitionPolicy,
    *,
    confidence_target: float = 0.9,
    seed: int = 0,
    decision_value_fn: Callable[[AcquisitionQuery, Mapping[str, list[bool]]], float] | None = None,
) -> AcquisitionResult:
    """Select until the observable posterior is confident or the pool ends.

    The sealed labels are revealed only after each query is selected.  No
    ground-truth edge labels or target error are accepted here; those belong to
    :func:`evaluate_trajectory`, which is an offline evaluator.
    """
    if not 0.0 < confidence_target <= 1.0:
        raise ValueError("confidence_target must be in (0, 1]")
    by_id = {query.query_id: query for query in queries}
    if len(by_id) != len(queries) or set(by_id) != set(sealed_labels):
        raise ValueError("sealed labels must cover the unique query pool")
    available = list(sorted(queries, key=lambda query: query.query_id))
    edge_ids = {query.edge_id for query in queries}
    observations: dict[str, list[bool]] = {}
    selected: list[str] = []
    cumulative: list[float] = []
    trace: list[dict[str, object]] = []
    rng = random.Random(seed)
    cost = 0.0
    stopped_by_policy = False
    while available:
        if observations and all(
            edge_id in observations and _posterior(observations, edge_id)[1] >= confidence_target
            for edge_id in edge_ids
        ):
            stopped_by_policy = True
            break
        frontier = [
            query for query in available
            if _posterior(observations, query.edge_id)[1] < confidence_target
        ]
        query, uncertainty, decision = _choose(frontier or available, policy, observations, rng, decision_value_fn)
        available.remove(query)
        # The observation is revealed only after selection; selection saw only
        # the posterior state accumulated from earlier revealed observations.
        observations.setdefault(query.edge_id, []).append(bool(sealed_labels[query.query_id]))
        selected.append(query.query_id)
        cost += query.cost
        cumulative.append(cost)
        trace.append({
            "query_id": query.query_id,
            "edge_id": query.edge_id,
            "uncertainty": uncertainty,
            "decision_value": decision,
            "cost": query.cost,
        })
    posterior = {
        edge_id: {"mean": _posterior(observations, edge_id)[0], "confidence": _posterior(observations, edge_id)[1]}
        for edge_id in sorted(edge_ids)
    }
    return AcquisitionResult(policy.value, tuple(selected), tuple(cumulative), stopped_by_policy, posterior, tuple(trace))


def evaluate_trajectory(
    trajectory: AcquisitionResult,
    queries: Sequence[AcquisitionQuery],
    sealed_labels: Mapping[str, bool],
    edge_truths: Mapping[str, bool],
    *,
    target_error: float,
) -> OfflineEvaluation:
    """Evaluate a completed trajectory with hidden truth, after selection."""
    if not 0.0 <= target_error <= 1.0:
        raise ValueError("target_error must be in [0, 1]")
    by_id = {query.query_id: query for query in queries}
    observations: dict[str, list[bool]] = {}
    errors: list[float] = []
    cost_to_target: float | None = None
    cost = 0.0
    for index, query_id in enumerate(trajectory.selected_query_ids):
        query = by_id[query_id]
        observations.setdefault(query.edge_id, []).append(bool(sealed_labels[query_id]))
        cost += query.cost
        wrong = 0
        for edge_id, truth in edge_truths.items():
            values = observations.get(edge_id, [])
            if not values or (sum(values) * 2 >= len(values)) != truth:
                wrong += 1
        error = wrong / len(edge_truths) if edge_truths else 0.0
        errors.append(error)
        if cost_to_target is None and error <= target_error:
            cost_to_target = cost
    final_error = errors[-1] if errors else (1.0 if edge_truths else 0.0)
    return OfflineEvaluation(cost_to_target, final_error, cost_to_target is not None, tuple(errors))
