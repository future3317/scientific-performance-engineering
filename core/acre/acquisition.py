"""Decision-aware acquisition over a finite, label-sealed query pool."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


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
    cost_to_target: float | None
    final_error: float
    target_reached: bool
    selection_trace: tuple[dict[str, object], ...]

    @property
    def total_cost(self) -> float:
        return self.cumulative_cost[-1] if self.cumulative_cost else 0.0


def _choose(
    available: Sequence[AcquisitionQuery],
    policy: AcquisitionPolicy,
    rng: random.Random,
) -> AcquisitionQuery:
    if not available:
        raise ValueError("no queries available")
    if policy is AcquisitionPolicy.RANDOM:
        return available[rng.randrange(len(available))]
    if policy is AcquisitionPolicy.UNCERTAINTY_ONLY:
        key = lambda query: (query.uncertainty, query.query_id)
    else:
        key = lambda query: (query.uncertainty * query.decision_value / query.cost, query.uncertainty, query.query_id)
    return max(available, key=key)


def _identification_error(
    observations: Mapping[str, list[bool]], truths: Mapping[str, bool]
) -> float:
    if not truths:
        return 0.0
    errors = 0
    for edge_id, truth in truths.items():
        values = observations.get(edge_id, [])
        if not values:
            errors += 1
            continue
        positives = sum(values)
        prediction = positives * 2 >= len(values)
        errors += prediction != truth
    return errors / len(truths)


def run_acquisition(
    queries: Sequence[AcquisitionQuery],
    sealed_labels: Mapping[str, bool],
    edge_truths: Mapping[str, bool],
    policy: AcquisitionPolicy,
    *,
    target_error: float = 0.1,
    seed: int = 0,
) -> AcquisitionResult:
    """Run one policy; sealed labels are revealed only after selection."""
    if not 0.0 <= target_error <= 1.0:
        raise ValueError("target_error must be in [0, 1]")
    by_id = {query.query_id: query for query in queries}
    if len(by_id) != len(queries) or set(by_id) != set(sealed_labels):
        raise ValueError("sealed labels must cover the unique query pool")
    if any(query.edge_id not in edge_truths for query in queries):
        raise ValueError("edge_truths must cover every query edge")
    available = list(sorted(queries, key=lambda query: query.query_id))
    observations: dict[str, list[bool]] = {}
    selected: list[str] = []
    cumulative: list[float] = []
    trace: list[dict[str, object]] = []
    rng = random.Random(seed)
    cost = 0.0
    reached: float | None = None
    error = _identification_error(observations, edge_truths)
    while available:
        unresolved = [query for query in available if query.edge_id not in observations]
        query = _choose(unresolved or available, policy, rng)
        available.remove(query)
        # Selection receives only public query metadata and prior observations.
        # The sealed label is read after the choice, at the experiment boundary.
        observations.setdefault(query.edge_id, []).append(bool(sealed_labels[query.query_id]))
        selected.append(query.query_id)
        cost += query.cost
        cumulative.append(cost)
        error = _identification_error(observations, edge_truths)
        trace.append({
            "query_id": query.query_id,
            "edge_id": query.edge_id,
            "uncertainty": query.uncertainty,
            "decision_value": query.decision_value,
            "cost": query.cost,
            "error": error,
        })
        if reached is None and error <= target_error:
            reached = cost
            break
    return AcquisitionResult(
        policy=policy.value,
        selected_query_ids=tuple(selected),
        cumulative_cost=tuple(cumulative),
        cost_to_target=reached,
        final_error=error,
        target_reached=reached is not None,
        selection_trace=tuple(trace),
    )
