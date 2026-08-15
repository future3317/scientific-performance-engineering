"""Adaptive query acquisition with sealed, offline-only evaluation."""

from __future__ import annotations

import random
import math
from functools import lru_cache
from dataclasses import dataclass, field
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
    cost: float
    context: Mapping[str, object] = field(default_factory=dict)
    subject_ids: tuple[str, ...] = ()
    experiment_type: str = "observation"
    risk: float = 0.5
    provenance_novelty: float = 0.5

    def __post_init__(self) -> None:
        if not self.query_id or not self.edge_id:
            raise ValueError("query_id and edge_id must be non-empty")
        if self.cost <= 0.0:
            raise ValueError("cost must be positive")
        if not 0.0 <= self.risk <= 1.0 or not 0.0 <= self.provenance_novelty <= 1.0:
            raise ValueError("risk and provenance_novelty must be in [0, 1]")
        if self.experiment_type == "":
            raise ValueError("experiment_type must be non-empty")


@dataclass(frozen=True)
class AcquisitionResult:
    policy: str
    selected_query_ids: tuple[str, ...]
    cumulative_cost: tuple[float, ...]
    stopped_by_policy: bool
    posterior: Mapping[str, Mapping[str, float]]
    selection_trace: tuple[dict[str, object], ...]
    identification_certificate: bool = False

    @property
    def total_cost(self) -> float:
        return self.cumulative_cost[-1] if self.cumulative_cost else 0.0


@dataclass(frozen=True)
class OfflineEvaluation:
    cost_to_target: float | None
    final_error: float
    target_reached: bool
    error_trajectory: tuple[float, ...]
    confusion_trajectory: tuple[Mapping[str, int], ...] = ()
    harmful_fp_trajectory: tuple[float, ...] = ()


@dataclass(frozen=True)
class IdentificationCertificate:
    """Observable stopping certificate; it never contains hidden truth."""

    certified: bool
    covered_edges: tuple[str, ...]
    min_confidence: float
    observations: int
    intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)


def _certificate(
    observations: Mapping[str, list[bool]],
    edge_ids: set[str],
    delta_family: float = 0.05,
) -> IdentificationCertificate:
    intervals = {edge_id: _confidence_sequence(observations.get(edge_id, []), delta_family / max(1, len(edge_ids))) for edge_id in edge_ids}
    covered_set = {edge_id for edge_id, (lower, upper) in intervals.items() if lower > 0.5 or upper < 0.5}
    covered = tuple(sorted(covered_set))
    confidences = [1.0 - min(1.0, 2.0 * (delta_family / max(1, len(edge_ids))) / (len(observations.get(edge_id, [])) * (len(observations.get(edge_id, [])) + 1))) for edge_id in edge_ids if observations.get(edge_id)]
    return IdentificationCertificate(
        certified=bool(edge_ids) and len(covered) == len(edge_ids),
        covered_edges=covered,
        min_confidence=min(confidences) if confidences else 0.0,
        observations=sum(len(values) for values in observations.values()),
        intervals=intervals,
    )


@lru_cache(maxsize=4096)
def _confidence_sequence_cached(values: tuple[bool, ...], delta_edge: float) -> tuple[float, float]:
    """Anytime Bernoulli CS via KL inversion and alpha spending.

    At look ``n`` we spend ``delta_edge/(n(n+1))``; the sum over all looks is
    at most ``delta_edge``.  The KL inversion is tighter than a fixed-sample
    Hoeffding interval while retaining the time-uniform guarantee.
    """
    n = len(values)
    if n == 0:
        return 0.0, 1.0
    mean = sum(values) / n
    delta_n = min(0.5, delta_edge / (n * (n + 1)))
    threshold = math.log(2.0 / delta_n) / n
    def kl(q: float, p: float) -> float:
        eps = 1e-12
        q = min(1.0 - eps, max(eps, q)); p = min(1.0 - eps, max(eps, p))
        return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    if mean <= 0.0:
        lower = 0.0
    else:
        lo, hi = 1e-12, mean
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if kl(mean, mid) > threshold: lo = mid
            else: hi = mid
        lower = hi
    if mean >= 1.0:
        upper = 1.0
    else:
        lo, hi = mean, 1.0 - 1e-12
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if kl(mean, mid) > threshold: hi = mid
            else: lo = mid
        upper = lo
    return lower, upper


def _confidence_sequence(values: Sequence[bool], delta_edge: float) -> tuple[float, float]:
    return _confidence_sequence_cached(tuple(values), round(delta_edge, 12))


def _posterior(observations: Mapping[str, list[bool]], edge_id: str) -> tuple[float, float]:
    values = observations.get(edge_id, [])
    if not values:
        return 0.5, 0.0
    # Beta(1, 1) posterior mean plus an observable agreement confidence.  The
    # latter is deliberately derived only from revealed observations; hidden
    # truth is never used as a stopping signal.
    mean = (1.0 + sum(values)) / (2.0 + len(values))
    agreement = max(sum(values), len(values) - sum(values)) / len(values)
    # A single observation is not a certificate.  The shrinkage term keeps
    # confidence observable while requiring repeated evidence before stopping.
    confidence = agreement * len(values) / (len(values) + 2.0)
    return mean, confidence


def _entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))


def _dynamic_uncertainty(query: AcquisitionQuery, observations: Mapping[str, list[bool]]) -> float:
    values = observations.get(query.edge_id, [])
    if not values:
        return 1.0
    mean, _ = _posterior(observations, query.edge_id)
    return _entropy(mean)


def _information_gain(query: AcquisitionQuery, observations: Mapping[str, list[bool]]) -> float:
    values = observations.get(query.edge_id, [])
    prior_mean, _ = _posterior(observations, query.edge_id)
    prior_entropy = _entropy(prior_mean)
    if not values:
        return prior_entropy
    expected = 0.0
    for outcome, probability in ((True, prior_mean), (False, 1.0 - prior_mean)):
        updated = list(values) + [outcome]
        posterior = (1.0 + sum(updated)) / (2.0 + len(updated))
        expected += probability * _entropy(posterior)
    return max(0.0, prior_entropy - expected)


def _choose(
    available: Sequence[AcquisitionQuery],
    policy: AcquisitionPolicy,
    observations: Mapping[str, list[bool]],
    rng: random.Random,
    decision_sensitivity_fn: Callable[[AcquisitionQuery, Mapping[str, list[bool]]], float] | None,
) -> tuple[AcquisitionQuery, float, float, float]:
    if not available:
        raise ValueError("no queries available")
    if policy is AcquisitionPolicy.RANDOM:
        query = available[rng.randrange(len(available))]
        uncertainty = _dynamic_uncertainty(query, observations)
        return query, uncertainty, _information_gain(query, observations), 0.0
    scored: list[tuple[tuple[float, str], AcquisitionQuery, float, float, float]] = []
    for query in available:
        uncertainty = _dynamic_uncertainty(query, observations)
        information_gain = _information_gain(query, observations)
        decision = decision_sensitivity_fn(query, observations) if decision_sensitivity_fn else (1.0 if not observations.get(query.edge_id) else 0.5)
        decision = min(1.0, max(0.0, decision))
        if not 0.0 <= decision <= 1.0:
            raise ValueError("decision_sensitivity_fn must return a value in [0, 1]")
        if policy is AcquisitionPolicy.UNCERTAINTY_ONLY:
            score = information_gain / query.cost
        else:
            score = (information_gain + decision + query.risk + query.provenance_novelty) / (query.cost + 0.1)
        scored.append(((score, query.query_id), query, uncertainty, information_gain, decision))
    _, query, uncertainty, information_gain, decision = max(scored, key=lambda item: item[0])
    return query, uncertainty, information_gain, decision


def run_acquisition(
    queries: Sequence[AcquisitionQuery],
    sealed_labels: Mapping[str, bool],
    policy: AcquisitionPolicy,
    *,
    confidence_target: float = 0.9,
    delta_family: float = 0.05,
    seed: int = 0,
    decision_sensitivity_fn: Callable[[AcquisitionQuery, Mapping[str, list[bool]]], float] | None = None,
) -> AcquisitionResult:
    """Select until the observable posterior is confident or the pool ends.

    The sealed labels are revealed only after each query is selected.  No
    ground-truth edge labels or target error are accepted here; those belong to
    :func:`evaluate_trajectory`, which is an offline evaluator.
    """
    if not 0.0 < confidence_target <= 1.0 or not 0.0 < delta_family < 1.0:
        raise ValueError("confidence_target must be in (0, 1] and delta_family in (0, 1)")
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
    certificate = _certificate(observations, edge_ids, delta_family)
    while available:
        certificate = _certificate(observations, edge_ids, delta_family)
        if certificate.certified:
            stopped_by_policy = True
            break
        frontier = [
            query for query in available
            if query.edge_id not in certificate.covered_edges
        ]
        query, uncertainty, information_gain, decision = _choose(frontier or available, policy, observations, rng, decision_sensitivity_fn)
        available.remove(query)
        # The observation is revealed only after selection; selection saw only
        # the posterior state accumulated from earlier revealed observations.
        observations.setdefault(query.edge_id, []).append(bool(sealed_labels[query.query_id]))
        selected.append(query.query_id)
        cost += query.cost
        cumulative.append(cost)
        current_certificate = _certificate(observations, edge_ids, delta_family)
        trace.append({
            "query_id": query.query_id,
            "edge_id": query.edge_id,
            "uncertainty": uncertainty,
            "information_gain": information_gain,
            "decision_sensitivity": decision,
            "decision_model": "bounded_decision_surrogate" if decision_sensitivity_fn else "none",
            "risk": query.risk,
            "provenance_novelty": query.provenance_novelty,
            "cost": query.cost,
            "certificate_min_confidence": current_certificate.min_confidence,
            "certificate_covered_edges": list(current_certificate.covered_edges),
            "confidence_sequence": {key: list(value) for key, value in current_certificate.intervals.items()},
        })
    posterior = {
        edge_id: {"mean": _posterior(observations, edge_id)[0], "confidence": _posterior(observations, edge_id)[1]}
        for edge_id in sorted(edge_ids)
    }
    certificate = _certificate(observations, edge_ids, delta_family)
    return AcquisitionResult(policy.value, tuple(selected), tuple(cumulative), stopped_by_policy, posterior, tuple(trace), certificate.certified)


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
    confusion_trajectory: list[Mapping[str, int]] = []
    harmful_fp_trajectory: list[float] = []
    cost_to_target: float | None = None
    cost = 0.0
    for index, query_id in enumerate(trajectory.selected_query_ids):
        query = by_id[query_id]
        observations.setdefault(query.edge_id, []).append(bool(sealed_labels[query_id]))
        cost += query.cost
        wrong = 0
        tp = fp = fn = tn = 0
        for edge_id, truth in edge_truths.items():
            values = observations.get(edge_id, [])
            if not values:
                continue
            prediction = sum(values) * 2 >= len(values)
            if prediction and truth: tp += 1
            elif prediction and not truth: fp += 1
            elif not prediction and truth: fn += 1
            else: tn += 1
            wrong += prediction != truth
        observed = tp + fp + fn + tn
        # Unknown edges remain unresolved for the offline identification
        # metric, but are never counted as false positives in the confusion
        # or harmful-FP trajectories.
        error = (wrong + len(edge_truths) - observed) / len(edge_truths) if edge_truths else 0.0
        negative_total = sum(not truth for truth in edge_truths.values())
        harmful_fp_trajectory.append(fp / negative_total if negative_total else 0.0)
        confusion_trajectory.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        errors.append(error)
        if cost_to_target is None and error <= target_error:
            cost_to_target = cost
    final_error = errors[-1] if errors else 0.0
    return OfflineEvaluation(cost_to_target, final_error, cost_to_target is not None, tuple(errors), tuple(confusion_trajectory), tuple(harmful_fp_trajectory))
