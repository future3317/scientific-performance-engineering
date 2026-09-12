"""Shared policy and offline metric contracts for provider-free boundary studies.

Policies receive only the public ``context`` field from an evaluation row.
Held-out utility and scientific-gate fields are supplied separately through
``score_records`` and are consumed here, at the evaluator boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Decision:
    """A policy's auditable reuse/abstain decision."""

    action: str
    score: float
    uncertainty: float
    reason: str

    def __post_init__(self) -> None:
        if self.action not in {"reuse", "abstain"}:
            raise ValueError(f"unknown policy action: {self.action!r}")
        if not math.isfinite(float(self.score)):
            raise ValueError("decision score must be finite")
        if not math.isfinite(float(self.uncertainty)) or self.uncertainty < 0:
            raise ValueError("decision uncertainty must be finite and non-negative")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("decision reason must be a non-empty string")


Policy = Callable[[Mapping[str, Any]], Decision]


_BLOCKING_REASON_ORDER = ("A", "E", "S", "L", "R")


def first_blocking_reason(reasons: Sequence[str]) -> str | None:
    """Return the registered first blocking clause without interpretation."""

    present = set(reasons)
    for reason in _BLOCKING_REASON_ORDER:
        if reason in present:
            return reason
    return None


def support_decision_regret(
    action: str,
    tau: float,
    scientific_gate: bool,
    delta: float,
) -> float:
    """Return the registered cross-context support-decision regret.

    The oracle reuses exactly when the scientific gate passes and the paired
    effect clears ``delta``.  Abstention has zero fallback utility.  Reuse on
    a failed scientific gate receives zero utility; reuse below the practical
    margin retains its measured utility, as specified for this estimand.
    """

    if action not in {"reuse", "abstain"}:
        raise ValueError(f"unknown policy action: {action!r}")
    effect = float(tau)
    oracle_supported = bool(scientific_gate) and effect > float(delta)
    oracle_utility = effect if oracle_supported else 0.0
    if action == "abstain":
        policy_utility = 0.0
    elif scientific_gate:
        policy_utility = effect
    else:
        policy_utility = 0.0
    return oracle_utility - policy_utility


def _decision(policy: Any, context: Mapping[str, Any]) -> Decision:
    selected = policy.decide(context) if hasattr(policy, "decide") else policy(context)
    if not isinstance(selected, Decision):
        raise TypeError("policy must return benchmark.boundary.policy_eval.Decision")
    return selected


def evaluate_policy(
    policy: Any,
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    score_records: Mapping[str, Mapping[str, Any]],
    *,
    delta: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one already-fitted policy with a single metric contract.

    ``calibration`` is accepted to make the contract explicit and to allow a
    caller to retain the exact freeze input in its report.  Fitting is not
    performed here: policy construction and parameter selection must happen
    before evaluation, using calibration only.  The evaluator passes each
    policy only ``row['context']`` and never forwards other row fields.
    """

    del calibration
    if not evaluation:
        return {
            "n": 0,
            "deployed": 0,
            "coverage": 0.0,
            "unsupported": 0,
            "unsupported_all": 0.0,
            "unsupported_deployed": None,
            "unsupported_deployed_defined": False,
            "oracle_supported": 0,
            "recall": None,
            "support_decision_regret": 0.0,
            "support_decision_regret_mean": None,
            "correct_abstention": 0,
            "missed_supported_deployment": 0,
            "reason_counts": {},
            "abstention_reason_counts": {},
            "ci_width_summary": {"missed_supported_mean": None, "correct_abstention_mean": None},
            "uncertainty_interpretation": "fact_only_until_registered_comparison",
            "decisions": [],
        }

    decisions: list[dict[str, Any]] = []
    unsupported = 0
    deployed = 0
    oracle_supported_count = 0
    supported_deployed = 0
    total_regret = 0.0
    correct_abstention = 0
    missed_supported = 0
    reason_counts: Counter[str] = Counter()
    abstention_reason_counts: Counter[str] = Counter()
    missed_ci_widths: list[float] = []
    correct_ci_widths: list[float] = []

    for row in evaluation:
        observation_id = str(row["id"])
        if observation_id not in score_records:
            raise KeyError(f"missing score record for evaluation id {observation_id!r}")
        score = score_records[observation_id]
        if "tau" not in score or "scientific_gate" not in score:
            raise KeyError(f"score record {observation_id!r} lacks tau/scientific_gate")
        selected = _decision(policy, row["context"])
        tau = float(score["tau"])
        scientific_gate = bool(score["scientific_gate"])
        supported = scientific_gate and tau > float(delta)
        if supported:
            oracle_supported_count += 1
        if selected.action == "reuse":
            deployed += 1
            if supported:
                supported_deployed += 1
            else:
                unsupported += 1
        elif supported:
            missed_supported += 1
            if "ci_width" in score and score["ci_width"] is not None:
                missed_ci_widths.append(float(score["ci_width"]))
        else:
            correct_abstention += 1
            if "ci_width" in score and score["ci_width"] is not None:
                correct_ci_widths.append(float(score["ci_width"]))
        total_regret += support_decision_regret(selected.action, tau, scientific_gate, delta)
        reason_counts[selected.reason] += 1
        if selected.action == "abstain":
            abstention_reason_counts[selected.reason] += 1
        decisions.append({
            "id": observation_id,
            "family": row.get("family"),
            "seed": row.get("seed"),
            "action": selected.action,
            "score": selected.score,
            "uncertainty": selected.uncertainty,
            "ci_width": score.get("ci_width"),
            "reason": selected.reason,
            "oracle_supported": supported,
        })

    n = len(evaluation)
    return {
        "n": n,
        "deployed": deployed,
        "coverage": deployed / n,
        "unsupported": unsupported,
        "unsupported_all": unsupported / n,
        "unsupported_deployed": unsupported / deployed if deployed else None,
        "unsupported_deployed_defined": bool(deployed),
        "oracle_supported": oracle_supported_count,
        "recall": supported_deployed / oracle_supported_count if oracle_supported_count else None,
        "support_decision_regret": total_regret,
        "support_decision_regret_mean": total_regret / n,
        "correct_abstention": correct_abstention,
        "missed_supported_deployment": missed_supported,
        "reason_counts": dict(sorted(reason_counts.items())),
        "abstention_reason_counts": dict(sorted(abstention_reason_counts.items())),
        "ci_width_summary": {
            "missed_supported_mean": mean(missed_ci_widths) if missed_ci_widths else None,
            "correct_abstention_mean": mean(correct_ci_widths) if correct_ci_widths else None,
        },
        "uncertainty_interpretation": "fact_only_until_registered_comparison",
        "decisions": decisions,
    }


def clustered_frontier(
    rows: Sequence[Mapping[str, Any]],
    cluster_keys: Sequence[str] = ("family", "seed"),
) -> dict[str, Any]:
    """Summarize frontier rows by predeclared family/seed clusters.

    Context rows are never treated as independent resampling units here.  The
    returned means first average within each cluster and then average the
    cluster means, leaving the raw row count and cluster count auditable.
    """

    clusters: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[tuple(row.get(key) for key in cluster_keys)].append(row)
    cluster_values: list[dict[str, float]] = []
    for members in clusters.values():
        values: dict[str, float] = {}
        numeric_keys = {
            key
            for member in members
            for key, value in member.items()
            if key not in cluster_keys and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for key in numeric_keys:
            values[key] = mean(float(member[key]) for member in members if key in member)
        cluster_values.append(values)
    means: dict[str, float] = {}
    for key in sorted({key for values in cluster_values for key in values}):
        means[key] = mean(values[key] for values in cluster_values if key in values)
    return {
        "n_rows": len(rows),
        "n_clusters": len(clusters),
        "cluster_keys": list(cluster_keys),
        "cluster_means": means,
    }


__all__ = ["Decision", "clustered_frontier", "evaluate_policy", "first_blocking_reason", "support_decision_regret"]
