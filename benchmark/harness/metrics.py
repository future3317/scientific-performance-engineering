"""Pure evolution metrics.

The episode runner owns state transitions and persistence; this module owns
only calculations over already materialized records.
"""

from __future__ import annotations

import json
from typing import Any


def transfer_gain(paired_results: list[dict[str, Any]]) -> float | None:
    deltas = [
        float(record["task_score_on"]) - float(record["task_score_off"])
        for record in paired_results
        if "task_score_on" in record and "task_score_off" in record
    ]
    return sum(deltas) / len(deltas) if deltas else None


def negative_transfer_rate(applications: list[dict[str, Any]], noise_floor: float | None = None) -> float | None:
    if not applications:
        return None
    if noise_floor is None and not any(app.get("noise_floor") is not None for app in applications):
        return None
    regressions = sum(
        float(app.get("delta", 0.0)) < -abs(float(app.get("noise_floor", noise_floor or 0.0)))
        for app in applications
    )
    return regressions / len(applications)


def rule_reuse_utility(applications: list[dict[str, Any]]) -> float | None:
    reused = [float(app["delta"]) for app in applications if app.get("reused") and "delta" in app]
    return sum(reused) / len(reused) if reused else None


def rule_precision(admitted: int, survived: int) -> float | None:
    return survived / admitted if admitted > 0 else None


def library_growth(canonical_rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "canonical_rule_count": len(canonical_rules),
        "description_length": sum(
            len(json.dumps(rule, ensure_ascii=False, default=str)) for rule in canonical_rules
        ),
    }


def utility_per_rule(total_gain: float | None, rule_count: int) -> float | None:
    return total_gain / rule_count if total_gain is not None and rule_count > 0 else None


def utility_per_token(total_gain: float | None, prompt_tokens: int) -> float | None:
    return total_gain / prompt_tokens if total_gain is not None and prompt_tokens > 0 else None


def conflict_rate(conflicting_pairs: int, canonical_pairs: int) -> float | None:
    return conflicting_pairs / canonical_pairs if canonical_pairs > 0 else None


def drift_recovery_latency(utility_series: list[float], drift_start: int) -> int | None:
    if drift_start <= 0 or drift_start >= len(utility_series):
        return None
    pre_drift = max(utility_series[:drift_start])
    dropped = False
    for index in range(drift_start, len(utility_series)):
        if utility_series[index] < pre_drift:
            dropped = True
        elif dropped:
            return index - drift_start
    return None


def poisoning_survival_rate(
    poison_ids: list[str], canonical_rule_ids: list[str], regressions_caused: int = 0
) -> float | None:
    if not poison_ids:
        return None
    canonized = sum(1 for poison_id in poison_ids if poison_id in set(canonical_rule_ids))
    return max(0, len(poison_ids) - canonized - regressions_caused) / len(poison_ids)
