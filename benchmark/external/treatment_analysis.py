"""Analysis of completed A/C/D artifacts without provider calls."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any

from .evaluator import summarize_policy


def _ci(values: list[float], seed: int, reps: int) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "low": None, "high": None}
    rng = random.Random(int(seed))
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(int(reps))]
    samples.sort()
    return {
        "mean": mean(values),
        "low": samples[max(0, int(0.025 * len(samples)) - 1)],
        "high": samples[min(len(samples) - 1, int(0.975 * len(samples)))],
    }


def _condition_summary(condition: str, artifact: dict[str, Any], seed: int, reps: int) -> dict[str, Any]:
    rows = list(artifact.get("rows", []))
    for row in rows:
        row.setdefault("policy_decisions", {}).setdefault(condition, {"reuse": condition != "A", "executed": True})
    summary = summarize_policy(rows, condition)
    deltas = [float(row["skill_score"]) - float(row["baseline_score"]) for row in rows]
    summary["delta_ci"] = _ci(deltas, seed, reps)
    summary["negative_transfer"] = sum(delta < 0.0 for delta in deltas)
    summary["positive_improvements"] = sum(delta > 0.0 for delta in deltas)
    summary["skill_version_count"] = len(artifact.get("skill_versions", []))
    summary["provider_calls"] = sum(int(item.get("calls", 1)) for item in artifact.get("usage", []))
    summary["total_tokens"] = sum(int(item.get("total_tokens", 0) or 0) for item in artifact.get("usage", []))
    return summary


def analyze_treatment_artifacts(artifacts: dict[str, dict[str, Any]], *, bootstrap_seed: int, bootstrap_reps: int) -> dict[str, Any]:
    paired_rows = {}
    for condition in ("A", "C", "D"):
        rows = artifacts[condition].get("rows", [])
        by_id = {str(row["instance_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate instance IDs in {condition}")
        paired_rows[condition] = by_id
    if not (set(paired_rows["A"]) == set(paired_rows["C"]) == set(paired_rows["D"])):
        raise ValueError("paired treatment instance IDs must match across A/C/D")
    summaries = {
        condition: _condition_summary(condition, artifacts[condition], bootstrap_seed + index, bootstrap_reps)
        for index, condition in enumerate(("A", "C", "D"))
    }
    c_rows = paired_rows["C"]
    d_rows = paired_rows["D"]
    paired_deltas = [float(d_rows[key]["skill_score"]) - float(c_rows[key]["skill_score"]) for key in sorted(set(c_rows) & set(d_rows))]
    paired = _ci(paired_deltas, bootstrap_seed + 10, bootstrap_reps)
    paired.update({
        "n": len(paired_deltas),
        "mean_delta": paired["mean"],
        "ci_low": paired["low"],
        "ci_high": paired["high"],
        "negative_delta_count": sum(delta < 0.0 for delta in paired_deltas),
        "positive_delta_count": sum(delta > 0.0 for delta in paired_deltas),
        "deltas": paired_deltas,
    })
    return {
        "schema_version": 1,
        "conditions": summaries,
        "paired_D_minus_C": paired,
        "resource_views": {
            "equal_task_execution_budget": "paired task scores use the same task order and baseline cache; evolution/provider cost is reported separately",
            "equal_total_compute": {condition: {"provider_calls": summaries[condition]["provider_calls"], "total_tokens": summaries[condition]["total_tokens"]} for condition in summaries},
        },
    }
