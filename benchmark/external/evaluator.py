"""Common scoring for fixed-skill and evolving-skill paired outcomes."""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean
from typing import Any, Iterable


def _decision(row: dict[str, Any], policy_name: str) -> dict[str, Any]:
    decisions = row.get("policy_decisions", row.get("policies", {}))
    if policy_name not in decisions:
        raise ValueError(f"missing policy decision: {policy_name}")
    return dict(decisions[policy_name])


def _delta(row: dict[str, Any]) -> float:
    return float(row["skill_score"]) - float(row["baseline_score"])


def _bootstrap_mean(values: list[float], seed: int, reps: int) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(int(seed))
    samples = []
    for _ in range(int(reps)):
        samples.append(mean(values[rng.randrange(len(values))] for _ in values))
    samples.sort()
    low = samples[max(0, int(0.025 * len(samples)) - 1)]
    high = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return (mean(values), low, high)


def summarize_policy(rows: Iterable[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    rows = list(rows)
    deployed = [row for row in rows if bool(_decision(row, policy_name).get("reuse", False))]
    unsupported = [row for row in deployed if _delta(row) < 0.0]
    repairs = [row for row in rows if _delta(row) > 0.0]
    retained_repairs = [row for row in repairs if bool(_decision(row, policy_name).get("reuse", False))]
    regret = []
    for row in rows:
        baseline = float(row["baseline_score"])
        skill = float(row["skill_score"])
        chosen = skill if bool(_decision(row, policy_name).get("reuse", False)) else baseline
        regret.append(max(baseline, skill) - chosen)
    return {
        "policy": policy_name,
        "n": len(rows),
        "deployed": len(deployed),
        "coverage": len(deployed) / len(rows) if rows else 0.0,
        "abstentions": len(rows) - len(deployed),
        "unsupported_deployed": len(unsupported),
        "unsupported_deployed_rate": len(unsupported) / len(deployed) if deployed else None,
        "unsupported_all": len(unsupported),
        "unsupported_all_rate": len(unsupported) / len(rows) if rows else 0.0,
        "repairs": len(repairs),
        "repair_recall": len(retained_repairs) / len(repairs) if repairs else None,
        "negative_transfer_rate": len(unsupported) / len(deployed) if deployed else 0.0,
        "support_decision_regret": mean(regret) if regret else 0.0,
        "net_score_delta": sum(_delta(row) for row in deployed),
        "observation_completion": sum(bool(row.get("measurement_reached", True)) for row in deployed) / len(deployed) if deployed else 0.0,
    }


def _macro(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    keys = ("coverage", "unsupported_deployed_rate", "unsupported_all_rate", "repair_recall", "support_decision_regret", "negative_transfer_rate", "observation_completion")
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(item[key]) for item in summaries if item.get(key) is not None]
        result[key] = mean(values) if values else None
    return result


def evaluate_paired_rows(
    rows: Iterable[dict[str, Any]],
    policy_names: Iterable[str],
    *,
    bootstrap_seed: int,
    bootstrap_reps: int = 10_000,
) -> dict[str, Any]:
    rows = list(rows)
    policy_names = [str(name) for name in policy_names]
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row.get("benchmark", "all")), str(row.get("model", "all")))].append(row)
    policy_results: dict[str, dict[str, Any]] = {}
    cell_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for policy_name in policy_names:
        overall = summarize_policy(rows, policy_name)
        regret_values = []
        for row in rows:
            baseline = float(row["baseline_score"])
            skill = float(row["skill_score"])
            chosen = skill if bool(_decision(row, policy_name).get("reuse", False)) else baseline
            regret_values.append(max(baseline, skill) - chosen)
        _, low, high = _bootstrap_mean(regret_values, bootstrap_seed, bootstrap_reps)
        overall["bootstrap"] = {"support_decision_regret": {"low": low, "high": high}}
        policy_results[policy_name] = overall
        for cell_rows in cells.values():
            cell_summaries[policy_name].append(summarize_policy(cell_rows, policy_name))
    return {
        "schema_version": 1,
        "n_rows": len(rows),
        "bootstrap_unit": "task_instance",
        "bootstrap_reps": int(bootstrap_reps),
        "policies": policy_results,
        "macro_cells": len(cells),
        "macro": {name: _macro(values) for name, values in cell_summaries.items()},
    }

