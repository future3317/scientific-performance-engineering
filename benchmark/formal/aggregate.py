"""Lineage-aware aggregation of formal trial records.

Task-score effects and performance effects are deliberately separate units:
the former is a linear score difference, while the latter is a paired
log-ratio of raw median speedups.  Intervals resample family, lineage, task,
and outer-trial levels rather than treating all rows as independent.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def _ci(values: list[float], seed: int = 0, samples: int = 2000) -> dict[str, float | None]:
    if not values:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(samples)]
    means.sort()
    low = means[max(0, int(0.025 * (len(means) - 1)))]
    high = means[min(len(means) - 1, int(0.975 * (len(means) - 1)))]
    return {"estimate": sum(values) / len(values), "ci_low": low, "ci_high": high, "n": len(values)}


def _hierarchical_ci(
    observations: list[tuple[str, str, str, str, float]],
    seed: int = 0,
    samples: int = 2000,
) -> dict[str, float | None]:
    """Bootstrap family -> lineage -> task -> outer-trial observations."""
    if not observations:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    hierarchy: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for family, lineage, task, _trial, value in observations:
        hierarchy[family][lineage][task].append(value)
    families = sorted(hierarchy)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(samples):
        sampled_values: list[float] = []
        for _family_draw in families:
            family = rng.choice(families)
            lineages = sorted(hierarchy[family])
            for _lineage_draw in lineages:
                lineage = rng.choice(lineages)
                tasks = sorted(hierarchy[family][lineage])
                for _task_draw in tasks:
                    task = rng.choice(tasks)
                    trials = hierarchy[family][lineage][task]
                    sampled_values.append(rng.choice(trials))
        bootstrap_means.append(sum(sampled_values) / len(sampled_values))
    bootstrap_means.sort()
    return {
        "estimate": sum(item[-1] for item in observations) / len(observations),
        "ci_low": bootstrap_means[int(0.025 * (len(bootstrap_means) - 1))],
        "ci_high": bootstrap_means[int(0.975 * (len(bootstrap_means) - 1))],
        "n": len(observations),
    }


def _task_score(record: dict[str, Any]) -> float | None:
    value = record.get("score", record.get("task_score"))
    if isinstance(value, dict):
        value = value.get("task_score")
    return float(value) if isinstance(value, (int, float)) else None


def _speedup(record: dict[str, Any]) -> float | None:
    score = record.get("score")
    if isinstance(score, dict) and "gates_passed" in score and not bool(score["gates_passed"]):
        return None
    result = record.get("result")
    if isinstance(result, dict) and "correctness_pass" in result and not bool(result["correctness_pass"]):
        return None
    candidates = [
        record.get("median_speedup"),
        record.get("verified_speedup", {}).get("median_speedup") if isinstance(record.get("verified_speedup"), dict) else None,
        result.get("verified_speedup", {}).get("median_speedup")
        if isinstance(result, dict)
        else None,
        score.get("verified_speedup", {}).get("median_speedup")
        if isinstance(score, dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    return None


def _identity(record: dict[str, Any], task_id: str) -> tuple[str, str, str]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    family = str(record.get("family", task.get("family", "unknown")))
    lineage = str(record.get("lineage_id", task.get("lineage_id", task_id)))
    trial = str(record.get("outer_trial_id", record.get("experiment", {}).get("outer_trial_id", "")))
    return family, lineage, trial


def aggregate_trials(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("validity") == "invalid" or record.get("status") == "invalid":
            continue
        task_id = str(record.get("task_id", ""))
        condition = str(record.get("condition", record.get("experiment", {}).get("condition", "")))
        trial = str(record.get("outer_trial_id", record.get("experiment", {}).get("outer_trial_id", "")))
        mode = str(record.get("context_mode", "reset"))
        if task_id and condition and trial:
            indexed[(task_id, trial, mode)][condition] = record

    effects: dict[str, list[float]] = {"D-C": [], "D-B": [], "B-A": []}
    speed_effects: dict[str, list[float]] = {"D-C": [], "D-B": [], "B-A": []}
    family_effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    hierarchical: dict[str, list[tuple[str, str, str, str, float]]] = defaultdict(list)
    for (task_id, trial, _mode), values in indexed.items():
        for label, left, right in (("D-C", "D", "C"), ("D-B", "D", "B"), ("B-A", "B", "A")):
            if left not in values or right not in values:
                continue
            left_score, right_score = _task_score(values[left]), _task_score(values[right])
            if left_score is not None and right_score is not None:
                delta = left_score - right_score
                effects[label].append(delta)
                family, lineage, _ = _identity(values[left], task_id)
                family_effects[family][label].append(delta)
                hierarchical["task_score:" + label].append((family, lineage, task_id, trial, delta))
            left_speed, right_speed = _speedup(values[left]), _speedup(values[right])
            if left_speed is not None and right_speed is not None:
                log_delta = math.log(left_speed) - math.log(right_speed)
                speed_effects[label].append(log_delta)
                family, lineage, _ = _identity(values[left], task_id)
                hierarchical["speed:" + label].append((family, lineage, task_id, trial, log_delta))

    invalid_count = sum(1 for record in records if record.get("validity") == "invalid" or record.get("status") == "invalid")
    return {
        "num_records": len(records),
        "num_valid_records": len(records) - invalid_count,
        "num_invalid_records": invalid_count,
        "paired_effects": {label: _ci(values) for label, values in effects.items()},
        "task_score_effects": {label: _ci(values) for label, values in effects.items()},
        "family_stratified": {
            family: {label: _ci(values) for label, values in metrics.items()}
            for family, metrics in sorted(family_effects.items())
        },
        "hierarchical_effects": {
            label.removeprefix("task_score:"): _hierarchical_ci(values, seed=index)
            for index, (label, values) in enumerate(sorted((k, v) for k, v in hierarchical.items() if k.startswith("task_score:")))
        },
        "paired_log_speedups": {
            label: _ci(values) for label, values in speed_effects.items()
        },
        "hierarchical_log_speedups": {
            label.removeprefix("speed:"): _hierarchical_ci(values, seed=index)
            for index, (label, values) in enumerate(sorted((k, v) for k, v in hierarchical.items() if k.startswith("speed:")))
        },
        # Backward-compatible name with corrected log-ratio semantics.
        "log_speedup": {label: _ci(values) for label, values in speed_effects.items()},
        "counterexample_records": [
            record for record in records
            if record.get("kind") in {"counterexample", "do_not_apply"}
        ],
    }
