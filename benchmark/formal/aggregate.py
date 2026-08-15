"""Lineage-aware aggregation of formal trial records."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def _ci(values: list[float], seed: int = 0, samples: int = 2000) -> dict[str, float | None]:
    if not values:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    low = means[max(0, int(0.025 * (len(means) - 1)))]
    high = means[min(len(means) - 1, int(0.975 * (len(means) - 1)))]
    return {"estimate": sum(values) / len(values), "ci_low": low, "ci_high": high, "n": len(values)}


def _metric(record: dict[str, Any]) -> float | None:
    score = record.get("score", record.get("task_score"))
    if isinstance(score, dict):
        score = score.get("task_score")
    return float(score) if isinstance(score, (int, float)) else None


def aggregate_trials(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    families: dict[str, str] = {}
    for record in records:
        task_id = str(record.get("task_id", ""))
        condition = str(record.get("condition", record.get("experiment", {}).get("condition", "")))
        trial = str(record.get("outer_trial_id", record.get("experiment", {}).get("outer_trial_id", "")))
        value = _metric(record)
        if task_id and condition and trial and value is not None:
            indexed[(task_id, trial, str(record.get("context_mode", "reset")))][condition] = value
            families[task_id] = str(record.get("family", record.get("task", {}).get("family", "unknown")))

    effects: dict[str, list[float]] = {"D-C": [], "D-B": [], "B-A": []}
    family_effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (task_id, _trial, _mode), values in indexed.items():
        family = families.get(task_id, "unknown")
        for label, left, right in (("D-C", "D", "C"), ("D-B", "D", "B"), ("B-A", "B", "A")):
            if left in values and right in values:
                delta = values[left] - values[right]
                effects[label].append(delta)
                family_effects[family][label].append(delta)

    def summarize(values: list[float]) -> dict[str, Any]:
        return _ci(values)

    return {
        "num_records": len(records),
        "paired_effects": {label: summarize(values) for label, values in effects.items()},
        "family_stratified": {
            family: {label: summarize(values) for label, values in metrics.items()}
            for family, metrics in sorted(family_effects.items())
        },
        "log_speedup": {
            label: summarize([math.log(max(1e-12, 1.0 + value)) for value in values])
            for label, values in effects.items()
        },
        "counterexample_records": [
            record for record in records
            if record.get("kind") in {"counterexample", "do_not_apply"}
        ],
    }
