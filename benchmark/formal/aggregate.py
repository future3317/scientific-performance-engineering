"""Lineage-aware aggregation of formal trial records.

Task-score effects and performance effects are deliberately separate units:
the former is a linear score difference, while the latter is a paired
log-ratio of raw median speedups.  Intervals resample family, lineage, task,
and outer-trial levels rather than treating all rows as independent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections import defaultdict
from typing import Any


@dataclass(frozen=True)
class RegretStep:
    """Canonical longitudinal record shared by all benchmark views."""

    context_id: str
    oracle_bundle: tuple[str, ...]
    deployed_bundle: tuple[str, ...]
    oracle_utility: float
    deployed_utility: float
    experiment_cost: float = 0.0
    failure_source: str | None = None
    acquisition_regret: float = 0.0
    negative_transfer_regret: float = 0.0
    interaction_regret: float = 0.0
    drift_recovery_regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "oracle_bundle": list(self.oracle_bundle),
            "deployed_bundle": list(self.deployed_bundle),
            "oracle_utility": self.oracle_utility,
            "deployed_utility": self.deployed_utility,
            "experiment_cost": self.experiment_cost,
            "failure_source": self.failure_source,
            "acquisition_regret": self.acquisition_regret,
            "negative_transfer_regret": self.negative_transfer_regret,
            "interaction_regret": self.interaction_regret,
            "drift_recovery_regret": self.drift_recovery_regret,
        }


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


def performance_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return track-native metrics without collapsing them into one score."""
    profile: dict[str, Any] = {
        "spe_core": {"verified_optimization_rate": None, "geomean_speedup_all_valid": None, "semantic_failure_rate": None},
        "boundary": {"harmful_false_positive_rate": None, "boundary_error": None, "experiment_cost": None},
        "interaction": {"relation_identification_rate": None, "bundle_regret": None},
        "evolution": {"negative_transfer_rate": None, "drift_recovery_latency": None, "poison_robustness": None, "regret": None},
        "efficiency": {"tokens": 0.0, "gpu_time": 0.0, "experiment_cost": 0.0, "library_complexity": 0.0},
    }
    optimization = [record for record in records if str(record.get("track", "spe_core")) == "spe_core"]
    valid_speedups = [value for record in optimization if (value := _speedup(record)) is not None]
    verified = [record for record in optimization if bool(record.get("verified", record.get("score", {}).get("verified", False) if isinstance(record.get("score"), dict) else False))]
    semantic_failures = [record for record in optimization if record.get("validity") == "invalid" or record.get("status") == "invalid" or (isinstance(record.get("score"), dict) and record["score"].get("gates_passed") is False)]
    if optimization:
        profile["spe_core"]["verified_optimization_rate"] = len(verified) / len(optimization)
        if valid_speedups:
            profile["spe_core"]["geomean_speedup_all_valid"] = math.exp(sum(math.log(value) for value in valid_speedups) / len(valid_speedups))
        profile["spe_core"]["semantic_failure_rate"] = len(semantic_failures) / len(optimization)
    boundaries = [record for record in records if record.get("track") == "boundary"]
    if boundaries:
        profile["boundary"]["harmful_false_positive_rate"] = sum(bool(record.get("harmful_false_positive", False)) for record in boundaries) / len(boundaries)
        profile["boundary"]["boundary_error"] = sum(float(record.get("boundary_error", 0.0)) for record in boundaries) / len(boundaries)
        profile["boundary"]["experiment_cost"] = sum(float(record.get("experiment_cost", 0.0)) for record in boundaries)
    interactions = [record for record in records if record.get("track") == "interaction"]
    if interactions:
        profile["interaction"]["relation_identification_rate"] = sum(bool(record.get("relation_correct", False)) for record in interactions) / len(interactions)
        profile["interaction"]["bundle_regret"] = sum(float(record.get("bundle_regret", 0.0)) for record in interactions) / len(interactions)
    evolutions = [record for record in records if record.get("track") == "evolution"]
    if evolutions:
        profile["evolution"]["negative_transfer_rate"] = sum(float(record.get("negative_transfer_rate", 0.0)) for record in evolutions) / len(evolutions)
        latencies = [float(record["drift_recovery_latency"]) for record in evolutions if record.get("drift_recovery_latency") is not None]
        profile["evolution"]["drift_recovery_latency"] = sum(latencies) / len(latencies) if latencies else None
        profile["evolution"]["poison_robustness"] = sum(float(record.get("poisoning_survival_rate", 0.0)) for record in evolutions) / len(evolutions)
        profile["evolution"]["regret"] = evolution_regret(evolutions)
    profile["efficiency"]["tokens"] = sum(float(record.get("tokens", record.get("prompt_tokens", 0.0))) for record in records)
    profile["efficiency"]["gpu_time"] = sum(float(record.get("gpu_time", 0.0)) for record in records)
    profile["efficiency"]["experiment_cost"] = sum(float(record.get("experiment_cost", 0.0)) for record in records)
    profile["efficiency"]["library_complexity"] = sum(float(record.get("library_complexity", 0.0)) for record in records)
    return profile


def evolution_regret(records: list[RegretStep | dict[str, Any]], lambda_cost: float = 1.0) -> dict[str, float | None]:
    """Canonical lifecycle regret used by formal and episode aggregation.

    The utility gap is hindsight-valid and the experiment cost is reported in
    the same record.  Component regrets are optional but, when present, are
    kept separate so profiles cannot hide acquisition, interaction, or drift
    costs inside one scalar.
    """
    steps = []
    for record in records:
        value = record.to_dict() if isinstance(record, RegretStep) else record
        if "oracle_utility" in value and "deployed_utility" in value:
            steps.append(value)
    if not steps:
        return {
            "total": None, "utility_gap": None, "acquisition": None,
            "negative_transfer": None, "interaction": None,
            "drift_recovery": None, "experiment_cost": None,
        }
    utility_gap = sum(float(item["oracle_utility"]) - float(item["deployed_utility"]) for item in steps)
    cost = lambda_cost * sum(float(item.get("experiment_cost", 0.0)) for item in steps)
    components = {
        "acquisition": sum(float(item.get("acquisition_regret", 0.0)) for item in steps),
        "negative_transfer": sum(float(item.get("negative_transfer_regret", 0.0)) for item in steps),
        "interaction": sum(float(item.get("interaction_regret", 0.0)) for item in steps),
        "drift_recovery": sum(float(item.get("drift_recovery_regret", 0.0)) for item in steps),
    }
    return {"total": utility_gap + cost, "utility_gap": utility_gap, **components, "experiment_cost": cost}
