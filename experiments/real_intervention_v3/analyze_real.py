from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


PAIR_STATUSES = (
    "VALID_PAIR",
    "BASELINE_INVALID",
    "INTERVENTION_MODEL_UNSUPPORTED",
    "INTERVENTION_NUMERIC_MISMATCH",
    "PLATFORM_LINK_OR_INSTALL",
    "SESSION_NOISE_UNRESOLVED",
)
_TIME_KEYS = ("total_cold_inclusive_s", "total_cold_inclusive_seconds", "total_cold_inclusive_timing_seconds")


def _time(receipt: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = receipt.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
            return float(value)
    return None


def _valid_pair(pair: Mapping[str, Any]) -> tuple[bool, float | None, float | None]:
    s0, s1 = pair.get("S0"), pair.get("S1")
    if pair.get("status") != "VALID_PAIR" or not isinstance(s0, Mapping) or not isinstance(s1, Mapping):
        return False, None, None
    if s0.get("status") != "valid" or s1.get("status") != "valid":
        return False, None, None
    baseline = _time(s0, _TIME_KEYS)
    candidate = _time(s1, _TIME_KEYS)
    return baseline is not None and candidate is not None, baseline, candidate


def _family_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(float(row["log_benefit"]))
    return {family: statistics.fmean(values) for family, values in by_family.items()}


def _bootstrap_ci(values: Mapping[str, float], *, seed: int = 101, replicates: int = 10_000) -> dict[str, Any] | None:
    if not values:
        return None
    families = sorted(values)
    if len(families) == 1:
        return {"lower": None, "upper": None, "status": "INSUFFICIENT_INDEPENDENT_FAMILIES",
                "inferential": False, "unit": "family", "families": 1, "replicates": replicates, "seed": seed}
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(values[families[rng.randrange(len(families))]] for _ in families)
        for _ in range(replicates)
    )
    return {"lower": draws[int(0.025 * (replicates - 1))], "upper": draws[int(0.975 * (replicates - 1))],
            "unit": "family", "families": len(families), "replicates": replicates, "seed": seed}


def _effect_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_pairs": 0, "n_families": 0, "context_equal_weighted_mean_log_benefit": None,
                "family_equal_weighted_mean_log_benefit": None, "median_speedup": None, "family_bootstrap_95_ci": None}
    family_means = _family_means(rows)
    context_mean = statistics.fmean(float(row["log_benefit"]) for row in rows)
    family_mean = statistics.fmean(family_means.values())
    return {"n_pairs": len(rows), "n_families": len(family_means),
            "context_equal_weighted_mean_log_benefit": context_mean,
            "family_equal_weighted_mean_log_benefit": family_mean,
            "median_speedup": statistics.median(float(row["speedup"]) for row in rows),
            "family_bootstrap_95_ci": _bootstrap_ci(family_means)}


def analyze(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize every paired receipt while keeping family as the CI unit."""
    status_counts = Counter(str(pair.get("status", "MISSING_STATUS")) for pair in pairs)
    for status in PAIR_STATUSES:
        status_counts.setdefault(status, 0)
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    valid_effects: list[dict[str, Any]] = []
    horizons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    resources: dict[str, dict[str, float]] = defaultdict(lambda: {"gpu_seconds": 0.0, "cpu_seconds": 0.0, "wall_seconds": 0.0,
                                                                 "peak_allocated_bytes": 0.0, "peak_reserved_bytes": 0.0})
    for pair in pairs:
        split = str(pair.get("split", "UNKNOWN"))
        family = str(pair.get("family", "UNKNOWN"))
        by_split[split].append(pair)
        by_source[str(pair.get("source", "UNKNOWN"))].append(pair)
        valid, baseline, candidate = _valid_pair(pair)
        if valid:
            assert baseline is not None and candidate is not None
            if baseline > 0 and candidate > 0:
                feature = pair.get("features") if isinstance(pair.get("features"), Mapping) else {}
                effect = {"pair_id": pair.get("pair_id"), "family": family, "split": split,
                          "source": str(pair.get("source", "UNKNOWN")),
                          "horizon": feature.get("horizon", pair.get("horizon")),
                          "baseline_seconds": baseline, "candidate_seconds": candidate,
                          "log_benefit": math.log(baseline / candidate), "speedup": baseline / candidate}
                valid_effects.append(effect)
                horizons[str(effect["horizon"])].append(effect)
        for arm in (pair.get("S0"), pair.get("S1")):
            if not isinstance(arm, Mapping):
                continue
            for target, names in (("gpu_seconds", ("gpu_seconds", "gpu_session_window_seconds")),
                                  ("cpu_seconds", ("child_cpu_s", "cpu_seconds", "child_cpu_seconds")),
                                  ("wall_seconds", ("parent_wall_s", "wall_seconds", "parent_wall_seconds")),
                                  ("peak_allocated_bytes", ("peak_allocated_bytes",)),
                                  ("peak_reserved_bytes", ("peak_reserved_bytes",))):
                value = _time(arm, names)
                if value is not None:
                    resources[split][target] += value

    steady_state: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"S0": [], "S1": []})
    for pair in pairs:
        split = str(pair.get("split", "UNKNOWN"))
        for arm_name in ("S0", "S1"):
            arm = pair.get(arm_name)
            if isinstance(arm, Mapping):
                value = _time(arm, ("steady_state_per_step_s",))
                if value is not None:
                    steady_state[split][arm_name].append(value)

    split_summary = {}
    for split, rows in sorted(by_split.items()):
        split_effects = [row for row in valid_effects if row["split"] == split]
        split_summary[split] = {"pairs": len(rows), "families": len({str(row.get("family", "UNKNOWN")) for row in rows}),
                                "status_counts": dict(sorted(Counter(str(row.get("status", "MISSING_STATUS")) for row in rows).items())),
                                "effects": _effect_summary(split_effects), "resources": resources[split],
                                "steady_state_per_step_seconds": {
                                    arm: {"n": len(values), "median": statistics.median(values) if values else None,
                                          "mean": statistics.fmean(values) if values else None}
                                    for arm, values in steady_state[split].items()}}

    source_summary = {}
    for source, rows in sorted(by_source.items()):
        source_effects = [row for row in valid_effects if row["source"] == source]
        source_summary[source] = {
            "input_kind": "REAL_MODEL_MATERIALIZED_INPUT" if source == "torchvision" else "TORCHBENCH_MINIATURE_INPUT" if source == "torchbench" else "UNSPECIFIED_INPUT_KIND",
            "pairs": len(rows), "status_counts": dict(sorted(Counter(str(row.get("status", "MISSING_STATUS")) for row in rows).items())),
            "effects": _effect_summary(source_effects),
        }

    candidate_failure_statuses = {"INTERVENTION_MODEL_UNSUPPORTED", "INTERVENTION_NUMERIC_MISMATCH"}
    return {
        "n_pairs": len(pairs),
        "failure_denominator": {"all_pairs": len(pairs), "status_counts": dict(sorted(status_counts.items())),
                                "candidate_failure_pairs": sum(status_counts[s] for s in candidate_failure_statuses),
                                "platform_invalid_pairs": status_counts["PLATFORM_LINK_OR_INSTALL"],
                                "baseline_invalid_pairs": status_counts["BASELINE_INVALID"],
                                "session_noise_unresolved_pairs": status_counts["SESSION_NOISE_UNRESOLVED"],
                                "valid_pairs": status_counts["VALID_PAIR"]},
        "split_summary": split_summary,
        "source_strata": source_summary,
        "overall_effects": _effect_summary(valid_effects),
        "per_horizon": {horizon: _effect_summary(rows) for horizon, rows in sorted(horizons.items())},
        "statistical_unit": "architecture/model family; contexts and horizons are repeated observations",
        "estimand": "log(S0 total cold-inclusive seconds / S1 total cold-inclusive seconds), only for valid correct pairs",
    }
