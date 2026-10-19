"""Family-clustered summaries for planned v4 paired measurements."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


_TERMINAL = {"COMPLETE", "FAILED", "TIMEOUT"}
_KNOWN_EXECUTION = _TERMINAL | {"UNKNOWN", "NOT_STARTED"}


def _positive_finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def family_cluster_bootstrap_ci(
    family_values: Mapping[str, float], *, seed: int, replicates: int
) -> list[float] | None:
    values = list(family_values.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _classify(pair: Mapping[str, Any]) -> tuple[str, dict[str, str], bool, bool]:
    classification = pair.get("classification")
    classification = classification if isinstance(classification, Mapping) else {}
    arms = {arm: pair.get(arm) for arm in ("S0", "S1")}
    states: dict[str, str] = {}
    for arm, receipt in arms.items():
        receipt = receipt if isinstance(receipt, Mapping) else {}
        state = receipt.get("arm_execution", "UNKNOWN")
        states[arm] = state if state in _KNOWN_EXECUTION else "UNKNOWN"

    def has_event(receipt: Any, kind: str) -> bool:
        if not isinstance(receipt, Mapping):
            return False
        events = receipt.get("events")
        return isinstance(events, list) and any(
            isinstance(event, Mapping) and event.get("kind") == kind
            for event in events
        )

    started = any(has_event(receipt, "started") for receipt in arms.values())
    closed = all(
        states[arm] in _TERMINAL
        and isinstance(arms[arm], Mapping)
        and arms[arm].get("receipt_verified") is True
        and has_event(arms[arm], "terminal")
        for arm in ("S0", "S1")
    )
    correctness = [
        classification.get("scientific_correctness"),
        *(receipt.get("scientific_correctness") if isinstance(receipt, Mapping) else None
          for receipt in arms.values()),
    ]
    policy = classification.get("policy_evidence")
    comparability = classification.get("paired_comparability")

    s1 = arms["S1"] if isinstance(arms["S1"], Mapping) else {}
    s0 = arms["S0"] if isinstance(arms["S0"], Mapping) else {}
    if s1.get("scientific_correctness") == "FAIL" or policy == "UNSAFE":
        status = "unsafe"
    elif any(
        isinstance(arms[arm], Mapping)
        and arms[arm].get("infrastructure_classification") == "INFRA"
        for arm in ("S0", "S1")
    ):
        status = "infra"
    elif s0.get("scientific_correctness") == "FAIL":
        status = "baseline_invalid"
    elif states["S0"] in {"FAILED", "TIMEOUT"} and states["S1"] in {"FAILED", "TIMEOUT"}:
        status = "infra"
    elif states["S0"] in {"FAILED", "TIMEOUT"}:
        status = "baseline_invalid"
    elif states["S1"] in {"FAILED", "TIMEOUT"}:
        status = "candidate_invalid"
    elif any(state == "UNKNOWN" for state in states.values()) or "UNKNOWN" in correctness:
        status = "unknown"
    elif not started:
        status = "unstarted"
    elif comparability == "CONFOUNDED":
        status = "noise"
    elif (not started or not closed or policy in {None, "MISSING"}
          or comparability in {None, "UNKNOWN", "INCOMPLETE"}
          or any(value in {None, "UNASSESSED"} for value in correctness)):
        status = "unknown"
    elif not all(
        isinstance(receipt.get("raw_receipt_sha256"), str)
        and len(receipt["raw_receipt_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in receipt["raw_receipt_sha256"].lower())
        for receipt in arms.values()
    ):
        status = "unknown"
    else:
        s0, s1 = arms["S0"], arms["S1"]
        cold0 = s0.get("timing", {}).get("cold_seconds") if isinstance(s0, Mapping) else None
        cold1 = s1.get("timing", {}).get("cold_seconds") if isinstance(s1, Mapping) else None
        if (states == {"S0": "COMPLETE", "S1": "COMPLETE"}
                and classification.get("arm_execution") == "COMPLETE"
                and classification.get("scientific_correctness") == "PASS"
                and all(receipt.get("scientific_correctness") == "PASS"
                        for receipt in arms.values())
                and comparability == "MATCHED" and policy == "ELIGIBLE"
                and _positive_finite(cold0) and _positive_finite(cold1)):
            status = "valid"
        else:
            status = "ineligible"
    return status, states, started, closed


def summarize_effects(
    planned_pairs: Sequence[Mapping[str, Any]], *, seed: int = 101, replicates: int = 10_000
) -> dict[str, Any]:
    """Classify every planned pair and summarize effects separately by split.

    The primary estimand is the mean of family means, so repeated contexts or
    sessions within a family do not increase the inferential sample size.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")

    rows: list[dict[str, Any]] = []
    family_splits: dict[str, set[str]] = defaultdict(set)
    for pair in planned_pairs:
        family = pair.get("family_id")
        if family is not None and str(family):
            family_splits[str(family)].add(str(pair.get("split", "UNKNOWN")))
        status, arm_states, started, closed = _classify(pair)
        row = dict(pair)
        row["effect_status"] = status
        row["arm_states"] = arm_states
        row["started"] = started
        row["closed"] = closed
        row["log_benefit"] = None
        row["steady_log_benefit"] = None
        row["cold_seconds_saved_per_pair"] = None
        if status == "valid":
            s0, s1 = pair["S0"], pair["S1"]
            cold0, cold1 = s0["timing"]["cold_seconds"], s1["timing"]["cold_seconds"]
            row["log_benefit"] = math.log(cold0 / cold1)
            row["cold_seconds_saved_per_pair"] = cold0 - cold1
            steady0 = s0.get("timing", {}).get("steady_seconds")
            steady1 = s1.get("timing", {}).get("steady_seconds")
            if _positive_finite(steady0) and _positive_finite(steady1):
                row["steady_log_benefit"] = math.log(steady0 / steady1)
        rows.append(row)

    overlapping = {family: splits for family, splits in family_splits.items() if len(splits) > 1}
    if overlapping:
        raise ValueError(f"independent families cross frozen splits: {overlapping}")

    splits = sorted({str(row.get("split", "UNKNOWN")) for row in rows})
    summaries: dict[str, Any] = {}
    for split in splits:
        selected = [row for row in rows if str(row.get("split", "UNKNOWN")) == split]
        counts = Counter(row["effect_status"] for row in selected)
        family_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            if row["effect_status"] == "valid":
                family_id = row.get("family_id")
                if family_id is None or str(family_id) == "":
                    row["effect_status"] = "unknown"
                    row["log_benefit"] = None
                    row["steady_log_benefit"] = None
                    row["cold_seconds_saved_per_pair"] = None
                    counts["valid"] -= 1
                    counts["unknown"] += 1
                    continue
                family_pairs[str(family_id)].append(row)

        family_effects: dict[str, float] = {}
        family_steady: dict[str, float] = {}
        family_cold_saved: dict[str, float] = {}
        family_summary = []
        for family_id, family_rows in sorted(family_pairs.items()):
            logs = [row["log_benefit"] for row in family_rows]
            steady = [row["steady_log_benefit"] for row in family_rows
                      if row["steady_log_benefit"] is not None]
            saved = [row["cold_seconds_saved_per_pair"] for row in family_rows]
            family_effects[family_id] = _mean(logs)
            family_cold_saved[family_id] = _mean(saved)
            if steady:
                family_steady[family_id] = _mean(steady)
            family_summary.append({
                "family_id": family_id,
                "valid_pair_count": len(family_rows),
                "mean_log_benefit": family_effects[family_id],
                "direction_counts": {
                    "positive_pairs": sum(value > 0 for value in logs),
                    "negative_pairs": sum(value < 0 for value in logs),
                    "zero_pairs": sum(value == 0 for value in logs),
                    "family_mean_positive": family_effects[family_id] > 0,
                    "family_mean_negative": family_effects[family_id] < 0,
                    "family_mean_zero": family_effects[family_id] == 0,
                },
                "mean_S0_cold_seconds_per_pair": _mean(
                    [row["S0"]["timing"]["cold_seconds"] for row in family_rows]
                ),
                "mean_S1_cold_seconds_per_pair": _mean(
                    [row["S1"]["timing"]["cold_seconds"] for row in family_rows]
                ),
                "mean_cold_seconds_saved_per_pair": family_cold_saved[family_id],
            })

        effects = list(family_effects.values())
        leave_one_out = {
            family_id: _mean([value for other, value in family_effects.items()
                              if other != family_id])
            for family_id in family_effects
        }
        lofo_values = [value for value in leave_one_out.values() if value is not None]
        summaries[split] = {
            "denominators": {
                "planned": len(selected),
                "started": sum(row["started"] for row in selected),
                "closed": sum(row["closed"] for row in selected),
                "valid": counts["valid"],
                "unsafe": counts["unsafe"],
                "baseline_invalid": counts["baseline_invalid"],
                "candidate_invalid": counts["candidate_invalid"],
                "infra": counts["infra"],
                "noise": counts["noise"],
                "unknown": counts["unknown"],
                "unstarted": counts["unstarted"],
                "ineligible": counts["ineligible"],
            },
            "paired_valid_rate_over_planned": counts["valid"] / len(selected) if selected else None,
            "independent_family_count": len(family_effects),
            "valid_pair_count": counts["valid"],
            "family_equal_mean_log_benefit": _mean(effects),
            "family_cluster_bootstrap_ci95_family_equal_mean_log_benefit":
                family_cluster_bootstrap_ci(family_effects, seed=seed, replicates=replicates),
            "mean_family_steady_log_benefit": _mean(list(family_steady.values())),
            "leave_one_family_out_mean_log_benefit": leave_one_out,
            "leave_one_family_out_range_log_benefit":
                [min(lofo_values), max(lofo_values)] if lofo_values else None,
            "family_direction_counts": {
                "positive": sum(value > 0 for value in effects),
                "negative": sum(value < 0 for value in effects),
                "zero": sum(value == 0 for value in effects),
            },
            "raw_cold_seconds": {
                "scope": "family-equal mean of per-pair seconds; intervals are not total seconds",
                "S0_mean_seconds_per_pair": _mean([
                    item["mean_S0_cold_seconds_per_pair"] for item in family_summary
                ]),
                "S1_mean_seconds_per_pair": _mean([
                    item["mean_S1_cold_seconds_per_pair"] for item in family_summary
                ]),
                "seconds_saved_mean_per_pair": _mean(list(family_cold_saved.values())),
                "family_cluster_bootstrap_ci95_seconds_saved_mean_per_pair":
                    family_cluster_bootstrap_ci(family_cold_saved, seed=seed, replicates=replicates),
            },
            "families": family_summary,
            "bootstrap_seed": seed,
            "bootstrap_replicates": replicates,
        }

    return {"pairs": rows, "by_split": summaries}
