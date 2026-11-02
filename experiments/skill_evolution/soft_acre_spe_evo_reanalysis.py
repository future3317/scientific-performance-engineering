"""Retrospective Soft-ACRE comparison on the frozen SPE-EvoBench generator."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

from benchmark.boundary.families import family_cases
from benchmark.boundary.soft_acre import SoftAcrePolicy, fit_soft_acre
from benchmark.families import family_predicate_grammar
from scripts.run_support_boundary_analysis import _acre_predictions


STUDY_ID = "EVOSPE_SOFT_ACRE_SPE_EVO_REANALYSIS_20261102_02"
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = (
    "autograd", "checkpoint", "compile", "crystal_generation", "crystal_sampling",
    "episode", "equivariant_head", "graph_cache", "h2d_pipeline", "repeated_compute",
    "scalar_sync",
)
SURFACE_COUNT = 64
RISK_WEIGHT = 1.0
COMPLEXITY_PENALTY = 0.002
MAX_LITERALS = 3


def _family_bootstrap_interval(values: Sequence[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(10_000)
    )
    return [draws[249], draws[9749]]


def _bucket(family: str, seed: int, observation_id: str) -> int:
    token = f"{family}|{seed}|{observation_id}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % 4


def _split(family: str, seed: int) -> dict[str, list[Any]]:
    pools = family_cases(family, surface_count=SURFACE_COUNT, seed=seed)
    unique = {item.observation_id: item for pool in pools.values() for item in pool}
    return {
        "calibration": [item for item in unique.values() if _bucket(family, seed, item.observation_id) == 2],
        "evaluation": [item for item in unique.values() if _bucket(family, seed, item.observation_id) == 3],
        "synthesis": [item for item in unique.values() if _bucket(family, seed, item.observation_id) in (0, 1)],
    }


def _context_and_cutpoints(
    contexts: Sequence[Mapping[str, Any]], feature_specs: Sequence[Mapping[str, Any]]
):
    workload_rows = [context.get("workload", {}) for context in contexts]
    cutpoints = {
        str(spec["path"]).removeprefix("workload."): median(
            float(row[str(spec["path"]).removeprefix("workload.")])
            for row in workload_rows
            if isinstance(row.get(str(spec["path"]).removeprefix("workload.")), (int, float))
            and not isinstance(row.get(str(spec["path"]).removeprefix("workload.")), bool)
        )
        for spec in feature_specs if str(spec["type"]).lower() == "numeric"
    }
    feature_names = tuple(str(spec["path"]) for spec in feature_specs)

    def transform(context: Mapping[str, Any]) -> dict[str, Any]:
        workload = context.get("workload", {})
        output: dict[str, Any] = {"scientific_gate": context.get("scientific_gate", True)}
        for spec in feature_specs:
            path = str(spec["path"])
            key = path.removeprefix("workload.")
            value = workload.get(key)
            if str(spec["type"]).lower() == "numeric":
                value = "<MISSING>" if value is None else ("low" if float(value) <= cutpoints[key] else "high")
            output[path] = value
        return output

    return feature_names, transform, cutpoints


def _policy_metrics(
    policy: Any,
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    reuse_by_id: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    selected = [
        bool(reuse_by_id[row["id"]]) if reuse_by_id is not None
        else policy.decide(row["context"]).action == "reuse"
        for row in rows
    ]
    n = len(rows)
    deployed = sum(selected)
    harmful = sum(
        use and (not bool(scores[row["id"]]["scientific_gate"]) or float(scores[row["id"]]["tau"]) <= 0.0)
        for use, row in zip(selected, rows)
    )
    utility = sum(
        float(scores[row["id"]]["tau"])
        for use, row in zip(selected, rows)
        if use and bool(scores[row["id"]]["scientific_gate"])
    ) / n
    supported = sum(
        bool(scores[row["id"]]["scientific_gate"]) and float(scores[row["id"]]["tau"]) > 0.0
        for row in rows
    )
    retained = sum(
        use and bool(scores[row["id"]]["scientific_gate"]) and float(scores[row["id"]]["tau"]) > 0.0
        for use, row in zip(selected, rows)
    )
    return {
        "contexts": n,
        "deployed": deployed,
        "harmful_deployed": harmful,
        "supported": supported,
        "supported_deployed": retained,
        "utility_sum": utility * n,
        "coverage": deployed / n if n else None,
        "unsupported_all": harmful / n if n else None,
        "unsupported_deployed": harmful / deployed if deployed else None,
        "support_recall": retained / supported if supported else None,
        "mean_incremental_utility": utility,
    }


def run() -> dict[str, Any]:
    units = []
    for family in FAMILIES:
        feature_specs = family_predicate_grammar(family)["features"]
        for seed in SEEDS:
            split = _split(family, seed)
            feature_names, transform, cutpoints = _context_and_cutpoints(
                [item.context for item in split["calibration"]], feature_specs
            )
            cal_rows = [
                {"id": item.observation_id, "family": f"{family}:{seed}",
                 "context": transform(item.context)}
                for item in split["calibration"]
            ]
            cal_scores = {
                item.observation_id: {"tau": item.effect, "scientific_gate": item.gate_passed}
                for item in split["calibration"]
            }
            soft = fit_soft_acre(
                cal_rows, cal_scores, feature_names=feature_names,
                risk_weight=RISK_WEIGHT, complexity_penalty=COMPLEXITY_PENALTY,
                max_literals=min(MAX_LITERALS, len(feature_names)),
            )
            simple = fit_soft_acre(
                cal_rows, cal_scores, feature_names=feature_names,
                risk_weight=0.0, complexity_penalty=0.0, max_literals=1,
            )
            evaluation_rows = [
                {"id": item.observation_id, "family": family, "seed": seed,
                 "context": transform(item.context)}
                for item in split["evaluation"]
            ]
            eval_scores = {
                item.observation_id: {"tau": item.effect, "scientific_gate": item.gate_passed}
                for item in split["evaluation"]
            }
            full_acre_predictions, _ = _acre_predictions(
                family, split["synthesis"], split["evaluation"], 1.0
            )
            global_policy = SoftAcrePolicy("global", (), 0.0, 0.0, 0.0, len({row["family"] for row in cal_rows}))
            policies = {
                "soft_acre": soft,
                "single_feature": simple,
                "global": global_policy,
            }
            metrics = {
                name: _policy_metrics(policy, evaluation_rows, eval_scores)
                for name, policy in policies.items()
            }
            metrics["full_acre"] = _policy_metrics(
                None, evaluation_rows, eval_scores, reuse_by_id=full_acre_predictions
            )
            units.append({
                "family": family, "seed": seed,
                "calibration_contexts": len(cal_rows), "evaluation_contexts": len(evaluation_rows),
                "numeric_cutpoints_fit_on_calibration": cutpoints,
                "soft_route": soft.route, "soft_literals": [[k, v] for k, v in soft.literals],
                "simple_route": simple.route, "simple_literals": [[k, v] for k, v in simple.literals],
                "metrics": metrics,
            })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[unit["family"]].append(unit)
    family_differences = {
        name: [
            sum(unit["metrics"]["soft_acre"]["mean_incremental_utility"] - unit["metrics"][name]["mean_incremental_utility"] for unit in rows) / len(rows)
            for _, rows in sorted(grouped.items())
        ]
        for name in ("single_feature", "global", "full_acre")
    }
    comparisons = {
        name: {
            "mean_family_utility_difference": sum(values) / len(values),
            "family_bootstrap_95_ci": _family_bootstrap_interval(values, seed=20261102 + index),
            "positive_families": sum(value > 0 for value in values),
            "negative_families": sum(value < 0 for value in values),
            "tied_families": sum(value == 0 for value in values),
            "independent_families": len(values),
        }
        for index, (name, values) in enumerate(family_differences.items())
    }
    policy_names = ("soft_acre", "single_feature", "global", "full_acre")
    totals = {}
    for name in policy_names:
        rows = [unit["metrics"][name] for unit in units]
        contexts = sum(row["contexts"] for row in rows)
        deployed = sum(row["deployed"] for row in rows)
        harmful = sum(row["harmful_deployed"] for row in rows)
        supported = sum(row["supported"] for row in rows)
        retained = sum(row["supported_deployed"] for row in rows)
        totals[name] = {
            "contexts": contexts,
            "deployed": deployed,
            "coverage": deployed / contexts,
            "unsupported_all": harmful / contexts,
            "unsupported_deployed": harmful / deployed if deployed else None,
            "support_recall": retained / supported if supported else None,
            "mean_incremental_utility": sum(row["utility_sum"] for row in rows) / contexts,
        }
    return {
        "study_id": STUDY_ID,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "analysis_class": "retrospective_provider_free_benchmark_reanalysis",
        "data_origin": "frozen SPE-EvoBench deterministic family generator and observation split",
        "resource_contract": {"provider_calls": 0, "gpu": "none"},
        "design": {
            "families": list(FAMILIES), "independent_families": len(FAMILIES),
            "seeds_per_family": len(SEEDS), "family_seed_units": len(units),
            "surface_count": SURFACE_COUNT,
            "split": "existing deterministic SHA256 observation split: 50% synthesis, 25% calibration, 25% evaluation",
            "policy_fit": "Soft-ACRE fits paired effects and scientific gate on calibration rows only; numeric cutpoints are calibration medians",
            "feature_grammar": "registered family feature paths; categorical equality; conjunctions up to three literals",
            "risk_weight": RISK_WEIGHT, "complexity_penalty": COMPLEXITY_PENALTY,
        },
        "micro_metrics_over_evaluation_contexts": totals,
        "soft_acre_minus_baseline_family_bootstrap": comparisons,
        "units": units,
        "claim_boundary": "This reuses a previously reported deterministic benchmark generator and evaluation split. It is a retrospective algorithm comparison, not a new independent or prospective efficacy result.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, default=str))
