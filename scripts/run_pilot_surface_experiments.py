#!/usr/bin/env python3
"""Run the 100--500 context hidden-surface pilot without formal-50 claims."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import random
from statistics import median
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.families import family_cases, run_boundary_family
from benchmark.families import PILOT_FAMILIES, validate_cross_view_consistency
from benchmark.harness.evolution import run_episode
from benchmark.interaction.acquisition_bench import run_acquisition_benchmark
from benchmark.interaction.factorial_bench import run_family_factorial_benchmark, run_higher_order_benchmark, run_interaction_power_curve
from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, evaluate_trajectory, run_acquisition
from core.acre.decision_probe import decision_sensitivity_callback


def run_active_boundary(family: str, *, surface_count: int, seed: int = 7) -> dict[str, Any]:
    pools = family_cases(family, surface_count=surface_count, seed=seed)
    queries: list[AcquisitionQuery] = []
    labels: dict[str, bool] = {}
    truths: dict[str, bool] = {}
    candidate_contexts = list(pools["representative_pool"]) + list(pools["active_query_pool"])
    context_rng = random.Random(seed * 1009 + len(family))
    visible = sorted(context_rng.sample(candidate_contexts, min(8, len(candidate_contexts))), key=lambda item: item.observation_id)
    for index, item in enumerate(visible):
        edge_id = f"{family}:{item.observation_id}"
        truths[edge_id] = item.positive_anchor()
        # Repeated, noisy observations make the certificate a real stopping
        # problem.  The noise is part of the visible experimental protocol;
        # the applicability label remains hidden from the acquisition policy.
        for replicate in range(24):
            query_id = f"{edge_id}:q{replicate}"
            queries.append(AcquisitionQuery(
                query_id,
                edge_id,
                0.6 + ((index * 3 + replicate) % 9) / 5.0,
                experiment_type="boundary",
                risk=0.2 + ((index + replicate) % 5) / 10.0,
                provenance_novelty=0.4 + (replicate % 3) / 5.0,
                context=dict(item.context),
            ))
            labels[query_id] = item.positive_anchor()
    policy_results: dict[str, Any] = {}
    for policy in AcquisitionPolicy:
        trials: list[dict[str, Any]] = []
        for trial_seed in range(seed, seed + 32):
            noisy_labels = dict(labels)
            noise_rng = random.Random(trial_seed * 7919 + len(family))
            for query_id in noisy_labels:
                if noise_rng.random() < 0.02:
                    noisy_labels[query_id] = not noisy_labels[query_id]
            decision_probe = decision_sensitivity_callback(queries) if policy is AcquisitionPolicy.DECISION_AWARE else None
            trajectory = run_acquisition(queries, noisy_labels, policy, seed=trial_seed, decision_sensitivity_fn=decision_probe)
            evaluation = evaluate_trajectory(trajectory, queries, noisy_labels, truths, target_error=0.0)
            curve = evaluation.error_trajectory
            auc = sum((curve[i - 1] + curve[i]) / 2.0 for i in range(1, len(curve))) if len(curve) > 1 else (curve[0] if curve else 0.0)
            trials.append({
                "experiments_to_certificate": len(trajectory.selected_query_ids) if trajectory.identification_certificate else None,
                "experiment_cost": trajectory.total_cost,
                "cost_to_target": evaluation.cost_to_target,
                "final_error": evaluation.final_error,
                "harmful_fp_during_learning": max(evaluation.harmful_fp_trajectory, default=0.0),
                "confusion_final": dict(evaluation.confusion_trajectory[-1]) if evaluation.confusion_trajectory else {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                "learning_curve_auc": auc,
            })
        policy_results[policy.value] = {
            "seeds": 32,
            "median_experiments_to_certificate": median([item["experiments_to_certificate"] for item in trials if item["experiments_to_certificate"] is not None]) if any(item["experiments_to_certificate"] is not None for item in trials) else None,
            "median_experiment_cost": median(item["experiment_cost"] for item in trials),
            "median_cost_to_target": median([item["cost_to_target"] for item in trials if item["cost_to_target"] is not None]) if any(item["cost_to_target"] is not None for item in trials) else None,
            "harmful_fp_during_learning": sum(item["harmful_fp_during_learning"] for item in trials) / len(trials),
            "learning_curve_auc": sum(item["learning_curve_auc"] for item in trials) / len(trials),
        }
    return {"family": family, "surface_count": surface_count, "acquisition_context_count": len(visible), "query_pool_registration": "seeded_random_subset", "replicates_per_context": 24, "policies": policy_results, "certificate": "time-uniform-bernoulli-cs"}


def run_drift_poison(*, root: Path, seed: int = 0) -> dict[str, Any]:
    episode = root / "benchmark" / "tasks" / "EVOL-COMPILER-DRIFT-20" / "episodes" / "compiler_drift_episode.yaml"
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="spe-evolution-pilot-") as temp:
        for condition in ("C", "D"):
            out = Path(temp) / condition
            try:
                result = run_episode(episode, condition, out, core_repo=root, context_mode="reset")
                rules_dir = out / "store" / "rules"
                canonical_rules = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(rules_dir.rglob("*.json"))
                    if not path.name.endswith(".state.json")
                ] if rules_dir.is_dir() else []
                promotions_dir = out / "store" / "evolution" / "promotions"
                promotion_records = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(promotions_dir.rglob("*.json"))
                ] if promotions_dir.is_dir() else []
                results[condition] = {
                    "status": "complete",
                    "context_mode": result["context_mode"],
                    "metrics": result["metrics"],
                    "promoted_rules": list(result["raw"].get("promoted_rules", [])),
                    "canonical_rules": canonical_rules,
                    "promotion_records": promotion_records,
                    "family_transformations": result["raw"]["family_transformations"],
                }
            except Exception as exc:
                results[condition] = {"status": "failed", "error": str(exc)}
    return results


def run_pilot(*, root: Path, surface_count: int = 100, seed: int = 7) -> dict[str, Any]:
    if not 100 <= surface_count <= 500:
        raise ValueError("surface_count must be between 100 and 500")
    consistency = validate_cross_view_consistency(tasks_root=root / "benchmark" / "tasks", surface_count=surface_count, seed=0)
    boundaries = [run_boundary_family(family, surface_count=surface_count, seed=seed) for family in PILOT_FAMILIES]
    active_boundary = [run_active_boundary(family, surface_count=surface_count, seed=seed) for family in PILOT_FAMILIES]
    interactions = run_family_factorial_benchmark(count=surface_count, seed=seed)
    return {
        "schema_version": 1,
        "pilot_id": "SPE-EvoBench-family-context-pilot",
        "surface_count": surface_count,
        "families": list(PILOT_FAMILIES),
        "formal_50_results": "not_claimed",
        "cross_view_consistency": consistency,
        "active_boundary": active_boundary,
        "boundary_cegis": boundaries,
        "causal_interaction": interactions,
        "higher_order_interaction": run_higher_order_benchmark(count=max(20, surface_count // 5), seed=seed),
        "interaction_power_curve": run_interaction_power_curve(seed=seed),
        "drift_poison": run_drift_poison(root=root),
        "legacy_acquisition_calibration": run_acquisition_benchmark(seed=seed),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=root / "benchmark" / "pilot_surface_report.json")
    args = parser.parse_args()
    report = run_pilot(root=root, surface_count=args.surface_count, seed=args.seed)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "surface_count": report["surface_count"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
