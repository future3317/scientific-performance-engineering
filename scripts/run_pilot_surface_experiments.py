#!/usr/bin/env python3
"""Run the 100--500 context hidden-surface pilot without formal-50 claims."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import random
import math
from statistics import median
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.families import family_cases, run_boundary_family
from benchmark.boundary.baselines import (
    _distance,
    _grammar_for,
    _workload,
    predict_applicability,
)
from benchmark.boundary.evaluator import evaluate_applicability
from benchmark.families import PILOT_FAMILIES, validate_cross_view_consistency
from benchmark.harness.evolution import run_episode
from benchmark.interaction.acquisition_bench import run_acquisition_benchmark
from benchmark.interaction.acquisition_bench import run_budget_aware_acquisition_benchmark
from benchmark.interaction.factorial_bench import run_family_factorial_benchmark, run_higher_order_benchmark, run_interaction_power_curve
from benchmark.interaction.router_bench import run_router_benchmark
from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, evaluate_trajectory, run_acquisition
from core.acre.decision_probe import decision_sensitivity_callback
from core.acre.cegis import StatisticalCEGIS
from core.acre.predicates import predicate_complexity
from core.predicates import match_predicate


METHOD_POLICIES = [
    "always_apply",
    "observed_positive_only",
    "nearest_experience",
    "untyped_positive_summary",
    "shortest_without_counterexamples",
    "acre_full",
]
METHOD_EVIDENCE_PREFIXES: list[int | str] = [1, 2, 4, 8, 16, "all"]


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
                can_change_decisions=("deploy", "abstain"),
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
    episode_paths = (
        root / "benchmark" / "tasks" / "EVOL-EPISODE-POISON-10" / "episodes" / "poison_episode.yaml",
        root / "benchmark" / "tasks" / "EVOL-COMPILER-DRIFT-20" / "episodes" / "compiler_drift_episode.yaml",
        root / "benchmark" / "tasks" / "EVOL-EQUIVARIANT-SPECIALIZE-30" / "episodes" / "equivariant_specialization_episode.yaml",
    )
    results: dict[str, Any] = {"episodes": {}}
    with tempfile.TemporaryDirectory(prefix="spe-evolution-pilot-") as temp:
        for episode in episode_paths:
            episode_results: dict[str, Any] = {}
            for condition in ("C", "D"):
                out = Path(temp) / episode.stem / condition
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
                    episode_results[condition] = {
                        "status": "complete",
                        "context_mode": result["context_mode"],
                        "metrics": result["metrics"],
                        "promoted_rules": list(result["raw"].get("promoted_rules", [])),
                        "canonical_rules": canonical_rules,
                        "promotion_records": promotion_records,
                        "family_transformations": result["raw"]["family_transformations"],
                    }
                except Exception as exc:
                    episode_results[condition] = {"status": "failed", "error": str(exc)}
            results["episodes"][episode.stem] = episode_results
    compiler_result = results["episodes"]["compiler_drift_episode"]
    results.update(compiler_result)
    return results


def _bootstrap_mean(values: list[float], *, seed: int, draws: int = 1000) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "lower": None, "upper": None, "n_families": 0}
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    samples.sort()
    lower_index = max(0, int(0.025 * draws) - 1)
    upper_index = min(draws - 1, int(0.975 * draws))
    return {
        "mean": sum(values) / len(values),
        "lower": samples[lower_index],
        "upper": samples[upper_index],
        "n_families": len(values),
    }


def run_method_efficacy(*, surface_count: int = 100, seed: int = 7) -> dict[str, Any]:
    """Run provider-free applicability comparisons on the family catalog."""
    if surface_count < 6:
        raise ValueError("surface_count must be at least 6")
    family_results: list[dict[str, Any]] = []
    for family in PILOT_FAMILIES:
        pools = family_cases(family, surface_count=surface_count, seed=seed)
        visible = sorted(
            list(pools["representative_pool"]) + list(pools["active_query_pool"]),
            key=lambda item: item.observation_id,
        )
        heldout = sorted(pools["sealed_boundary_pool"], key=lambda item: item.observation_id)
        truth = {item.observation_id: item.expected_applicable for item in heldout}
        prefix_results: dict[str, dict[str, Any]] = {}
        for prefix in METHOD_EVIDENCE_PREFIXES:
            visible_prefix = visible if prefix == "all" else visible[: int(prefix)]
            policy_metrics: dict[str, Any] = {}
            for policy in METHOD_POLICIES:
                predictions = predict_applicability(
                    policy,
                    visible_prefix,
                    heldout,
                    family=family,
                )
                policy_metrics[policy] = evaluate_applicability(predictions, truth)
            prefix_results[str(prefix)] = {
                "visible_evidence_count": len(visible_prefix),
                "heldout_count": len(heldout),
                "policies": policy_metrics,
            }
        family_results.append({
            "family": family,
            "visible_count": len(visible),
            "heldout_count": len(heldout),
            "prefixes": prefix_results,
        })

    aggregate: dict[str, dict[str, Any]] = {policy: {} for policy in METHOD_POLICIES}
    for policy in METHOD_POLICIES:
        for prefix in METHOD_EVIDENCE_PREFIXES:
            key = str(prefix)
            metrics = [
                result["prefixes"][key]["policies"][policy]
                for result in family_results
            ]
            aggregate[policy][key] = {
                metric: _bootstrap_mean(
                    [float(item[metric]) for item in metrics if item[metric] is not None],
                    seed=seed * 1000 + len(policy) * 17 + len(key),
                )
                for metric in ("precision", "recall", "f1", "unsafe_generalization_rate", "abstention_rate")
            }
    pairwise_acre: dict[str, dict[str, Any]] = {}
    for policy in METHOD_POLICIES:
        if policy == "acre_full":
            continue
        pairwise_acre[policy] = {}
        for prefix in METHOD_EVIDENCE_PREFIXES:
            key = str(prefix)
            differences = {
                metric: [
                    float(result["prefixes"][key]["policies"]["acre_full"][metric])
                    - float(result["prefixes"][key]["policies"][policy][metric])
                    for result in family_results
                    if result["prefixes"][key]["policies"]["acre_full"][metric] is not None
                    and result["prefixes"][key]["policies"][policy][metric] is not None
                ]
                for metric in ("recall", "f1", "unsafe_generalization_rate")
            }
            pairwise_acre[policy][key] = {
                metric: _bootstrap_mean(
                    values,
                    seed=seed * 2000 + len(policy) * 31 + len(key) * 7 + len(metric),
                )
                for metric, values in differences.items()
            }
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "surface_count": surface_count,
        "family_count": len(family_results),
        "families": list(PILOT_FAMILIES),
        "policies": list(METHOD_POLICIES),
        "evidence_prefixes": list(METHOD_EVIDENCE_PREFIXES),
        "uncertainty": "family_cluster_bootstrap_95_percent",
        "family_results": family_results,
        "aggregate": aggregate,
        "pairwise_acre": pairwise_acre,
    }


def run_provider_free_ablation(*, root: Path, surface_count: int = 100, seed: int = 7) -> dict[str, Any]:
    """Collect component comparisons over the same deterministic fixtures."""
    method = run_method_efficacy(surface_count=surface_count, seed=seed)
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "surface_count": surface_count,
        "applicability": {
            "acre_full": method["aggregate"]["acre_full"]["all"],
            "minus_counterexample": method["aggregate"]["shortest_without_counterexamples"]["all"],
            "minus_typed": method["aggregate"]["untyped_positive_summary"]["all"],
        },
        "router": run_router_benchmark(),
        "lifecycle": run_drift_poison(root=root, seed=seed),
    }


def run_budget_aware_ablation(*, root: Path | None = None, seed: int = 7) -> dict[str, Any]:
    """Run the compact provider-free resource/acquisition policy ablation."""
    from benchmark.calibration.split import build_calibration_split
    from core.acre.resource_budget import ExecutionReserveBudget

    # This admission fixture is intentionally public and deterministic: it
    # measures the cost of the eligibility policy, not relation truth.
    candidates = range(16)
    eager_invocations = len(tuple(candidates))
    lazy_invocations = sum(
        1 for index in candidates
        if index % 2 == 0 and index % 4 == 0
    )
    calibration = build_calibration_split(
        ("compile", "episode"), calibration_seed=101, final_seed=seed, count=4,
    )
    reserve_ledger = ExecutionReserveBudget(total_budget=3.0, scheduled_minimums=(1.0, 1.0))
    reserve_ledger, admission = reserve_ledger.admit_evolution(2.0, query_id="reserve-limited-query")
    completed = 0
    while completed < len(reserve_ledger.scheduled_minimums):
        reserve_ledger = reserve_ledger.advance_task()
        completed += 1
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "variants": [
            "legacy_acre", "execution_reserve", "certificate_critical",
            "lazy_relation", "full_budget_aware_acre",
        ],
        "acquisition": run_budget_aware_acquisition_benchmark(seed=seed),
        "relation": {
            "eager_reference": {
                "factorial_invocations": eager_invocations,
                "fixture": "public_relation_admission_v1",
            },
            "lazy_relation": {
                "factorial_invocations": lazy_invocations,
                "factorial_invocations_saved": eager_invocations - lazy_invocations,
                "fixture": "public_relation_admission_v1",
            },
        },
        "execution_reserve": {
            "total_budget": 3.0,
            "evolution_frozen": reserve_ledger.evolution_frozen,
            "evolution_queries_admitted": int(admission.accepted),
            "scheduled_cells": len(reserve_ledger.scheduled_minimums),
            "scheduled_cells_completed": completed,
            "missing_scheduled_cells": len(reserve_ledger.scheduled_minimums) - completed,
            "admission_reason": admission.reason,
        },
        "calibration_split": calibration,
    }


def _support_split(family: str, *, surface_count: int, seed: int) -> dict[str, Any]:
    pools = family_cases(family, surface_count=surface_count, seed=seed)
    evidence = list(pools["representative_pool"]) + list(pools["active_query_pool"])
    sealed = list(pools["sealed_boundary_pool"])
    # The first half is a calibration context set used only to choose the
    # matched-coverage threshold; the second half is never consulted until
    # final policy evaluation.
    calibration = sealed[::2]
    evaluation = sealed[1::2]
    return {
        "evidence": evidence,
        "calibration": calibration,
        "evaluation": evaluation,
    }


def _positive_similarity(observation: Any, positives: list[Any]) -> float:
    if not positives:
        return 0.0
    return max(
        1.0 / (1.0 + _distance(_workload(anchor), _workload(observation)))
        for anchor in positives
    )


def _matched_positive_threshold(
    evidence: list[Any], calibration: list[Any], *, family: str,
) -> tuple[float, float, dict[str, bool]]:
    positives = [item for item in evidence if item.positive_anchor()]
    full_calibration = predict_applicability(
        "acre_full", evidence, calibration, family=family,
    )
    target_coverage = sum(full_calibration.values()) / len(calibration) if calibration else 0.0
    scores = [_positive_similarity(item, positives) for item in calibration]
    candidates = sorted(set(scores + [0.0, 1.0, math.nextafter(max(scores, default=1.0), math.inf)]))
    threshold = min(
        candidates,
        key=lambda value: (
            abs(sum(score >= value for score in scores) / len(scores) - target_coverage)
            if scores else abs(target_coverage),
            -value,
        ),
    )
    return threshold, target_coverage, full_calibration


def _matched_positive_predictions(
    evidence: list[Any], contexts: list[Any], threshold: float,
) -> dict[str, bool]:
    positives = [item for item in evidence if item.positive_anchor()]
    return {
        item.observation_id: _positive_similarity(item, positives) >= threshold
        for item in contexts
    }


def _synthesized_predicate(
    positives: list[Any], counterexamples: list[Any], family: str,
) -> tuple[dict[str, Any] | None, int | None]:
    synthesis = StatisticalCEGIS(_grammar_for(family)).synthesize(
        positive=positives,
        counterexamples=counterexamples,
        parent_predicate=None,
    )
    if synthesis.predicate is None:
        return None, None
    return synthesis.predicate, int(predicate_complexity(synthesis.predicate)["description_length"])


def _subset_seed_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def run_support_boundary_package(
    *, surface_count: int = 100, seed: int = 7, subset_seeds: int = 10,
) -> dict[str, Any]:
    """Run provider-free evidence-layer, matched-coverage, and CE-dose analyses."""
    if surface_count < 12:
        raise ValueError("surface_count must be at least 12")
    layers = {
        "activation_only": "always_apply",
        "global_utility_only": "shortest_without_counterexamples",
        "typed_applicability_only": "untyped_positive_summary",
        "positive_anchor_only_no_counterexample": "observed_positive_only",
        "nearest_experience": "nearest_experience",
        "full_acre": "acre_full",
    }
    layer_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in layers}
    family_rows: list[dict[str, Any]] = []
    dose_rows: dict[str, list[dict[str, Any]]] = {str(dose): [] for dose in (0.0, 0.25, 0.5, 1.0)}

    for family_index, family in enumerate(PILOT_FAMILIES):
        split = _support_split(family, surface_count=surface_count, seed=seed)
        evidence = split["evidence"]
        calibration = split["calibration"]
        evaluation = split["evaluation"]
        truth = {item.observation_id: item.expected_applicable for item in evaluation}
        family_record: dict[str, Any] = {
            "family": family,
            "evidence_count": len(evidence),
            "calibration_count": len(calibration),
            "evaluation_count": len(evaluation),
            "layers": {},
            "matched_coverage": {},
            "counterexample_dose": {},
        }
        for name, policy in layers.items():
            predictions = predict_applicability(policy, evidence, evaluation, family=family)
            metrics = evaluate_applicability(predictions, truth)
            family_record["layers"][name] = metrics
            layer_rows[name].append({"family": family, **metrics})

        threshold, target_coverage, calibration_predictions = _matched_positive_threshold(
            evidence, calibration, family=family,
        )
        matched_predictions = _matched_positive_predictions(evidence, evaluation, threshold)
        matched_metrics = evaluate_applicability(matched_predictions, truth)
        family_record["matched_coverage"] = {
            "threshold": threshold,
            "calibration_target_coverage": target_coverage,
            "calibration_predicted_count": sum(calibration_predictions.values()),
            "calibration_count": len(calibration),
            "evaluation": matched_metrics,
        }

        positives = [item for item in evidence if item.positive_anchor()]
        counterexamples = [item for item in evidence if item.certified_counterexample()]
        for dose in (0.0, 0.25, 0.5, 1.0):
            per_family: list[dict[str, Any]] = []
            for subset_seed in range(subset_seeds):
                rng = random.Random(seed * 100000 + family_index * 1000 + subset_seed)
                count = round(dose * len(counterexamples))
                selected = sorted(
                    rng.sample(counterexamples, count) if count else [],
                    key=lambda item: item.observation_id,
                )
                predicate, description_length = _synthesized_predicate(
                    positives, selected, family,
                )
                predictions = {
                    item.observation_id: bool(predicate and match_predicate(predicate, item.context))
                    for item in evaluation
                }
                metrics = evaluate_applicability(predictions, truth)
                row = {
                    "family": family,
                    "subset_seed": subset_seed,
                    "dose": dose,
                    "counterexample_count": count,
                    "description_length": description_length,
                    **metrics,
                }
                per_family.append(row)
                dose_rows[str(dose)].append(row)
            family_record["counterexample_dose"][str(dose)] = {
                "subset_seeds": subset_seeds,
                "subset_seed_summary": {
                    metric: _subset_seed_summary(per_family, metric)
                    for metric in ("unsafe_generalization_rate", "recall", "abstention_rate", "description_length")
                },
                "rows": per_family,
            }
        family_rows.append(family_record)

    raw_counts: dict[str, Any] = {}
    for name, rows in layer_rows.items():
        raw_counts[name] = {
            metric: sum(int(row[metric]) for row in rows)
            for metric in ("tp", "fp", "fn", "tn")
        }
        counts = raw_counts[name]
        total = sum(counts.values())
        raw_counts[name].update({
            "n": total,
            "coverage_n": counts["tp"] + counts["fp"],
            "unsupported_n": counts["fp"],
        })
    dose_summary: dict[str, Any] = {}
    for dose, rows in dose_rows.items():
        pooled = {
            metric: sum(int(row[metric]) for row in rows)
            for metric in ("tp", "fp", "fn", "tn")
        }
        total = sum(pooled.values())
        family_means = []
        for family in PILOT_FAMILIES:
            family_subset = [row for row in rows if row["family"] == family]
            def mean_metric(metric: str) -> float | None:
                values = [float(row[metric]) for row in family_subset if row[metric] is not None]
                return sum(values) / len(values) if values else None
            family_means.append({
                "family": family,
                **{
                    metric: mean_metric(metric)
                    for metric in ("unsafe_generalization_rate", "recall", "abstention_rate", "description_length")
                },
            })
        dose_summary[dose] = {
            "subset_seeds": subset_seeds,
            "runs": len(rows),
            "pooled_counts": {**pooled, "n": total, "coverage_n": pooled["tp"] + pooled["fp"], "unsupported_n": pooled["fp"]},
            "family_cluster_means": {
                metric: _bootstrap_mean(
                    [row[metric] for row in family_means if row[metric] is not None],
                    seed=9000 + len(metric),
                )
                for metric in ("unsafe_generalization_rate", "recall", "abstention_rate", "description_length")
            },
        }
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "analysis": "context_indexed_support_boundary_mechanism",
        "surface_count": surface_count,
        "family_count": len(PILOT_FAMILIES),
        "families": list(PILOT_FAMILIES),
        "seed": seed,
        "split": "family evidence and calibration contexts are used for fitting/threshold selection; untouched evaluation contexts are used for final metrics",
        "layers": list(layers),
        "raw_counts": raw_counts,
        "layer_family_rows": layer_rows,
        "family_rows": family_rows,
        "matched_coverage": {
            "baseline": "positive-anchor similarity using calibration-selected threshold",
            "evaluation_family_cluster": {
                metric: _bootstrap_mean(
                    [row["matched_coverage"]["evaluation"][metric] for row in family_rows if row["matched_coverage"]["evaluation"][metric] is not None],
                    seed=12000 + len(metric),
                )
                for metric in ("unsafe_generalization_rate", "recall", "abstention_rate")
            },
        },
        "counterexample_dose": {
            "doses": [0.0, 0.25, 0.5, 1.0],
            "subset_seeds": subset_seeds,
            "summary": dose_summary,
        },
        "interpretation": "provider-free mechanism diagnostics; family-cluster intervals use family means, subset seeds are descriptive repeats and pooled counts have no iid binomial confidence claim; no hosted efficacy claim",
    }


def run_budget_aware_reserve_stress(
    *, seed: int = 7, schedule_count: int = 256, reserve_enabled: bool = True,
) -> dict[str, Any]:
    """Stress the reserve contract and its matched no-reserve control.

    Each seeded schedule is replayed at three levels of optional-evidence
    slack.  The schedule and query sequence are identical across levels and
    reserve modes.  With ``reserve_enabled=False``, optional evolution may
    consume the scheduled execution envelope before tasks are replayed.  The
    measured outcome is operational: optional work admitted/abstained and
    scheduled cells completed.  It is not a utility or efficacy estimate.
    """
    from core.acre.resource_budget import ExecutionReserveBudget

    if schedule_count < 1:
        raise ValueError("schedule_count must be positive")
    rng = random.Random(int(seed))
    schedule_lengths = (1, 2, 4, 8, 16, 32)
    minimum_costs = (0.25, 0.5, 1.0, 2.0)
    query_costs = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    slack_by_level = {"tight": 0.0, "moderate": 2.0, "slack": 8.0}
    summary = {
        level: {
            "scheduled_cells": 0,
            "scheduled_cells_completed": 0,
            "missing_scheduled_cells": 0,
            "optional_query_opportunities": 0,
            "evolution_queries_admitted": 0,
            "abstentions": 0,
            "evolution_spent": 0.0,
            "evolution_frozen_schedules": 0,
        }
        for level in slack_by_level
    }

    for _ in range(int(schedule_count)):
        length = rng.choice(schedule_lengths)
        minimums = tuple(rng.choice(minimum_costs) for _ in range(length))
        queries = tuple(rng.choice(query_costs) for _ in range(rng.randint(2, 8)))
        reserve = float(sum(minimums))
        for level, slack in slack_by_level.items():
            total_budget = reserve + slack
            ledger = ExecutionReserveBudget(
                total_budget=total_budget,
                scheduled_minimums=minimums if reserve_enabled else (),
            )
            admitted = 0
            abstained = 0
            for query_index, cost in enumerate(queries):
                ledger, decision = ledger.admit_evolution(
                    cost, query_id=f"stress-{level}-{query_index}"
                )
                if decision.accepted:
                    admitted += 1
                else:
                    abstained += 1
            completed = 0
            if reserve_enabled:
                while ledger.task_index < len(ledger.scheduled_minimums):
                    ledger = ledger.advance_task()
                    completed += 1
            else:
                remaining = max(0.0, total_budget - ledger.evolution_spent)
                for minimum_cost in minimums:
                    if remaining + 1e-12 < minimum_cost:
                        break
                    remaining -= minimum_cost
                    completed += 1
            row = summary[level]
            row["scheduled_cells"] += length
            row["scheduled_cells_completed"] += completed
            row["missing_scheduled_cells"] += length - completed
            row["optional_query_opportunities"] += len(queries)
            row["evolution_queries_admitted"] += admitted
            row["abstentions"] += abstained
            row["evolution_spent"] += ledger.evolution_spent
            row["evolution_frozen_schedules"] += int(ledger.evolution_frozen)

    for row in summary.values():
        row["scheduled_completion_rate"] = (
            row["scheduled_cells_completed"] / row["scheduled_cells"]
            if row["scheduled_cells"] else 1.0
        )
        row["abstention_rate"] = (
            row["abstentions"] / row["optional_query_opportunities"]
            if row["optional_query_opportunities"] else 0.0
        )
    total_cells = summary["tight"]["scheduled_cells"]
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "seed": int(seed),
        "schedule_count": int(schedule_count),
        "reserve_enabled": bool(reserve_enabled),
        "resource_policy": "execution_reserve" if reserve_enabled else "no_execution_reserve",
        "design": {
            "schedule_lengths": list(schedule_lengths),
            "minimum_execution_costs": list(minimum_costs),
            "optional_query_costs": list(query_costs),
            "optional_budget_slack": slack_by_level,
            "interpretation": "operational reserve stress; not efficacy or utility evidence",
        },
        "pressure_levels": summary,
        "scheduled_cells": total_cells,
        "scheduled_cells_completed": summary["tight"]["scheduled_cells_completed"],
        "missing_scheduled_cells": summary["tight"]["missing_scheduled_cells"],
        "scheduled_completion_rate": summary["tight"]["scheduled_completion_rate"],
    }


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
    parser.add_argument("--method-efficacy-only", action="store_true")
    parser.add_argument("--component-ablation", action="store_true")
    parser.add_argument("--support-boundary-package", action="store_true")
    parser.add_argument("--budget-aware-ablation", action="store_true")
    parser.add_argument("--budget-aware-stress", action="store_true")
    parser.add_argument("--budget-aware-no-reserve", action="store_true")
    args = parser.parse_args()
    selected_modes = sum(bool(value) for value in (args.method_efficacy_only, args.component_ablation, args.support_boundary_package, args.budget_aware_ablation, args.budget_aware_stress, args.budget_aware_no_reserve))
    if selected_modes > 1:
        parser.error("provider-free experiment modes are mutually exclusive")
    if args.budget_aware_stress:
        report = run_budget_aware_reserve_stress(seed=args.seed)
    elif args.budget_aware_no_reserve:
        report = run_budget_aware_reserve_stress(seed=args.seed, reserve_enabled=False)
    elif args.budget_aware_ablation:
        report = run_budget_aware_ablation(root=root, seed=args.seed)
    elif args.component_ablation:
        report = run_provider_free_ablation(root=root, surface_count=args.surface_count, seed=args.seed)
    elif args.support_boundary_package:
        report = run_support_boundary_package(surface_count=args.surface_count, seed=args.seed)
    elif args.method_efficacy_only:
        report = run_method_efficacy(surface_count=args.surface_count, seed=args.seed)
    else:
        report = run_pilot(root=root, surface_count=args.surface_count, seed=args.seed)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "surface_count": report.get("surface_count"), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
