#!/usr/bin/env python3
"""Run the unified provider-free context-indexed support frontier."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.baselines import predict_applicability
from benchmark.boundary.families import family_cases
from benchmark.boundary.policy_eval import Decision, evaluate_policy
from benchmark.boundary.tree_baselines import fit_cart_policy, fit_catboost_ltt_policy
from benchmark.families import PILOT_FAMILIES
from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.predicates import match_predicate
from scripts.run_support_boundary_analysis import (
    _context_key,
    _grammar_for,
    _split_cases,
    _support_metrics,
)


def _prediction_policy(predictions: Mapping[str, bool], evaluation_cases: Sequence[Any], reason: str):
    by_context = {
        _context_key(item.context): bool(predictions.get(item.observation_id, False))
        for item in evaluation_cases
    }

    def decide(context: Mapping[str, Any]) -> Decision:
        admitted = by_context.get(_context_key(context), False)
        return Decision("reuse" if admitted else "abstain", 1.0 if admitted else 0.0, 0.0, reason)

    return decide


def _all_apply(_: Mapping[str, Any]) -> Decision:
    return Decision("reuse", 1.0, 0.0, "global_reuse")


def _policy_predictions(name: str, family: str, synthesis: Sequence[Any], calibration: Sequence[Any], evaluation: Sequence[Any]):
    if name == "global_reuse":
        return _all_apply, None
    if name == "positive_only":
        predictions = predict_applicability("observed_positive_only", synthesis, evaluation, family=family)
        return _prediction_policy(predictions, evaluation, "positive_only"), None
    if name == "nearest_experience":
        predictions = predict_applicability("nearest_experience", synthesis, evaluation, family=family)
        return _prediction_policy(predictions, evaluation, "nearest_experience"), None
    if name == "matched_positive":
        predictions = predict_applicability("shortest_without_counterexamples", synthesis, evaluation, family=family)
        return _prediction_policy(predictions, evaluation, "matched_positive"), None

    synthesis_rows = [{"context": dict(item.context), "label": item.positive_anchor()} for item in synthesis]
    calibration_rows = [{"context": dict(item.context), "unsupported": not item.positive_anchor()} for item in calibration]
    if name == "cart_ltt":
        return fit_cart_policy(synthesis_rows, calibration_rows, _schema(family, synthesis), max_depth=5, seed=0), None
    if name == "catboost_ltt":
        return fit_catboost_ltt_policy(synthesis_rows, calibration_rows, _schema(family, synthesis), depth=5, trees=300, learning_rate=0.05, seed=0), None
    if name == "acre_full":
        positives = [item for item in synthesis if item.positive_anchor()]
        counterexamples = [item for item in synthesis if item.certified_counterexample()]
        if not positives or not counterexamples:
            return _prediction_policy({}, evaluation, "acre_insufficient_evidence"), None
        result = StatisticalCEGIS(_grammar_for(family)).synthesize(
            positive=positives,
            counterexamples=counterexamples,
            parent_predicate=None,
            decision_contexts=[item.context for item in evaluation],
        )
        predictions = {
            item.observation_id: bool(result.predicate and match_predicate(result.predicate, item.context))
            for item in evaluation
        }
        return _prediction_policy(predictions, evaluation, "acre_certificate"), result.to_dict()
    raise ValueError(f"unknown policy {name}")


def _schema(family: str, synthesis: Sequence[Any]):
    from benchmark.boundary.typed_features import TypedFeatureSchema

    return TypedFeatureSchema.from_calibration(family, [item.context for item in synthesis])


def _metric_row(policy: str, family: str, seed: int, policy_obj: Any, synthesis: Sequence[Any], calibration: Sequence[Any], evaluation: Sequence[Any], predicate: Mapping[str, Any] | None = None, freeze: Mapping[str, Any] | None = None) -> dict[str, Any]:
    eval_rows = [{"id": item.observation_id, "family": family, "seed": seed, "context": dict(item.context)} for item in evaluation]
    calibration_rows = [{"id": item.observation_id, "context": dict(item.context), "label": item.positive_anchor()} for item in calibration]
    score_records = {
        item.observation_id: {"tau": item.effect, "scientific_gate": item.gate_passed, "ci_width": item.effect_upper - item.effect_lower}
        for item in evaluation
    }
    metrics = evaluate_policy(policy_obj, calibration_rows, eval_rows, score_records, delta=0.0)
    predictions = {entry["id"]: entry["action"] == "reuse" for entry in metrics["decisions"]}
    support = _support_metrics(predictions, evaluation)
    return {
        "policy": policy,
        "family": family,
        "seed": int(seed),
        "evaluation_ids": [item.observation_id for item in evaluation],
        "predicate": dict(predicate) if predicate else None,
        "freeze": dict(freeze or {}),
        "provider_calls": 0,
        "predicate_description_length": None,
        **metrics,
        **support,
    }


def _permuted_observations(synthesis: Sequence[Any], seed: int) -> list[BoundaryObservation]:
    labels = [item.positive_anchor() for item in synthesis]
    random.Random(seed).shuffle(labels)
    return [
        BoundaryObservation(
            item.observation_id,
            item.context,
            1.0 if label else 0.0,
            True,
            1.0 if label else 0.0,
            1.0 if label else 0.0,
        )
        for item, label in zip(synthesis, labels)
    ]


def _permutation_row(family: str, seed: int, evaluation: Sequence[Any], synthesis: Sequence[Any], permutation: int) -> dict[str, Any]:
    permuted = _permuted_observations(synthesis, seed=20260914 + permutation)
    positives = [item for item in permuted if item.positive_anchor()]
    counterexamples = [item for item in permuted if item.certified_counterexample()]
    result = StatisticalCEGIS(_grammar_for(family)).synthesize(
        positive=positives,
        counterexamples=counterexamples,
        parent_predicate=None,
        decision_contexts=[item.context for item in evaluation],
    ) if positives and counterexamples else None
    predictions = {
        item.observation_id: bool(result and result.predicate and match_predicate(result.predicate, item.context))
        for item in evaluation
    }
    support = _support_metrics(predictions, evaluation)
    return {
        "family": family,
        "seed": int(seed),
        "permutation": int(permutation),
        "support_precision": support["support_precision"],
        "support_recall": support["support_recall"],
        "support_decision_regret_mean": None,
        "support_tp": support["support_tp"],
        "support_fp": support["support_fp"],
        "support_fn": support["support_fn"],
        "predicate": result.to_dict() if result else None,
    }


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], reps: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['family']}|{row['seed']}"] .append(row)
    metrics = ("coverage", "unsupported_deployed", "unsupported_all", "support_precision", "support_recall", "support_decision_regret_mean")
    output: dict[str, Any] = {"n_clusters": len(grouped), "bootstrap_reps": int(reps), "metrics": {}}
    rng = np.random.default_rng(seed)
    clusters = list(grouped.values())
    for metric in metrics:
        values = [float(group[0][metric]) for group in clusters if group[0].get(metric) is not None]
        if not values:
            output["metrics"][metric] = None
            continue
        samples = rng.choice(np.asarray(values, dtype=float), size=(int(reps), len(values)), replace=True).mean(axis=1)
        output["metrics"][metric] = {
            "estimate": float(np.mean(values)),
            "lower": float(np.quantile(samples, 0.025)),
            "upper": float(np.quantile(samples, 0.975)),
        }
    return output


def run_context_indexed_frontier(*, surface_count: int = 64, generator_seeds: Sequence[int] = (0, 1, 2, 3, 4), families: Sequence[str] | None = None, doses: Sequence[float] = (1.0,), permutation_reps: int = 100, bootstrap_reps: int = 10000) -> dict[str, Any]:
    selected_families = tuple(families or PILOT_FAMILIES)
    rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    policy_names = ("global_reuse", "positive_only", "nearest_experience", "matched_positive", "cart_ltt", "catboost_ltt", "acre_full")
    for family in selected_families:
        for seed in generator_seeds:
            split = _split_cases(family, int(seed), surface_count)
            synthesis, calibration, evaluation = split["synthesis"], split["calibration"], split["evaluation"]
            for name in policy_names:
                policy, receipt = _policy_predictions(name, family, synthesis, calibration, evaluation)
                rows.append(_metric_row(name, family, int(seed), policy, synthesis, calibration, evaluation, freeze={"receipt": receipt, "evaluation_used_for_fit": False}))
            for permutation in range(int(permutation_reps)):
                permutation_rows.append(_permutation_row(family, int(seed), evaluation, synthesis, permutation))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    return {
        "schema_version": 1,
        "analysis": "context_indexed_support_frontier_with_competitive_baselines",
        "provider_calls": 0,
        "registered_input_artifact": "canonical_family_generators_and_registered_split_v1",
        "families": list(selected_families),
        "generator_seeds": [int(seed) for seed in generator_seeds],
        "surface_count": int(surface_count),
        "doses": [float(value) for value in doses],
        "policy_registry": list(policy_names),
        "semantic_router": {"status": "provider_required", "included_in_provider_free_rows": False},
        "split": "deterministic sha256 observation split: 50% synthesis, 25% calibration, 25% evaluation",
        "hidden_truth_usage": "scoring_only",
        "rows": rows,
        "permutation_null": permutation_rows,
        "cluster_bootstrap": {name: _cluster_bootstrap(members, bootstrap_reps, 20260914 + index) for index, (name, members) in enumerate(sorted(grouped.items()))},
        "claim_boundary": "provider-free support-boundary comparison; semantic router and hosted efficacy require separate provider-backed artifact",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=64)
    parser.add_argument("--permutation-reps", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_context_indexed_frontier(surface_count=args.surface_count, permutation_reps=args.permutation_reps, bootstrap_reps=args.bootstrap_reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(report["rows"]), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
