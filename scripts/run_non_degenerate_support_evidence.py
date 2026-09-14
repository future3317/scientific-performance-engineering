#!/usr/bin/env python3
"""Run a revisioned provider-free support frontier with repeated measurements."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.baselines import predict_applicability
from benchmark.boundary.families import family_cases
from benchmark.boundary.policy_eval import Decision, evaluate_policy
from benchmark.boundary.tree_baselines import fit_cart_policy, fit_catboost_ltt_policy
from benchmark.families import PILOT_FAMILIES
from benchmark.external.non_degenerate_evidence import classify_observation, repeated_observation
from core.acre.cegis import StatisticalCEGIS
from core.acre.predicates import predicate_complexity
from core.predicates import match_predicate
from scripts.run_support_boundary_analysis import _bucket, _context_key, _grammar_for


def _split_cases(family: str, seed: int, surface_count: int) -> dict[str, list]:
    pools = family_cases(family, surface_count=surface_count, seed=seed)
    all_cases = {item.observation_id: item for values in pools.values() for item in values}
    synthesis = [item for item in all_cases.values() if _bucket(family, seed, item.observation_id) in (0, 1)]
    calibration = [item for item in all_cases.values() if _bucket(family, seed, item.observation_id) == 2]
    evaluation = [item for item in all_cases.values() if _bucket(family, seed, item.observation_id) == 3]
    return {"synthesis": synthesis, "calibration": calibration, "evaluation": evaluation}


def _policy_predictions(name: str, family: str, synthesis: list, calibration: list, evaluation: list):
    if name == "global_reuse":
        return lambda _context: Decision("reuse", 1.0, 0.0, "global_reuse"), None
    if name == "positive_only":
        predictions = predict_applicability("observed_positive_only", synthesis, evaluation, family=family)
    elif name == "nearest_experience":
        predictions = predict_applicability("nearest_experience", synthesis, evaluation, family=family)
    elif name == "matched_positive":
        predictions = predict_applicability("shortest_without_counterexamples", synthesis, evaluation, family=family)
    else:
        schema_rows = [{"context": dict(item.context), "label": item.positive_anchor()} for item in synthesis]
        calibration_rows = [{"context": dict(item.context), "unsupported": not item.positive_anchor()} for item in calibration]
        schema = __import__("benchmark.boundary.typed_features", fromlist=["TypedFeatureSchema"]).TypedFeatureSchema.from_calibration(family, [item.context for item in synthesis])
        if name == "cart_ltt":
            return fit_cart_policy(schema_rows, calibration_rows, schema, max_depth=5, seed=0), None
        if name == "catboost_ltt":
            return fit_catboost_ltt_policy(schema_rows, calibration_rows, schema, depth=5, trees=300, learning_rate=0.05, seed=0), None
        positives = [item for item in synthesis if item.positive_anchor()]
        counterexamples = [item for item in synthesis if item.certified_counterexample()]
        result = StatisticalCEGIS(_grammar_for(family)).synthesize(
            positive=positives,
            counterexamples=counterexamples,
            parent_predicate=None,
            decision_contexts=[item.context for item in evaluation],
        ) if positives and counterexamples else None
        predictions = {item.observation_id: bool(result and result.predicate and match_predicate(result.predicate, item.context)) for item in evaluation}
        return (lambda context: Decision("reuse" if predictions.get(_context_key(context), False) else "abstain", 1.0, 0.0, "acre_certificate")), result.to_dict() if result else None
    by_context = {_context_key(item.context): bool(predictions.get(item.observation_id, False)) for item in evaluation}
    return lambda context: Decision("reuse" if by_context.get(_context_key(context), False) else "abstain", 1.0, 0.0, name), None


def run(*, surface_count: int, generator_seeds: tuple[int, ...], repetitions: int, noise_scale: float, families: tuple[str, ...]) -> dict:
    rows = []
    status_counts = Counter()
    for family in families:
        for seed in generator_seeds:
            split = _split_cases(family, seed, surface_count)
            converted = {}
            for split_name, cases in split.items():
                converted[split_name] = [
                    repeated_observation(
                        observation_id=item.observation_id,
                        context=item.context,
                        effect=item.effect,
                        gate_passed=item.gate_passed,
                        seed=seed,
                        repetitions=repetitions,
                        noise_scale=noise_scale,
                        delta=0.05,
                    ) for item in cases
                ]
                status_counts.update(classify_observation(item, 0.0) for item in converted[split_name])
            synthesis, calibration, evaluation = converted["synthesis"], converted["calibration"], converted["evaluation"]
            eval_rows = [{"id": item.observation_id, "family": family, "seed": seed, "context": dict(item.context)} for item in evaluation]
            cal_rows = [{"id": item.observation_id, "context": dict(item.context), "label": item.positive_anchor()} for item in calibration]
            score_records = {item.observation_id: {"tau": item.effect, "scientific_gate": item.gate_passed, "ci_width": item.effect_upper - item.effect_lower} for item in evaluation}
            for policy_name in ("global_reuse", "positive_only", "nearest_experience", "matched_positive", "cart_ltt", "catboost_ltt", "acre_full"):
                policy, receipt = _policy_predictions(policy_name, family, synthesis, calibration, evaluation)
                metrics = evaluate_policy(policy, cal_rows, eval_rows, score_records)
                predictions = {entry["id"]: entry["action"] == "reuse" for entry in metrics["decisions"]}
                truth = {item.observation_id: item.positive_anchor() for item in evaluation}
                tp = sum(int(predictions.get(k, False) and v) for k, v in truth.items())
                fp = sum(int(predictions.get(k, False) and not v) for k, v in truth.items())
                fn = sum(int(not predictions.get(k, False) and v) for k, v in truth.items())
                deployed = sum(predictions.values())
                rows.append({
                    "family": family, "seed": seed, "policy": policy_name,
                    "evaluation_n": len(evaluation), "coverage": deployed / len(evaluation) if evaluation else 0.0,
                    "unsupported_deployed": fp, "unsupported_all": fp,
                    "support_precision": tp / (tp + fp) if tp + fp else None,
                    "support_recall": tp / (tp + fn) if tp + fn else None,
                    "identified": sum(classify_observation(item, 0.0) != "unresolved" for item in synthesis),
                    "underidentified": sum(classify_observation(item, 0.0) == "unresolved" for item in synthesis),
                    "predicate_description_length": predicate_complexity((receipt or {}).get("predicate"))["description_length"] if isinstance(receipt, dict) and receipt.get("predicate") else None,
                    "certificate": receipt,
                    "repetitions": repetitions,
                    "noise_scale": noise_scale,
                })
    return {
        "schema_version": 1,
        "analysis": "provider_free_non_degenerate_support_frontier_v1",
        "provider_calls": 0,
        "families": list(families),
        "generator_seeds": list(generator_seeds),
        "surface_count": surface_count,
        "repetitions": repetitions,
        "noise_scale": noise_scale,
        "confidence_delta": 0.05,
        "status_counts": dict(status_counts),
        "rows": rows,
        "hidden_truth_usage": "scoring_only_after_decisions",
        "claim_boundary": "provider-free repeated-measurement support frontier; measurement noise is a registered replay profile, not hosted agent efficacy",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-count", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(surface_count=args.surface_count, generator_seeds=(0, 1, 2, 3, 4), repetitions=args.repetitions, noise_scale=args.noise_scale, families=tuple(PILOT_FAMILIES))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(report["rows"]), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

