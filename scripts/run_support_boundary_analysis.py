#!/usr/bin/env python3
"""Run provider-free selective-risk and counterexample-dose analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.baselines import predict_applicability
from benchmark.boundary.families import family_cases
from benchmark.boundary.ltt import LTTConfig, LTTLogisticPolicy
from benchmark.boundary.policy_eval import Decision, evaluate_policy
from benchmark.boundary.rscb import RSCBConfig, RSCBModel
from benchmark.boundary.typed_features import TypedFeatureSchema
from benchmark.families import PILOT_FAMILIES
from core.acre.cegis import StatisticalCEGIS
from core.acre.predicates import PredicateGrammar, predicate_complexity
from core.predicates import match_predicate


def _bucket(family: str, seed: int, observation_id: str) -> int:
    token = f"{family}|{seed}|{observation_id}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % 4


def _grammar_for(family: str) -> PredicateGrammar:
    from benchmark.families import family_predicate_grammar

    return PredicateGrammar.from_dict(family_predicate_grammar(family))


def _context_key(context: Mapping[str, Any]) -> str:
    return json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)


def _split_cases(family: str, seed: int, surface_count: int) -> dict[str, list[Any]]:
    pools = family_cases(family, surface_count=surface_count, seed=seed)
    all_cases = [item for pool in pools.values() for item in pool]
    unique: dict[str, Any] = {item.observation_id: item for item in all_cases}
    synthesis = [item for item in unique.values() if _bucket(family, seed, item.observation_id) in (0, 1)]
    calibration = [item for item in unique.values() if _bucket(family, seed, item.observation_id) == 2]
    evaluation = [item for item in unique.values() if _bucket(family, seed, item.observation_id) == 3]
    return {"synthesis": synthesis, "calibration": calibration, "evaluation": evaluation}


def _prediction_policy(predictions: Mapping[str, bool], evaluation_cases: Sequence[Any]):
    by_context = {_context_key(item.context): bool(predictions.get(item.observation_id, False)) for item in evaluation_cases}

    def decide(context: Mapping[str, Any]) -> Decision:
        admitted = by_context.get(_context_key(context), False)
        return Decision("reuse" if admitted else "abstain", 1.0 if admitted else 0.0, 0.0, "frozen_prediction")

    return decide


def _support_metrics(predictions: Mapping[str, bool], evaluation_cases: Sequence[Any]) -> dict[str, Any]:
    truth = {item.observation_id: bool(item.positive_anchor()) for item in evaluation_cases}
    tp = sum(int(predictions.get(key, False) and value) for key, value in truth.items())
    fp = sum(int(predictions.get(key, False) and not value) for key, value in truth.items())
    fn = sum(int(not predictions.get(key, False) and value) for key, value in truth.items())
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "support_tp": tp,
        "support_fp": fp,
        "support_fn": fn,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
        "support_iou": tp / union if union else None,
        "oracle_supported": tp + fn,
    }


def _acre_predictions(family: str, synthesis: Sequence[Any], evaluation: Sequence[Any], dose: float) -> tuple[dict[str, bool], dict[str, Any] | None]:
    positives = [item for item in synthesis if item.positive_anchor()]
    counterexamples = [item for item in synthesis if item.certified_counterexample()]
    if not positives:
        return {item.observation_id: False for item in evaluation}, None
    if dose <= 0.0:
        predictions = predict_applicability("shortest_without_counterexamples", synthesis, evaluation, family=family)
        return predictions, None
    count = min(len(counterexamples), int(round(len(counterexamples) * dose)))
    if count == 0:
        return {item.observation_id: False for item in evaluation}, None
    result = StatisticalCEGIS(_grammar_for(family)).synthesize(
        positive=positives,
        counterexamples=counterexamples[:count],
        parent_predicate=None,
        decision_contexts=[item.context for item in evaluation],
    )
    predictions = {
        item.observation_id: bool(result.predicate and match_predicate(result.predicate, item.context))
        for item in evaluation
    }
    return predictions, result.to_dict()


def _add_row(
    rows: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
    name: str,
    policy: Any,
    calibration: Sequence[Any],
    evaluation: Sequence[Any],
    predicate: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
) -> None:
    eval_rows = [{"id": item.observation_id, "family": family, "seed": seed, "context": dict(item.context)} for item in evaluation]
    calibration_rows = [{"id": item.observation_id, "context": dict(item.context), "label": item.positive_anchor()} for item in calibration]
    scores = {
        item.observation_id: {
            "tau": item.effect,
            "scientific_gate": item.gate_passed,
            "ci_width": item.effect_upper - item.effect_lower,
        }
        for item in evaluation
    }
    metrics = evaluate_policy(policy, calibration_rows, eval_rows, scores)
    predictions = {entry["id"]: entry["action"] == "reuse" for entry in metrics["decisions"]}
    support = _support_metrics(predictions, evaluation)
    complexity = predicate_complexity(predicate)["description_length"] if predicate else None
    rows.append({
        "family": family,
        "seed": seed,
        "policy": name,
        "evaluation_ids": [item.observation_id for item in evaluation],
        "predicate_description_length": complexity,
        "freeze": dict(freeze or {}),
        **metrics,
        **support,
    })


def run_support_boundary_analysis(
    *,
    surface_count: int = 64,
    generator_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    families: Sequence[str] | None = None,
    doses: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
) -> dict[str, Any]:
    selected_families = tuple(families or PILOT_FAMILIES)
    rows: list[dict[str, Any]] = []
    for family in selected_families:
        for seed in generator_seeds:
            split = _split_cases(family, int(seed), surface_count)
            synthesis = split["synthesis"]
            calibration = split["calibration"]
            evaluation = split["evaluation"]
            contexts = [item.context for item in synthesis]
            schema = TypedFeatureSchema.from_calibration(family, contexts)
            synthesis_rows = [{"context": dict(item.context), "label": item.positive_anchor()} for item in synthesis]
            calibration_rows = [{"context": dict(item.context), "unsupported": not item.positive_anchor()} for item in calibration]
            ltt = LTTLogisticPolicy.fit(synthesis_rows, calibration_rows, schema, LTTConfig.default())
            rscb = RSCBModel.fit(synthesis_rows, schema, RSCBConfig.default())
            _add_row(rows, family=family, seed=int(seed), name="ltt_logistic", policy=ltt, calibration=calibration, evaluation=evaluation,
                     freeze={"model": ltt.to_dict(), "evaluation_used_for_fit": False})
            _add_row(rows, family=family, seed=int(seed), name="rscb_default", policy=rscb, calibration=calibration, evaluation=evaluation,
                     freeze={"model": rscb.to_dict(), "evaluation_used_for_fit": False})
            for dose in doses:
                predictions, synthesis_result = _acre_predictions(family, synthesis, evaluation, float(dose))
                _add_row(rows, family=family, seed=int(seed), name=f"acre_dose_{int(round(float(dose) * 100))}",
                         policy=_prediction_policy(predictions, evaluation), calibration=calibration, evaluation=evaluation,
                         predicate=(synthesis_result or {}).get("predicate"),
                         freeze={"counterexample_dose": float(dose), "synthesis_result": synthesis_result, "evaluation_used_for_fit": False})
    return {
        "schema_version": 1,
        "analysis": "provider_free_context_indexed_support_boundary_p0",
        "provider_calls": 0,
        "surface_count": surface_count,
        "generator_seeds": [int(seed) for seed in generator_seeds],
        "families": list(selected_families),
        "doses": [float(dose) for dose in doses],
        "split": "deterministic sha256 observation split: 50% synthesis, 25% calibration, 25% evaluation",
        "hidden_truth_usage": "offline scoring only",
        "rows": rows,
        "claim_boundary": "provider-free support-boundary and selective-risk comparison; no hosted efficacy claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "support_boundary_analysis.json")
    args = parser.parse_args()
    report = run_support_boundary_analysis(surface_count=args.surface_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(report["rows"]), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
