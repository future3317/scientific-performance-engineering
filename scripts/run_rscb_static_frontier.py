#!/usr/bin/env python3
"""Run the frozen provider-free support-policy frontier.

This driver uses the existing family evidence/calibration/evaluation split and
never exposes evaluation score fields to a policy.  It does not call a
provider and is not a hosted or treatment-efficacy experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.baselines import predict_applicability
from benchmark.boundary.families import family_cases
from benchmark.boundary.policy_eval import Decision, clustered_frontier, evaluate_policy
from benchmark.boundary.rscb import RSCBConfig, RSCBModel, matched_threshold
from benchmark.boundary.static_gate import StaticTestGate
from benchmark.families import PILOT_FAMILIES


def _key(context: Mapping[str, Any]) -> str:
    return json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)


def _prediction_policy(predictions: Mapping[str, bool], contexts: Sequence[Mapping[str, Any]]):
    by_context = {_key(context): bool(prediction) for context, prediction in zip(contexts, predictions.values())}
    # ``predictions`` is built in evaluation order by this driver.  The map is
    # only used as a fixed lookup during offline evaluation.
    def decide(context: Mapping[str, Any]) -> Decision:
        admitted = by_context.get(_key(context), False)
        return Decision("reuse" if admitted else "abstain", 1.0 if admitted else 0.0, 0.0, "frozen_prediction")
    return decide


def _rows_for_family(family: str, surface_count: int, seed: int) -> dict[str, Any]:
    pools = family_cases(family, surface_count=surface_count, seed=seed)
    evidence = list(pools["representative_pool"]) + list(pools["active_query_pool"])
    sealed = list(pools["sealed_boundary_pool"])
    calibration_cases = sealed[::2]
    evaluation_cases = sealed[1::2]
    calibration = [{"id": item.observation_id, "context": dict(item.context), "label": item.positive_anchor()} for item in calibration_cases]
    evaluation = [
        {"id": item.observation_id, "family": family, "seed": seed, "context": dict(item.context)}
        for item in evaluation_cases
    ]
    score_records = {
        item.observation_id: {"tau": item.effect, "scientific_gate": item.gate_passed}
        for item in evaluation_cases
    }
    return {
        "evidence": evidence,
        "calibration": calibration,
        "calibration_cases": calibration_cases,
        "evaluation": evaluation,
        "score_records": score_records,
    }


def _evaluate_policy_set(family: str, split: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = split["evidence"]
    calibration = split["calibration"]
    evaluation = split["evaluation"]
    scores = split["score_records"]
    eval_contexts = [row["context"] for row in evaluation]
    calibration_contexts = [row["context"] for row in calibration]
    rows: list[dict[str, Any]] = []

    def add(name: str, policy: Any, freeze: Mapping[str, Any]) -> None:
        metrics = evaluate_policy(policy, calibration, evaluation, scores)
        rows.append({"family": family, "seed": evaluation[0]["seed"] if evaluation else None, "policy": name, **metrics, "freeze": dict(freeze)})

    for name in ("always_apply", "observed_positive_only", "nearest_experience", "shortest_without_counterexamples", "acre_full"):
        predictions = predict_applicability(name, evidence, evaluation_cases_from_rows(evaluation), family=family)
        add(name, _prediction_policy(predictions, eval_contexts), {"source": "existing_baseline", "evaluation_used_for_fit": False})

    static_gate = StaticTestGate.from_calibration_suite([
        {"test_id": "public_contract", "passed": True, "category": "contract"},
        {"test_id": "scientific_regression", "passed": True, "category": "correctness"},
    ])
    add("static_test_gate", static_gate, {"suite": static_gate.to_dict(), "evaluation_used_for_fit": False})

    schema = __import__("benchmark.boundary.typed_features", fromlist=["TypedFeatureSchema"]).TypedFeatureSchema.from_calibration(
        family, calibration_contexts
    )
    model = RSCBModel.fit(calibration, schema, RSCBConfig.default())
    add("rscb_default", model, {"model": model.to_dict(), "evaluation_used_for_fit": False})
    target_coverage = 0.0
    if calibration:
        full_calibration = predict_applicability("acre_full", evidence, split["calibration_cases"], family=family)
        target_coverage = sum(full_calibration.values()) / len(full_calibration) if full_calibration else 0.0
    theta = matched_threshold(model, calibration_contexts, target_coverage)
    matched = model.with_threshold(theta)
    add("rscb_matched", matched, {"theta": theta, "target_calibration_coverage": target_coverage, "evaluation_used_for_fit": False})

    for k in (0.0, 0.67448975, 1.28155157, 1.64485363, 2.32634787):
        for theta_value in RSCBConfig.default().threshold_grid:
            variant = RSCBModel(
                schema=model.schema,
                config=RSCBConfig(k=k, theta=theta_value, max_iterations=model.config.max_iterations, tolerance=model.config.tolerance, threshold_grid=model.config.threshold_grid),
                coefficients=model.coefficients,
                covariance=model.covariance,
                iterations=model.iterations,
            )
            add(f"rscb_frontier_k={k:.8g}_theta={theta_value:.2f}", variant, {"k": k, "theta": theta_value, "evaluation_used_for_fit": False})
    return rows, {"schema": schema.to_dict(), "rscb_model": model.to_dict(), "matched_theta": theta, "target_calibration_coverage": target_coverage}


def evaluation_cases_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Reconstruct the baseline observation view without exposing score fields."""
    from benchmark.boundary.families import BoundaryCase
    return [BoundaryCase(str(row["id"]), row["context"], 0.0, True, 0.0, 0.0) for row in rows]


def run_frozen_frontier(
    *,
    surface_count: int = 24,
    seed: int = 7,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_families = tuple(families or PILOT_FAMILIES)
    all_rows: list[dict[str, Any]] = []
    family_freezes: dict[str, Any] = {}
    for family in selected_families:
        split = _rows_for_family(family, surface_count, seed)
        rows, freeze = _evaluate_policy_set(family, split)
        all_rows.extend(rows)
        family_freezes[family] = freeze
    policy_names = sorted({row["policy"] for row in all_rows})
    summaries = []
    for policy in policy_names:
        selected = [row for row in all_rows if row["policy"] == policy]
        aggregate = clustered_frontier(selected, cluster_keys=("family", "seed"))
        summaries.append({"policy": policy, **aggregate})
    return {
        "schema_version": 1,
        "analysis": "provider_free_rscb_static_support_frontier",
        "provider_calls": 0,
        "surface_count": surface_count,
        "seed": seed,
        "families": list(selected_families),
        "split": "existing family evidence; sealed pool split into calibration/evaluation by fixed alternating order",
        "parameter_freeze": {
            "evaluation_used_for_threshold_selection": False,
            "rscb_penalty_grid": [0.0, 0.67448975, 1.28155157, 1.64485363, 2.32634787],
            "rscb_threshold_grid": list(RSCBConfig.default().threshold_grid),
            "cluster_units": ["family", "seed"],
        },
        "family_freezes": family_freezes,
        "rows": all_rows,
        "summaries": summaries,
        "claim_boundary": "provider-free mechanism and support-decision comparison; no hosted efficacy claim",
        "environment": {"python": sys.executable, "platform": platform.platform()},
    }


def write_frontier_artifacts(output_root: Path, report: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rscb_static_frontier.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (output_root / "rscb_static_frontier_manifest.json").write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report = run_frozen_frontier(surface_count=args.surface_count, seed=args.seed)
    manifest = {"schema_version": 1, "provider_calls": 0, "analysis": report["analysis"], "split": report["split"], "claim_boundary": report["claim_boundary"]}
    write_frontier_artifacts(args.out_root, report, manifest)
    print(json.dumps({"status": "complete", "policies": len(report["summaries"]), "out_root": str(args.out_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
