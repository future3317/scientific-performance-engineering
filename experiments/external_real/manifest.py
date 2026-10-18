from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_GENERATION_SEEDS = (0, 1, 2)
DEFAULT_EVIDENCE_BUDGETS = (0, 8, 16, 32, 64)
DEFAULT_FEATURE_FIELDS = (
    "task_family",
    "goal_operation",
    "object_category",
    "material_category",
    "target_state_change",
    "requires_measurement",
    "requires_container",
    "variation_bucket",
)


def build_protocol_manifests(
    *,
    benchmark: str,
    benchmark_revision: str,
    protocol_revision: str,
    model_id: str,
    final_test_ids: Sequence[str],
    burned_evaluation_ids: Sequence[str],
    generation_seeds: Sequence[int] = DEFAULT_GENERATION_SEEDS,
    evidence_budgets: Sequence[int] = DEFAULT_EVIDENCE_BUDGETS,
    provider_calls_authorized: bool = False,
    code_commit: str | None = None,
    input_artifacts: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    final_ids = tuple(sorted({str(item) for item in final_test_ids}))
    burned_ids = tuple(sorted({str(item) for item in burned_evaluation_ids}))
    overlap = sorted(set(final_ids) & set(burned_ids))
    if overlap:
        raise ValueError(f"final and burned evaluation IDs overlap: {overlap}")
    if not generation_seeds:
        raise ValueError("at least one generation seed is required")
    if any(int(seed) < 0 for seed in generation_seeds):
        raise ValueError("generation seeds must be non-negative")
    if tuple(int(budget) for budget in evidence_budgets) != tuple(sorted(set(int(budget) for budget in evidence_budgets))):
        raise ValueError("evidence budgets must be unique and sorted")
    if any(int(budget) < 0 for budget in evidence_budgets):
        raise ValueError("evidence budgets must be non-negative")

    identity = {
        "benchmark": benchmark,
        "benchmark_revision": benchmark_revision,
        "protocol_revision": protocol_revision,
        "code_commit": code_commit,
        "input_artifacts": dict(input_artifacts or {}),
    }
    protocol = {
        **identity,
        "model_id": model_id,
        "generation_seeds": [int(seed) for seed in generation_seeds],
        "evidence_budgets": [int(budget) for budget in evidence_budgets],
        "split_policy": {
            "unit": "canonical_task_id",
            "construction": "official_train_only",
            "verification": "official_train_only",
            "synthesis": "official_train_only",
            "calibration": "official_train_only",
            "support": "train_side_only",
            "evaluation": "held_out_and_frozen_before_outcomes",
        },
        "feature_fields": list(DEFAULT_FEATURE_FIELDS),
        "final_test_ids": list(final_ids),
        "burned_evaluation_ids": list(burned_ids),
        "primary_metrics": [
            "coverage",
            "regression_rate_deployed",
            "repair_recall",
            "paired_score_gain",
            "incremental_policy_utility",
            "official_score",
            "fallback_rate",
            "additional_provider_cost",
        ],
        "provider_calls": 1 if provider_calls_authorized else 0,
    }
    return {
        "protocol_manifest": protocol,
        "burned_evaluation_ids": {**identity, "ids": list(burned_ids)},
        "benchmark_version_manifest": {
            **identity,
            "official_evaluator": "benchmark_native",
            "final_test_ids": list(final_ids),
            "burned_evaluation_ids": list(burned_ids),
        },
        "provider_model_manifest": {
            **identity,
            "model_id": model_id,
            "provider_calls_authorized": bool(provider_calls_authorized),
            "temperature": 0.0,
            "parser_revision": None,
        },
        "statistics_manifest": {
            **identity,
            "cluster_unit": "family_seed",
            "task_cluster_unit": "task_id",
            "bootstrap_replicates": 10000,
            "permutation_replicates": 100,
            "permutation_target": "counterexample_assignment_only",
            "confidence_level": 0.95,
        },
    }


def write_protocol_manifests(out_dir: Path, manifests: Mapping[str, Mapping[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "protocol_manifest",
        "burned_evaluation_ids",
        "benchmark_version_manifest",
        "provider_model_manifest",
        "statistics_manifest",
    ):
        if name not in manifests:
            raise ValueError(f"missing manifest: {name}")
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(dict(manifests[name]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
