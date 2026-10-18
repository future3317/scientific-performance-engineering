from __future__ import annotations

import json

from scripts.run_support_boundary_sensitivity import (
    classify_boundary_observation,
    run_support_boundary_sensitivity,
    sensitivity_grid,
)
from scripts.summarize_support_boundary_sensitivity import build_summary


def test_boundary_classification_matches_registered_three_way_rules() -> None:
    assert classify_boundary_observation(0.11, 0.11, True, 0.10, 0.10) == "positive_anchor"
    assert classify_boundary_observation(0.09, 0.10, True, 0.10, 0.10) == "counterexample"
    assert classify_boundary_observation(0.09, 0.11, True, 0.10, 0.10) == "unresolved"
    assert classify_boundary_observation(0.50, 0.60, False, 0.10, 0.10) == "counterexample"


def test_sensitivity_grid_is_fixed_and_records_unavailable_dimensions() -> None:
    grid = sensitivity_grid()
    assert grid["confidence_level"] == (0.90, 0.95, 0.99)
    assert grid["repetition_fraction"] == (0.50, 0.75, 1.00)
    assert grid["delta_multiplier"] == (0.5, 1.0, 1.5)
    assert grid["utility_scale_multiplier"] == (0.5, 1.0, 2.0)
    assert grid["grammar_depth_offset"] == (-1, 0, 1)
    assert grid["grammar_literal_offset"] == (-1, 0, 1)


def test_small_sensitivity_report_has_common_grid_metric_deltas_and_summary(tmp_path) -> None:
    report = run_support_boundary_sensitivity(
        surface_count=24,
        generator_seeds=(0, 1),
        families=("compile",),
        doses=(1.0,),
    )
    assert report["provider_calls"] == 0
    assert report["hidden_truth_usage"] == "scoring_only"
    assert report["availability"]["repetition_fraction"]["status"] == "not_applicable"
    assert report["availability"]["utility_scale"]["status"] == "not_applicable"
    assert report["availability"]["delta"]["registered_value"] == 0.0
    registry = report["configuration_registry"]
    assert any(item["configuration_id"] == "default" for item in registry)
    assert all("confidence_level" in item and "grammar" in item for item in registry)
    metric_row = report["metric_rows"][0]
    for key in (
        "coverage",
        "unsupported_deployed",
        "support_precision",
        "support_recall",
        "support_decision_regret_mean",
        "delta_coverage",
        "delta_unsupported_deployed",
        "delta_recall",
    ):
        assert key in metric_row
    assert report["stability"]["canonical_context_count"] > 0
    assert report["stability"]["family_seed_pair_rows"]
    pair_rows = report["stability"]["family_seed_pair_rows"]
    assert all(row["family"] == "compile" for row in pair_rows)
    assert all("jaccard" in row for row in pair_rows)
    assert report["stability"]["across_family_summary"]["family_count"] == 1
    artifact = tmp_path / "sensitivity.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    assert build_summary(artifact)["grid_accounting"] == report["grid_accounting"]
