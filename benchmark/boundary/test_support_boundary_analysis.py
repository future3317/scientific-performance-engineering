from __future__ import annotations

from scripts.run_support_boundary_analysis import run_support_boundary_analysis
from scripts.run_support_boundary_analysis import _support_metrics


def test_counterexample_doses_keep_the_same_evaluation_contexts() -> None:
    report = run_support_boundary_analysis(
        surface_count=24,
        generator_seeds=(0,),
        families=("compile",),
        doses=(0.0, 1.0),
    )
    rows = [row for row in report["rows"] if row["policy"].startswith("acre_dose_")]
    assert len(rows) == 2
    assert all(row["evaluation_ids"] for row in rows)
    assert len({tuple(row["evaluation_ids"]) for row in rows}) == 1
    assert report["provider_calls"] == 0


def test_support_metrics_expose_raw_supported_counts_and_boundary_scores() -> None:
    report = run_support_boundary_analysis(
        surface_count=24,
        generator_seeds=(0,),
        families=("compile",),
        doses=(0.0,),
    )
    row = next(item for item in report["rows"] if item["policy"] == "acre_dose_0")
    for key in ("support_tp", "support_fp", "support_fn", "support_precision", "support_recall", "support_iou", "predicate_description_length"):
        assert key in row
    assert row["support_tp"] + row["support_fn"] == row["oracle_supported"]


def test_explicit_scoring_truth_preserves_support_denominators() -> None:
    metrics = _support_metrics({"a": True, "b": False}, [], truth={"a": False, "b": True})
    assert metrics["support_tp"] == 0
    assert metrics["support_fp"] == metrics["support_fn"] == 1
    assert metrics["oracle_supported"] == 1
