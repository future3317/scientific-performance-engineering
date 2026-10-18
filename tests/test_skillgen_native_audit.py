from __future__ import annotations

from scripts.run_skillgen_native_audit import (
    area_under_progress_curve,
    paired_summary,
)


def test_area_under_progress_curve_is_normalized() -> None:
    assert area_under_progress_curve([0.0, 0.5, 1.0]) == 0.5
    assert area_under_progress_curve([]) == 0.0


def test_paired_summary_reports_effect_and_endpoint_counts() -> None:
    summary = paired_summary([
        {"baseline_score": 10.0, "skill_score": 20.0, "baseline_success": False, "skill_success": False},
        {"baseline_score": 50.0, "skill_score": 40.0, "baseline_success": False, "skill_success": True},
        {"baseline_score": 30.0, "skill_score": 30.0, "baseline_success": False, "skill_success": False},
    ])
    assert summary == {
        "n": 3,
        "mean_paired_score_delta": 0.0,
        "median_paired_score_delta": 0.0,
        "positive_score_deltas": 1,
        "negative_score_deltas": 1,
        "ties": 1,
        "completion_repairs": 1,
        "completion_regressions": 0,
    }
