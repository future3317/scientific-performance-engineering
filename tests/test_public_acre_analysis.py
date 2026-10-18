from __future__ import annotations

import pytest

from benchmark.external.treatment_analysis import analyze_treatment_artifacts


@pytest.mark.parametrize("duplicate", [False, True])
def test_treatment_analysis_rejects_incomplete_or_duplicate_pairs(duplicate) -> None:
    row = {"instance_id": "x", "baseline_score": 0.0, "skill_score": 0.0}
    artifacts = {condition: {"rows": [dict(row)], "usage": []} for condition in ("A", "C", "D")}
    artifacts["D"]["rows"] = [dict(row), dict(row)] if duplicate else []
    with pytest.raises(ValueError, match="duplicate|must match"):
        analyze_treatment_artifacts(artifacts, bootstrap_seed=1, bootstrap_reps=20)


def test_treatment_analysis_separates_negative_transfer_and_paired_effect() -> None:
    artifacts = {
        "A": {"rows": [{"instance_id": "x", "baseline_score": 0.5, "skill_score": 0.5}], "usage": [], "skill_versions": []},
        "C": {"rows": [{"instance_id": "x", "baseline_score": 0.5, "skill_score": 0.4}], "usage": [], "skill_versions": []},
        "D": {"rows": [{"instance_id": "x", "baseline_score": 0.5, "skill_score": 0.7}], "usage": [], "skill_versions": [{"version_id": "v0"}]},
    }
    result = analyze_treatment_artifacts(artifacts, bootstrap_seed=1, bootstrap_reps=20)
    assert result["conditions"]["C"]["negative_transfer"] == 1
    assert result["conditions"]["D"]["negative_transfer"] == 0
    assert result["paired_D_minus_C"]["mean_delta"] == pytest.approx(0.3)
    assert result["paired_D_minus_C"]["negative_delta_count"] == 0
