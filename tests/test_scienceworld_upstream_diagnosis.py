from __future__ import annotations

import json

from scripts.diagnose_scienceworld_train_formal import (
    build_frozen_diagnosis,
    completion_transition,
)


def test_completion_transition_reports_all_four_states() -> None:
    assert completion_transition(False, False) == "fail->fail"
    assert completion_transition(False, True) == "fail->success"
    assert completion_transition(True, False) == "success->fail"
    assert completion_transition(True, True) == "success->success"


def test_frozen_diagnosis_preserves_missing_trajectory_fields(tmp_path) -> None:
    task_manifest = {
        "verification": [
            {
                "task_id": "scienceworld:boil:7",
                "task_name": "boil",
                "task_family": "Chemistry",
                "variation_idx": 7,
            }
        ]
    }
    (tmp_path / "candidate_seed_0.skill.txt").write_text("skill\n", encoding="utf-8")
    (tmp_path / "verification_seed_0.json").write_text(
        json.dumps({
            "skill_hash": "abc",
            "outcome": "WITHHOLD_SKILL",
            "pair_count": 1,
            "execution_valid_count": 1,
            "skill_success_count": 0,
            "failure_reasons": [],
        }),
        encoding="utf-8",
    )

    rows, summary = build_frozen_diagnosis(task_manifest, tmp_path, generation_seeds=(0,))

    assert len(rows) == 1
    assert rows[0]["task_id"] == "scienceworld:boil:7"
    assert rows[0]["execution_valid"] is True
    assert rows[0]["skill_full_success"] is False
    assert rows[0]["baseline_score"] is None
    assert rows[0]["skill_score"] is None
    assert rows[0]["data_status"] == "trajectory_not_persisted"
    assert summary["known_pair_count"] == 1
    assert summary["unreconstructable_pair_fields"]
