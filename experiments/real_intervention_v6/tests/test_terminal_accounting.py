from pathlib import Path

import pytest

from experiments.real_intervention_v5.common import write_json, read_json, digest
from experiments.real_intervention_v6.execution_adapter import run_arm
from experiments.real_intervention_v6.run_all import incomplete_rows
from experiments.real_intervention_v6.report import finish


def test_partial_pair_with_started_prefill_is_unknown_not_unstarted(tmp_path):
    cell = {"pair_id": "family-h128-s0", "family_id": "family", "split": "development", "horizon": 128}
    config = {"study": "B", "reported_actual_reuse_counts": [1, 10], "projected_reuse_counts": [100]}
    folder = tmp_path / "pairs/development/family-h128-s0/PREFILL-0"
    write_json(folder / "intent.json", {})
    write_json(folder / "started.json", {"pid": 123})
    rows = incomplete_rows(tmp_path, cell, config, "publication failed")
    assert len(rows) == 4
    assert all(r["primary_pair_status"] == "UNKNOWN" for r in rows)
    assert all(r["known_started_arms"] == 1 for r in rows)


def test_resume_refuses_changed_arm_or_final_identity(tmp_path):
    spec = {"source_sha": "test", "final_identity": {"policy_sha256": "first"}, "timeout_s": 60}
    write_json(tmp_path / "terminal.json", {"spec_identity": digest({k: v for k, v in spec.items() if k != "timeout_s"}), "primary_pair_status": "VALID"})
    assert run_arm({**spec, "timeout_s": 30}, tmp_path)["primary_pair_status"] == "VALID"
    with pytest.raises(RuntimeError, match="identity differs"):
        run_arm({**spec, "final_identity": {"policy_sha256": "changed"}}, tmp_path)


def test_terminal_index_includes_orphan_and_does_not_claim_final_pilot(tmp_path):
    orphan = tmp_path / "pairs/development/a/EAGER-0"
    write_json(orphan / "intent.json", {})
    write_json(orphan / "started.json", {"pid": 123})
    write_json(tmp_path / "RUN_START.json", {"source_sha": "test", "config": {"study_id": "V6_TEST"}})
    row = {"pair_id": "a", "family_id": "a", "context_id": "a-h128", "split": "development", "contrast": "COLD",
           "primary_pair_status": "EXPOSURE_UNCERTAIN", "analytically_comparable": True, "s0_s": 2., "s1_s": 3., "log_benefit": -.405}
    result = finish(tmp_path, "ENGINEERING_OR_RESOURCE_NO_GO", [row], elapsed_s=1, blocker="fixture")
    assert result["definite_worker_starts"] == 1
    assert result["unresolved_intents"] == 1
    assert result["scientific_status"] == "DEVELOPMENT_CALIBRATION_DESCRIPTIVE_ONLY"
    assert read_json(tmp_path / "RAW_ATTEMPT_INDEX.json")[0]["status"] == "UNKNOWN"
    assert read_json(tmp_path / "BUNDLE_VALIDATION.json")["status"] == "PASS"
