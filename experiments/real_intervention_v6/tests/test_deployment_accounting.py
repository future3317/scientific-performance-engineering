import pytest
from pathlib import Path

from experiments.real_intervention_v5.common import read_json, write_json
from experiments.real_intervention_v6 import study


def test_actual_R_jobs_and_publication_cost_are_not_replaced_by_steady(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    write_json(root / "FROZEN_MANIFEST.json", {"source_sha": "fixture", "versions": {"torch": "fixture"}})
    row = {"family_id": "family", "split": "development", "shape": [3, 8, 8], "checks": [{"batch": 1, "payload": {
        "weights_sha256": "w", "inputs_sha256": "i", "reference_sha256": "r"}}]}
    cell = {"family_id": "family", "split": "development", "horizon": 128, "session": 0, "order": "AB", "pair_id": "family-h128-s0", "features": {}}
    config = {"study_id": "V6_FIXTURE", "study": "B", "precision": "BF16", "construction_seed": 1,
              "correctness": {}, "arm_timeout_s": 60, "idle_wait_s": 0, "actual_deployment_jobs": 2,
              "reported_actual_reuse_counts": [1, 2], "projected_reuse_counts": [100]}
    launched = []
    def fake_arm(spec, folder, **kwargs):
        arm = spec["arm"]
        launched.append(arm)
        if arm == "PREFILL":
            write_json(Path(spec["cache_path"]) / "inductor/entry.json", {"test": "compiler-artifact"})
        return {"primary_pair_status": "EXPOSURE_UNCERTAIN", "parent_wall_s": {"PREFILL": 5., "EAGER": 2., "RELOAD": 1., "COLD": 9.}[arm],
                "timer_crosscheck": {"passed": True}, "timing": {"in_worker_job_s": .5, "steady_per_call_s": .002}}
    monkeypatch.setattr(study, "run_arm", fake_arm)
    result = study.measure(root, cell, row, config, {"uuid": "GPU-FIXTURE"})
    by_contrast = {r["contrast"]: r for r in result["records"]}
    publication = read_json(root / "pairs/development/family-h128-s0/PREFILL_COST.json")
    full_prefill = publication["full_prefill_job_s"]
    assert full_prefill >= 5
    assert by_contrast["PERSISTENT_R1"]["s0_s"] == 2
    assert by_contrast["PERSISTENT_R1"]["s1_s"] == pytest.approx(full_prefill + 1)
    assert by_contrast["PERSISTENT_R2"]["s0_s"] == 4
    assert by_contrast["PERSISTENT_R2"]["s1_s"] == pytest.approx(full_prefill + 2)
    assert by_contrast["PROJECTED_R100"]["s1_s"] == pytest.approx(full_prefill + 100)
    assert by_contrast["PROJECTED_R100"]["row_kind"] == "PROJECTION"
    assert launched.count("PREFILL") == 1
    assert launched.count("RELOAD") == launched.count("EAGER") == 2
    assert launched.count("COLD") == 1
    assert launched[:4] == ["PREFILL", "EAGER", "COLD", "RELOAD"]
    study.measure(root, cell, row, config, {"uuid": "GPU-FIXTURE"})
    assert len(launched) == 6  # closed identities are loaded, never repeated
