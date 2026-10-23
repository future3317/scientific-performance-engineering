import pytest

from experiments.real_intervention_v5.common import read_json, write_json
from experiments.real_intervention_v6 import run_all


def test_resume_after_policy_freeze_reuses_identical_release_and_predictions(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    root.mkdir()
    config = {"study_id": "V6_FIXTURE", "study": "A", "wall_cap_s": 1000, "disk_cap_bytes": 10**7,
              "minimum_observed_dev_families": 2, "minimum_observed_cal_families": 2, "minimum_registered_final_families": 2}
    plan = [{"family_id": f"{split}{i}", "split": split, "pair_id": f"{split}{i}-h128-s0", "horizon": 128,
             "features": {"parameters": 100, "conv_fraction": .5}}
            for split in ("development", "calibration", "final") for i in range(2)]
    roster = {p["family_id"]: {} for p in plan}
    manifest = {"source_sha": "fixture", "study_id": "V6_FIXTURE", "gpu": {"uuid": "GPU-FIXTURE"}, "cohort": {"split_counts": {"final": 2}}}
    def fake_freeze(*args):
        write_json(root / "FROZEN_MANIFEST.json", manifest)
        return manifest, roster, plan
    monkeypatch.setattr(run_all, "preflight", lambda: "fixture")
    monkeypatch.setattr(run_all, "idle_device", lambda: ({"uuid": "GPU-FIXTURE"}, []))
    monkeypatch.setattr(run_all, "audit", lambda root: write_json(root / "V6_ROOT_CAUSE_REGISTER.json", {}))
    monkeypatch.setattr(run_all, "platform_validation", lambda *args: None)
    monkeypatch.setattr(run_all, "engineering_validation", lambda *args: None)
    monkeypatch.setattr(run_all, "freeze", fake_freeze)
    state = {"crash": True}
    def fake_measure(root, cell, *args, **kwargs):
        if cell["split"] == "final":
            assert (root / "POLICY_FREEZE.json").exists()
            assert (root / "FINAL_RELEASE.json").exists()
            if state["crash"]:
                raise KeyboardInterrupt("fixture controller crash")
        return {"records": [{**cell, "contrast": "LAYOUT", "reuse_count": 1,
            "context_id": cell["family_id"] + "-h128-LAYOUT", "analytically_comparable": True,
            "primary_pair_status": "EXPOSURE_UNCERTAIN", "s0_s": 2., "s1_s": 3., "log_benefit": -.405}]}
    monkeypatch.setattr(run_all, "measure", fake_measure)
    with pytest.raises(KeyboardInterrupt):
        run_all.run(root, config)
    policy_bytes = (root / "POLICY_FREEZE.json").read_bytes()
    release_bytes = (root / "FINAL_RELEASE.json").read_bytes()
    state["crash"] = False
    def forbid_refit(*args):
        pytest.fail("policy may not refit or remeasure routing overhead after prefinal freeze")
    monkeypatch.setattr(run_all, "freeze_policies", forbid_refit)
    result = run_all.run(root, config)
    assert result["status"] == "PILOT_COMPLETE"
    assert (root / "POLICY_FREEZE.json").read_bytes() == policy_bytes
    assert (root / "FINAL_RELEASE.json").read_bytes() == release_bytes
