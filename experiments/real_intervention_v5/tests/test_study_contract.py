from pathlib import Path
import tarfile

import pytest

from experiments.real_intervention_v5.common import digest, require_final_access, write_json, read_json
from experiments.real_intervention_v5.registry import assert_split_integrity
from experiments.real_intervention_v5.report import bundle, finish


def test_final_requires_frozen_matching_study_and_policy(tmp_path):
    spec = {"mode": "measurement", "split": "final", "study_id": "new-study", "final_release_path": str(tmp_path / "FINAL_RELEASE.json")}
    with pytest.raises(PermissionError): require_final_access(spec)
    write_json(tmp_path / "POLICY_FREEZE.json", {"predictions": []})
    write_json(tmp_path / "FROZEN_MANIFEST.json", {"study_id": "new-study"})
    release = {"study_id": "new-study", "policy_sha256": digest(read_json(tmp_path / "POLICY_FREEZE.json")), "manifest_sha256": digest(read_json(tmp_path / "FROZEN_MANIFEST.json"))}
    write_json(tmp_path / "FINAL_RELEASE.json", release)
    require_final_access(spec)
    write_json(tmp_path / "POLICY_FREEZE.json", {"predictions": ["changed"]})
    with pytest.raises(PermissionError): require_final_access(spec)
    require_final_access({**spec, "mode": "qualification"})


def test_same_lineage_cannot_enter_two_splits():
    with pytest.raises(ValueError):
        assert_split_integrity([{"family_id": "resnet", "split": "development"}, {"family_id": "resnet", "split": "final"}])


def test_export_hashes_sanitized_bytes_and_excludes_weights_cache(tmp_path):
    write_json(tmp_path / "receipt.json", {"root": "/home/workspace/lrh/RESULTS/new", "status": "UNKNOWN"})
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "secret.json").write_text("not exported")
    (tmp_path / "weights.pt").write_bytes(b"not exported")
    bundle(tmp_path)
    assert read_json(tmp_path / "BUNDLE_VALIDATION.json")["status"] == "PASS"
    with tarfile.open(tmp_path / "EVIDENCE_BUNDLE.tar.gz") as f:
        assert set(f.getnames()) == {"receipt.json", "MEMBER_SHA256.json"}
        assert b"/home/workspace/lrh" not in f.extractfile("receipt.json").read()


def test_platform_stop_reports_unknown_qualification_not_zero(tmp_path):
    from experiments.real_intervention_v5.certificate_feasibility import feasibility
    config = {"study_id": "new-study", "study": "A", "batches": [1, 8], "horizons": [1, 1024], "sessions_per_context": 2}
    manifest = {"config": config, "source_sha": "test", "gpu": None, "versions": {}}
    registry = [{"family_id": "resnet", "split": "final"}]
    finish(tmp_path, manifest, registry, [], [], "PLATFORM_BLOCKED_NO_IDLE_GPU", feasibility(0), {"required": 12, "actual": 0}, .01)
    metrics = read_json(tmp_path / "V5_METRICS.json")
    assert metrics["families"]["baseline_valid"] is None
    assert metrics["pairs"]["planned_eligible_sessions"] is None
    assert not metrics["final_opened"]


def test_compiler_cache_cannot_be_shared_between_scopes(tmp_path, monkeypatch):
    from experiments.real_intervention_v5.run_all import execute
    from experiments.real_intervention_v5 import runtime
    monkeypatch.setattr(runtime, "run_session", lambda *args: {"analysis_status": "VALID_PAIRED"})
    base = {"cache_path": str(tmp_path / "cache"), "study_id": "A", "split": "development", "family_id": "resnet", "batch": 1, "shape": [3, 224, 224], "source_version": "1"}
    config = {"precision": "BF16", "session_timeout_seconds": 30}
    execute([base], tmp_path / "one", config, {"uuid": "GPU-a"})
    with pytest.raises(ValueError, match="different experimental scope"):
        execute([{**base, "study_id": "B"}], tmp_path / "two", config, {"uuid": "GPU-a"})
