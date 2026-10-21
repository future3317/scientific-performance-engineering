from pathlib import Path
import tarfile

import pytest

from experiments.real_intervention_v5.common import digest, require_final_access, write_json, read_json, file_digest
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
    registry = [{"candidate_id": "torchvision:resnet18", "family_id": "resnet", "split": "final"}]
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


def test_bf16_absolute_tolerance_cannot_hide_collapsed_near_zero_output():
    import torch
    from experiments.real_intervention_v5.worker import check
    reference = torch.tensor([.001], dtype=torch.bfloat16)
    collapsed = torch.zeros_like(reference)
    observed = check(reference, collapsed, {"rtol": .02, "atol": .002, "relative_l2_max": .02})
    assert not observed["passed"]
    assert observed["reason"] == "relative_l2_limit_exceeded"


def test_qualification_rejects_different_reconstructed_weights(tmp_path, monkeypatch):
    from experiments.real_intervention_v5 import run_all
    import time
    payload = {"baseline": {"passed": True}, "candidate": {"passed": True},
               "weights_sha256": "first", "inputs_sha256": "same", "reference_sha256": "same",
               "kernel_exposure": {"active": True, "cuda_kernel_events": 1}}
    monkeypatch.setattr(run_all, "execute", lambda *args: {"analysis_status": "VALID_PAIRED", "arms": [
        {"worker_result": payload.copy()}, {"worker_result": {**payload, "weights_sha256": "different"}}]})
    row = {"representative": True, "available": True, "family_id": "resnet", "split": "development"}
    config = {"study_id": "A", "study": "A", "seed": 1, "batches": [1], "correctness": {}, "session_timeout_seconds": 30}
    observed = run_all.qualify([row], config, {"uuid": "GPU-a"}, tmp_path, time.monotonic() + 30)
    assert not observed[0]["baseline_valid"]
    assert observed[0]["eligibility"] == "QUALIFICATION_FAILED"


def test_two_studies_export_together_without_pooling_effects(tmp_path):
    from experiments.real_intervention_v5.campaign import combine_results
    import json
    results = []
    for study in ("A", "B"):
        root = tmp_path / study
        root.mkdir()
        write_json(root / "V5_METRICS.json", {"study_id": study})
        (root / "V5_SUMMARY.md").write_text(study)
        (root / "PAIRS.jsonl").write_text(json.dumps({"pair_id": "one", "contrast": study}) + "\n")
        results.append({"root": str(root), "study_id": study})
    combine_results(tmp_path, "source", results)
    assert read_json(tmp_path / "V5_METRICS.json")["study_effects_never_pooled"]
    assert set(read_json(tmp_path / "V5_METRICS.json")["studies"]) == {"A", "B"}
    rows = [json.loads(line) for line in (tmp_path / "PAIRS.jsonl").read_text().splitlines()]
    assert [r["study_id"] for r in rows] == ["A", "B"]
    assert read_json(tmp_path / "BUNDLE_VALIDATION.json")["status"] == "PASS"


@pytest.mark.parametrize("study,horizons", [("A", [1,16,128,1024]), ("B", [128,1024,4096])])
def test_each_context_has_both_ab_and_ba(study, horizons):
    from experiments.real_intervention_v5.run_all import plan_pairs
    row = {"family_id": "one-lineage", "split": "development", "eligibility": "ELIGIBLE", "checks": [{"batch": 1, "payload": {"features": {"parameters": 1, "conv2d": 1}}}]}
    config = {"study_id": "new-study", "study": study, "seed": 20261021, "batches": [1], "horizons": horizons, "sessions_per_context": 2}
    plan = plan_pairs([row], config)
    for h in horizons:
        assert {p["order"] for p in plan if p["horizon"] == h} == {"AB", "BA"}
    assert plan_pairs([row], config) == plan
    assert len({p["family_id"] for p in plan}) == 1


def test_qualification_reuse_requires_identical_worker_and_original_bytes(tmp_path):
    from experiments.real_intervention_v5 import run_all
    old = tmp_path / "qualification" / "original.json"
    write_json(old, {"baseline": {"passed": True}, "candidate": {"passed": False}})
    registry, versions = [{"family_id": "one-lineage"}], {"torch": "frozen"}
    reuse = {"prior_source_sha": "old-source", "prior_study_id": "preparation", "worker_sha256": file_digest(Path(run_all.__file__).with_name("worker.py")),
             "registry_sha256": digest(registry), "versions": versions,
             "copied_files": {"qualification/original.json": {"bytes": old.stat().st_size, "sha256": file_digest(old)}}}
    write_json(tmp_path / "QUALIFICATION_REUSE.json", reuse)
    assert run_all.validate_qualification_reuse(tmp_path, registry, versions)["prior_source_sha"] == "old-source"
    with pytest.raises(RuntimeError, match="identical numerical worker"):
        run_all.validate_qualification_reuse(tmp_path, registry, {"torch": "changed"})
    write_json(old, {"candidate": {"passed": True}})
    with pytest.raises(RuntimeError, match="bytes changed"):
        run_all.validate_qualification_reuse(tmp_path, registry, versions)
