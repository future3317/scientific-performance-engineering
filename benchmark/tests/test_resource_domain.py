from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.formal.resource_domain import (
    MANIFEST_ID,
    build_resource_domain_manifest,
    classify_baseline_probe,
    validate_resource_domain_manifest,
    validate_runtime_contract,
)


def _probe(**overrides):
    probe = {
        "task_id": "T1",
        "task_package_digest": "a" * 64,
        "execution_class": "atomic_performance",
        "baseline_only": True,
        "clean_admission": True,
        "foreign_processes": [],
        "protocol_valid": True,
        "baseline_completed": True,
        "resource_exhausted": False,
        "total_device_memory": 24 * 1024**3,
        "preflight_free_memory": 24 * 1024**3,
        "peak_allocated": 512 * 1024**2,
        "peak_reserved": 512 * 1024**2,
        "reserved_unallocated": 0,
        "failed_allocation_bytes": None,
        "exception_class": None,
        "failure_signature": None,
        "confirmation_attempt_count": 0,
        "capacity_failure_reproduced": False,
        "cleanup_quiescent": True,
        "baseline_probe_session_id": "session-t1",
    }
    probe.update(overrides)
    return probe


def test_clean_baseline_is_in_domain():
    assert classify_baseline_probe(_probe()) == "in_domain"


def test_first_clean_capacity_oom_requires_confirmation():
    result = classify_baseline_probe(_probe(
        baseline_completed=False,
        resource_exhausted=True,
        peak_allocated=23 * 1024**3,
        peak_reserved=23 * 1024**3,
        preflight_free_memory=6 * 1024**2,
        reserved_unallocated=95 * 1024**2,
        failed_allocation_bytes=16 * 1024**2,
        exception_class="torch.OutOfMemoryError",
        failure_signature="cuda oom",
        confirmation_attempt_count=1,
    ))
    assert result == "unresolved"


def test_confirmed_clean_capacity_oom_is_out_of_domain():
    assert classify_baseline_probe(_probe(
        baseline_completed=False,
        resource_exhausted=True,
        peak_allocated=23 * 1024**3,
        peak_reserved=23 * 1024**3,
        preflight_free_memory=6 * 1024**2,
        reserved_unallocated=95 * 1024**2,
        failed_allocation_bytes=16 * 1024**2,
        exception_class="torch.OutOfMemoryError",
        failure_signature="cuda oom",
        confirmation_attempt_count=2,
        capacity_failure_reproduced=True,
    )) == "out_of_domain_capacity"


def test_repeated_clean_capacity_oom_does_not_require_ninety_percent_peak():
    assert classify_baseline_probe(_probe(
        baseline_completed=False,
        resource_exhausted=True,
        total_device_memory=20 * 1024**3,
        peak_allocated=15 * 1024**3,
        peak_reserved=15 * 1024**3,
        preflight_free_memory=6 * 1024**2,
        reserved_unallocated=95 * 1024**2,
        failed_allocation_bytes=16 * 1024**2,
        exception_class="torch.OutOfMemoryError",
        failure_signature="cuda oom",
        confirmation_attempt_count=2,
        capacity_failure_reproduced=True,
    )) == "out_of_domain_capacity"


def test_foreign_process_telemetry_does_not_change_resource_domain():
    assert classify_baseline_probe(_probe(foreign_processes=[1234])) == "in_domain"
    assert classify_baseline_probe(_probe(protocol_valid=False)) == "unresolved"


def test_scientific_failure_does_not_change_resource_domain():
    assert classify_baseline_probe(_probe(scientific_valid=False, correctness_pass=False)) == "in_domain"


def test_manifest_is_frozen_and_binds_runtime_and_population(tmp_path: Path):
    repo_root = Path(__file__).parents[2]
    active = json.loads((repo_root / "benchmark" / "pilot_population.json").read_text(encoding="utf-8"))
    probes = []
    for task_id in active["task_ids"]:
        task_dir = repo_root / "benchmark" / "tasks" / task_id
        probes.append(_probe(
            task_id=task_id,
            task_package_digest=__import__("benchmark.calibration.identity", fromlist=["task_package_digest"]).task_package_digest(task_dir),
            baseline_probe_session_id=f"session-{task_id}",
        ))
    runtime = {"cuda_visible_devices": "5", "gpu_uuid": "GPU-test", "gpu_name": "RTX 4090", "total_device_memory": 24564 * 1024**2}
    path = tmp_path / "resource_domain_eligibility.json"
    manifest = build_resource_domain_manifest(repo_root=repo_root, task_probes=probes, runtime_contract=runtime, out_path=path)
    assert manifest["manifest_id"] == MANIFEST_ID
    assert manifest["status"] == "frozen"
    assert manifest["admission_contract"]["foreign_processes_allowed"] is True
    assert len(manifest["task_ids"]) == 30
    assert len(manifest["in_domain_task_ids"]) == 30
    assert validate_resource_domain_manifest(manifest, repo_root=repo_root) == []
    assert json.loads(path.read_text(encoding="utf-8"))["artifact_digest"] == manifest["artifact_digest"]


@pytest.mark.parametrize("forbidden", ["agent_outcome", "oracle_effect", "candidate_result", "d_b_effect"])
def test_manifest_rejects_agent_oracle_candidate_fields(forbidden):
    with pytest.raises(ValueError, match="agent/outcome"):
        classify_baseline_probe(_probe(**{forbidden: {"value": 1}}))


def test_manifest_rejects_stale_runtime_and_task_identity(tmp_path: Path):
    repo_root = Path(__file__).parents[2]
    active = json.loads((repo_root / "benchmark" / "pilot_population.json").read_text(encoding="utf-8"))
    probes = []
    for task_id in active["task_ids"]:
        task_dir = repo_root / "benchmark" / "tasks" / task_id
        probes.append(_probe(
            task_id=task_id,
            task_package_digest=__import__("benchmark.calibration.identity", fromlist=["task_package_digest"]).task_package_digest(task_dir),
            baseline_probe_session_id=f"session-{task_id}",
        ))
    runtime = {"cuda_visible_devices": "5", "gpu_uuid": "GPU-test", "gpu_name": "RTX 4090", "total_device_memory": 24564 * 1024**2}
    manifest = build_resource_domain_manifest(repo_root=repo_root, task_probes=probes, runtime_contract=runtime, out_path=tmp_path / "manifest.json")
    assert validate_resource_domain_manifest({**manifest, "benchmark_revision": "stale"}, repo_root=repo_root)
    changed = dict(manifest)
    changed["runtime_contract"] = {**runtime, "gpu_uuid": "GPU-other"}
    assert validate_resource_domain_manifest(changed, repo_root=repo_root)
    changed_task = json.loads(json.dumps(manifest))
    changed_task["tasks"][0]["task_package_digest"] = "b" * 64
    assert validate_resource_domain_manifest(changed_task, repo_root=repo_root)


def test_runtime_contract_mismatch_is_blocking():
    manifest = {"runtime_contract": {"cuda_visible_devices": "5", "gpu_uuid": "GPU-5"}}
    assert validate_runtime_contract(
        manifest, {"cuda_visible_devices": "3", "gpu_uuid": "GPU-3"}
    )
