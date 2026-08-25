from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmark.calibration.identity import task_package_digest
from benchmark.formal.eligibility import build_manifest, validate_manifest
from benchmark.harness import miniyaml


ROOT = Path(__file__).resolve().parents[1]


def _pilot(task_id: str, *, stable: bool = True, typical_noise: float = 3.0) -> dict:
    task_dir = ROOT / "tasks" / task_id
    spec = miniyaml.load(str(task_dir / "task.yaml"))
    fingerprint = {
        "python_version": "3.12", "platform": "linux", "torch_version": "2.0",
        "cuda_version": "12.1", "gpu_name": "GPU", "gpu_count": 1,
        "torch_geometric_version": None, "cpu_affinity": [0],
        "driver_version": "550", "cuda_visible_devices": "0", "gpu_uuid": "GPU-x",
        "thread_topology": {"omp_num_threads": "8"}, "cuda_available": True,
    }
    session = "session-" + task_id
    trial = {
        "task_id": task_id, "outer_trial_id": "outer-000", "measurement_session_id": session,
        "execution_validity": "valid", "protocol_failure": False, "timeout": False,
        "correctness_pass": True, "scientific_gates": {"gate": True},
    }
    task = {
        "task_id": task_id, "task_digest": task_package_digest(task_dir), "revision": "r" * 40,
        "environment": fingerprint, "outer_trials": [trial], "baseline_stability": stable,
        "noise_control": {"artifacts": [{"outer_trial_id": "outer-000", "measurement_session_id": session, "typical_noise_percent": typical_noise}]},
        "semantic_gates": {"correctness_pass": True, "scientific_gates": True, "execution_valid": True},
        "anti_cheat": {"status": "pass"},
    }
    return {"active_task_ids": [task_id], "tasks": [task]}


def test_manifest_is_agent_blind_and_revision_bound():
    task_id = "CORE-H2D-PIPELINE-03R2"
    manifest = build_manifest(
        pilot_calibration=_pilot(task_id), tasks_root=ROOT / "tasks", benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64, calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == [task_id]
    assert manifest["tasks"][0]["effective_noise_floor_percent"] == 3.0
    assert "oracle_ci" not in manifest["tasks"][0]
    assert validate_manifest(manifest) == []

    tampered = copy.deepcopy(manifest)
    tampered["tasks"][0]["d_b_effect"] = 1.2
    assert validate_manifest(tampered)


def test_unstable_atomic_task_is_diagnostic_not_efficacy():
    task_id = "CORE-H2D-PIPELINE-03R2"
    manifest = build_manifest(
        pilot_calibration=_pilot(task_id, stable=False, typical_noise=40.0), tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == [task_id]
    assert manifest["tasks"][0]["baseline_stability"] == "blocked"


def test_agent_outcome_cannot_enter_manifest_input():
    task_id = "CORE-H2D-PIPELINE-03R2"
    pilot = _pilot(task_id)
    pilot["tasks"][0]["agent_outcome"] = {"condition": "D", "score": 1.0}
    with pytest.raises(ValueError, match="agent/outcome"):
        build_manifest(
            pilot_calibration=pilot, tasks_root=ROOT / "tasks", benchmark_revision="r" * 40,
            population_manifest_digest="p" * 64, calibration_protocol_digest="c" * 64,
        )


def test_evolution_record_remains_diagnostic_when_atomic_efficacy_is_blocked():
    task_id = "EVOL-EQUIVARIANT-SPECIALIZE-30"
    manifest = build_manifest(
        pilot_calibration=_pilot(task_id, stable=False), tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == [task_id]
    assert manifest["tasks"][0]["baseline_stability"] == "not_applicable"
