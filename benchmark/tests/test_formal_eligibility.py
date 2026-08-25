from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmark.calibration.identity import task_package_digest
from benchmark.calibration.report import build_efficacy_eligibility_input
from benchmark.formal.eligibility import build_manifest, validate_manifest
from benchmark.harness import miniyaml, stats


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
    noise_artifact = {
        "task_id": task_id, "outer_trial_id": "outer-000", "measurement_session_id": session,
        "benchmark_revision": "r" * 40, "task_package_digest": task_package_digest(task_dir),
        "population_manifest_digest": "p" * 64, "calibration_protocol_digest": "c" * 64,
        "typical_noise_percent": typical_noise,
    }
    noise_artifact["artifact_digest"] = stats._noise_control_digest(noise_artifact)
    task = {
        "task_id": task_id, "task_digest": task_package_digest(task_dir), "revision": "r" * 40,
        "environment": fingerprint, "outer_trials": [trial], "baseline_stability": stable,
        "noise_control": {"artifacts": [noise_artifact]},
        "semantic_gates": {"correctness_pass": True, "scientific_gates": True, "execution_valid": True},
        "anti_cheat": {"status": "pass"},
    }
    return {
        "schema_version": 1,
        "benchmark_revision": "r" * 40,
        "population_manifest_digest": "p" * 64,
        "calibration_protocol_digest": "c" * 64,
        "active_task_ids": [task_id],
        "tasks": [task],
    }


def test_canonical_pilot_projection_is_end_to_end_and_outcome_blind():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    source["tasks"][0]["oracle_ci"] = {"ci_low": 1.9, "ci_high": 9.9}
    source["tasks"][0]["empirical_evidence"] = {"verified_speedup": 9.9}
    projected = build_efficacy_eligibility_input(source)
    manifest = build_manifest(
        pilot_calibration=projected, tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == ["CORE-H2D-PIPELINE-03R2"]
    assert "oracle_ci" not in projected["tasks"][0]
    assert "empirical_evidence" not in projected["tasks"][0]


def test_real_report_to_pilot_to_manifest_pipeline_is_sanitized(monkeypatch):
    from benchmark.calibration.report import build_pilot_calibration

    task_id = "CORE-H2D-PIPELINE-03R2"
    task = _pilot(task_id)["tasks"][0]
    record = {
        **task,
        "calibration_status": "eligible",
        "calibration_protocol_digest": "c" * 64,
        "population_manifest_digest": "p" * 64,
        "oracle_ci": {"ci_low": 1.8, "ci_high": 2.0},
    }
    report = {
        "population_id": "test",
        "active_task_ids": [task_id],
        "empirical_calibration": {
            "status": "observed", "calibration_gate": "ready_for_review",
            "records_by_task": {task_id: record}, "missing_task_ids": [],
        },
        "empirical_rejection_flags": {}, "semantic_gate_failures": [],
        "calibration_protocol_digest": "c" * 64,
    }
    monkeypatch.setattr(
        "benchmark.calibration.report.load_active_manifest",
        lambda _root: {"task_ids": [task_id]},
    )
    pilot = build_pilot_calibration(report, ROOT / "tasks")
    projected = build_efficacy_eligibility_input(pilot)
    manifest = build_manifest(
        pilot_calibration=projected, tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == [task_id]
    assert "oracle_ci" not in projected["tasks"][0]


def test_outcome_fields_in_sanitized_input_fail_closed():
    projected = build_efficacy_eligibility_input(_pilot("CORE-H2D-PIPELINE-03R2"))
    for field in ("oracle_ci", "verified_speedup", "d_b_effect"):
        tampered = copy.deepcopy(projected)
        tampered["tasks"][0][field] = {"value": 99.0}
        with pytest.raises(ValueError, match="agent/outcome"):
            build_manifest(
                pilot_calibration=tampered, tasks_root=ROOT / "tasks",
                benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
                calibration_protocol_digest="c" * 64,
            )


def test_revision_population_protocol_and_package_mismatches_fail_closed():
    projected = build_efficacy_eligibility_input(_pilot("CORE-H2D-PIPELINE-03R2"))
    with pytest.raises(ValueError, match="benchmark revision"):
        build_manifest(
            pilot_calibration=projected, tasks_root=ROOT / "tasks",
            benchmark_revision="x" * 40, population_manifest_digest="p" * 64,
            calibration_protocol_digest="c" * 64,
        )
    with pytest.raises(ValueError, match="population digest"):
        build_manifest(
            pilot_calibration=projected, tasks_root=ROOT / "tasks",
            benchmark_revision="r" * 40, population_manifest_digest="x" * 64,
            calibration_protocol_digest="c" * 64,
        )
    with pytest.raises(ValueError, match="protocol digest"):
        build_manifest(
            pilot_calibration=projected, tasks_root=ROOT / "tasks",
            benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
            calibration_protocol_digest="x" * 64,
        )
    tampered = copy.deepcopy(projected)
    tampered["tasks"][0]["task_package_digest"] = "x" * 64
    with pytest.raises(ValueError, match="task package digest"):
        build_manifest(
            pilot_calibration=tampered, tasks_root=ROOT / "tasks",
            benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
            calibration_protocol_digest="c" * 64,
        )


def _eligibility(task_id: str, *, stable: bool = True, typical_noise: float = 3.0) -> dict:
    return build_efficacy_eligibility_input(_pilot(task_id, stable=stable, typical_noise=typical_noise))


def test_manifest_is_agent_blind_and_revision_bound():
    task_id = "CORE-H2D-PIPELINE-03R2"
    manifest = build_manifest(
        pilot_calibration=_eligibility(task_id), tasks_root=ROOT / "tasks", benchmark_revision="r" * 40,
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
        pilot_calibration=_eligibility(task_id, stable=False, typical_noise=40.0), tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == [task_id]
    assert manifest["tasks"][0]["baseline_stability"] == "blocked"


def test_agent_outcome_cannot_enter_manifest_input():
    task_id = "CORE-H2D-PIPELINE-03R2"
    pilot = _eligibility(task_id)
    pilot["tasks"][0]["agent_outcome"] = {"condition": "D", "score": 1.0}
    with pytest.raises(ValueError, match="agent/outcome"):
        build_manifest(
            pilot_calibration=pilot, tasks_root=ROOT / "tasks", benchmark_revision="r" * 40,
            population_manifest_digest="p" * 64, calibration_protocol_digest="c" * 64,
        )


def test_evolution_record_remains_diagnostic_when_atomic_efficacy_is_blocked():
    task_id = "EVOL-EQUIVARIANT-SPECIALIZE-30"
    manifest = build_manifest(
        pilot_calibration=_eligibility(task_id, stable=False), tasks_root=ROOT / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == [task_id]
    assert manifest["tasks"][0]["baseline_stability"] == "not_applicable"
