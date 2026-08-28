from __future__ import annotations

import json
from pathlib import Path

from benchmark.calibration.report import _empirical_flags, build_efficacy_eligibility_input
from benchmark.formal.eligibility import build_manifest
from benchmark.formal.resource_domain import classify_baseline_probe
from benchmark.harness import stats


ROOT = Path(__file__).resolve().parents[2]


def _positive_spec() -> dict:
    return {
        "task_id": "T",
        "track": "spe_core",
        "kind": "positive",
        "scientific_gates": ["finite"],
        "measurement": {"noise_floor_percent": 2.0, "min_improvement_percent": 5.0},
    }


def test_valid_but_ineligible_attrition_does_not_block_population_gate(tmp_path: Path):
    payload = {"tasks": [{
        "task_id": "T",
        "calibration_status": "eligible",
        "oracle_ci_low_percent": 1.0,
        "oracle_ci_high_percent": 4.0,
        "control_noise_percent": [2.0],
    }]}
    path = tmp_path / "empirical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    flags, calibration = _empirical_flags([_positive_spec()], path, [])

    assert flags["effect_too_small"] == ["T"]
    assert calibration["hard_validity_flags"] == {}
    assert calibration["task_attrition_flags"]["effect_too_small"] == ["T"]
    assert calibration["population_integrity_gate"] == "pass"
    assert calibration["calibration_gate"] == "ready_for_review"


def test_protocol_invalid_still_blocks_population_gate(tmp_path: Path):
    payload = {"tasks": [{
        "task_id": "T",
        "calibration_status": "eligible",
        "protocol_failure": True,
        "execution_validity": "valid",
    }]}
    path = tmp_path / "empirical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, calibration = _empirical_flags([_positive_spec()], path, [])

    assert calibration["population_integrity_gate"] == "blocked"
    assert calibration["hard_validity_flags"]["protocol_invalid"] == ["T"]


def test_baseline_stability_is_descriptive_when_noise_is_complete(tmp_path: Path):
    artifact = stats.write_noise_control(
        tmp_path / "noise.json",
        {
            "task_id": "T",
            "outer_trial_id": "outer-000",
            "benchmark_revision": "r" * 40,
            "task_manifest_digest": "m" * 64,
            "task_package_digest": "p" * 64,
            "population_manifest_digest": "q" * 64,
            "control_implementation": "baseline",
            "hardware_fingerprint": {},
            "software_fingerprint": {},
            "compile_threads": 2,
            "compiler_cache_policy": "fresh",
            "primary_metric": "step_ms_p50",
            "higher_is_better": False,
            "control_a_runs": [1.0] * 5,
            "control_b_runs": [1.0] * 5,
            "declared_noise_floor_percent": 2.0,
            "expected_speedup_range": [1.0, 2.0],
        },
    )

    assert artifact["baseline_stability"] is True


def test_valid_positive_without_orphan_stability_flag_can_enter_manifest():
    from benchmark.tests.test_formal_eligibility import _pilot

    source = _pilot("CORE-H2D-PIPELINE-03R2", stable=None)
    projected = build_efficacy_eligibility_input(source)
    manifest = build_manifest(
        pilot_calibration=projected,
        tasks_root=ROOT / "benchmark" / "tasks",
        benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )

    assert manifest["efficacy_task_ids"] == ["CORE-H2D-PIPELINE-03R2"]
    assert manifest["tasks"][0]["baseline_stability"] == "descriptive_only"


def _timeout_probe(**overrides):
    probe = {
        "task_id": "T1",
        "task_package_digest": "a" * 64,
        "execution_class": "atomic_performance",
        "baseline_only": True,
        "clean_admission": True,
        "foreign_processes": [],
        "protocol_valid": True,
        "baseline_completed": False,
        "resource_exhausted": False,
        "timeout": True,
        "timeout_stage": "baseline_probe",
        "timeout_s": 600.0,
        "timeout_reproduced": True,
        "confirmation_attempt_count": 2,
        "total_device_memory": 24 * 1024**3,
        "preflight_free_memory": 24 * 1024**3,
        "peak_allocated": 512 * 1024**2,
        "peak_reserved": 512 * 1024**2,
        "reserved_unallocated": 0,
        "failed_allocation_bytes": None,
        "exception_class": None,
        "failure_signature": "baseline probe timed out",
        "cleanup_quiescent": True,
        "baseline_probe_session_id": "session-t1",
    }
    probe.update(overrides)
    return probe


def test_repeated_baseline_timeout_is_walltime_attrition():
    assert classify_baseline_probe(_timeout_probe()) == "out_of_domain_walltime"


def test_unconfirmed_baseline_timeout_remains_unresolved():
    assert classify_baseline_probe(_timeout_probe(timeout_reproduced=False)) == "unresolved"


def test_baseline_walltime_requires_baseline_only_evidence():
    assert classify_baseline_probe(_timeout_probe(baseline_only=False)) == "unresolved"
    assert classify_baseline_probe(_timeout_probe(timeout_stage="verifier")) == "unresolved"


def test_oracle_timeout_is_visible_attrition_not_resource_domain():
    payload = {"tasks": [{
        "task_id": "T",
        "calibration_status": "eligible",
        "baseline_reference_diagnostic": {
            "status": "calibration_reference_runtime_ineligible",
        },
        "outer_trials": [{
            "execution_validity": "resource_blocked",
            "timeout": True,
        }],
    }]}
    path = Path("/tmp/evospe-oracle-timeout.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        _, calibration = _empirical_flags([_positive_spec()], path, [])
    finally:
        path.unlink()
    assert calibration["population_integrity_gate"] == "pass"
    assert calibration["task_attrition_flags"][
        "calibration_reference_runtime_ineligible"
    ] == ["T"]
    assert "resource_blocked" not in calibration["hard_validity_flags"]
