from __future__ import annotations

import copy

import pytest

from benchmark.calibration.report import build_efficacy_eligibility_input
from benchmark.calibration.identity import task_package_digest
from benchmark.formal.eligibility import build_manifest
from benchmark.formal.schedule import build_task_block_schedule
from benchmark.provenance import json_digest
from benchmark.formal.resource_sensitivity import compute_resource_sensitivity
from benchmark.tests.test_formal_eligibility import _pilot


def _resource(task_id: str, status: str = "in_domain") -> dict:
    runtime = {"cuda_visible_devices": "5", "gpu_uuid": "GPU-test"}
    return {
        "manifest_id": "resource-domain-eligibility-v1",
        "status": "frozen",
        "benchmark_revision": "r" * 40,
        "population_manifest_digest": "p" * 64,
        "runtime_contract": runtime,
        "runtime_contract_digest": json_digest(runtime),
        "artifact_digest": "r" * 64,
        "task_ids": [task_id],
        "tasks": [{
            "task_id": task_id,
            "task_package_digest": task_package_digest(__import__("pathlib").Path(__file__).parents[1] / "tasks" / task_id),
            "resource_domain_status": status,
        }],
    }


def test_resource_out_of_domain_is_diagnostic_not_scientific_failure():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    projected = build_efficacy_eligibility_input(source, resource_domain_manifest=_resource("CORE-H2D-PIPELINE-03R2", "out_of_domain_capacity"))
    manifest = build_manifest(
        pilot_calibration=projected, tasks_root=__import__("pathlib").Path(__file__).parents[1] / "tasks",
        benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == ["CORE-H2D-PIPELINE-03R2"]
    assert "RESOURCE_OUT_OF_TARGET_RUNTIME_DOMAIN" in manifest["tasks"][0]["reason"]


def test_resource_excluded_task_can_build_manifest_without_performance_artifacts():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    source["tasks"][0]["outer_trials"] = []
    source["tasks"][0]["noise_control"] = {"artifacts": []}
    projected = build_efficacy_eligibility_input(
        source,
        resource_domain_manifest=_resource("CORE-H2D-PIPELINE-03R2", "out_of_domain_capacity"),
    )
    manifest = build_manifest(
        pilot_calibration=projected,
        tasks_root=__import__("pathlib").Path(__file__).parents[1] / "tasks",
        benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    assert manifest["efficacy_task_ids"] == []
    assert manifest["diagnostic_task_ids"] == ["CORE-H2D-PIPELINE-03R2"]


def test_resource_manifest_digest_is_bound_to_eligibility_projection():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    projected = build_efficacy_eligibility_input(source, resource_domain_manifest=_resource("CORE-H2D-PIPELINE-03R2"))
    assert projected["resource_domain_manifest_digest"] == "r" * 64
    assert projected["tasks"][0]["resource_domain_status"] == "in_domain"


def test_resource_status_cannot_be_added_without_resource_manifest():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    projected = build_efficacy_eligibility_input(source)
    tampered = copy.deepcopy(projected)
    tampered["tasks"][0]["resource_domain_status"] = "out_of_domain_capacity"
    with pytest.raises(ValueError, match="resource domain"):
        build_manifest(
            pilot_calibration=tampered, tasks_root=__import__("pathlib").Path(__file__).parents[1] / "tasks",
            benchmark_revision="r" * 40, population_manifest_digest="p" * 64,
            calibration_protocol_digest="c" * 64,
        )


def test_resource_out_of_domain_task_is_omitted_without_dummy_cells():
    manifest = {
        "task_ids": ["T-in", "T-out"],
        "in_domain_task_ids": ["T-in"],
        "out_of_domain_task_ids": ["T-out"],
    }
    schedule = build_task_block_schedule(
        [(1, "T-in"), (2, "T-out")], conditions=("B", "D"),
        context_modes=("reset",), outer_trials=3, schedule_seed=0,
        resource_domain_manifest=manifest,
    )
    assert len(schedule) == 6
    assert {item["task_id"] for item in schedule} == {"T-in"}


def test_not_applicable_evolution_task_remains_in_schedule():
    manifest = {
        "task_ids": ["T-in", "T-episode", "T-out"],
        "in_domain_task_ids": ["T-in"],
        "not_applicable_task_ids": ["T-episode"],
        "out_of_domain_task_ids": ["T-out"],
        "unresolved_task_ids": [],
    }
    schedule = build_task_block_schedule(
        [(1, "T-in"), (1, "T-episode"), (1, "T-out")], conditions=("B", "D"),
        context_modes=("reset",), outer_trials=1, schedule_seed=0,
        resource_domain_manifest=manifest,
    )
    assert {item["task_id"] for item in schedule} == {"T-in", "T-episode"}


def test_resource_sensitivity_uses_registered_weights_and_bounded_delta():
    result = compute_resource_sensitivity(
        registered_weights={"T-out": 0.25},
        observed_delta=0.4,
    )
    assert result["missing_weight"] == 0.25
    assert result["full_population_worst_case"] == pytest.approx([-0.1, 0.9])
    assert result["utility_bounds"] == [-1.0, 1.0]


def test_resource_excluded_task_has_no_calibration_execution_cell():
    from benchmark.calibration.campaign import calibration_task_ids

    manifest = {
        "task_ids": ["T-in", "T-episode", "T-out"],
        "tasks": [
            {"task_id": "T-in", "resource_domain_status": "in_domain"},
            {"task_id": "T-episode", "resource_domain_status": "not_applicable"},
            {"task_id": "T-out", "resource_domain_status": "out_of_domain_capacity"},
        ],
        "unresolved_task_ids": [],
    }
    assert calibration_task_ids(["T-in", "T-episode", "T-out"], manifest) == ["T-in", "T-episode"]


def test_pilot_projection_preserves_resource_attrition_without_performance_artifacts(monkeypatch):
    from benchmark.calibration import report as calibration_report

    task_id = "CORE-H2D-PIPELINE-03R2"
    task_dir = __import__("pathlib").Path(__file__).parents[1] / "tasks" / task_id
    monkeypatch.setattr(calibration_report, "load_active_manifest", lambda _: {"task_ids": [task_id]})
    source = {
        "benchmark_revision": "r" * 40,
        "resource_domain": {
            "manifest_digest": "m" * 64,
            "status_by_task": {task_id: "out_of_domain_capacity"},
            "runtime_contract": {"cuda_visible_devices": "5"},
        },
        "task_calibration": {},
        "empirical_calibration": {
            "status": "observed", "calibration_gate": "ready_for_review",
            "records_by_task": {}, "missing_task_ids": [], "empirical_digest": "e" * 64,
        },
        "empirical_rejection_flags": {}, "semantic_gate_failures": [],
        "population_id": "population", "calibration_protocol_digest": "c" * 64,
    }
    projected = calibration_report.build_pilot_calibration(source, task_dir.parent)
    task = projected["tasks"][0]
    assert task["status"] == "blocked"
    assert task["block_reason"] == ["RESOURCE_OUT_OF_TARGET_RUNTIME_DOMAIN"]
    assert task["outer_trials"] == []
    assert task["noise_control"] == {"artifacts": []}


def test_calibration_approval_allows_resource_domain_attrition():
    from benchmark.calibration.approval import validate_calibration_approval

    report = {
        "active_task_ids": ["T-in", "T-out"],
        "resource_domain": {"out_of_domain_task_ids": ["T-out"]},
        "empirical_calibration": {"empirical_digest": "e" * 64},
        "retired_for_formal": [],
    }
    calibration = {
        "artifact_digest": "p" * 64,
        "empirical_digest": "e" * 64,
        "tasks": [
            {"task_id": "T-in", "status": "eligible", "eligibility": True},
            {"task_id": "T-out", "status": "blocked", "eligibility": False},
        ],
    }
    approval = {
        "schema_version": 1,
        "approved": True,
        "approval_digest": "pending",
        "approver": "researcher",
        "timestamp": "now",
        "population_report_digest": json_digest(report),
        "pilot_calibration_digest": "p" * 64,
        "empirical_digest": "e" * 64,
        "calibration_protocol_digest": "c" * 64,
        "benchmark_revision": "r" * 40,
        "approved_task_ids": ["T-in"],
    }
    approval["approval_digest"] = json_digest({key: value for key, value in approval.items() if key != "approval_digest"})
    assert validate_calibration_approval(report, calibration, approval) == []
