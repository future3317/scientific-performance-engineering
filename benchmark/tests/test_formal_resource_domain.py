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


def test_resource_sensitivity_uses_registered_weights_and_bounded_delta():
    result = compute_resource_sensitivity(
        registered_weights={"T-out": 0.25},
        observed_delta=0.4,
    )
    assert result["missing_weight"] == 0.25
    assert result["full_population_worst_case"] == pytest.approx([-0.1, 0.9])
    assert result["utility_bounds"] == [-1.0, 1.0]
