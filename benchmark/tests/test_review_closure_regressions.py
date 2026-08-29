from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from benchmark.calibration.report import _empirical_flags, build_efficacy_eligibility_input
from benchmark.formal.aggregate import _lineage_task_outer_ci
from benchmark.formal.eligibility import build_formal_efficacy_projection, build_manifest, validate_formal_efficacy_projection
from benchmark.formal.resource_domain import classify_baseline_probe
from benchmark.formal.schedule import build_schedule
from benchmark.tests.test_formal_eligibility import _pilot
from benchmark.tests.test_formal_resource_domain import _resource
from scripts.preregister_formal_slots import build as build_preregistered
from scripts.run_resource_domain_census import _foreign_processes
from benchmark.provenance import json_digest


ROOT = Path(__file__).resolve().parents[2]


def test_scientific_gate_failure_is_task_attrition_not_semantic_gate_weakness(tmp_path: Path):
    payload = {
        "tasks": [{
            "task_id": "T",
            "calibration_status": "eligible",
            "semantic_gate_pass_rate": 0.5,
            "scientific_gates": {"finite": False},
        }]
    }
    path = tmp_path / "empirical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    flags, _ = _empirical_flags([{
        "task_id": "T",
        "track": "spe_core",
        "kind": "positive",
        "scientific_gates": ["finite"],
        "measurement": {},
    }], path, [])

    assert flags["semantic_gate_too_weak"] == []
    assert flags["scientific_gate_failed"] == ["T"]


def test_cotenant_oom_is_not_intrinsic_capacity_exclusion():
    probe = {
        "task_id": "T1",
        "task_package_digest": "a" * 64,
        "execution_class": "atomic_performance",
        "baseline_only": True,
        "clean_admission": True,
        "foreign_processes": [4242],
        "protocol_valid": True,
        "cleanup_quiescent": True,
        "baseline_completed": False,
        "resource_exhausted": True,
        "total_device_memory": 24 * 1024**3,
        "preflight_free_memory": 6 * 1024**2,
        "peak_allocated": 23 * 1024**3,
        "peak_reserved": 23 * 1024**3,
        "reserved_unallocated": 95 * 1024**2,
        "failed_allocation_bytes": 16 * 1024**2,
        "exception_class": "torch.OutOfMemoryError",
        "failure_signature": "cuda oom",
        "confirmation_attempt_count": 2,
        "capacity_failure_reproduced": True,
        "cleanup_quiescent": True,
    }

    assert classify_baseline_probe(probe) == "unresolved"


def test_resource_probe_merges_cotenant_telemetry_seen_after_admission():
    assert _foreign_processes(
        {"foreign_processes": []},
        {"foreign_processes": [{"pid": 4242, "used_memory": "512 MiB"}]},
    ) == [{"pid": 4242, "used_memory": "512 MiB"}]


def test_schedule_can_be_built_from_already_filtered_task_universe():
    plan = build_schedule(
        ROOT / "benchmark" / "split" / "sequential.yaml",
        task_items=[(7, "CORE-H2D-PIPELINE-03R2")],
        conditions=("B", "D"),
        outer_trials=1,
    )

    assert {item["task_id"] for item in plan} == {"CORE-H2D-PIPELINE-03R2"}


def test_formal_projection_maps_source_eligibility_to_slot_identity(tmp_path: Path):
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    eligibility = build_manifest(
        pilot_calibration=build_efficacy_eligibility_input(source),
        tasks_root=ROOT / "benchmark" / "tasks",
        benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    formal_population = {
        "schema_version": 1,
        "population_id": "formal",
        "status": "materialized_frozen",
        "slots": [{
            "slot_id": "SLOT-01",
            "task_id": "SLOT-01",
            "package_digest": "d" * 64,
            "source_task_id": "CORE-H2D-PIPELINE-03R2",
            "visibility": "sealed",
        }],
    }

    projection = build_formal_efficacy_projection(
        eligibility,
        formal_population,
        out_path=tmp_path / "formal_efficacy_projection.json",
    )

    assert projection["status"] == "frozen"
    assert projection["slots"][0]["source_task_id"] == "CORE-H2D-PIPELINE-03R2"
    assert projection["slots"][0]["formal_slot_id"] == "SLOT-01"
    assert projection["slots"][0]["slot_package_digest"] == "d" * 64
    assert projection["slots"][0]["inherited_eligibility"]["role"] == "efficacy"
    assert (tmp_path / "formal_efficacy_projection.json").is_file()


def test_formal_projection_binds_slot_package_digest_to_materialized_slot():
    source = _pilot("CORE-H2D-PIPELINE-03R2")
    eligibility = build_manifest(
        pilot_calibration=build_efficacy_eligibility_input(source),
        tasks_root=ROOT / "benchmark" / "tasks",
        benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )
    formal_population = {
        "status": "materialized_frozen",
        "slots": [{
            "slot_id": "SLOT-01",
            "task_id": "SLOT-01",
            "package_digest": "d" * 64,
            "source_task_id": "CORE-H2D-PIPELINE-03R2",
        }],
    }
    projection = build_formal_efficacy_projection(eligibility, formal_population)
    tampered = copy.deepcopy(projection)
    tampered["slots"][0]["slot_package_digest"] = "e" * 64
    tampered["artifact_digest"] = json_digest({key: value for key, value in tampered.items() if key != "artifact_digest"})

    errors = validate_formal_efficacy_projection(
        tampered, eligibility_manifest=eligibility, formal_population=formal_population,
    )

    assert "formal slot package digest mismatch: SLOT-01" in errors


def test_hierarchical_bootstrap_resamples_lineage_task_and_outer_levels():
    observations = [
        ("family", "L1", "T1", "outer-000", 0.0),
        ("family", "L1", "T1", "outer-001", 100.0),
        ("family", "L1", "T2", "outer-000", 0.0),
        ("family", "L1", "T2", "outer-001", 100.0),
        ("family", "L2", "T3", "outer-000", 50.0),
        ("family", "L2", "T3", "outer-001", 50.0),
        ("family", "L2", "T4", "outer-000", 50.0),
        ("family", "L2", "T4", "outer-001", 50.0),
    ]

    result = _lineage_task_outer_ci(observations, seed=0, samples=300)

    assert result["estimate"] == pytest.approx(50.0)
    assert result["ci_low"] < result["ci_high"]


def test_sealed_slots_are_seeded_template_instances_not_independent_lineages():
    manifest = build_preregistered(ROOT)
    sealed = [slot for slot in manifest["slots"] if slot["visibility"] == "sealed"]

    assert sealed
    assert {slot["lineage_class"] for slot in sealed} == {"seeded-template-instance"}


def test_workflow_actions_are_pinned_to_commit_shas():
    workflow = (ROOT / ".github" / "workflows" / "validate.yml").read_text(encoding="utf-8")

    assert not re.search(r"uses:\s+actions/[^@]+@v\d", workflow)
    assert "actions/checkout@" in workflow
    assert "actions/setup-python@" in workflow


def test_not_applicable_evolution_keeps_role_without_unresolved_reason():
    task_id = "EVOL-EQUIVARIANT-SPECIALIZE-30"
    source = _pilot(task_id)
    projected = build_efficacy_eligibility_input(
        source,
        resource_domain_manifest=_resource(task_id, "not_applicable"),
    )

    manifest = build_manifest(
        pilot_calibration=projected,
        tasks_root=ROOT / "benchmark" / "tasks",
        benchmark_revision="r" * 40,
        population_manifest_digest="p" * 64,
        calibration_protocol_digest="c" * 64,
    )

    task = manifest["tasks"][0]
    assert task["role"] == "diagnostic"
    assert "RESOURCE_DOMAIN_UNRESOLVED" not in task["reason"]
