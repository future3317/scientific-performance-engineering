from __future__ import annotations

import json
from pathlib import Path

from benchmark.formal import aggregate
from benchmark.formal.attest import validate_experiment
from benchmark.taskgen.validate_population import validate_formal_readiness


ROOT = Path(__file__).resolve().parents[2]


def test_preregistered_slots_are_explicit_and_sealed_primary():
    payload = json.loads((ROOT / "benchmark" / "manifests" / "v1.0-50-slots.json").read_text(encoding="utf-8"))
    assert len(payload["slots"]) == 50
    assert payload["public_dev_total"] == 15
    assert payload["sealed_total"] == 35
    required = {"slot_id", "visibility", "track", "family_id", "mechanism", "polarity", "difficulty", "lineage_class", "generator_version", "seed", "scientific_contract", "replacement_class"}
    assert all(required <= set(slot) for slot in payload["slots"])


def test_blocked_approval_cannot_open_formal_readiness():
    report = {"empirical_calibration": {"calibration_gate": "blocked"}, "semantic_gate_failures": [], "empirical_rejection_flags": {}, "retired_for_formal": []}
    calibration = {"calibration_gate": "blocked", "tasks": []}
    approval = {"approved": False}
    assert validate_formal_readiness(report, calibration, approval)


def test_confirmatory_aggregate_withholds_incomplete_matrix():
    result = aggregate.aggregate_confirmatory([], required_cells=[("t", "o", "reset", "D")], claims={"primary": {"id": "D-B"}})
    assert result["status"] == "withheld"


def test_formal_manifest_requires_agent_provenance_fields():
    manifest = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-50",
        "experiment_id": "x",
        "benchmark_revision": "a" * 40,
        "skill_view_digest": "a" * 64,
        "task_manifest_digest": "b" * 64,
        "agent_model_id": "model",
        "agent_config": {},
        "condition": "D",
        "context_mode": "reset",
        "task_order": ["T"],
        "worker_isolation": {"mode": "external_namespace_executor", "network_mode": "none", "mount_allowlist": ["task"]},
        "outer_trial_id": "outer-000",
        "budgets": {"tokens": 1, "tool_calls": 1, "wall_time_s": 1},
        "hardware_fingerprint": {},
        "software_fingerprint": {},
        "torch_version": None,
        "cuda_version": None,
    }
    assert any("formal agent_config missing provider" in error for error in validate_experiment(manifest))
