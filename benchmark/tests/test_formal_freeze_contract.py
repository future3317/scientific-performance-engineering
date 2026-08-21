from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark.formal import aggregate
from benchmark.formal import release_manifest, run_campaign
from benchmark.formal.attest import validate_experiment
from benchmark.calibration.report import validate_formal_readiness
from benchmark.calibration import report as validate_population


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


def test_strict_formal_cli_fails_closed_without_crashing(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "population_report.json"
    calibration_path = tmp_path / "pilot_calibration.json"
    approval_path = tmp_path / "calibration_approval.json"
    report_path.write_text(json.dumps({
        "empirical_calibration": {"calibration_gate": "blocked"},
        "semantic_gate_failures": [],
        "empirical_rejection_flags": {},
        "retired_for_formal": [],
    }), encoding="utf-8")
    calibration_path.write_text(json.dumps({"calibration_gate": "blocked", "tasks": []}), encoding="utf-8")
    approval_path.write_text(json.dumps({"approved": False}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_population", "--strict-formal", "--out", str(report_path),
        "--empirical", str(calibration_path), "--pilot-calibration", str(calibration_path),
        "--approval", str(approval_path),
    ])
    assert validate_population.main() == 1


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


def test_campaign_config_accepts_repo_relative_and_absolute_same_manifest(tmp_path: Path):
    config = ROOT / "benchmark" / "formal" / "campaign_config.yaml"
    manifest = (ROOT / "benchmark" / "manifests" / "v1.0-50-slots.json").resolve()
    validated = run_campaign._validate_campaign_config(
        config,
        conditions=("A", "B", "C", "D"),
        context_modes=("reset",),
        outer_trials=3,
        schedule_seed=0,
        population_manifest=manifest,
        repo_root=ROOT,
    )
    assert validated["population_manifest"] == "benchmark/manifests/v1.0-50-slots.json"
    bad_config = tmp_path / "campaign.yaml"
    bad_config.write_text("mode: formal\npopulation_manifest: benchmark/manifests/other.json\n", encoding="utf-8")
    try:
        run_campaign._validate_campaign_config(
            bad_config,
            conditions=("A", "B", "C", "D"), context_modes=("reset",),
            outer_trials=3, schedule_seed=0, population_manifest=manifest, repo_root=ROOT,
        )
    except ValueError as exc:
        assert "population_manifest" in str(exc)
    else:
        raise AssertionError("different manifest path unexpectedly accepted")


def test_release_manifest_rejects_empty_executor_image(tmp_path: Path):
    paths = []
    for name in ("claims", "protocol", "campaign"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)
    try:
        release_manifest.build_release_manifest(
            repo_root=ROOT,
            population={},
            approval={},
            contamination={},
            claims_path=paths[0],
            protocol_path=paths[1],
            campaign_config=paths[2],
            executor_image="",
        )
    except ValueError as exc:
        assert "executor_image" in str(exc)
    else:
        raise AssertionError("empty executor_image unexpectedly accepted")
