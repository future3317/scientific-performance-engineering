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


def test_confirmatory_aggregate_uses_frozen_efficacy_subset_only():
    records = []
    for task_id in ("T1", "T2"):
        for condition, score in (("B", 0.4), ("D", 0.8)):
            records.append({
                "task_id": task_id, "outer_trial_id": "outer-000", "context_mode": "reset",
                "condition": condition, "visibility": "sealed", "track": "spe_core",
                "efficacy_eligible": True, "execution_validity": "valid", "validity": "valid",
                "score": {"task_score": score, "gates_passed": True},
            })
    required = [(task, "outer-000", "reset", condition) for task in ("T1", "T2") for condition in ("B", "D")]
    result = aggregate.aggregate_confirmatory(
        records, required_cells=required,
        claims={"primary": {"id": "D-B", "tracks": ["spe_core"]}},
        efficacy_task_ids={"T1"}, eligibility_manifest_digest="e" * 64,
    )
    assert result["status"] == "available"
    assert result["primary_db_estimator"]["lineage_count"] == 1
    assert result["eligibility_manifest_digest"] == "e" * 64


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


def test_provider_managed_agent_config_requires_current_provenance_contract():
    manifest = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-50",
        "experiment_id": "x",
        "benchmark_revision": "a" * 40,
        "skill_view_digest": "a" * 64,
        "task_manifest_digest": "b" * 64,
        "agent_model_id": "deepseek-v4-flash",
        "agent_config": {
            "provider": "deepseek",
            "model_id": "deepseek-v4-flash",
            "provider_model_version": "DeepSeek-V4-Flash",
            "provider_model_list_response_digest": "f" * 64,
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 8192,
            "system_prompt_digest": "c" * 64,
            "agent_code_commit": "a" * 40,
            "tool_versions": {},
            "tool_schema_digest": "d" * 64,
            "tool_allowlist": ["read_file", "write_file", "run_command"],
            "retry_policy": {"max_attempts": 2},
            "timeout_s": 900,
            "container_digest": "e" * 64,
            "pricing_revision": "deepseek-pricing-2026-08-24",
            "backend_weight_immutability_claim": False,
        },
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
        "slot_id": "SPE-v1.0-50-01",
        "visibility": "public_dev",
    }
    errors = validate_experiment(manifest)
    assert errors == []
    assert not any("model_snapshot" in error for error in errors)


def test_provider_managed_agent_config_rejects_missing_new_attestation_fields():
    manifest = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-50",
        "experiment_id": "x",
        "benchmark_revision": "a" * 40,
        "skill_view_digest": "a" * 64,
        "task_manifest_digest": "b" * 64,
        "agent_model_id": "deepseek-v4-flash",
        "agent_config": {"provider": "deepseek", "model_id": "deepseek-v4-flash"},
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
        "slot_id": "SPE-v1.0-50-01",
        "visibility": "public_dev",
    }
    errors = validate_experiment(manifest)
    assert any("provider_model_list_response_digest" in error for error in errors)
    assert any("tool_schema_digest" in error for error in errors)


def test_host_controller_exposes_stable_prefix_and_tool_schema():
    from benchmark.formal import host_agent

    assert host_agent.TOOL_NAMES == ("read_file", "write_file", "run_command")
    assert host_agent.fixed_prefix_digest() == host_agent.fixed_prefix_digest()
    assert "task_id" not in host_agent.FIXED_PREFIX_TEXT
    assert "trial" not in host_agent.FIXED_PREFIX_TEXT.lower()


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


def test_nonformal_eligibility_keeps_diagnostic_tasks_in_schedule(tmp_path: Path, monkeypatch):
    task_ids = [
        "CORE-AUTOGRAD-VJP-SMALL-25R2",
        "CORE-DATALOADER-FANOUT-16R2",
        "CORE-KERNEL-FUSION-09R2",
        "SCIML-EQUIV-LOWORDER-26R2",
    ]
    split = tmp_path / "pilot_split.yaml"
    split.write_text(
        "split_id: pilot\nphases:\n  - index: 1\n    name: calibration\n    tasks:\n"
        + "".join(f"      - {task_id}\n" for task_id in task_ids),
        encoding="utf-8",
    )
    skill_source = tmp_path / "skill"
    skill_source.mkdir()
    (skill_source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    eligibility_manifest = {
        "efficacy_task_ids": task_ids[2:],
        "diagnostic_task_ids": task_ids[:2],
        "tasks": [
            {"task_id": task_id, "role": "efficacy" if index >= 2 else "diagnostic"}
            for index, task_id in enumerate(task_ids)
        ],
        "resource_domain_manifest_digest": None,
    }
    monkeypatch.setattr(run_campaign.eligibility, "load_manifest", lambda *args, **kwargs: eligibility_manifest)
    monkeypatch.setattr(run_campaign.eligibility, "validate_runtime_contract", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_campaign.attest, "benchmark_revision", lambda _root: "b" * 40)

    args = type("Args", (), {
        "repo_root": ROOT,
        "tasks_root": ROOT / "benchmark" / "tasks",
        "split": split,
        "skill_source": skill_source,
        "skill_view": None,
        "out": tmp_path / "out",
        "conditions": "A",
        "context_modes": "reset",
        "outer_trials": 1,
        "model_id": "test",
        "agent_config": "{}",
        "budgets": None,
        "agent_command": None,
        "population_manifest": ROOT / "benchmark" / "manifests" / "v1.0-50-slots.json",
        "efficacy_eligibility": tmp_path / "eligibility.json",
        "resource_domain_manifest": None,
        "formal": False,
        "resume": False,
        "schedule_seed": 2000699898,
        "claims": None,
        "analysis_plan": None,
        "campaign_config": None,
    })()
    args.efficacy_eligibility.write_text("{}\n", encoding="utf-8")

    campaign = run_campaign.run_campaign(args)
    assert campaign["schedule_size"] == len(task_ids)
    assert campaign["task_order"] == task_ids


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
