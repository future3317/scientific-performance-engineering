from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.real_intervention_v4 import matched_sessions, run_all
from experiments.real_intervention_v4.outcomes import atomic_json, canonical_digest, record_event


_FEATURES = {
    "batch": 1, "horizon": 1, "conv2d_module_count": 2,
    "input_channels": 3, "input_height": 16, "input_width": 16,
}
_RUNTIME = {
    "torch_version": "2.6.0", "torch_cuda_version": "12.6",
    "driver": {"driver_version": "535.247.01"}, "packages": {},
}


def _paired_row(pair_id, family_id, split):
    identity = {
        "runtime_gpu_uuid": "GPU-test", "inputs_sha256": "a" * 64,
        "weights_sha256": "b" * 64, "source_commit": "source-test",
        "config_digest": "c" * 64, "context_digest": "d" * 64,
    }
    runtime = {
        "device_uuid": "GPU-test", "device_name": "test GPU",
        "torch_version": "2.6.0", "torch_cuda_version": "12.6",
        "driver": {"driver_version": "535.247.01"}, "packages": {},
    }
    receipt = {
        **identity, "raw_receipt_sha256": "e" * 64,
        "runtime_identity": runtime, "arm_execution": "COMPLETE",
        "scientific_correctness": "PASS",
        "timing": {"cold_seconds": 2.0, "steady_seconds": 1.0},
    }
    treatment = {**receipt, "raw_receipt_sha256": "f" * 64,
                 "timing": {"cold_seconds": 1.8, "steady_seconds": 0.9}}
    return {
        "pair_id": pair_id, "family_id": family_id, "split": split,
        "context_id": f"{pair_id}-context", "features": dict(_FEATURES),
        "context": {
            **_FEATURES,
            "qualification": {"eligibility": "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY",
                              "conv2d_count": 2},
            "actual_input_shapes": {"input": [1, 3, 16, 16]},
        },
        "classification": {
            "arm_execution": "COMPLETE", "scientific_correctness": "PASS",
            "paired_comparability": "MATCHED", "policy_evidence": "ELIGIBLE",
        },
        "S0": receipt, "S1": treatment, "terminal_pair": True,
    }


def _freeze_inputs():
    development = _paired_row("dev-1", "family-dev", "development")
    calibration = _paired_row("cal-1", "family-cal", "calibration")
    # Keep this caller regression fast and model-free: incomplete pairs do not
    # enter synthesis/calibration, while the production route still runs.
    for row in (development, calibration):
        row["classification"]["paired_comparability"] = "INCOMPLETE"
    planned = [
        {"pair_id": "dev-1", "family_id": "family-dev", "split": "development",
         "context_id": development["context_id"], "context": development["context"]},
        {"pair_id": "cal-1", "family_id": "family-cal", "split": "calibration",
         "context_id": calibration["context_id"], "context": calibration["context"]},
        {"pair_id": "final-1", "family_id": "family-final", "split": "final",
         "context_id": "final-context", "context": development["context"]},
    ]
    config = {
        "study_id": "v4-runner-test", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "bootstrap_seed": 101, "bootstrap_repetitions": 20,
        "effect_contract": {
            "primary_estimand": "family_equal",
            "minimum_material_log_benefit": 0.05,
            "target_95pct_family_ci_half_width_log_benefit": 0.1,
        },
        "acre": {
            "minimum_calibration_deploy_families": 10,
            "minimum_calibration_support_families": 5,
            "minimum_calibration_non_support_families": 5,
        },
    }
    manifest = {
        "manifest_digest": "manifest-test", "source_commit": "source-test",
        "runtime_identity": _RUNTIME, "planned_pairs": planned,
    }
    return config, manifest, [development, calibration]


def test_qualification_cache_setup_is_inside_the_supervised_deadline(tmp_path, monkeypatch):
    attempt_id = "qual-tv-resnet18-B1-S0REF"
    attempt_dir = tmp_path / "raw" / attempt_id
    cache_root = str((attempt_dir / "cache").resolve())
    details = {
        "identity": {"study_id": "test-study"},
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_dir),
        "cache_scope_path": cache_root,
        "cache_environment": {"TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/inductor"},
    }
    monkeypatch.setattr(run_all, "_qualification_spec", lambda *args, **kwargs: details)
    monkeypatch.setattr(run_all, "wait_for_selected_gpu", lambda *args, **kwargs: True)
    monkeypatch.setattr(run_all.time, "monotonic", lambda: 1.0)
    recovered = {
        "arm_execution": "COMPLETE", "receipt_verified": True,
        "worker_result": {"status": "VALID", "eligibility": "S0_REFERENCE_COMPLETE",
                          "arm_execution": "COMPLETE"},
    }
    monkeypatch.setattr(run_all, "recover_attempt", lambda *args, **kwargs: recovered)
    observed = {}

    def supervised_execute(**kwargs):
        observed.update(kwargs)
        return {"parent_wall_seconds": 0.1, "within_deadline_plus_grace": True}

    monkeypatch.setattr(run_all, "execute_arm", supervised_execute)
    row = run_all._run_qualification_phase(
        {"candidate_id": "torchvision:resnet18", "family_id": "resnet",
         "source": "torchvision", "model": "resnet18"},
        1, "s0-reference",
        config={
            "study_id": "test-study", "workload_dependency_overlay": "/unused/overlay",
            "arm_timeout_seconds": 5, "cleanup_grace_seconds": 1,
            "term_grace_seconds": 0.2, "watchdog_heartbeat_seconds": 1,
            "gpu_sample_seconds": 1,
            "gpu_selection": {"idle_memory_limit_mib": 1024},
        },
        root=tmp_path, gpu_uuid="GPU-test", wall_deadline=10,
        poll_seconds=0.1, repo=tmp_path,
    )

    assert observed["cache_setup"] == {"mode": "EMPTY"}
    assert observed["study_deadline_monotonic"] == 10
    assert row["arm_execution"] == "COMPLETE"


def test_freeze_policies_routes_planned_final_context_with_selected_gpu(tmp_path):
    config, manifest, rows = _freeze_inputs()

    freeze = run_all._freeze_policies(
        rows, manifest, config, tmp_path, "source-test", gpu_uuid="GPU-test",
    )
    status = json.loads((tmp_path / "ACRE_STATUS.json").read_text(encoding="utf-8"))

    assert status["selected_gpu_uuid"] == "GPU-test"
    assert status["engine_route_invoked"] is True
    assert status["planned_final_context_count"] == 1
    assert freeze["policies"]["ORIGINAL_ACRE"]["route_decisions"]["final-context"]
    assert freeze["final_outcomes_read"] is False


def test_production_run_passes_selected_gpu_through_real_policy_freeze_and_route(
    tmp_path, monkeypatch,
):
    repo = Path(__file__).resolve().parents[3]
    config_path = repo / "experiments" / "real_intervention_v4" / "config" / "PRIMARY_LAYOUT_STUDY.json"
    config, manifest, rows = _freeze_inputs()
    config.update({
        "wall_cap_seconds": 60, "torchbench_root": "unused",
        "torchbench_commit": "torchbench-test", "batches": [1],
        "gpu_selection": {"preferred_uuids": [], "scan_interval_seconds": 1},
        "split": {
            "minimum_total_families_for_confirmatory": 23,
            "minimum_development_families": 8,
            "minimum_calibration_families": 10,
            "minimum_final_families": 5,
        },
        "acre": {**config["acre"], "minimum_development_families": 8},
        "engineering_shakedown": {"minimum_model_level_pairs": 12},
        "disk_cap_bytes": 1_000_000,
        "resource_accounting": {"full_scan_every_pairs": 16},
    })
    manifest.update({
        "registered_families": [], "eligible_families": [],
        "split_family_counts": {}, "eligible_family_counts_by_split": {},
    })

    monkeypatch.setattr(run_all, "read_config", lambda _path: config)
    monkeypatch.setattr(run_all, "_validated_results_root", lambda path: Path(path).resolve())
    monkeypatch.setattr(run_all, "_assert_server_contract", lambda *_args: (repo, "source-test"))
    monkeypatch.setattr(run_all, "_validate_split_freeze", lambda *_args: "split-freeze-test")
    monkeypatch.setattr(run_all, "_write_cpu_suite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_all, "_write_v3_audit", lambda *_args: {})
    monkeypatch.setattr(run_all, "select_study_gpu", lambda *_args, **_kwargs: "GPU-test")
    monkeypatch.setattr(run_all, "_source_environment", lambda *_args: {})
    monkeypatch.setattr(run_all, "_resolve_server_path", lambda _path: tmp_path)
    from experiments.real_intervention_v4 import source_registry
    monkeypatch.setattr(source_registry, "candidate_universe", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(run_all, "_freeze_registered_family_split", lambda **_kwargs: {})
    monkeypatch.setattr(run_all, "_qualification_rows", lambda *_args: [])
    monkeypatch.setattr(run_all, "_write_qualification_matrix", lambda *_args: None)
    monkeypatch.setattr(run_all, "_load_or_build_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        run_all, "_shakedown_roster",
        lambda *_args: [{"pair_id": f"shake-{index}", "split": "development"}
                        for index in range(12)],
    )
    monkeypatch.setattr(run_all, "_update_resource_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_all, "run_pair", lambda *_args, **_kwargs: {
        "result_storage_delta_bytes_this_invocation": 0,
    })
    monkeypatch.setattr(run_all, "_write_shakedown", lambda *_args: {
        "passes": True, "planned_opportunities": 12, "closed_opportunities": 12,
    })
    monkeypatch.setattr(run_all, "_run_formal_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "_write_primary_pairs", lambda *_args: rows)
    from experiments.real_intervention_v4 import generate_report
    monkeypatch.setattr(generate_report, "write_final_outputs", lambda **_kwargs: {
        "closeout_complete": True, "confirmatory_analysis_complete": False,
    })

    result = run_all.run(
        config_path, tmp_path / config["study_id"], resume_safe=False,
    )
    status = json.loads(
        (tmp_path / config["study_id"] / "ACRE_STATUS.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (tmp_path / config["study_id"] / "POLICY_FREEZE.json").read_text(encoding="utf-8")
    )

    assert result["selected_gpu_uuid"] == "GPU-test"
    assert status["selected_gpu_uuid"] == "GPU-test"
    assert status["engine_route_invoked"] is True
    assert freeze["policies"]["ORIGINAL_ACRE"]["route_decisions"]["final-context"]


def test_primary_pair_ledger_rebuilds_stale_valid_cache_from_invalid_raw_receipt(tmp_path):
    pair = {
        "pair_id": "pair-stale-valid", "family_id": "family-a",
        "split": "development", "context_id": "context-a",
        "order": ["S0", "S1"], "context": {"batch": 1},
    }
    config = {
        "study_id": "v4-stale-pair-test",
        "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
    }
    manifest = {
        "manifest_digest": "manifest-test", "source_commit": "source-test",
        "planned_pairs": [pair], "registered_families": [],
        "eligible_families": [], "split_family_counts": {},
        "eligible_family_counts_by_split": {},
    }
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    cached = _paired_row(pair["pair_id"], pair["family_id"], pair["split"])
    cached.update({
        "study_id": config["study_id"],
        "intervention_id": config["intervention_id"], "purpose": "PRIMARY",
        "manifest_digest": manifest["manifest_digest"],
        "planned_digest": canonical_digest({
            "manifest_digest": manifest["manifest_digest"], "pair": pair,
            "purpose": "PRIMARY", "selected_gpu_uuid": "GPU-test",
        }),
        "selected_gpu_uuid": "GPU-test",
    })
    atomic_json(pair_dir / f"{pair['pair_id']}.json", cached)

    attempt_id = f"primary-{pair['pair_id']}-S0"
    attempt_dir = tmp_path / "raw" / attempt_id
    spec = {"arm": "S0", "context": pair["context"]}
    atomic_json(attempt_dir / "spec.json", spec)
    identity = matched_sessions._arm_identity(
        pair={
            **pair, "study_id": config["study_id"],
            "intervention_id": config["intervention_id"],
            "attempt_prefix": f"primary-{pair['pair_id']}",
        },
        arm="S0", source_commit="source-test",
        config_digest=canonical_digest(config), gpu_uuid="GPU-test",
    )
    identity = {
        **identity, "attempt_id": attempt_id, "spec_digest": canonical_digest(spec),
    }
    record_event(attempt_dir, "intent", {"identity": identity, "command": ["worker"]})
    record_event(attempt_dir, "started", {
        "identity": identity, "pid": 123, "process_group_id": 123,
    })
    receipt = {
        "identity": identity, "arm_execution": "COMPLETE",
        "scientific_correctness": "PASS", "process_returncode": 0,
    }
    atomic_json(attempt_dir / "receipt.json", receipt)
    receipt_digest = hashlib.sha256((attempt_dir / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt_dir, "terminal", {
        "identity": identity, "arm_execution": "COMPLETE",
        "process_returncode": 0, "receipt_sha256": receipt_digest,
    })
    (attempt_dir / "receipt.json").write_text("{corrupt", encoding="utf-8")

    rows = run_all._write_primary_pairs(tmp_path, manifest, config, "GPU-test")
    refreshed = json.loads((pair_dir / f"{pair['pair_id']}.json").read_text(encoding="utf-8"))

    assert rows[0]["S0"]["arm_execution"] == "UNKNOWN"
    assert rows[0]["classification"]["arm_execution"] == "UNKNOWN"
    assert rows[0]["terminal_pair"] is False
    assert refreshed["classification"]["arm_execution"] == "UNKNOWN"
    assert json.loads((tmp_path / "PAIRS.jsonl").read_text(encoding="utf-8").splitlines()[0]) == rows[0]
    assert run_all._dev_cal_closed(manifest, rows) is False

    closed_gate = run_all._gate(
        manifest, rows, {"passes": True},
        {
            "split": {
                "minimum_total_families_for_confirmatory": 0,
                "minimum_development_families": 0,
                "minimum_calibration_families": 0,
                "minimum_final_families": 0,
            },
            "acre": {
                "minimum_development_families": 0,
                "minimum_calibration_deploy_families": 0,
                "minimum_calibration_support_families": 0,
                "minimum_calibration_non_support_families": 0,
            },
        },
        False,
        {"acre_bridge": {
            "status": "FITTED", "canonical_support_certificate_created": True,
            "support_evidence": {"certificate_scope_bound": True},
        }},
    )
    assert closed_gate["final_opened"] is False
    assert closed_gate["checks"]["development_calibration_closed"] is False


def test_final_gate_requires_five_independent_support_and_non_support_families():
    config = {
        "split": {
            "minimum_total_families_for_confirmatory": 23,
            "minimum_development_families": 8,
            "minimum_calibration_families": 10,
            "minimum_final_families": 5,
        },
        "acre": {
            "minimum_development_families": 8,
            "minimum_calibration_deploy_families": 10,
            "minimum_calibration_support_families": 5,
            "minimum_calibration_non_support_families": 5,
        },
    }
    rows = [
        {"family_id": f"dev-{i}", "split": "development",
         "classification": {"policy_evidence": "ELIGIBLE"}}
        for i in range(8)
    ] + [
        {"family_id": f"cal-{i}", "split": "calibration",
         "classification": {"policy_evidence": "ELIGIBLE"}}
        for i in range(10)
    ]
    manifest = {
        "registered_families": [f"family-{i}" for i in range(23)],
        "eligible_families": [f"family-{i}" for i in range(23)],
        "split_family_counts": {"development": 8, "calibration": 10, "final": 5},
        "eligible_family_counts_by_split": {"development": 8, "calibration": 10, "final": 5},
    }
    shakedown = {"passes": True}
    bridge = {
        "status": "FITTED", "canonical_support_certificate_created": True,
        "support_evidence": {
            "certificate_scope_bound": True,
            "support_families": [f"cal-{i}" for i in range(5)],
            "non_support_families": [f"cal-{i}" for i in range(5, 10)],
        },
    }
    freeze = {"acre_bridge": bridge}

    passing = run_all._gate(manifest, rows, shakedown, config, True, freeze)
    bridge["support_evidence"]["non_support_families"] = ["cal-5"]
    failing = run_all._gate(manifest, rows, shakedown, config, True, freeze)

    assert passing["final_opened"] is True
    assert passing["calibration_support_families"] == 5
    assert passing["calibration_non_support_families"] == 5
    assert failing["final_opened"] is True
    assert failing["acre_certificate_qualified"] is False
    assert failing["checks"]["calibration_non_support_families_minimum"] is False


def test_primary_final_does_not_misrepresent_an_unfitted_acre_certificate():
    config = {
        "split": {
            "minimum_total_families_for_confirmatory": 23,
            "minimum_development_families": 8,
            "minimum_calibration_families": 10,
            "minimum_final_families": 5,
        },
        "acre": {
            "minimum_development_families": 8,
            "minimum_calibration_deploy_families": 10,
            "minimum_calibration_support_families": 5,
            "minimum_calibration_non_support_families": 5,
        },
    }
    rows = [
        {"family_id": f"dev-{i}", "split": "development",
         "classification": {"policy_evidence": "ELIGIBLE"}}
        for i in range(8)
    ] + [
        {"family_id": f"cal-{i}", "split": "calibration",
         "classification": {"policy_evidence": "ELIGIBLE"}}
        for i in range(10)
    ]
    manifest = {
        "registered_families": [f"family-{i}" for i in range(23)],
        "eligible_families": [f"family-{i}" for i in range(23)],
        "split_family_counts": {"development": 8, "calibration": 10, "final": 5},
        "eligible_family_counts_by_split": {"development": 8, "calibration": 10, "final": 5},
    }

    gate = run_all._gate(
        manifest, rows, {"passes": True}, config, True,
        {"acre_bridge": {"status": "ACRE_NOT_FITTED",
                          "canonical_support_certificate_created": False}},
    )

    assert gate["final_opened"] is True
    assert gate["acre_certificate_qualified"] is False
    assert gate["acre_certificate_required_for_primary_final"] is False


def test_power_feasibility_is_frozen_from_performance_blind_roster(tmp_path):
    config = {
        "study_id": "v4-feasibility-test",
        "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "split": {
            "minimum_total_families_for_confirmatory": 23,
            "minimum_development_families": 8,
            "minimum_calibration_families": 10,
            "minimum_final_families": 5,
        },
        "acre": {
            "minimum_calibration_support_families": 5,
            "minimum_calibration_non_support_families": 5,
        },
        "effect_contract": {
            "minimum_material_log_benefit": 0.05,
            "target_95pct_family_ci_half_width_log_benefit": 0.1,
        },
    }
    manifest = {
        "manifest_digest": "manifest", "registered_families": [str(i) for i in range(23)],
        "eligible_family_counts_by_split": {"development": 8, "calibration": 10, "final": 5},
        "planned_pairs": [
            {"split": split} for split, count in (("development", 8),
                                                   ("calibration", 10), ("final", 5))
            for _ in range(count)
        ],
    }

    result = run_all._write_power_feasibility(
        tmp_path, manifest, [{"performance_timing_recorded": False}], config, "source",
    )

    assert result["confirmatory_family_count_feasible"] is True
    assert result["planned_primary_pairs_total"] == 23
    assert result["planned_primary_arm_invocations"] == 46
    assert result["acre_certificate_sample_feasible"] is False
    assert result["canonical_acre_promotion_gate"]["minimum_all_success_promotion_families"] == 36
    assert result["canonical_acre_promotion_gate"]["minimum_calibration_families_for_certificate"] == 42
    assert result["statistical_power"] == "NOT_ESTIMABLE_BEFORE_TREATMENT_EFFECTS"
    with pytest.raises(ValueError, match="before performance-effect timing"):
        run_all._write_power_feasibility(
            tmp_path / "after-timing", manifest,
            [{"performance_timing_recorded": True}], config, "source",
        )


def test_successful_cpu_suite_is_reused_on_resume(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "2 passed", "")

    monkeypatch.setattr(run_all.subprocess, "run", run)

    assert run_all._write_cpu_suite(
        tmp_path, source_commit="source", config_digest="config",
    )
    assert run_all._write_cpu_suite(
        tmp_path, source_commit="source", config_digest="config",
    )

    assert len(calls) == 1


def test_cpu_suite_identity_drift_is_not_retested_in_the_same_root(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "2 passed", "")

    monkeypatch.setattr(run_all.subprocess, "run", run)
    run_all._write_cpu_suite(tmp_path, source_commit="source", config_digest="config")

    with pytest.raises(ValueError, match="identity or log differs"):
        run_all._write_cpu_suite(tmp_path, source_commit="other", config_digest="config")

    assert len(calls) == 1


def test_failed_cpu_suite_is_not_silently_retried_on_resume(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "1 failed", "")

    monkeypatch.setattr(run_all.subprocess, "run", run)
    assert not run_all._write_cpu_suite(
        tmp_path, source_commit="source", config_digest="config",
    )
    assert not run_all._write_cpu_suite(
        tmp_path, source_commit="source", config_digest="config",
    )

    assert len(calls) == 1


def test_v3_audit_is_written_before_failed_cpu_fault_suite(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    config = {"study_id": "audit-before-cpu-failure"}
    config_path = repo / "experiments" / "real_intervention_v4" / "config" / "PRIMARY_LAYOUT_STUDY.json"
    order = []
    monkeypatch.setattr(run_all, "read_config", lambda _path: config)
    monkeypatch.setattr(run_all, "_validated_results_root", lambda path: Path(path).resolve())
    monkeypatch.setattr(run_all, "_assert_server_contract", lambda *_args: (repo, "source-test"))
    monkeypatch.setattr(run_all, "_validate_split_freeze", lambda *_args: "split-freeze-test")

    def write_audit(_config, _root):
        order.append("audit")
        return {"closure_check": {"matches_expected": True}}

    def failed_suite(*_args, **_kwargs):
        order.append("fault-suite")
        return False

    monkeypatch.setattr(run_all, "_write_v3_audit", write_audit)
    monkeypatch.setattr(run_all, "_write_cpu_suite", failed_suite)

    result = run_all.run(config_path, tmp_path / config["study_id"], resume_safe=False)
    root = tmp_path / config["study_id"]

    assert order == ["audit", "fault-suite"]
    assert result["status"] == "PLATFORM_BLOCKED"
    assert (root / "V3_AUDIT_SUMMARY.json").is_file()
    assert json.loads((root / "V3_AUDIT_SUMMARY.json").read_text(encoding="utf-8"))[
        "closure_check"]["matches_expected"] is True


def test_resource_progress_uses_pair_deltas_between_periodic_full_scans(tmp_path):
    config = {"disk_cap_bytes": 100_000,
              "resource_accounting": {"full_scan_every_pairs": 4}}
    (tmp_path / "baseline.bin").write_bytes(b"baseline")

    baseline = run_all._update_resource_progress(
        tmp_path, config, pair_id="baseline", split="PRE_TIMING",
        pair_index=0, attempt_storage_delta=0,
    )
    estimated = run_all._update_resource_progress(
        tmp_path, config, pair_id="pair-1", split="development",
        pair_index=1, attempt_storage_delta=128,
    )

    assert baseline["status"] == "COMPLETE"
    assert estimated["status"] == "ESTIMATED"
    assert estimated["complete"] is False
    assert estimated["estimated_root_bytes"] == baseline["last_complete_root_scan_bytes"] + 128
    assert estimated["unreconciled_pair_attempt_growth_bytes"] == 128

    reconciled = run_all._update_resource_progress(
        tmp_path, config, pair_id="pair-4", split="development",
        pair_index=4, attempt_storage_delta=64,
    )
    assert reconciled["status"] == "COMPLETE"
    assert reconciled["complete"] is True
    assert reconciled["unreconciled_pair_attempt_growth_bytes"] == 0


def test_unknown_result_root_storage_stops_new_work(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "cache_usage", lambda _root: {
        "bytes": None, "observed_bytes_lower_bound": 17,
        "complete": False, "status": "UNKNOWN",
    })
    with pytest.raises(RuntimeError, match="RESULT_ROOT_STORAGE_UNKNOWN"):
        run_all._update_resource_progress(
            tmp_path, {"disk_cap_bytes": 100}, pair_id="baseline",
            split="PRE_TIMING", pair_index=0,
            attempt_storage_delta=0, force_full_scan=True,
        )
