from __future__ import annotations

from pathlib import Path

import pytest

from experiments.real_intervention_v4 import matched_sessions
from experiments.real_intervention_v4.outcomes import atomic_json, canonical_digest, record_event


def _run_inputs(tmp_path: Path):
    pair = {
        "study_id": "v4-test",
        "pair_id": "pair-001",
        "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "family_id": "family-a",
        "split": "dev",
        "context": {"qualification": {"eligibility": "NOT_QUALIFIED"}},
        "attempt_prefix": "primary-pair-001",
    }
    config = {
        "correctness": {"rtol": 1e-4, "atol": 1e-5},
        "arm_timeout_seconds": 10.0, "cleanup_grace_seconds": 1.0,
        "term_grace_seconds": 0.1, "watchdog_heartbeat_seconds": 1.0,
        "gpu_sample_seconds": 0.1,
    }
    return pair, config, tmp_path


def test_study_wall_deadline_prevents_supervisor_launch(tmp_path, monkeypatch):
    pair, config, root = _run_inputs(tmp_path)
    clock = iter((1.0, 2.0))
    monkeypatch.setattr(matched_sessions.time, "monotonic", lambda: next(clock))

    monkeypatch.setattr(
        matched_sessions, "execute_arm",
        lambda **_: pytest.fail("worker supervisor must not launch after the study deadline"),
    )

    result = matched_sessions._run_arm(
        pair=pair, arm="S0", results_root=root, source_commit="source",
        config_digest="config", gpu_uuid="GPU-test", config=config,
        wall_deadline=1.5,
    )

    assert result["arm_execution"] == "NOT_STARTED"
    assert result["first_failure_stage"] == "STUDY_WALL_CAP"


def test_concurrent_cache_initialization_is_unresolved_not_candidate_failure(tmp_path, monkeypatch):
    pair, config, root = _run_inputs(tmp_path)
    monkeypatch.setattr(matched_sessions.time, "monotonic", lambda: 1.0)

    monkeypatch.setattr(matched_sessions, "execute_arm", lambda **_: {
        "arm_execution": "NOT_STARTED",
        "reason": "attempt cache initialization is in progress",
    })
    result = matched_sessions._run_arm(
        pair=pair, arm="S0", results_root=root, source_commit="source",
        config_digest="config", gpu_uuid="GPU-test", config=config,
        wall_deadline=10.0,
    )

    attempt_dir = root / "raw" / "primary-pair-001-S0"
    assert result["arm_execution"] == "UNKNOWN"
    assert result["first_failure_stage"] == "ATTEMPT_CACHE_LOCKED"
    assert not (attempt_dir / "receipt.json").exists()
    assert not (attempt_dir / "events").exists()


def test_cache_only_attempt_without_intent_recovers_as_not_started(tmp_path):
    planned = {
        "pair_id": "pair-001", "family_id": "family-a", "split": "dev",
        "order": ["S0", "S1"], "context": {},
    }
    config = {"study_id": "v4-test", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1"}
    attempt_dir = tmp_path / "raw" / "primary-pair-001-S0"
    cache_root = attempt_dir / "cache"
    cache_root.mkdir(parents=True)
    matched_sessions.atomic_json(attempt_dir / "CACHE_SCOPE.json", {
        "root": str(cache_root.resolve()), "environment": {},
    })

    observation = matched_sessions.load_pair_observation(
        planned_pair=planned, manifest_digest="manifest", source_commit="source",
        config=config, results_root=tmp_path, gpu_uuid="GPU-test",
    )

    assert observation["S0"]["arm_execution"] == "NOT_STARTED"
    assert observation["S1"]["arm_execution"] == "NOT_STARTED"
    assert observation["terminal_pair"] is False


def test_started_unknown_pair_is_unresolved_and_never_replayed(tmp_path, monkeypatch):
    planned = {
        "pair_id": "pair-unknown", "family_id": "family-a", "split": "dev",
        "order": ["S0", "S1"], "context": {"qualification": {}},
    }
    config = {
        "study_id": "v4-test", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "correctness": {"rtol": 1e-4, "atol": 1e-5},
        "gpu_selection": {"idle_memory_limit_mib": 100},
    }
    monkeypatch.setattr(matched_sessions, "wait_for_selected_gpu", lambda *_, **__: True)
    calls = []

    def start_unknown(**kwargs):
        calls.append(kwargs["arm"])
        pair = kwargs["pair"]
        arm = kwargs["arm"]
        attempt_id = f"{pair['attempt_prefix']}-{arm}"
        folder = tmp_path / "raw" / attempt_id
        spec = {"arm": arm, "context": pair["context"]}
        atomic_json(folder / "spec.json", spec)
        identity = matched_sessions._arm_identity(
            pair=pair, arm=arm, source_commit="source",
            config_digest=canonical_digest(config), gpu_uuid="GPU-test",
        )
        identity = {**identity, "attempt_id": attempt_id, "spec_digest": canonical_digest(spec)}
        record_event(folder, "intent", {"identity": identity, "command": ["injected-worker"]})
        record_event(folder, "started", {"identity": identity, "pid": 9001,
                                          "process_group_id": 9001})
        return {"arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN"}

    monkeypatch.setattr(matched_sessions, "_run_arm", start_unknown)
    first = matched_sessions.run_pair(
        planned_pair=planned, manifest_digest="manifest", source_commit="source",
        config=config, results_root=tmp_path, gpu_uuid="GPU-test", wall_deadline=float("inf"),
    )
    second = matched_sessions.run_pair(
        planned_pair=planned, manifest_digest="manifest", source_commit="source",
        config=config, results_root=tmp_path, gpu_uuid="GPU-test", wall_deadline=float("inf"),
    )

    assert calls == ["S0"]
    assert first["S0"]["arm_execution"] == second["S0"]["arm_execution"] == "UNKNOWN"
    assert second["S1"]["arm_execution"] == "NOT_STARTED"
    assert second["terminal_pair"] is False
    assert second["classification"]["arm_execution"] == "UNKNOWN"
