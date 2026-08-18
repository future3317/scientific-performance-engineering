from __future__ import annotations

import os
import json
from pathlib import Path

from benchmark.formal import aggregate, attest
from benchmark.formal.run_campaign import _build_required_experiment_executor, _cleanup_process_group, _resume_stream_prefix, _trial_compiler_cache, post_task_update
from benchmark.harness import conditions
from core.cost import BudgetedContextRenderer


def test_readiness_withholds_incomplete_cells() -> None:
    required = [("T1", "outer-000", "reset", condition) for condition in ("A", "B", "C", "D")]
    records = [
        {"task_id": "T1", "outer_trial_id": "outer-000", "context_mode": "reset", "condition": "A", "execution_validity": "valid", "task_outcome": "pass"},
        {"task_id": "T1", "outer_trial_id": "outer-000", "context_mode": "reset", "condition": "B", "execution_validity": "resource_blocked", "task_outcome": "error"},
    ]
    readiness = aggregate.aggregate_trials(records, required_cells=required)["readiness"]
    assert readiness["missing_cells"]
    assert readiness["resource_blocked_count"] == 1
    assert readiness["efficacy_aggregate_status"] == "withheld"


def test_protocol_valid_task_failure_remains_efficacy_eligible() -> None:
    cell = [("T1", "outer-000", "reset", "A")]
    record = {
        "task_id": "T1", "outer_trial_id": "outer-000", "context_mode": "reset", "condition": "A",
        "execution_validity": "valid", "task_outcome": "fail", "efficacy_eligible": True,
        "validity": "valid", "score": {"task_score": 0.0},
    }
    summary = aggregate.aggregate_trials([record], required_cells=cell)
    assert summary["readiness"]["efficacy_aggregate_status"] == "available"
    assert summary["readiness"]["outcome_counts"] == {"fail": 1}


def test_withheld_readiness_hides_paired_effect_values() -> None:
    records = [
        {"task_id": "T1", "outer_trial_id": "outer-000", "context_mode": "reset", "condition": "B", "execution_validity": "valid", "efficacy_eligible": True, "score": 0.2},
        {"task_id": "T1", "outer_trial_id": "outer-000", "context_mode": "reset", "condition": "D", "execution_validity": "resource_blocked", "efficacy_eligible": False, "score": 0.4},
    ]
    summary = aggregate.aggregate_trials(records, required_cells=[("T1", "outer-000", "reset", "B"), ("T1", "outer-000", "reset", "D")])
    assert summary["readiness"]["efficacy_aggregate_status"] == "withheld"
    assert summary["paired_effects"]["D-B"]["status"] == "withheld"


def test_context_renderer_drops_payload_and_reports_only_identity() -> None:
    payload = {"context": {"domain": "x"}, "rule_views": [{"rule_id": f"R{i}", "payload": "x" * 1000} for i in range(4)]}
    rendered = BudgetedContextRenderer(180).render(payload)
    assert rendered["token_cost"] <= 180
    assert rendered["renderer"]["dropped_count"] > 0
    assert "payload" not in rendered["renderer"]


def test_protocol_invalid_does_not_mutate_condition_store(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    from scripts.render_skill_view import render_skill_view

    render_skill_view(skill, bundle)
    store = tmp_path / "store"
    conditions.materialize_condition("C", bundle, store, context_mode="reset")
    before = conditions.store_digest(store)
    transition = post_task_update(
        condition="C",
        store=store,
        task_id="T1",
        result={"verdict": "pass", "task_id": "T1"},
        scored={"task_score": 0.5},
        core_repo=repo_root,
        out_dir=tmp_path,
        execution_validity="invalid",
    )
    assert transition["transition"] == "protocol_invalid_no_mutation"
    assert transition["pre_store_digest"] == before == transition["post_store_digest"]


def test_trial_compiler_cache_restores_environment(tmp_path: Path) -> None:
    old_torch = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    old_triton = os.environ.get("TRITON_CACHE_DIR")
    with _trial_compiler_cache(tmp_path / "trial", "candidate") as manifest:
        assert manifest["policy"] == "verifier-invocation-scoped"
        assert Path(manifest["torchinductor_cache_dir"]).is_dir()
        assert Path(manifest["triton_cache_dir"]).is_dir()
    with _trial_compiler_cache(tmp_path / "trial", "control") as other:
        assert other["torchinductor_cache_dir"] != manifest["torchinductor_cache_dir"]
        assert other["triton_cache_dir"] != manifest["triton_cache_dir"]
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == old_torch
    assert os.environ.get("TRITON_CACHE_DIR") == old_triton


def test_required_experiment_timeout_is_resource_blocked(tmp_path: Path) -> None:
    executor = _build_required_experiment_executor(
        '"{python}" -c "import time; time.sleep(1)"'.format(python=os.sys.executable),
        tmp_path,
        timeout=0.01,
    )
    result = executor({"experiment_id": "E1", "required_arms": ["00"]})
    assert result["status"] == "resource_blocked"


def test_verifier_process_group_cleanup_reaps_grandchild() -> None:
    import subprocess
    import sys

    child_code = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); raise SystemExit(3)"
    process = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)
    process.wait(timeout=5)
    assert process.returncode == 3
    assert _cleanup_process_group(process) == []


def test_resume_uses_final_digest_and_persists_stream_block(tmp_path: Path) -> None:
    from scripts.render_skill_view import render_skill_view

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    render_skill_view(skill, bundle)
    store = tmp_path / "stream" / "condition-store"
    conditions.materialize_condition("C", bundle, store, context_mode="reset")
    final_digest = conditions.store_digest(store)
    stream = store.parent
    for index, task_id in enumerate(("T1", "T2")):
        task_dir = stream / task_id
        task_dir.mkdir()
        (task_dir / "trial.json").write_text(
            json.dumps({
                "task_id": task_id,
                "transition": {"status": "raw_experience_capture", "post_store_digest": "stale" if index == 0 else final_digest},
                "attestation_ok": True,
            }), encoding="utf-8",
        )
    blocked = set()
    prefix, records, _ = _resume_stream_prefix(
        stream, [("phase", "T1"), ("phase", "T2"), ("phase", "T3")],
        {"T1": 0, "T2": 1, "T3": 2}, "C-reset", blocked,
    )
    assert prefix == 2 and set(records) == {("C-reset", "T1"), ("C-reset", "T2")}
    (stream / "T2" / "trial.json").write_text(
        json.dumps({
            "task_id": "T2",
            "transition": {"status": "state_mutation_error", "post_store_digest": final_digest},
            "attestation_ok": False,
        }), encoding="utf-8",
    )
    _resume_stream_prefix(stream, [("phase", "T1"), ("phase", "T2")], {"T1": 0, "T2": 1}, "C-reset", blocked)
    assert "C-reset" in blocked
