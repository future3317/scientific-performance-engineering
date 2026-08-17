from __future__ import annotations

import os
from pathlib import Path

from benchmark.formal import aggregate, attest
from benchmark.formal.run_campaign import _trial_compiler_cache, post_task_update
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
    with _trial_compiler_cache(tmp_path / "trial") as manifest:
        assert manifest["policy"] == "trial-scoped"
        assert Path(manifest["torchinductor_cache_dir"]).is_dir()
        assert Path(manifest["triton_cache_dir"]).is_dir()
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == old_torch
    assert os.environ.get("TRITON_CACHE_DIR") == old_triton
