#!/usr/bin/env python3
"""Contract tests for the formal v1.0-20 driver and paired aggregation."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.formal import aggregate, attest, budget, run_campaign, schedule
from benchmark.harness import conditions


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    split_path = repo_root / "benchmark" / "split" / "sequential.yaml"
    ordered = schedule.task_order(split_path)
    assert len(ordered) == 20, len(ordered)
    plan = schedule.build_schedule(split_path, outer_trials=3)
    assert len(plan) == 20 * 4 * 3
    assert {item["condition"] for item in plan} == {"A", "B", "C", "D"}
    assert {item["context_mode"] for item in plan} == {"reset"}
    assert {item["outer_trial_id"] for item in plan} == {"outer-000", "outer-001", "outer-002"}

    carry_plan = schedule.build_schedule(split_path, conditions=("B", "D"), context_modes=("reset", "carry"), outer_trials=1)
    assert len(carry_plan) == 20 * 2 * 2
    assert {item["context_mode"] for item in carry_plan} == {"reset", "carry"}

    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-1",
        "benchmark_revision": "1234567",
        "skill_view_digest": "a" * 64,
        "task_manifest_digest": "b" * 64,
        "agent_model_id": "test-agent",
        "agent_config": {},
        "condition": "D",
        "context_mode": "reset",
        "task_order": [task_id for _, task_id in ordered],
        "outer_trial_id": "outer-000",
        "budgets": budget.parse_budget(None).as_dict(),
        "hardware_fingerprint": {},
        "software_fingerprint": {},
        "torch_version": None,
        "cuda_version": None,
    }
    assert attest.validate_experiment(manifest) == []

    records = []
    for condition, score, speedup in (("B", 0.40, 1.10), ("C", 0.50, 1.20), ("D", 0.70, 1.50)):
        records.append({"task_id": "T1", "family": "compiler", "lineage_id": "L1", "condition": condition, "context_mode": "reset", "outer_trial_id": "outer-000", "score": score, "median_speedup": speedup, "kind": "positive"})
    summary = aggregate.aggregate_trials(records)
    assert math.isclose(summary["paired_effects"]["D-C"]["estimate"], 0.20)
    assert math.isclose(summary["paired_effects"]["D-B"]["estimate"], 0.30)
    assert summary["family_stratified"]["compiler"]["D-C"]["n"] == 1
    assert math.isclose(summary["paired_log_speedups"]["D-C"]["estimate"], math.log(1.5) - math.log(1.2))
    assert math.isclose(summary["task_score_effects"]["D-C"]["estimate"], 0.20)
    assert summary["hierarchical_effects"]["D-C"]["n"] == 1
    invalid_summary = aggregate.aggregate_trials(records + [{"task_id": "T1", "family": "compiler", "condition": "D", "outer_trial_id": "outer-001", "score": 1.0, "validity": "invalid"}])
    assert invalid_summary["num_invalid_records"] == 1
    assert invalid_summary["paired_effects"]["D-C"]["n"] == 1
    semantic_invalid = {"task_id": "T1", "family": "compiler", "condition": "D", "outer_trial_id": "outer-002", "score": {"task_score": 0.9, "gates_passed": False, "verified_speedup": {"median_speedup": 4.0}}}
    assert aggregate.aggregate_trials(records + [semantic_invalid])["paired_log_speedups"]["D-C"]["n"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skill"
        root.mkdir()
        (root / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        bundle = Path(tmp) / "bundle"
        from scripts.render_skill_view import render_skill_view
        render_skill_view(root, bundle)
        store = Path(tmp) / "store"
        attest_store = conditions.materialize_condition("C", bundle, store, context_mode="reset")
        transition = run_campaign.post_task_update(
            condition="C",
            store=store,
            task_id="T1",
            result={"verdict": "pass", "task_id": "T1"},
            scored={"task_score": 0.5},
            core_repo=repo_root,
            out_dir=Path(tmp),
        )
        assert transition["pre_store_digest"] != transition["post_store_digest"]
        assert transition["added_experience_ids"] == ["EXP-T1"]
        assert conditions.verify_attestation(store)[0]

    frozen_budget = budget.parse_budget({"tokens": 100, "tool_calls": 2, "wall_time_s": 5})
    assert any("missing" in item for item in frozen_budget.validate_usage({}))
    assert frozen_budget.validate_usage({"input_tokens": 40, "output_tokens": 40, "tool_calls": 2, "wall_time_s": 1}) == []
    assert any("tokens exceeded" in item for item in frozen_budget.validate_usage({"input_tokens": 80, "output_tokens": 40, "tool_calls": 2, "wall_time_s": 1}))
    timed_out = run_campaign._run_agent(
        f'"{sys.executable}" -c "import time; time.sleep(0.1)"', {}, repo_root, 0.01
    )
    assert timed_out["returncode"] == 124

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dry-run"
        args = Namespace(
            repo_root=repo_root,
            tasks_root=repo_root / "benchmark" / "tasks",
            split=split_path,
            skill_source=repo_root,
            skill_view=None,
            out=out,
            conditions="A,B,C,D",
            context_modes="reset",
            outer_trials=3,
            model_id="dry-run-agent",
            agent_config="{}",
            budgets=None,
            agent_command=None,
        )
        campaign = run_campaign.run_campaign(args)
        assert campaign["status"] == "planned"
        assert campaign["results_claimed"] is False
        saved = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
        assert saved["schedule_size"] == 240
        assert saved["results_claimed"] is False
        assert (out / "schedule.json").is_file()

    print("test_formal: OK")


if __name__ == "__main__":
    main()
