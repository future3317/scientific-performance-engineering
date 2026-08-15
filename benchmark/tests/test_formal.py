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
    for condition, score in (("B", 0.40), ("C", 0.50), ("D", 0.70)):
        records.append({"task_id": "T1", "family": "compiler", "condition": condition, "context_mode": "reset", "outer_trial_id": "outer-000", "score": score, "kind": "positive"})
    summary = aggregate.aggregate_trials(records)
    assert math.isclose(summary["paired_effects"]["D-C"]["estimate"], 0.20)
    assert math.isclose(summary["paired_effects"]["D-B"]["estimate"], 0.30)
    assert summary["family_stratified"]["compiler"]["D-C"]["n"] == 1

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
