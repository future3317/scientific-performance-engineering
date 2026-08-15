#!/usr/bin/env python3
"""Run or dry-run a complete SPE-EvoBench formal campaign.

Without ``--agent-command`` this writes only a frozen campaign plan and never
claims benchmark results. With a command, the driver gives the agent a fresh
solution workspace for each task and then invokes the immutable verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.harness import conditions, miniyaml, scoring, verifier
from benchmark.harness.fingerprint import capture_fingerprint
from benchmark.formal import aggregate, attest, budget, schedule
from scripts.render_skill_view import render_skill_view, validate_skill_view_bundle


def _copy_workspace(task_dir: Path, destination: Path) -> None:
    source = task_dir / "workspace"
    if not source.is_dir():
        raise FileNotFoundError(f"workspace missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _run_agent(command_template: str, env: dict[str, str], cwd: Path, timeout: float) -> dict[str, Any]:
    command = command_template.format(**env)
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _experiment_manifest(
    *,
    repo_root: Path,
    skill_digest: str,
    task_digest: str,
    task_ids: list[str],
    item: dict[str, Any],
    model_id: str,
    agent_config: dict[str, Any],
    budgets: budget.Budget,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": f"SPE-EvoBench-v1.0-20-{item['stream_id']}-{item['task_id']}",
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "agent_model_id": model_id,
        "agent_config": agent_config,
        "condition": item["condition"],
        "context_mode": item["context_mode"],
        "task_order": task_ids,
        "outer_trial_id": item["outer_trial_id"],
        "budgets": budgets.as_dict(),
        "hardware_fingerprint": fingerprint,
        "software_fingerprint": {
            "python_version": fingerprint.get("python_version"),
            "platform": fingerprint.get("platform"),
            "torch_geometric_version": fingerprint.get("torch_geometric_version"),
        },
        "torch_version": fingerprint.get("torch_version"),
        "cuda_version": fingerprint.get("cuda_version"),
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    tasks_root = Path(args.tasks_root).resolve()
    split_path = Path(args.split).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = schedule.task_order(split_path)
    task_ids = [task_id for _, task_id in task_items]
    if args.skill_view:
        skill_view = Path(args.skill_view).resolve()
        if not (skill_view / "skill_view_manifest.json").is_file():
            raise ValueError("--skill-view must point to a rendered skill bundle")
        if validate_skill_view_bundle(skill_view):
            raise ValueError("--skill-view failed bundle validation")
    else:
        skill_view = out_dir / "skill-view"
        render_skill_view(args.skill_source, skill_view)
    skill_digest = attest.skill_view_digest(skill_view)
    task_digest = attest.task_manifest_digest(tasks_root, task_ids)
    conditions_list = tuple(item.strip().upper() for item in args.conditions.split(",") if item.strip())
    context_modes = tuple(item.strip() for item in args.context_modes.split(",") if item.strip())
    plan = schedule.build_schedule(
        split_path,
        conditions=conditions_list,
        context_modes=context_modes,
        outer_trials=args.outer_trials,
    )
    budgets = budget.parse_budget(json.loads(args.budgets) if args.budgets else None)
    fingerprint = capture_fingerprint()
    campaign = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-20-pilot",
        "status": "planned" if not args.agent_command else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "conditions": list(conditions_list),
        "context_modes": list(context_modes),
        "outer_trials": args.outer_trials,
        "task_order": task_ids,
        "budgets": budgets.as_dict(),
        "agent_model_id": args.model_id,
        "agent_config": json.loads(args.agent_config),
        "schedule_size": len(plan),
        "results_claimed": False,
    }
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.agent_command:
        (out_dir / "schedule.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return campaign

    records: list[dict[str, Any]] = []
    stores: dict[str, Path] = {}
    context_paths: dict[str, Path] = {}
    for item in plan:
        stream_id = str(item["stream_id"])
        task_id = str(item["task_id"])
        trial_dir = out_dir / "trials" / stream_id / task_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        manifest = _experiment_manifest(
            repo_root=repo_root,
            skill_digest=skill_digest,
            task_digest=task_digest,
            task_ids=task_ids,
            item=item,
            model_id=args.model_id,
            agent_config=json.loads(args.agent_config),
            budgets=budgets,
            fingerprint=fingerprint,
        )
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        state_path = context_paths.setdefault(stream_id, trial_dir.parent / "context.json")
        if item["context_mode"] == "reset" and state_path.exists():
            state_path.unlink()
        store = stores.get(stream_id)
        if store is None:
            store = trial_dir.parent / "condition-store"
            conditions.materialize_condition(item["condition"], skill_view if item["condition"] != "A" else None, store, item["context_mode"])
            stores[stream_id] = store
        solution_dir = trial_dir / "solution"
        _copy_workspace(tasks_root / task_id, solution_dir)
        env = {
            "SPE_TASK_ID": task_id,
            "SPE_TASK_DIR": str(tasks_root / task_id),
            "SPE_SOLUTION_DIR": str(solution_dir),
            "SPE_CONDITION": str(item["condition"]),
            "SPE_CONTEXT_MODE": str(item["context_mode"]),
            "SPE_CONDITION_STORE": str(store),
            "SPE_CONTEXT_STATE": str(state_path),
            "SPE_RESULT_PATH": str(trial_dir / "result.json"),
            "SPE_BUDGET_JSON": json.dumps(budgets.as_dict(), sort_keys=True),
            "SPE_OUTER_TRIAL_ID": str(item["outer_trial_id"]),
        }
        agent = _run_agent(args.agent_command, env, repo_root, budgets.wall_time_s)
        result = verifier.verify_task(
            tasks_root / task_id,
            solution_dir,
            out_path=trial_dir / "result.json",
            condition=str(item["condition"]),
            context_mode=str(item["context_mode"]),
            seed=int(item["outer_trial_index"]),
        )
        scored = scoring.score_task(result)
        record = {
            "experiment": manifest,
            "task_id": task_id,
            "family": miniyaml.load(str(tasks_root / task_id / "task.yaml")).get("family"),
            "kind": miniyaml.load(str(tasks_root / task_id / "task.yaml")).get("kind"),
            "condition": item["condition"],
            "context_mode": item["context_mode"],
            "outer_trial_id": item["outer_trial_id"],
            "phase": item["phase"],
            "agent": agent,
            "budget_errors": budgets.check_cost(result.get("cost", {})),
            "score": scored,
        }
        (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        records.append(record)
    campaign["status"] = "complete"
    campaign["results_claimed"] = bool(records)
    campaign["aggregate"] = aggregate.aggregate_trials(records)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--tasks-root", type=Path, default=root / "benchmark" / "tasks")
    parser.add_argument("--split", type=Path, default=root / "benchmark" / "split" / "sequential.yaml")
    parser.add_argument("--skill-source", type=Path, default=root)
    parser.add_argument("--skill-view", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--context-modes", default="reset")
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--model-id", default="unidentified-agent")
    parser.add_argument("--agent-config", default="{}")
    parser.add_argument("--budgets", default=None, help='JSON, e.g. {"tokens":12000,"tool_calls":80,"wall_time_s":900}')
    parser.add_argument("--agent-command", default=None, help="shell template; receives SPE_* environment variables")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "schedule_size": result["schedule_size"], "results_claimed": result["results_claimed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
