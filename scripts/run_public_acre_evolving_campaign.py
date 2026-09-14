#!/usr/bin/env python3
"""Run the new online evolving-skill A/C/D ScienceWorld treatment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmark.external.evaluator import evaluate_paired_rows
from benchmark.external.evolving_campaign import run_sequential_campaign
from scripts.run_skillgen_external_calibration import _chat, _coverage


def _load_tasks(path: Path, n: int) -> list[dict[str, Any]]:
    rows = list(json.loads(path.read_text(encoding="utf-8"))["instances"])
    return rows[: int(n)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _load_tasks(args.test, args.n)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=120.0)
    baseline_cache: dict[str, tuple[str, dict[str, Any]]] = {}

    def agent_call(prompt: str, task: dict[str, Any], condition: str) -> tuple[str, dict[str, Any]]:
        task_id = str(task["instance_id"])
        if condition == "A":
            if task_id not in baseline_cache:
                baseline_cache[task_id] = _chat(client, args.model, prompt, max_tokens=args.max_tokens)
            return baseline_cache[task_id]
        return _chat(client, args.model, prompt, max_tokens=args.max_tokens)

    def score_call(output: str, task: dict[str, Any]) -> float:
        return _coverage(args.skillgen_root, output, task)

    artifacts = {}
    for condition in ("A", "C", "D"):
        condition_dir = args.out / condition
        artifacts[condition] = run_sequential_campaign(
            tasks,
            condition=condition,
            run_id=f"{args.run_id}-{condition}",
            out_dir=condition_dir,
            agent_call=agent_call,
            score_call=score_call,
            minimum_effect=args.minimum_effect,
        )
        (condition_dir / "campaign.json").write_text(json.dumps(artifacts[condition], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries = {
        condition: evaluate_paired_rows(
            artifact["rows"], [condition], bootstrap_seed=args.seed, bootstrap_reps=args.bootstrap_reps
        )
        for condition, artifact in artifacts.items()
    }
    paired = []
    c_by_task = {row["instance_id"]: row for row in artifacts["C"]["rows"]}
    d_by_task = {row["instance_id"]: row for row in artifacts["D"]["rows"]}
    for task_id in sorted(set(c_by_task) & set(d_by_task), key=lambda value: int(c_by_task[value]["task_index"])):
        paired.append({
            "instance_id": task_id,
            "baseline_score": c_by_task[task_id]["skill_score"],
            "skill_score": d_by_task[task_id]["skill_score"],
            "policy_decisions": {"D_minus_C": {"reuse": True, "reason": "paired_condition_difference"}},
        })
    summaries["D_minus_C"] = evaluate_paired_rows(paired, ["D_minus_C"], bootstrap_seed=args.seed, bootstrap_reps=args.bootstrap_reps)
    return {
        "schema_version": 1,
        "revision": args.revision,
        "status": "complete",
        "benchmark": "scienceworld",
        "model": args.model,
        "task_ids": [task["instance_id"] for task in tasks],
        "conditions": ["A", "C", "D"],
        "provider_calls": sum(len(artifact["usage"]) for artifact in artifacts.values()),
        "summaries": summaries,
        "artifacts": {condition: str(args.out / condition / "campaign.json") for condition in artifacts},
        "claim_boundary": "online evolving-skill calibration treatment; held-out efficacy wording requires complete paired observations and preregistered analysis",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--skillgen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--run-id", default="public-acre-evolving-20260915")
    parser.add_argument("--revision", default="public-acre-treatment-20260915-v1")
    args = parser.parse_args()
    args.api_key = args.api_key or os.environ["MP_API_KEY"]
    args.base_url = args.base_url or os.environ["DEEPSEEK_BASE_URL"]
    args.out.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (args.out / "treatment_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "provider_calls": result["provider_calls"], "output": str(args.out / 'treatment_summary.json')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

