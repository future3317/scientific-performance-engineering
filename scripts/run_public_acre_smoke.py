#!/usr/bin/env python3
"""Validate the public fixed-skill path without making provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.external.contracts import validate_public_context
from benchmark.external.manifest import PublicTreatmentManifest
from benchmark.external.skillgen_adapter import SCORING_ONLY_FIELDS, build_public_context, load_pinned_benchmark


def run(args: argparse.Namespace) -> dict:
    view = load_pinned_benchmark(args.benchmark, args.dataset, expected_commit=args.benchmark_commit)
    for instance in view.instances:
        validate_public_context(build_public_context(instance), SCORING_ONLY_FIELDS)
    ids = [str(instance["instance_id"]) for instance in view.instances]
    split = {"synthesis": ids[:1], "calibration": ids[1:2], "evaluation": ids[2:]}
    manifest = PublicTreatmentManifest.create(
        revision=args.revision,
        implementation_commit=args.implementation_commit,
        benchmark_revisions={args.benchmark: args.benchmark_commit},
        task_split=split,
        model_config={"provider": "deepseek", "model": args.model, "provider_calls": 0},
        budgets={"max_tokens": args.max_tokens, "max_tool_calls": args.max_tool_calls},
        seeds=[args.seed],
        policies=["global_skill", "semantic_router", "nearest", "catboost_ltt", "acre"],
    )
    return {"status": "protocol_valid", "provider_calls": 0, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="scienceworld")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--revision", default="public-acre-treatment-20260915-v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": report["provider_calls"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

