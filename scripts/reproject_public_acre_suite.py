#!/usr/bin/env python3
"""Recompute public policy decisions from a completed paired-outcome cache.

Only the train-side support evidence is freshly acquired.  Held-out no-skill and
skill outcomes are immutable inputs; the script never uses their scores to make
decisions and records that reuse explicitly in the new manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmark.external.contracts import DecisionRecord, validate_public_context
from benchmark.external.decision_ledger import DecisionLedger
from benchmark.external.evaluator import evaluate_paired_rows
from benchmark.external.skillgen_adapter import SCORING_ONLY_FIELDS, build_public_context
from scripts.run_public_acre_suite import SKILLS, _agent_prompt, _canonical, _chat, _load, _score, _support_key


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _decision(run_id: str, benchmark: str, index: int, instance: dict[str, Any], policy: str, reuse: bool, reason: str) -> DecisionRecord:
    context = build_public_context(instance)
    validate_public_context(context, SCORING_ONLY_FIELDS)
    return DecisionRecord(
        run_id=run_id,
        condition="fixed",
        task_id=str(instance["instance_id"]),
        task_index=index,
        stage="public_fixed_skill_test",
        policy=policy,
        decision_id=f"{run_id}:{index}:{policy}",
        reuse=reuse,
        score=1.0 if reuse else 0.0,
        uncertainty=None,
        reason=reason,
        certificate_id="fixed-intervention-v2" if policy == "acre" and reuse else None,
        skill_version=f"{benchmark}-skill-v1",
        public_context_digest=_digest(context),
        available_evidence_ids=(),
    )


def reproject_benchmark(args: argparse.Namespace, benchmark: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.skillgen_root).resolve()
    source = json.loads((Path(args.source_root) / benchmark / "report.json").read_text(encoding="utf-8"))
    source_digest = _digest(source)
    source_rows = {str(row["instance_id"]): row for row in source["rows"]}
    test = _load(root / "data" / benchmark / "test.json", None, args.seed + 1)
    test = [row for row in test if str(row["instance_id"]) in source_rows]
    train = _load(root / "data" / benchmark / "train.json", args.train_n, args.seed)
    client = OpenAI(api_key=os.environ["MP_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], timeout=120.0)
    skill_text = SKILLS[benchmark]
    train_deltas: dict[str, list[float]] = {}
    usage: list[dict[str, Any]] = []
    for item in train:
        base, ub = _chat(client, args.model, _agent_prompt(item, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(item, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(ub, stage="reprojection_train_baseline"), dict(us, stage="reprojection_train_skill")])
        train_deltas.setdefault(_support_key(benchmark, item), []).append(_score(root, benchmark, skilled, item) - _score(root, benchmark, base, item))
    support = {key: bool(values) and sum(values) / len(values) > 0 and min(values) >= 0 for key, values in train_deltas.items()}
    run_id = f"{args.revision}-{benchmark}"
    ledger = DecisionLedger(out_dir / "decisions.jsonl")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(test):
        old = source_rows[str(item["instance_id"])]
        group = _support_key(benchmark, item)
        semantic = old["policy_decisions"]["semantic_router"]
        exact_support = bool(support.get(group, False))
        positive = bool(any(v > 0 for v in train_deltas.get(group, [])))
        decisions = [
            _decision(run_id, benchmark, index, item, "global_skill", True, "global_reuse"),
            _decision(run_id, benchmark, index, item, "semantic_router", bool(semantic.get("reuse", False)), str(semantic.get("reason", "cached_semantic_router"))),
            _decision(run_id, benchmark, index, item, "positive_only", positive, "positive_train_support" if positive else "no_positive_train_support"),
            _decision(run_id, benchmark, index, item, "acre", exact_support, "typed_support" if exact_support else "unresolved_support"),
        ]
        for decision in decisions:
            ledger.append(decision)
        row = {
            "instance_id": item["instance_id"],
            "benchmark": benchmark,
            "model": args.model,
            "baseline_score": old["baseline_score"],
            "skill_score": old["skill_score"],
            "measurement_reached": old.get("measurement_reached", True),
            "policy_decisions": {d.policy: d.to_dict() for d in decisions},
            "outcome_source_revision": source["revision"],
        }
        rows.append(row)
        for decision in decisions:
            ledger.append_outcome(decision.decision_id, {"task_index": index, "official_score": row["skill_score"]})
    report = {
        "schema_version": 2,
        "status": "complete",
        "benchmark": benchmark,
        "skillgen_commit": args.skillgen_commit,
        "model": args.model,
        "model_config": {"temperature": 0.0, "max_tokens": args.max_tokens, "router_max_tokens": 96},
        "revision": args.revision,
        "source_report_revision": source["revision"],
        "source_report_sha256": source_digest,
        "outcome_reuse": "paired held-out outcomes reused; only train-side support evidence reacquired",
        "skill_text_sha256": _digest(skill_text),
        "train_instances": [x["instance_id"] for x in train],
        "test_instances": [x["instance_id"] for x in test],
        "public_context_fields": source["public_context_fields"],
        "scoring_only_fields": sorted(SCORING_ONLY_FIELDS),
        "provider_calls": len(usage),
        "test_outcomes_reused": len(rows),
        "usage": usage,
        "policy_summary": evaluate_paired_rows(rows, ["global_skill", "semantic_router", "positive_only", "acre"], bootstrap_seed=args.seed, bootstrap_reps=2000),
        "rows": rows,
        "claim_boundary": "public fixed-intervention audit on pinned SkillGen data; test outcomes reused from a completed paired cache; not prospective SkillGen generation efficacy",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillgen-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--skillgen-commit", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--revision", default="public-acre-skillgen-suite-20260915-v2")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for benchmark in SKILLS:
        reports[benchmark] = reproject_benchmark(args, benchmark, args.out_root / benchmark)
    manifest = {
        "schema_version": 1,
        "revision": args.revision,
        "source_root": str(args.source_root),
        "source_reports": {b: {"revision": r["source_report_revision"], "sha256": r["source_report_sha256"]} for b, r in reports.items()},
        "skillgen_commit": args.skillgen_commit,
        "model": args.model,
        "train_n": args.train_n,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "new_provider_calls": sum(r["provider_calls"] for r in reports.values()),
        "reused_test_outcomes": sum(r["test_outcomes_reused"] for r in reports.values()),
        "claim_boundary": "public fixed-intervention audit; held-out outcomes reused from v1 paired cache; not prospective SkillGen generation efficacy",
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "new_provider_calls": manifest["new_provider_calls"], "reused_test_outcomes": manifest["reused_test_outcomes"], "output": str(args.out_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
