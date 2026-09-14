#!/usr/bin/env python3
"""Run a pinned fixed-skill paired audit with the DeepSeek adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmark.external.contracts import DecisionRecord, validate_public_context
from benchmark.external.decision_ledger import DecisionLedger
from benchmark.external.evaluator import evaluate_paired_rows
from benchmark.external.skillgen_adapter import SCORING_ONLY_FIELDS, build_public_context
from scripts.run_skillgen_external_calibration import _chat, _coverage


ROUTER_VERSION = "public-acre-semantic-router-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_instances(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = list(json.loads(path.read_text(encoding="utf-8"))["instances"])
    return rows if limit is None else rows[: int(limit)]


def _router_prompt(context: dict[str, Any], skill_text: str) -> str:
    return (
        "Decide whether to apply this already globally verified skill to the public context.\n"
        f"Router version: {ROUTER_VERSION}\n"
        f"Public context JSON: {_canonical(context)}\n"
        f"Skill text:\n{skill_text}\n"
        'Return JSON only: {"apply": true|false, "reason": "short explanation"}.\n'
    )


def _router_decision(client: OpenAI, model: str, context: dict[str, Any], skill_text: str) -> tuple[bool, str, dict[str, Any]]:
    raw, usage = _chat(client, model, _router_prompt(context, skill_text), max_tokens=96)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, "router_invalid_json", usage
    value = parsed.get("apply", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "yes", "1"}
    return bool(value), str(parsed.get("reason", "no_reason"))[:240], usage


def _agent_prompt(instance: dict[str, Any], skill_text: str | None) -> str:
    prompt = str(instance["input"])
    return prompt if not skill_text else prompt + "\n\n## Verified skill\n" + skill_text


def _make_decision(run_id: str, condition: str, index: int, instance: dict[str, Any], policy: str, reuse: bool, score: float | None, reason: str, skill_version: str) -> DecisionRecord:
    context = build_public_context(instance)
    validate_public_context(context, SCORING_ONLY_FIELDS)
    return DecisionRecord(
        run_id=run_id,
        condition=condition,
        task_id=str(instance["instance_id"]),
        task_index=index,
        stage="fixed_skill_test",
        policy=policy,
        decision_id=f"{run_id}:{index}:{policy}",
        reuse=reuse,
        score=score,
        uncertainty=None,
        reason=reason,
        certificate_id="fixed-skill-certificate-v1" if policy == "acre" and reuse else None,
        skill_version=skill_version,
        public_context_digest=_digest(context),
        available_evidence_ids=(),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    skill_text = args.skill_text.read_text(encoding="utf-8").strip()
    train = _load_instances(args.train, args.train_n)
    test = _load_instances(args.test, args.test_n)
    train_ids = {str(item["instance_id"]) for item in train}
    test_ids = {str(item["instance_id"]) for item in test}
    if train_ids & test_ids:
        raise ValueError("train/test instance overlap")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=120.0)
    usage: list[dict[str, Any]] = []
    train_deltas: dict[str, list[float]] = defaultdict(list)
    for item in train:
        base, ub = _chat(client, args.model, _agent_prompt(item, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(item, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(ub, stage="train_baseline"), dict(us, stage="train_skill")])
        name = str((item.get("metadata") or {}).get("task_name") or (item.get("metadata") or {}).get("task_id"))
        train_deltas[name].append(_coverage(args.skillgen_root, skilled, item) - _coverage(args.skillgen_root, base, item))
    support = {
        name: bool(values) and sum(values) / len(values) > 0.0 and min(values) >= 0.0
        for name, values in train_deltas.items()
    }
    ledger = DecisionLedger(args.out.with_suffix(".decisions.jsonl"))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(test):
        context = build_public_context(item)
        name = str(context.get("task_name") or context.get("task_id"))
        semantic, semantic_reason, ur = _router_decision(client, args.model, context, skill_text)
        usage.append(dict(ur, stage="semantic_router"))
        acre = bool(support.get(name, False))
        decisions = [
            _make_decision(args.run_id, "fixed", index, item, "global_skill", True, 1.0, "global_reuse", "skill-v0"),
            _make_decision(args.run_id, "fixed", index, item, "semantic_router", semantic, 0.5, semantic_reason, "skill-v0"),
            _make_decision(args.run_id, "fixed", index, item, "acre", acre, 1.0 if acre else 0.0, "typed_support" if acre else "unresolved_support", "skill-v0"),
        ]
        for decision in decisions:
            ledger.append(decision)
        baseline, ub = _chat(client, args.model, _agent_prompt(item, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(item, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(ub, stage="test_baseline"), dict(us, stage="test_skill")])
        row = {
            "instance_id": item["instance_id"],
            "benchmark": (item.get("metadata") or {}).get("benchmark", args.benchmark),
            "model": args.model,
            "baseline_score": _coverage(args.skillgen_root, baseline, item),
            "skill_score": _coverage(args.skillgen_root, skilled, item),
            "measurement_reached": True,
            "policy_decisions": {decision.policy: decision.to_dict() for decision in decisions},
        }
        rows.append(row)
        for decision in decisions:
            ledger.append_outcome(decision.decision_id, {"task_index": index, "official_score": row["skill_score"]})
    return {
        "schema_version": 1,
        "revision": args.revision,
        "status": "complete",
        "benchmark": args.benchmark,
        "model": args.model,
        "skill_version": "skill-v0",
        "skill_text_sha256": _digest(skill_text),
        "train_instances": [row["instance_id"] for row in train],
        "test_instances": [row["instance_id"] for row in test],
        "provider_calls": len(usage),
        "usage": usage,
        "policy_summary": evaluate_paired_rows(rows, ["global_skill", "semantic_router", "acre"], bootstrap_seed=args.seed, bootstrap_reps=2000),
        "rows": rows,
        "claim_boundary": "fixed-skill public paired audit; not the online evolving-skill treatment estimand",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--skill-text", type=Path, required=True)
    parser.add_argument("--skillgen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--benchmark", default="scienceworld")
    parser.add_argument("--train-n", type=int, default=1)
    parser.add_argument("--test-n", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--run-id", default="public-acre-fixed-smoke-20260915")
    parser.add_argument("--revision", default="public-acre-treatment-20260915-v1")
    args = parser.parse_args()
    import os
    args.api_key = args.api_key or os.environ["MP_API_KEY"]
    args.base_url = args.base_url or os.environ["DEEPSEEK_BASE_URL"]
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": report["provider_calls"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

