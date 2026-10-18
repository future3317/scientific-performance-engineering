#!/usr/bin/env python3
"""Run the pinned SkillGen public fixed-intervention audit across benchmarks.

The intervention is frozen before held-out execution.  All policy decisions are
serialized to a decision ledger before paired outcomes are attached, and every
policy is scored from the same no-skill/skill rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmark.external.contracts import DecisionRecord, validate_public_context
from benchmark.external.decision_ledger import DecisionLedger
from benchmark.external.evaluator import evaluate_paired_rows
from benchmark.external.skillgen_adapter import SCORING_ONLY_FIELDS, build_public_context, grade_output


SKILLS = {
    "scienceworld": (
        "Plan the public task before acting. Follow its declared action grammar literally, "
        "satisfy navigation and possession prerequisites, include required focus/measurement "
        "steps, and emit only the requested plan format. Do not invent task-specific facts."
    ),
    "mind2web": (
        "Ground each next action in the visible page and candidate elements. Preserve the "
        "requested action schema and use the exact available backend node identifiers; do not "
        "invent elements or skip a required intermediate action."
    ),
    "toolbench": (
        "Solve API-use tasks by selecting only tools exposed in the task, checking required "
        "parameters and dependencies, and returning the benchmark's exact requested format. "
        "Do not invent tool names, arguments, or observations."
    ),
    "mcp_bench": (
        "Follow the task's MCP planning format exactly. Use only listed servers and tools, "
        "respect dependencies and parallelism, and ground every parameter in the public "
        "request. Return only the requested structured answer."
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _chat(client: OpenAI, model: str, prompt: str, *, max_tokens: int) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Follow the benchmark task's output contract exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return content, {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _router_prompt(context: dict[str, Any], skill_text: str) -> str:
    return (
        "Decide whether to apply this already verified fixed intervention to the public context.\n"
        "Use only the supplied public context and intervention description.\n"
        f"Public context JSON: {_canonical(context)}\n"
        f"Intervention description: {skill_text}\n"
        'Return JSON only: {"apply": true|false, "reason": "short explanation"}.\n'
    )


def _router_decision(client: OpenAI, model: str, context: dict[str, Any], skill_text: str) -> tuple[bool, str, dict[str, Any]]:
    raw, usage = _chat(client, model, _router_prompt(context, skill_text), max_tokens=96)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, "router_invalid_json", usage
    if not isinstance(parsed, dict):
        return False, "router_invalid_response", usage
    value = parsed.get("apply")
    if not isinstance(value, bool):
        return False, "router_invalid_apply", usage
    return value, str(parsed.get("reason", "no_reason"))[:240], usage


def _verify_skillgen_revision(skillgen_root: Path, expected_commit: str) -> str:
    """Return HEAD only when the checkout matches the requested revision."""
    try:
        actual = subprocess.run(
            ["git", "-C", str(skillgen_root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        expected = subprocess.run(
            ["git", "-C", str(skillgen_root), "rev-parse", "--verify", f"{expected_commit}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "skillgen_root must be a Git checkout and --skillgen-commit must resolve to a commit"
        ) from exc
    if actual != expected:
        raise ValueError(
            f"SkillGen checkout revision mismatch: expected {expected}, found {actual}"
        )
    return actual


def _agent_prompt(instance: dict[str, Any], skill_text: str | None) -> str:
    prompt = str(instance["input"])
    return prompt if not skill_text else prompt + "\n\n## Verified intervention\n" + skill_text


def _load(path: Path, count: int | None, seed: int) -> list[dict[str, Any]]:
    rows = list(json.loads(path.read_text(encoding="utf-8"))["instances"])
    random.Random(seed).shuffle(rows)
    return rows if count is None else rows[: int(count)]


def _support_key(benchmark: str, instance: dict[str, Any]) -> str:
    """Return a public, task-type key shared across train/test variants."""
    metadata = instance.get("metadata") or {}
    if benchmark == "scienceworld":
        value: Any = ("task_id", metadata.get("task_id"))
    elif benchmark == "mind2web":
        value = ("web", metadata.get("domain"), metadata.get("subdomain"), metadata.get("website"), metadata.get("op"))
    elif benchmark == "toolbench":
        value = ("tool", metadata.get("subset"), metadata.get("split_source"))
    elif benchmark == "mcp_bench":
        value = ("mcp", metadata.get("combination_type"), tuple(sorted(metadata.get("servers") or [])))
    else:
        value = (benchmark,)
    return _canonical(value)


def _score(skillgen_root: Path, benchmark: str, output: str, instance: dict[str, Any]) -> float:
    result = grade_output(skillgen_root, benchmark, output, instance)
    for key in ("score", "coverage_rate", "overall_score", "reward", "success", "passed", "is_correct"):
        if key in result:
            value = result[key]
            return float(value if not isinstance(value, bool) else int(value))
    raise ValueError(f"grader for {benchmark} returned no scalar score: {sorted(result)}")


def _make_decision(run_id: str, benchmark: str, index: int, instance: dict[str, Any], policy: str, reuse: bool, reason: str) -> DecisionRecord:
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
        certificate_id="fixed-intervention-v1" if policy == "acre" and reuse else None,
        skill_version=f"{benchmark}-skill-v1",
        public_context_digest=_digest(context),
        available_evidence_ids=(),
    )


def run_benchmark(args: argparse.Namespace, benchmark: str, out_dir: Path) -> dict[str, Any]:
    root = Path(args.skillgen_root).resolve()
    skill_text = SKILLS[benchmark]
    out_dir.mkdir(parents=True, exist_ok=True)
    skill_path = out_dir / f"{benchmark}.skill.txt"
    skill_path.write_text(skill_text + "\n", encoding="utf-8")
    train = _load(root / "data" / benchmark / "train.json", args.train_n, args.seed)
    test = _load(root / "data" / benchmark / "test.json", args.test_n, args.seed + 1)
    if {str(x["instance_id"]) for x in train} & {str(x["instance_id"]) for x in test}:
        raise ValueError(f"train/test overlap for {benchmark}")
    client = OpenAI(api_key=os.environ["MP_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], timeout=120.0)
    run_id = f"{args.revision}-{benchmark}"
    ledger = DecisionLedger(out_dir / "decisions.jsonl")
    usage: list[dict[str, Any]] = []
    train_deltas: dict[str, list[float]] = defaultdict(list)
    for item in train:
        base, ub = _chat(client, args.model, _agent_prompt(item, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(item, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(ub, stage="train_baseline"), dict(us, stage="train_skill")])
        group = _support_key(benchmark, item)
        train_deltas[group].append(_score(root, benchmark, skilled, item) - _score(root, benchmark, base, item))
    support = {group: bool(values) and sum(values) / len(values) > 0 and min(values) >= 0 for group, values in train_deltas.items()}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(test):
        context = build_public_context(item)
        group = _support_key(benchmark, item)
        routed, router_reason, ur = _router_decision(client, args.model, context, skill_text)
        usage.append(dict(ur, stage="semantic_router"))
        exact_support = bool(support.get(group, False))
        decisions = [
            _make_decision(run_id, benchmark, index, item, "global_skill", True, "global_reuse"),
            _make_decision(run_id, benchmark, index, item, "semantic_router", routed, router_reason),
            _make_decision(run_id, benchmark, index, item, "positive_only", bool(any(v > 0 for v in train_deltas.get(group, []))), "positive_train_support" if any(v > 0 for v in train_deltas.get(group, [])) else "no_positive_train_support"),
            _make_decision(run_id, benchmark, index, item, "acre", exact_support, "typed_support" if exact_support else "unresolved_support"),
        ]
        for decision in decisions:
            ledger.append(decision)
        baseline, ub = _chat(client, args.model, _agent_prompt(item, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(item, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(ub, stage="test_baseline"), dict(us, stage="test_skill")])
        row = {
            "instance_id": item["instance_id"],
            "benchmark": benchmark,
            "model": args.model,
            "baseline_score": _score(root, benchmark, baseline, item),
            "skill_score": _score(root, benchmark, skilled, item),
            "measurement_reached": True,
            "policy_decisions": {d.policy: d.to_dict() for d in decisions},
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
        "skill_text_sha256": _digest(skill_text),
        "train_instances": [x["instance_id"] for x in train],
        "test_instances": [x["instance_id"] for x in test],
        "public_context_fields": ["benchmark", "task_id", "task_name", "fold", "task_description", "initial_observation", "category", "difficulty", "model_harness_identity", "domain", "subdomain", "website", "operation", "subset", "split_source", "combination_type", "servers"],
        "scoring_only_fields": sorted(SCORING_ONLY_FIELDS),
        "provider_calls": len(usage),
        "usage": usage,
        "policy_summary": evaluate_paired_rows(rows, ["global_skill", "semantic_router", "positive_only", "acre"], bootstrap_seed=args.seed, bootstrap_reps=2000),
        "rows": rows,
        "claim_boundary": "public fixed-intervention audit on pinned SkillGen data; not prospective SkillGen skill-generation efficacy",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillgen-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--skillgen-commit", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(SKILLS))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--revision", default="public-acre-skillgen-suite-20260915-v1")
    args = parser.parse_args()
    args.skillgen_commit_claimed = args.skillgen_commit
    args.skillgen_commit = _verify_skillgen_revision(
        args.skillgen_root.resolve(), args.skillgen_commit
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reports = {}
    for benchmark in args.benchmarks:
        if benchmark not in SKILLS:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        reports[benchmark] = run_benchmark(args, benchmark, args.out_root / benchmark)
    manifest = {
        "schema_version": 1,
        "revision": args.revision,
        "skillgen_commit": args.skillgen_commit,
        "skillgen_commit_claimed": args.skillgen_commit_claimed,
        "benchmarks": list(args.benchmarks),
        "model": args.model,
        "train_n": args.train_n,
        "test_n": args.test_n,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "provider": "deepseek",
        "reports": {benchmark: {"provider_calls": report["provider_calls"], "n_test": len(report["test_instances"]), "sha256": _digest(report)} for benchmark, report in reports.items()},
        "claim_boundary": "external public fixed-intervention deployment audit; not SkillGen original generation efficacy",
        "elapsed_s": round(time.time() - started, 3),
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "benchmarks": list(reports), "provider_calls": sum(r["provider_calls"] for r in reports.values()), "output": str(args.out_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
