#!/usr/bin/env python3
"""Run a small provider-backed fixed-skill ScienceWorld calibration audit.

This is a new, independent calibration artifact. It uses only public
execution-before-visible context for routing and joins gold key actions only
after all decisions and paired plans have been serialized in memory. It is
not the frozen provider-free frontier and it is not a hosted efficacy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


ROUTER_VERSION = "skillgen-scienceworld-calibration-router-v1"
MODEL_DEFAULT = "deepseek-v4-flash"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _public_context(instance: dict[str, Any]) -> dict[str, Any]:
    metadata = instance.get("metadata") or {}
    # These fields are visible before execution. Gold actions, key actions,
    # ground truth, and path-derived scores are deliberately excluded.
    return {
        "benchmark": metadata.get("benchmark"),
        "task_id": metadata.get("task_id"),
        "task_name": metadata.get("task_name"),
        "variation_idx": metadata.get("variation_idx"),
        "fold": metadata.get("fold"),
        "task_description": metadata.get("task_description"),
        "initial_observation": metadata.get("initial_observation"),
    }


def _router_prompt(context: dict[str, Any], skill_text: str) -> str:
    return (
        "Decide whether to apply the fixed intervention to this public context.\n"
        f"Router version: {ROUTER_VERSION}\n"
        f"Public context JSON: {_canonical(context)}\n"
        f"Intervention description: {skill_text}\n"
        'Return JSON only: {"apply": true|false, "reason": "short explanation"}.\n'
    )


def _chat(client: OpenAI, model: str, prompt: str, *, max_tokens: int) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce deterministic ScienceWorld action plans."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return content, {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _router_decision(client: OpenAI, model: str, context: dict[str, Any], skill_text: str) -> tuple[bool, str, dict[str, Any]]:
    raw, usage = _chat(client, model, _router_prompt(context, skill_text), max_tokens=96)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, "router_invalid_json", usage
    apply_value = parsed.get("apply", False)
    if isinstance(apply_value, str):
        apply_value = apply_value.strip().lower() in {"true", "yes", "1"}
    return bool(apply_value), str(parsed.get("reason", "no_reason"))[:240], usage


def _agent_prompt(instance: dict[str, Any], skill_text: str | None) -> str:
    prompt = str(instance["input"])
    if skill_text:
        prompt += "\n\n## Fixed intervention\n" + skill_text
    return prompt


def _load_instances(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["instances"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _coverage(skillgen_root: Path, output: str, instance: dict[str, Any]) -> float:
    sys.path.insert(0, str(skillgen_root))
    from benchmarks.scienceworld_grader import _extract_plan, compute_key_coverage

    plan = _extract_plan(output)
    return float(compute_key_coverage(plan, (instance.get("metadata") or {}).get("key_actions") or []).get("coverage_rate", 0.0))


def _summarize_policy(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deployed = [row for row in rows if row["policies"][name]["apply"]]
    unsupported = [row for row in deployed if row["skill_score"] + 1e-12 < row["baseline_score"]]
    repairs = [row for row in deployed if row["skill_score"] > row["baseline_score"]]
    return {
        "policy": name,
        "n": len(rows),
        "deployed": len(deployed),
        "coverage": len(deployed) / len(rows) if rows else 0.0,
        "unsupported_deployed": len(unsupported),
        "unsupported_deployed_rate": len(unsupported) / len(deployed) if deployed else None,
        "repairs": len(repairs),
        "net_score_delta": sum(row["skill_score"] - row["baseline_score"] for row in deployed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    skillgen_root = Path(args.skillgen_root).resolve()
    skill_text = Path(args.skill_text).read_text(encoding="utf-8").strip()
    train = _load_instances(Path(args.train), args.train_n, args.seed)
    test = _load_instances(Path(args.test), args.test_n, args.seed + 1)
    if {row["instance_id"] for row in train} & {row["instance_id"] for row in test}:
        raise ValueError("train/test instance overlap")
    client = OpenAI(api_key=os.environ["MP_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], timeout=120.0)

    usage: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in train:
        base, u0 = _chat(client, args.model, _agent_prompt(row, None), max_tokens=args.max_tokens)
        skilled, u1 = _chat(client, args.model, _agent_prompt(row, skill_text), max_tokens=args.max_tokens)
        usage.extend([dict(u0, stage="train_baseline"), dict(u1, stage="train_skill")])
        train_rows.append({
            "task_name": (row.get("metadata") or {}).get("task_name"),
            "delta": _coverage(skillgen_root, skilled, row) - _coverage(skillgen_root, base, row),
        })

    group_deltas: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        group_deltas[str(row["task_name"])].append(float(row["delta"]))
    # Evidence support is derived only from paired train outcomes. A group is
    # supported when its observed mean is positive and has no harmful example.
    group_support = {key: bool(values) and sum(values) / len(values) > 0 and min(values) >= 0 for key, values in group_deltas.items()}

    rows: list[dict[str, Any]] = []
    for instance in test:
        baseline, ub = _chat(client, args.model, _agent_prompt(instance, None), max_tokens=args.max_tokens)
        skilled, us = _chat(client, args.model, _agent_prompt(instance, skill_text), max_tokens=args.max_tokens)
        context = _public_context(instance)
        routed, reason, ur = _router_decision(client, args.model, context, skill_text)
        usage.extend([dict(ub, stage="test_baseline"), dict(us, stage="test_skill"), dict(ur, stage="semantic_router")])
        baseline_score = _coverage(skillgen_root, baseline, instance)
        skill_score = _coverage(skillgen_root, skilled, instance)
        task_name = str(context.get("task_name"))
        support = bool(group_support.get(task_name, False))
        rows.append({
            "instance_id": instance["instance_id"],
            "task_name": task_name,
            "task_id": context.get("task_id"),
            "variation_idx": context.get("variation_idx"),
            "public_context_digest": _sha256_text(_canonical(context)),
            "baseline_score": baseline_score,
            "skill_score": skill_score,
            "policies": {
                "global_skillgen": {"apply": True, "reason": "global_reuse"},
                "semantic_router": {"apply": routed, "reason": reason},
                "positive_only": {"apply": support, "reason": "positive_train_support" if support else "no_positive_train_support"},
                "evidence_support": {"apply": support, "reason": "positive_and_no_harm_train_support" if support else "unresolved_or_harmful_train_support"},
            },
        })

    policy_rows = [_summarize_policy(name, rows) for name in ("global_skillgen", "semantic_router", "positive_only", "evidence_support")]
    return {
        "schema_version": 1,
        "status": "complete",
        "analysis": "skillgen_scienceworld_fixed_skill_calibration",
        "provider": "deepseek",
        "model": args.model,
        "temperature": 0.0,
        "skill_text_sha256": _sha256_text(skill_text),
        "skillgen_commit": args.skillgen_commit,
        "train_instances": [row["instance_id"] for row in train],
        "test_instances": [row["instance_id"] for row in test],
        "public_context_fields": ["benchmark", "task_id", "task_name", "variation_idx", "fold", "task_description", "initial_observation"],
        "scoring_only_fields": ["key_actions", "gold_actions", "ground_truth", "gold_path_length"],
        "provider_calls": len(usage),
        "usage": usage,
        "group_support": group_support,
        "policy_summary": policy_rows,
        "rows": rows,
        "claim_boundary": "small provider-backed calibration evidence; not prospective SkillGen efficacy and not a sealed treatment comparison",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--skill-text", type=Path, required=True)
    parser.add_argument("--skillgen-root", type=Path, required=True)
    parser.add_argument("--skillgen-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--train-n", type=int, default=6)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    started = time.time()
    report = run(args)
    report["elapsed_s"] = round(time.time() - started, 3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": report["provider_calls"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
