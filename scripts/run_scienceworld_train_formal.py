#!/usr/bin/env python3
"""Generate and globally verify ScienceWorld skills on train-side data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_real.adapters.scienceworld_native import ScienceWorldAdapter
from experiments.external_real.skill_pipeline import freeze_skill, verify_skill
from experiments.external_real.run_scienceworld import ScienceWorldProviderBackend
from benchmark.formal.host_agent import _api_call_with_retry


def generation_prompt(tasks: Sequence[Mapping[str, object]], *, seed: int) -> str:
    rows = [
        {
            "task_id": str(task["task_id"]),
            "task_name": str(task.get("task_name", "")),
            "task_description": str(task.get("task_description", "")),
            "initial_observation": str(task.get("initial_observation", "")),
        }
        for task in tasks
    ]
    return (
        "Create one concise reusable ScienceWorld action-planning skill from these public "
        f"construction tasks. seed={int(seed)}. Do not include task-specific answers. "
        "Return only the skill text, with public action grammar and a final wait1 step.\n"
        + json.dumps(rows, ensure_ascii=False, sort_keys=True)
    )


def summarize_candidate(*, generation_seed: int, skill_hash: str, verification: Mapping[str, object]) -> dict[str, object]:
    return {
        "generation_seed": int(generation_seed),
        "skill_hash": skill_hash,
        "outcome": str(verification.get("outcome", "WITHHOLD_SKILL")),
        "failure_reasons": sorted(str(item) for item in verification.get("failure_reasons", [])),
    }


def _generate_skill(
    *, prompt: str, model_id: str, on_attempt: Callable[[], None] | None = None
) -> str:
    response = _api_call_with_retry(
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MP_API_KEY", ""),
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You write reusable benchmark skills."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": 768,
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        },
        timeout_s=120,
        on_attempt=on_attempt,
    )
    return str(response["choices"][0]["message"].get("content", "")).strip()


def run_train_formal(*, task_manifest: Mapping[str, Sequence[Mapping[str, object]]], scienceworld_root: Path, out_dir: Path, model_id: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = ScienceWorldProviderBackend(scienceworld_root=scienceworld_root, model_id=model_id, max_actions=20)
    adapter = ScienceWorldAdapter(
        task_manifest,
        backend=backend,
        benchmark_revision="scienceworld-1.2.3-5d569f9e4ff43ab6242a3622f7e25f9bc367ccd1",
        evaluator_revision="ScienceWorld-official-score-1.2.3",
    )
    candidates: list[dict[str, object]] = []
    generation_attempts = 0

    def count_generation_attempt() -> None:
        nonlocal generation_attempts
        generation_attempts += 1

    for seed in (0, 1, 2):
        body = _generate_skill(
            prompt=generation_prompt(task_manifest["construction"], seed=seed),
            model_id=model_id,
            on_attempt=count_generation_attempt,
        )
        skill = freeze_skill(body, (), (), "ScienceWorld train-side generated skill", "scienceworld", seed)
        verification = verify_skill(adapter, skill, adapter.load_tasks("verification"), repeat_seed=seed)
        (out_dir / f"candidate_seed_{seed}.skill.txt").write_text(body + "\n", encoding="utf-8")
        (out_dir / f"verification_seed_{seed}.json").write_text(json.dumps(verification.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidates.append(summarize_candidate(generation_seed=seed, skill_hash=skill.skill_hash, verification=verification.__dict__))
    admitted = [candidate for candidate in candidates if candidate["outcome"] == "ADMITTED"]
    report = {
        "schema_version": 1,
        "stage": "scienceworld_train_formal",
        "provider": "deepseek",
        "model_id": model_id,
        "generation_seeds": [0, 1, 2],
        "candidates": candidates,
        "admitted_candidates": len(admitted),
        "provider_calls": generation_attempts + int(getattr(backend, "provider_calls", 0)),
        "acre_started": False,
        "final_test_started": False,
        "claim_boundary": "train-side SkillGen construction and verification; no held-out treatment efficacy",
        "next_stage_authorized": "acre_train_side_only" if admitted else "stop_before_acre_and_final_test",
    }
    (out_dir / "train_formal_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--scienceworld-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    args = parser.parse_args()
    report = run_train_formal(
        task_manifest=json.loads(args.task_manifest.read_text(encoding="utf-8")),
        scienceworld_root=args.scienceworld_root,
        out_dir=args.out_dir,
        model_id=args.model,
    )
    print(json.dumps({"stage": report["stage"], "provider_calls": report["provider_calls"], "admitted_candidates": report["admitted_candidates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
