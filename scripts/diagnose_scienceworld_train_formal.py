#!/usr/bin/env python3
"""Audit what can and cannot be reconstructed from the frozen train-side run.

The original verification runner persisted aggregate reports only.  This tool
does not rerun provider calls and never imputes trajectory values that were not
persisted in that run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


TRAJECTORY_FIELDS = (
    "baseline_score",
    "skill_score",
    "baseline_completed",
    "skill_completed",
    "baseline_max_score",
    "skill_max_score",
    "baseline_positive_reward_steps",
    "skill_positive_reward_steps",
    "baseline_valid_action_count",
    "skill_valid_action_count",
    "baseline_invalid_action_count",
    "skill_invalid_action_count",
    "baseline_valid_action_ratio",
    "skill_valid_action_ratio",
    "baseline_steps",
    "skill_steps",
    "baseline_termination_reason",
    "skill_termination_reason",
    "paired_score_delta",
    "completion_transition",
    "progress_improvement",
    "regression",
    "provider",
    "model",
    "skill_injected",
    "skill_token_count",
    "system_prompt_hash",
    "user_prompt_hash",
    "max_steps",
    "token_budget",
    "parser_failures",
    "environment_errors",
)


def completion_transition(baseline_completed: bool, skill_completed: bool) -> str:
    return f"{'success' if baseline_completed else 'fail'}->{'success' if skill_completed else 'fail'}"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_frozen_diagnosis(
    task_manifest: Mapping[str, Sequence[Mapping[str, object]]],
    train_formal_dir: Path,
    *,
    generation_seeds: Sequence[int] = (0, 1, 2),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    tasks = list(task_manifest.get("verification", ()))
    if not tasks:
        raise ValueError("verification task manifest is empty")
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("verification task IDs must be non-empty and unique")

    rows: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []
    for raw_seed in generation_seeds:
        seed = int(raw_seed)
        report = _read_json(train_formal_dir / f"verification_seed_{seed}.json")
        pair_count = int(report.get("pair_count", -1))
        valid_count = int(report.get("execution_valid_count", -1))
        successful_count = int(report.get("skill_success_count", -1))
        if pair_count != len(tasks):
            raise ValueError(f"seed {seed} pair count does not match manifest")
        if not 0 <= valid_count <= pair_count or not 0 <= successful_count <= valid_count:
            raise ValueError(f"invalid aggregate counts for seed {seed}")

        known_all_valid = valid_count == pair_count
        known_no_skill_success = successful_count == 0
        candidate_summary = {
            "generation_seed": seed,
            "skill_hash": report.get("skill_hash"),
            "pair_count": pair_count,
            "execution_valid_count": valid_count,
            "skill_success_count": successful_count,
            "aggregate_supports_per_pair_execution_valid": known_all_valid,
            "aggregate_supports_per_pair_skill_failure": known_no_skill_success,
            "trajectory_fields_reconstructable": False,
            "classification": "underdetermined_from_frozen_aggregate_artifact",
        }
        candidate_summaries.append(candidate_summary)

        for task in tasks:
            row: dict[str, object] = {
                "generation_seed": seed,
                "skill_hash": report.get("skill_hash"),
                "task_id": str(task["task_id"]),
                "task_name": task.get("task_name"),
                "task_family": task.get("task_family"),
                "variation_id": task.get("variation_idx", task.get("variation_id")),
                "evidence_source": "frozen_train_formal_aggregate",
                "execution_valid": True if known_all_valid else None,
                "skill_full_success": False if known_no_skill_success else None,
                "data_status": "trajectory_not_persisted",
            }
            row.update({field: None for field in TRAJECTORY_FIELDS})
            rows.append(row)

    summary = {
        "schema_version": 1,
        "stage": "scienceworld_upstream_diagnosis",
        "source": "frozen train-side verification summaries",
        "provider_calls": 0,
        "candidate_count": len(candidate_summaries),
        "verification_task_count": len(tasks),
        "known_pair_count": len(rows),
        "known_execution_valid_pairs": sum(
            int(item["execution_valid_count"]) for item in candidate_summaries
        ),
        "known_skill_success_pairs": sum(
            int(item["skill_success_count"]) for item in candidate_summaries
        ),
        "trajectory_fields_reconstructable": False,
        "unreconstructable_pair_fields": list(TRAJECTORY_FIELDS),
        "candidates": candidate_summaries,
        "root_cause_status": "underdetermined_from_frozen_artifact",
    }
    return rows, summary


def write_diagnosis(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "execution_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    (out_dir / "candidate_summary.json").write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# ScienceWorld upstream diagnosis",
        "",
        "## Frozen evidence boundary",
        "",
        "This report reads the existing train-side verification summaries only. "
        "It performs zero provider calls and does not reopen ACRE or final-test execution.",
        "",
        "## Known facts",
        "",
        f"- candidate count: {summary['candidate_count']}",
        f"- verification pairs: {summary['known_pair_count']}",
        f"- execution-valid pairs (aggregate): {summary['known_execution_valid_pairs']}",
        f"- skill-success pairs (aggregate): {summary['known_skill_success_pairs']}",
        "- all three candidates were withheld by the existing global verification gate",
        "",
        "## Root-cause classification",
        "",
        "The frozen artifacts contain only per-candidate aggregate counts. They do "
        "not contain baseline/skill scores, action traces, valid/invalid action "
        "counts, termination reasons, prompt receipts, token usage, or injection "
        "receipts. Therefore A–E and F cannot be distinguished from this artifact "
        "without rerunning provider calls, and no such rerun is used here.",
        "",
        "| Hypothesis | Status from frozen evidence |",
        "|---|---|",
        "| A: base-agent/harness floor | not identifiable; baseline scores were not persisted |",
        "| B: skill systematically harms execution | not identifiable; paired scores/actions were not persisted |",
        "| C: binary admission is too coarse | not identifiable; continuous native scores were not persisted |",
        "| D: positive partial-score effect | not identifiable; paired score deltas were not persisted |",
        "| E: valid-action ratio collapses with skill | not identifiable; action validity was not persisted |",
        "| F: skill was not injected | not established; source path accepts the skill, but request receipts were not persisted |",
        "",
        "The immediate root cause of the forensic gap is telemetry loss at the "
        "verification boundary: aggregate `VerificationReport` objects were saved, "
        "but the 30 paired trajectories were not. The correct scientific status is "
        "`underdetermined_from_frozen_artifact`, not a claim that the base agent or "
        "the skill method failed for a particular mechanistic reason.",
        "",
        "## Consequence",
        "",
        "This artifact is train-side diagnostic evidence only. It does not authorize "
        "ACRE acquisition, held-out evaluation, C/D, or a SkillGen efficacy claim.",
    ]
    (out_dir / "root_cause_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--train-formal-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_frozen_diagnosis(
        _read_json(args.task_manifest),
        args.train_formal_dir,
    )
    write_diagnosis(rows, summary, args.out_dir)
    print(json.dumps({"pairs": len(rows), "status": summary["root_cause_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
