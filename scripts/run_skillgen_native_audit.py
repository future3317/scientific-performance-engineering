#!/usr/bin/env python3
"""Small train-side native ScienceWorld audit of the released SkillGen shape.

This is intentionally not named or reported as an exact paper reproduction:
the server cannot download the released Qwen checkpoint, so the audit uses the
configured provider model while preserving the upstream task wrapper, prompt,
trajectory graph, and skill-conditioned execution structure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.formal.host_agent import _api_call_with_retry


SIMPLIFICATIONS = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"


def area_under_progress_curve(curve: Sequence[float]) -> float:
    if not curve:
        return 0.0
    return sum(float(value) for value in curve) / len(curve)


def paired_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    deltas = [float(row["skill_score"]) - float(row["baseline_score"]) for row in rows]
    return {
        "n": len(rows),
        "mean_paired_score_delta": statistics.mean(deltas) if deltas else 0.0,
        "median_paired_score_delta": statistics.median(deltas) if deltas else 0.0,
        "positive_score_deltas": sum(delta > 0 for delta in deltas),
        "negative_score_deltas": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "completion_repairs": sum(
            not bool(row["baseline_success"]) and bool(row["skill_success"]) for row in rows
        ),
        "completion_regressions": sum(
            bool(row["baseline_success"]) and not bool(row["skill_success"]) for row in rows
        ),
    }


def load_scienceworld_rows(path: Path, *, limit: int) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sorted(
        rows,
        key=lambda row: (
            str(row["additional_info"]["env_name"]),
            int(row["additional_info"]["var"]),
        ),
    )[:limit]
    if len(selected) < limit:
        raise ValueError(f"{path} contains only {len(selected)} rows; need {limit}")
    return selected


def _parse_action(raw: object) -> str:
    text = str(raw).strip().lower()
    if "action:" in text:
        text = text.split("action:", 1)[1]
    text = text.splitlines()[0].split(",", 1)[0].strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text.rstrip(".")


def _prompt(
    prompt_dict: Mapping[str, object],
    *,
    goal: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    skill_text: str,
) -> list[dict[str, str]]:
    history_text = "\n".join(f"{kind}: {value}" for kind, value in history[-20:])
    user = (
        str(prompt_dict["instruction"])
        + "\nHere is the upstream ScienceWorld example:\n"
        + str(prompt_dict["examples"])
        + f"\nYou should perform actions to accomplish the goal: {goal}\n"
        + "Use one action per response and return only the action.\n"
        + f"Current observation: {observation}\n"
        + f"History:\n{history_text}\n"
    )
    if skill_text:
        user += "\nFrozen SkillGen graph guidance:\n" + skill_text + "\n"
    user += "Action:"
    return [
        {"role": "system", "content": str(prompt_dict["system_msg"])},
        {"role": "user", "content": user},
    ]


def _call_action(*, model_id: str, messages: Sequence[Mapping[str, str]]) -> str:
    response = _api_call_with_retry(
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MP_API_KEY", ""),
        {
            "model": model_id,
            "messages": list(messages),
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        },
        timeout_s=120,
    )
    return _parse_action(response["choices"][0]["message"].get("content", ""))


def run_episode(
    *,
    task: Mapping[str, object],
    skill_text: str,
    model_id: str,
    scienceworld_root: Path,
    prompt_dict: Mapping[str, object],
    max_steps: int,
) -> dict[str, object]:
    if str(scienceworld_root) not in sys.path:
        sys.path.insert(0, str(scienceworld_root))
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(
        "",
        str(scienceworld_root / "scienceworld" / "scienceworld.jar"),
        envStepLimit=max_steps,
    )
    actions: list[str] = []
    scores: list[float] = [0.0]
    valid_action_count = 0
    invalid_action_count = 0
    completed = False
    error: str | None = None
    try:
        env.load(
            str(task["additional_info"]["env_name"]),
            int(task["additional_info"]["var"]),
            simplificationStr=SIMPLIFICATIONS,
            generateGoldPath=True,
        )
        observation, _ = env.reset()
        history: list[tuple[str, str]] = [("OBSERVATION", str(observation))]
        for _ in range(max_steps):
            valid_actions = env.get_valid_action_object_combinations()
            action = _call_action(
                model_id=model_id,
                messages=_prompt(
                    prompt_dict,
                    goal=str(task["goal"]),
                    observation=str(observation),
                    history=history,
                    skill_text=skill_text,
                ),
            )
            actions.append(action)
            if action in valid_actions:
                valid_action_count += 1
            else:
                invalid_action_count += 1
            observation, _reward, completed, info = env.step(action)
            score = float(info.get("score", scores[-1]))
            scores.append(score)
            history.extend([("ACTION", action), ("OBSERVATION", str(observation))])
            if completed:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    final_score = float(scores[-1])
    return {
        "task_id": f"{task['additional_info']['env_name']}:{task['additional_info']['var']}",
        "task_name": task["additional_info"]["env_name"],
        "variation_id": int(task["additional_info"]["var"]),
        "score": final_score,
        "max_score": max(scores),
        "progress_rate": max(scores) / 100.0,
        "success": bool(completed and final_score >= 100.0),
        "completed": bool(completed),
        "valid_action_count": valid_action_count,
        "invalid_action_count": invalid_action_count,
        "valid_action_ratio": valid_action_count / len(actions) if actions else 0.0,
        "steps": len(actions),
        "termination_reason": "completed" if completed else (error or "step_budget_exhausted"),
        "score_curve": [value / 100.0 for value in scores],
        "aupc": area_under_progress_curve([value / 100.0 for value in scores]),
        "actions": actions,
        "provider_calls": len(actions),
        "execution_valid": error is None,
        "error": error,
    }


def build_graph_skill(train_rows: Sequence[Mapping[str, object]], *, upstream_root: Path, work_dir: Path) -> str:
    skill_root = str(upstream_root / "skill_extraction")
    if skill_root not in sys.path:
        sys.path.insert(0, skill_root)
    from domain_graph import ActionContributionEstimator, TaskGraph

    work_dir.mkdir(parents=True, exist_ok=True)
    graph = TaskGraph(save_dir=str(work_dir), task_name="scienceworld_audit")
    trajectories = []
    for row in train_rows:
        actions = list(row.get("actions", []))
        scores = list(row.get("score_curve", []))
        if not actions:
            continue
        progress = [0.0] + [float(value) for value in scores[1 : len(actions) + 1]]
        progress += [progress[-1]] * max(0, len(actions) + 2 - len(progress))
        sequence = [("INIT_OBS", "INIT_STATE", 0.0)]
        sequence.extend(("OBS", action, progress[index + 1]) for index, action in enumerate(actions))
        sequence.append(("FINAL_OBS", "FINAL_STATE", progress[-1]))
        trajectories.append(sequence)
    if not trajectories:
        raise ValueError("no train trajectory reached the graph builder")
    graph.add_sample_trajectory(trajectories)
    estimator = ActionContributionEstimator(graph, reward_noise_std=0.0)
    estimator.compute_q_values(num_iterations=20, patience=3)
    ranking = estimator.combined_ranking_step_wise()
    edges = [
        (parent, child, score)
        for parent, children in ranking.items()
        for child, score in children
    ]
    edges.sort(key=lambda item: (-float(item[2]), item[0], item[1]))
    if not edges:
        raise ValueError("upstream graph produced no ranked skill transitions")
    return "\n".join(
        f"- after `{parent}`, prefer `{child}` (contribution={float(score):.6f})"
        for parent, child, score in edges[:12]
    )


def run_audit(
    *,
    upstream_root: Path,
    scienceworld_root: Path,
    out_dir: Path,
    model_id: str,
    train_count: int,
    evaluation_count: int,
    max_steps: int,
) -> dict[str, object]:
    prompt_dict = json.loads(
        (upstream_root / "prompt" / "task" / "scienceworld_base.json").read_text(encoding="utf-8")
    )
    train = load_scienceworld_rows(upstream_root / "data" / "sc" / "train_data_0.jsonl", limit=train_count)
    evaluation = load_scienceworld_rows(upstream_root / "data" / "sc" / "test_data_0.jsonl", limit=evaluation_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "stage": "skillgen_upstream_style_native_feasibility_audit",
        "upstream_repo_revision": "816c91f458ddb87be32657c882cb0431b9cfea01",
        "benchmark_revision": "ScienceWorld-1.2.3",
        "model_id": model_id,
        "prompt": "released ScienceWorld base instruction/example plus graph skill guidance",
        "simplifications": SIMPLIFICATIONS,
        "train_file": "data/sc/train_data_0.jsonl",
        "evaluation_file": "data/sc/test_data_0.jsonl",
        "train_count": train_count,
        "evaluation_count": evaluation_count,
        "max_steps": max_steps,
        "provider_calls_authorized": True,
        "final_test_started": False,
        "acre_started": False,
        "claim_boundary": "upstream-style native feasibility audit; not SkillGen paper reproduction or ACRE efficacy",
    }
    (out_dir / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_rows = [
        run_episode(
            task=task,
            skill_text="",
            model_id=model_id,
            scienceworld_root=scienceworld_root,
            prompt_dict=prompt_dict,
            max_steps=max_steps,
        )
        for task in train
    ]
    skill_text = build_graph_skill(
        train_rows,
        upstream_root=upstream_root,
        work_dir=out_dir / "graph_work",
    )
    baseline_rows: list[dict[str, object]] = []
    skill_rows: list[dict[str, object]] = []
    for task in evaluation:
        baseline_rows.append(run_episode(
            task=task, skill_text="", model_id=model_id,
            scienceworld_root=scienceworld_root, prompt_dict=prompt_dict, max_steps=max_steps,
        ))
        skill_rows.append(run_episode(
            task=task, skill_text=skill_text, model_id=model_id,
            scienceworld_root=scienceworld_root, prompt_dict=prompt_dict, max_steps=max_steps,
        ))
    paired = [
        {
            "task_id": base["task_id"],
            "baseline_score": base["score"],
            "skill_score": skill["score"],
            "baseline_success": base["success"],
            "skill_success": skill["success"],
            "paired_score_delta": float(skill["score"]) - float(base["score"]),
            "progress_delta": float(skill["progress_rate"]) - float(base["progress_rate"]),
            "valid_action_delta": float(skill["valid_action_ratio"]) - float(base["valid_action_ratio"]),
        }
        for base, skill in zip(baseline_rows, skill_rows)
    ]
    def policy_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "n": len(rows),
            "mean_score": statistics.mean(float(row["score"]) for row in rows) if rows else 0.0,
            "progress_rate": statistics.mean(float(row["progress_rate"]) for row in rows) if rows else 0.0,
            "success_rate": statistics.mean(bool(row["success"]) for row in rows) if rows else 0.0,
            "grounding_rate": statistics.mean(float(row["valid_action_ratio"]) for row in rows) if rows else 0.0,
            "aupc": statistics.mean(float(row["aupc"]) for row in rows) if rows else 0.0,
            "execution_valid_rate": statistics.mean(bool(row["execution_valid"]) for row in rows) if rows else 0.0,
        }
    report = {
        **protocol,
        "provider_calls": sum(int(row["provider_calls"]) for row in train_rows + baseline_rows + skill_rows),
        "train_base": policy_metrics(train_rows),
        "evaluation_base": policy_metrics(baseline_rows),
        "evaluation_skill": policy_metrics(skill_rows),
        "paired": paired,
        "paired_summary": paired_summary(paired),
        "skill_package": skill_text,
        "native_skill_nonzero_progress": policy_metrics(skill_rows)["progress_rate"] > 0.0,
        "positive_mean_treatment_effect": paired_summary(paired)["mean_paired_score_delta"] > 0.0,
        "claim_boundary": "provider-backed upstream-style native feasibility audit; no held-out ACRE treatment claim",
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "skill_package.txt").write_text(skill_text + "\n", encoding="utf-8")
    (out_dir / "train_base_rows.json").write_text(json.dumps(train_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "evaluation_base_rows.json").write_text(json.dumps(baseline_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "evaluation_skill_rows.json").write_text(json.dumps(skill_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--scienceworld-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--evaluation-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    report = run_audit(
        upstream_root=args.upstream_root,
        scienceworld_root=args.scienceworld_root,
        out_dir=args.out_dir,
        model_id=args.model,
        train_count=args.train_count,
        evaluation_count=args.evaluation_count,
        max_steps=args.max_steps,
    )
    print(json.dumps({
        "provider_calls": report["provider_calls"],
        "positive_mean_treatment_effect": report["positive_mean_treatment_effect"],
        "native_skill_nonzero_progress": report["native_skill_nonzero_progress"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
