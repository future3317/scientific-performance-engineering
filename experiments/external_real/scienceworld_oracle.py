from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


def action_is_valid_before_step(action: str, valid_actions: Sequence[str]) -> bool:
    return str(action) in {str(item) for item in valid_actions}


def select_oracle_cases(
    split_cases: Mapping[str, Sequence[tuple[str, int]]],
    *,
    limit: int,
) -> list[tuple[str, int, str]]:
    if limit <= 0:
        raise ValueError("oracle limit must be positive")
    cases = [
        (str(task_name), int(variation_idx), str(split))
        for split in ("train", "dev")
        for task_name, variation_idx in split_cases.get(split, ())
    ]
    cases.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(cases) < limit:
        raise ValueError(f"only {len(cases)} train/dev cases available; need {limit}")
    return cases[:limit]


def oracle_gate(rows: Sequence[Mapping[str, object]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not rows:
        return False, ["oracle case set is empty"]
    if any(not bool(row.get("completed", False)) for row in rows):
        reasons.append("one or more gold paths did not complete")
    if any(float(row.get("score", 0.0)) < 99.0 for row in rows):
        reasons.append("one or more gold paths scored below 99")
    return not reasons, reasons


def run_oracle(
    *,
    scienceworld_root: Path,
    out_dir: Path,
    limit: int = 10,
) -> dict[str, object]:
    root = scienceworld_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", str(root / "scienceworld" / "scienceworld.jar"), envStepLimit=200)
    split_cases: dict[str, list[tuple[str, int]]] = {"train": [], "dev": []}
    try:
        for task_name in sorted(env.get_task_names()):
            env.load(task_name, 0, "")
            split_cases["train"].extend((task_name, int(index)) for index in env.get_variations_train())
            split_cases["dev"].extend((task_name, int(index)) for index in env.get_variations_dev())
        cases = select_oracle_cases(split_cases, limit=limit)
        rows: list[dict[str, object]] = []
        for task_name, variation_idx, split in cases:
            env.load(task_name, variation_idx, "", True)
            env.reset()
            actions = env.get_gold_action_sequence()
            score = 0.0
            completed = False
            valid_action_count = 0
            invalid_action_count = 0
            max_score = 0.0
            steps = 0
            for action in actions:
                valid_actions_before_step = env.get_valid_action_object_combinations()
                action_valid = action_is_valid_before_step(action, valid_actions_before_step)
                _observation, _reward, completed, info = env.step(action)
                steps += 1
                score = float(info.get("score", score))
                max_score = max(max_score, score)
                if action_valid:
                    valid_action_count += 1
                else:
                    invalid_action_count += 1
                if completed:
                    break
            rows.append({
                "task_name": task_name,
                "variation_id": variation_idx,
                "split": split,
                "gold_action_count": len(actions),
                "steps": steps,
                "score": score,
                "max_score": max_score,
                "completed": bool(completed),
                "valid_action_count": valid_action_count,
                "invalid_action_count": invalid_action_count,
                "termination_reason": "completed" if completed else "gold_path_exhausted",
            })
        passed, reasons = oracle_gate(rows)
        report = {
            "schema_version": 1,
            "stage": "scienceworld_native_oracle_sanity",
            "benchmark_revision": "ScienceWorld-1.2.3",
            "provider_calls": 0,
            "final_test_started": False,
            "cases": rows,
            "gate_passed": passed,
            "gate_reasons": reasons,
            "claim_boundary": "train/dev native environment and official evaluator sanity only",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "oracle_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scienceworld-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    report = run_oracle(
        scienceworld_root=args.scienceworld_root,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print(json.dumps({"cases": len(report["cases"]), "gate_passed": report["gate_passed"]}))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
