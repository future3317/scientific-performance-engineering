"""Train-side paired viability evaluation for frozen ALFWorld skills."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Collection, Sequence

from experiments.external_real.alfworld_runner import (
    DEFAULT_MODEL,
    DeepSeekClient,
    _alfworld_config,
    run_episode,
)


FAMILY_TASK_TYPES = {
    "clean_place": "pick_clean_then_place_in_recep",
    "heat_place": "pick_heat_then_place_in_recep",
    "cool_place": "pick_cool_then_place_in_recep",
}
FAMILY_TASK_IDS = {"clean_place": 3, "heat_place": 4, "cool_place": 5}


def choose_viability_paths(
    candidates: Sequence[str | Path],
    excluded: Collection[str | Path],
    count: int,
    seed: int,
) -> list[str]:
    """Select a deterministic set of non-synthesis trajectories."""

    excluded_paths = {str(path) for path in excluded}
    available = sorted(str(path) for path in candidates if str(path) not in excluded_paths)
    if len(available) < count:
        raise ValueError(f"need {count} non-synthesis episodes, found {len(available)}")
    return random.Random(seed).sample(available, count)


def _load_frozen_skills(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    skills = payload["skills"]
    loaded: dict[str, dict[str, Any]] = {}
    for family in FAMILY_TASK_TYPES:
        row = skills[family]
        skill_text = str(row["skill_text"])
        digest = hashlib.sha256(skill_text.encode("utf-8")).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"frozen skill hash mismatch for {family}")
        loaded[family] = row
    return loaded


def _candidate_paths(train_root: Path, task_type: str) -> list[Path]:
    paths = []
    for path in train_root.rglob("traj_data.json"):
        if (
            json.loads(path.read_text(encoding="utf-8")).get("task_type") == task_type
            and (path.parent / "game.tw-pddl").is_file()
        ):
            paths.append(path)
    return sorted(paths)


def _run_condition(
    data_root: Path,
    game_files: Sequence[str],
    task_type_id: int,
    client: DeepSeekClient,
    skill: str | None,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = _alfworld_config(data_root, [task_type_id], max_steps)
    alfred = AlfredTWEnv(config, train_eval="train")
    alfred.game_files = list(game_files)
    alfred.num_games = len(game_files)
    environment = alfred.init_env(batch_size=1)
    environment.seed(seed)
    try:
        return [run_episode(environment, client, skill=skill, max_steps=max_steps) for _ in game_files]
    finally:
        environment.close()


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "invalid_action_count": sum(row["invalid_actions"] for row in rows),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({row["termination_reason"] for row in rows})
        },
    }


def _align_episode_rows(
    rows: Sequence[dict[str, Any]], expected_game_files: Sequence[str]
) -> list[dict[str, Any]]:
    by_gamefile = {row["gamefile"]: row for row in rows}
    if len(by_gamefile) != len(rows) or set(by_gamefile) != set(expected_game_files):
        raise RuntimeError("environment did not return the selected gamefiles exactly once")
    return [by_gamefile[gamefile] for gamefile in expected_game_files]


def run_viability(
    data_root: Path,
    skill_results: Path,
    output_dir: Path,
    client: DeepSeekClient,
    episodes_per_family: int = 20,
    max_steps: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    """Run paired base/skill episodes on train tasks excluded from synthesis."""

    frozen_skills = _load_frozen_skills(skill_results)
    train_root = data_root / "json_2.1.1" / "train"
    families: dict[str, Any] = {}
    for family, task_type in FAMILY_TASK_TYPES.items():
        candidates = _candidate_paths(train_root, task_type)
        excluded = frozen_skills[family]["demonstration_paths"]
        selected_traj = choose_viability_paths(candidates, excluded, episodes_per_family, seed)
        selected_games = [str(Path(path).parent / "game.tw-pddl") for path in selected_traj]
        base_rows = _align_episode_rows(_run_condition(
            data_root, selected_games, FAMILY_TASK_IDS[family], client, None, max_steps, seed
        ), selected_games)
        skill_rows = _align_episode_rows(_run_condition(
            data_root,
            selected_games,
            FAMILY_TASK_IDS[family],
            client,
            frozen_skills[family]["skill_text"],
            max_steps,
            seed,
        ), selected_games)
        families[family] = {
            "task_type": task_type,
            "skill_sha256": frozen_skills[family]["sha256"],
            "selected_traj_data": selected_traj,
            "selected_game_files": selected_games,
            "base": {"summary": _summary(base_rows), "episodes": base_rows},
            "skill": {"summary": _summary(skill_rows), "episodes": skill_rows},
            "success_rate_difference": _summary(skill_rows)["success_rate"] - _summary(base_rows)["success_rate"],
        }

    provider_calls = sum(
        len(episode["trajectory"])
        for family in families.values()
        for condition in (family["base"], family["skill"])
        for episode in condition["episodes"]
    )
    result = {
        "schema_version": 1,
        "benchmark": "ALFWorld",
        "split": "train",
        "mode": "paired_skill_viability",
        "families": families,
        "provider_calls": provider_calls,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "ALFWorld",
                "split": "train",
                "mode": "paired_skill_viability",
                "episodes_per_family": episodes_per_family,
                "families": list(FAMILY_TASK_TYPES),
                "selection_seed": seed,
                "episode_seed": seed,
                "max_steps": max_steps,
                "model": client.model,
                "temperature": client.temperature,
                "thinking": "disabled",
                "skill_results": str(skill_results),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# ALFWorld train-side skill viability",
        "",
        "Each frozen family skill was evaluated against the same DeepSeek base agent on 20",
        "train episodes not used for synthesis. The two conditions share the selected gamefiles",
        "and seed; valid episodes and full trajectories are stored in `results.json`.",
        "",
    ]
    for family, row in families.items():
        report_lines.extend(
            [
                f"## {family}",
                "",
                f"- Skill SHA256: `{row['skill_sha256']}`",
                f"- Base success: **{row['base']['summary']['success_count']}/{episodes_per_family}**",
                f"- Skill success: **{row['skill']['summary']['success_count']}/{episodes_per_family}**",
                f"- Success-rate difference (skill - base): **{row['success_rate_difference']:.3f}**",
                f"- Base invalid actions: **{row['base']['summary']['invalid_action_count']}**",
                f"- Skill invalid actions: **{row['skill']['summary']['invalid_action_count']}**",
                "",
            ]
        )
    report_lines.extend([f"- Provider calls: **{result['provider_calls']}**", ""])
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--skill-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-family", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
    client = DeepSeekClient(api_key, args.base_url, args.model)
    run_viability(
        args.data_root,
        args.skill_results,
        args.output_dir,
        client,
        args.episodes_per_family,
        args.max_steps,
        args.seed,
    )


if __name__ == "__main__":
    main()
