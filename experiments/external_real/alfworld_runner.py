"""Minimal text-only ALFWorld runner for the external audit."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_FAMILIES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "clean_place",
    4: "heat_place",
    5: "cool_place",
    6: "pick_two_obj_and_place",
}

TASK_TYPE_NAMES = {
    "pick_and_place_simple": "pick_and_place_simple",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean_place",
    "pick_heat_then_place_in_recep": "heat_place",
    "pick_cool_then_place_in_recep": "cool_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
}

SMOKE_TASK_TYPE_IDS = [3, 4, 5]


def task_family_name(task_type: int | str) -> str:
    """Return the stable family label for an official ALFWorld task type."""

    if isinstance(task_type, int):
        try:
            return TASK_FAMILIES[task_type]
        except KeyError as exc:
            raise ValueError(f"unknown ALFWorld task type: {task_type}") from exc
    value = str(task_type)
    if value.isdigit():
        return task_family_name(int(value))
    try:
        return TASK_TYPE_NAMES[value]
    except KeyError as exc:
        raise ValueError(f"unknown ALFWorld task type: {task_type}") from exc


def extract_action(raw_output: str, admissible_actions: Sequence[str]) -> str | None:
    """Extract one exact admissible command from a model response."""

    admissible = {str(action).strip().casefold(): str(action).strip() for action in admissible_actions}
    candidates: list[str] = []
    for line in str(raw_output).splitlines():
        match = re.match(r"^\s*(?:action|command)\s*:\s*(.*?)\s*$", line, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip().strip("`\"'"))
    candidates.append(str(raw_output).strip().strip("`\"'"))
    for candidate in candidates:
        if candidate.casefold() in admissible:
            return admissible[candidate.casefold()]
    return None


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def choose_action(self, observation: str, admissible_actions: Sequence[str], skill: str | None) -> dict[str, Any]:
        skill_text = skill or "No reusable skill is provided."
        prompt = (
            "You are playing ALFWorld text-only. Choose exactly one command from the admissible list. "
            "Reply with only `Action: <exact command>` and nothing else.\n\n"
            f"Reusable procedural skill:\n{skill_text}\n\n"
            f"Observation:\n{observation}\n\n"
            "Admissible commands:\n" + "\n".join(f"- {action}" for action in admissible_actions)
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.perf_counter() - started
        message = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return {
            "raw_output": message,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "latency_seconds": latency,
        }


def _alfworld_config(data_root: Path, task_types: Sequence[str], max_steps: int) -> dict[str, Any]:
    return {
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "task_types": list(task_types),
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "dataset": {
            "data_path": str(data_root / "json_2.1.1" / "train"),
            "eval_id_data_path": str(data_root / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(data_root / "json_2.1.1" / "valid_unseen"),
            "num_train_games": 0,
            "num_eval_games": 0,
        },
        "logic": {
            "domain": str(data_root / "logic" / "alfred.pddl"),
            "grammar": str(data_root / "logic" / "alfred.twl2"),
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


def _task_type_for_game(game_file: str) -> str:
    trajectory_path = Path(game_file).parent / "traj_data.json"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return str(data["task_type"])


def _select_game_files(game_files: Sequence[str], episodes: int, seed: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for game_file in sorted(game_files):
        grouped[_task_type_for_game(game_file)].append(game_file)
    ordered_types = sorted(grouped)
    if not ordered_types:
        raise ValueError("ALFWorld provided no eligible game files")
    selected: list[str] = []
    cursor = seed % len(ordered_types)
    while len(selected) < episodes:
        task_type = ordered_types[cursor % len(ordered_types)]
        if grouped[task_type]:
            selected.append(grouped[task_type].pop(0))
        cursor += 1
    return selected


def _scalar(info: Mapping[str, Sequence[Any]], key: str, index: int = 0) -> Any:
    values = info.get(key, [])
    return values[index] if values else None


def run_episode(environment: Any, client: DeepSeekClient, skill: str | None, max_steps: int) -> dict[str, Any]:
    observations, infos = environment.reset()
    observation = observations[0]
    admissible = list(_scalar(infos, "admissible_commands") or [])
    gamefile = _scalar(infos, "extra.gamefile")
    task_type = _task_type_for_game(gamefile)
    trajectory: list[dict[str, Any]] = []
    termination_reason = "step_limit"
    won = False

    for step_index in range(max_steps):
        completion = client.choose_action(observation, admissible, skill)
        action = extract_action(completion["raw_output"], admissible)
        if action is None:
            termination_reason = "invalid_action"
            trajectory.append({
                "step": step_index,
                "observation": observation,
                "admissible_actions": admissible,
                "raw_model_output": completion["raw_output"],
                "selected_action": None,
                "invalid_action": True,
                "reward": None,
                "done": True,
                "won": False,
                **{key: completion[key] for key in ("input_tokens", "output_tokens", "latency_seconds")},
            })
            break

        next_observations, rewards, dones, next_infos = environment.step([action])
        won = bool(_scalar(next_infos, "won") or False)
        done = bool(dones[0])
        trajectory.append({
            "step": step_index,
            "observation": observation,
            "admissible_actions": admissible,
            "raw_model_output": completion["raw_output"],
            "selected_action": action,
            "invalid_action": False,
            "reward": rewards[0],
            "done": done,
            "won": won,
            **{key: completion[key] for key in ("input_tokens", "output_tokens", "latency_seconds")},
        })
        if done:
            termination_reason = "success" if won else "environment_done"
            break
        observation = next_observations[0]
        admissible = list(_scalar(next_infos, "admissible_commands") or [])

    return {
        "gamefile": gamefile,
        "task_type": task_type,
        "task_family": task_family_name(task_type),
        "success": won,
        "steps": len(trajectory),
        "invalid_actions": sum(int(row["invalid_action"]) for row in trajectory),
        "termination_reason": termination_reason,
        "trajectory": trajectory,
    }


def run_smoke(
    data_root: Path,
    output_dir: Path,
    client: DeepSeekClient,
    episodes: int = 8,
    max_steps: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    task_types = list(SMOKE_TASK_TYPE_IDS)
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = _alfworld_config(data_root, task_types, max_steps)
    alfred = AlfredTWEnv(config, train_eval="train")
    selected = _select_game_files(alfred.game_files, episodes, seed)
    alfred.game_files = selected
    alfred.num_games = len(selected)
    environment = alfred.init_env(batch_size=1)
    environment.seed(seed)
    try:
        episode_rows = [run_episode(environment, client, skill=None, max_steps=max_steps) for _ in range(episodes)]
    finally:
        environment.close()

    result = {
        "schema_version": 1,
        "benchmark": "ALFWorld",
        "split": "train",
        "mode": "text_only_smoke",
        "episodes": episode_rows,
        "summary": {
            "episode_count": len(episode_rows),
            "success_rate": sum(row["success"] for row in episode_rows) / len(episode_rows),
            "invalid_action_count": sum(row["invalid_actions"] for row in episode_rows),
            "task_family_counts": {family: sum(row["task_family"] == family for row in episode_rows) for family in sorted({row["task_family"] for row in episode_rows})},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "benchmark": "ALFWorld",
        "split": "train",
        "mode": "text_only_smoke",
        "episodes": episodes,
        "task_types": task_types,
        "max_steps": max_steps,
        "seed": seed,
        "model": client.model,
        "temperature": client.temperature,
        "skill": None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(
        "# ALFWorld text-only smoke\n\n"
        f"- Episodes: **{len(episode_rows)}**\n"
        f"- Success rate: **{result['summary']['success_rate']:.3f}**\n"
        f"- Invalid actions: **{result['summary']['invalid_action_count']}**\n"
        "- Skill: none\n"
        "- All observations, admissible actions, model outputs, rewards, done/won signals, and termination reasons are persisted in `results.json`.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
    client = DeepSeekClient(api_key, args.base_url, args.model)
    run_smoke(args.data_root, args.output_dir, client, args.episodes, args.max_steps, args.seed)


if __name__ == "__main__":
    main()
