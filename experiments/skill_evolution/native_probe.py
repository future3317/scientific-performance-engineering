"""Exercise the official train environment with admissible commands, without a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.external_real.alfworld_runner import _alfworld_config, _select_game_files


def probe_game_files(available: list[str], plan: dict | None = None) -> list[str]:
    if plan is None:
        return _select_game_files(available, 3, 0)
    if plan.get("status") != "DRY_RUN_NOT_EXECUTED" or plan.get("provider_calls") != 0:
        raise ValueError("expected an unexecuted provider-free readiness plan")
    selected = [row["gamefile"] for row in plan["episodes"]]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("readiness probe requires three distinct planned episodes")
    if not set(selected).issubset(available):
        raise ValueError("planned episodes are not in the official train environment")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path, help="use the fresh episodes selected by skill_evolution.readiness")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text()) if args.plan else None
    config = _alfworld_config(args.data_root, [3, 4, 5], 60)
    (args.output / "config.json").write_text(json.dumps({
        "environment": config, "readiness_plan": str(args.plan) if args.plan else None,
        "readiness_revision": plan.get("code_revision") if plan else None,
    }, indent=2) + "\n")
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    official = AlfredTWEnv(config, train_eval="train")
    official.game_files = probe_game_files(official.game_files, plan)
    official.num_games = len(official.game_files)
    env = official.init_env(batch_size=1)
    env.seed(plan["selection_seed"] if plan else 0)
    rows = []
    started = time.monotonic()
    try:
        for _ in range(3):
            observations, info = env.reset()
            admissible = info["admissible_commands"][0]
            action = next((a for a in admissible if a != "look"), admissible[0])
            after, rewards, dones, next_info = env.step([action])
            rows.append({
                "gamefile": info.get("extra.gamefile", [None])[0],
                "observation": observations[0], "admissible_commands": admissible,
                "action_argument": [action], "next_observation": after[0],
                "reward": rewards[0], "done": bool(dones[0]),
                "won": next_info.get("won", [None])[0],
                "env_steps": 1, "admissible": action in admissible,
            })
    finally:
        env.close()
        result = {"mode": "native_train_environment_probe", "provider_calls": 0,
                  "episodes": rows, "wall_seconds": time.monotonic() - started,
                  "interpretation": "Environment integration only; no agent or skill efficacy."}
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if plan is not None and {row["gamefile"] for row in rows} != set(official.game_files):
        raise ValueError("native resets did not cover exactly the three planned train episodes")
    (args.output / "REPORT.md").write_text(
        f"# Native ALFWorld integration\n\n{len(rows)} resets and batched legal steps completed.\n"
        "Provider calls: 0. This probe makes no success-rate claim.\n")
    print(json.dumps({"resets": len(rows), "env_steps": sum(r["env_steps"] for r in rows), "provider_calls": 0}))


if __name__ == "__main__":
    main()
