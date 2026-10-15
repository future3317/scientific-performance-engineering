"""Exercise the official train environment with admissible commands, without a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.external_real.alfworld_runner import _alfworld_config, _select_game_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = _alfworld_config(args.data_root, [3, 4, 5], 60)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    official = AlfredTWEnv(config, train_eval="train")
    official.game_files = _select_game_files(official.game_files, 3, 0)
    official.num_games = len(official.game_files)
    env = official.init_env(batch_size=1)
    env.seed(0)
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
    (args.output / "REPORT.md").write_text(
        f"# Native ALFWorld integration\n\n{len(rows)} resets and batched legal steps completed.\n"
        "Provider calls: 0. This probe makes no success-rate claim.\n")
    print(json.dumps({"resets": len(rows), "env_steps": sum(r["env_steps"] for r in rows), "provider_calls": 0}))


if __name__ == "__main__":
    main()
