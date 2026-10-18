"""Train-only official ALFWorld outcome scoring and teacher diagnostics."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .split_manifest import _train_parts


def official_outcome(done: Any, won: Any) -> dict[str, Any]:
    """Score the official terminal contract without inferring success from reward."""
    done = _native_bool(done)
    won = _native_bool(won)
    if done is None or won is None:
        return {"success": None, "status": "INCONCLUSIVE"}
    if done and won:
        return {"success": True, "status": "VERIFIED_SUCCESS"}
    if done:
        return {"success": False, "status": "VERIFIED_FAILURE"}
    if won:
        return {"success": None, "status": "VERIFIER_INCONSISTENT"}
    return {"success": None, "status": "IN_PROGRESS"}


def _native_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    # TextWorld/Gym versions may expose NumPy's scalar boolean for vectorized envs.
    if type(value).__name__ == "bool_":
        return bool(value)
    return None


def make_selected_environment(gamefile: str, *, max_steps: int, seed: int) -> Any:
    """Official native wrappers for exactly one frozen game, without a train scan.

    AlfredTWEnv.__init__ collects metadata for the entire train population.
    Registering the selected game directly keeps reserved task files unopened.
    The wrappers/infos/horizon match its dqn init_env path, with no expert.
    """
    _train_parts(gamefile)
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    env_id = textworld.gym.register_games(
        [gamefile], textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"]),
        batch_size=1, asynchronous=True, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    environment = textworld.gym.make(env_id)
    environment.seed(seed)
    return environment


def score_official_episode(
    *,
    done: Any,
    won: Any,
    reward: Any = None,
    steps: int = 0,
    max_steps: int,
    action_admissible: bool = True,
    environment_available: bool = True,
) -> dict[str, Any]:
    """Add terminal, budget and API-contract semantics to the official scorer."""
    if not environment_available:
        status = "ENVIRONMENT_FAILURE"
    elif not action_admissible:
        status = "API_CONTRACT_FAILURE"
    else:
        outcome = official_outcome(done, won)
        status = outcome["status"]
        if status == "IN_PROGRESS" and steps >= max_steps:
            status = "VERIFIED_FAILURE"
    success = status == "VERIFIED_SUCCESS"
    if status not in {"VERIFIED_SUCCESS", "VERIFIED_FAILURE"}:
        success_value: bool | None = None
    else:
        success_value = success
    return {
        "success": success_value,
        "status": status,
        "done": _native_bool(done),
        "won": _native_bool(won),
        "reward_observed": reward is not None,
        "steps": steps,
        "max_steps": max_steps,
    }


def _first(info: Mapping[str, Any], key: str) -> Any:
    value = info.get(key)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _expert_action(info: Mapping[str, Any]) -> Any:
    plan = _first(info, "extra.expert_plan")
    if isinstance(plan, (list, tuple)):
        return plan[0] if plan else None
    return plan


def teacher_diagnostic_episode(environment: Any, *, split: str, max_steps: int) -> dict[str, Any]:
    """Run one official train expert episode; its outcome is never agent evidence."""
    if split != "train":
        raise ValueError("teacher/oracle diagnostics are restricted to ALFWorld train")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    _, info = environment.reset()
    gamefile = _first(info, "extra.gamefile")
    if not isinstance(gamefile, str):
        return _teacher_result(None, [], score_official_episode(
            done=None, won=None, max_steps=max_steps, steps=0, environment_available=False,
        ))
    _train_parts(gamefile)
    trace = []
    done: Any = False
    won: Any = False
    reward: Any = None
    status = "IN_PROGRESS"

    for step in range(max_steps):
        admissible = _first(info, "admissible_commands") or []
        if not isinstance(admissible, Sequence) or isinstance(admissible, str):
            admissible = []
        action = _expert_action(info)
        if not isinstance(action, str) or action not in admissible:
            outcome = score_official_episode(
                done=False, won=False, max_steps=max_steps, steps=step,
                action_admissible=False,
            )
            return _teacher_result(gamefile, trace, outcome)

        next_observations, rewards, dones, next_info = environment.step([action])
        reward = rewards[0] if rewards else None
        done = dones[0] if dones else None
        won = _first(next_info, "won")
        trace.append({"step": step, "action": action, "reward": reward,
                      "done": done, "won": won})
        info = next_info
        status = official_outcome(done, won)["status"]
        if status != "IN_PROGRESS":
            break

    outcome = score_official_episode(
        done=done, won=won, reward=reward, max_steps=max_steps,
        steps=len(trace),
    )
    return _teacher_result(gamefile, trace, outcome)


def _teacher_result(gamefile: str | None, trace: list[dict[str, Any]], outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_kind": "teacher_or_scripted_diagnostic",
        "split": "train",
        "gamefile": gamefile,
        "teacher_success": outcome["success"],
        "agent_success": False,
        "status": outcome["status"],
        "outcome": outcome,
        "provider_calls": 0,
        "trace": trace,
    }


def _official_teacher_config(data_root: Path, gamefile: Path, max_steps: int) -> dict[str, Any]:
    _train_parts(str(gamefile))
    metadata_path = gamefile.parent / "traj_data.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_type = metadata.get("task_type")
    from experiments.external_real.alfworld_runner import TASK_FAMILIES, TASK_TYPE_NAMES, _alfworld_config

    family = TASK_TYPE_NAMES.get(task_type)
    task_type_id = next((key for key, name in TASK_FAMILIES.items() if name == family), None)
    if family is None or task_type_id is None:
        raise ValueError(f"unsupported ALFWorld task_type in train metadata: {task_type!r}")
    config = _alfworld_config(data_root, [task_type_id], max_steps)
    config["general"]["training_method"] = "dagger"
    config["env"]["expert_type"] = "handcoded"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps
    return config


def run_official_train_teacher(data_root: Path, gamefile: Path, *, max_steps: int = 60) -> dict[str, Any]:
    """Build the official train wrapper for one game and run its handcoded expert."""
    config = _official_teacher_config(data_root, gamefile, max_steps)
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    official = AlfredTWEnv(config, train_eval="train")
    official.game_files = [str(gamefile)]
    official.num_games = 1
    environment = official.init_env(batch_size=1)
    try:
        return teacher_diagnostic_episode(environment, split="train", max_steps=max_steps)
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gamefile", type=Path, required=True,
                        help="one compiled game.tw-pddl under json_2.1.1/train")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="fresh output directory for config, native logs, result, and report")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = _official_teacher_config(args.data_root, args.gamefile, args.max_steps)
    (args.output_dir / "config.json").write_text(json.dumps({
        "mode": "teacher_or_scripted_diagnostic", "split": "train",
        "gamefile": str(args.gamefile), "max_steps": args.max_steps,
        "provider_calls": 0, "environment_config": config,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "official_environment.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            result = run_official_train_teacher(args.data_root, args.gamefile, max_steps=args.max_steps)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    summary = (
        "# Official ALFWorld train teacher diagnostic\n\n"
        f"- Status: `{result['status']}`\n"
        f"- Teacher success: `{result['teacher_success']}`\n"
        f"- Agent success counted: `{result['agent_success']}`\n"
        f"- Environment steps: {len(result['trace'])}\n"
        "- Provider calls: 0\n"
        "- Evidence kind: `teacher_or_scripted_diagnostic` (not agent or Skill efficacy)\n"
    )
    (args.output_dir / "REPORT.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": result["status"], "steps": len(result["trace"]),
                      "provider_calls": 0, "agent_success": False,
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
