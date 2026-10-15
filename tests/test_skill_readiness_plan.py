import json
from pathlib import Path

import pytest

from experiments.skill_evolution.readiness import plan_readiness
from experiments.skill_evolution.native_probe import probe_game_files


def _game(root, template, trial):
    path = root / "json_2.1.1" / "train" / template / trial / "game.tw-pddl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_plan_excludes_entire_old_template_across_trials_and_relocated_roots(tmp_path):
    root = tmp_path / "data"
    prefix = "pick_clean_then_place_in_recep"
    old = _game(root, prefix + "-old", "trial_1")
    _game(root, prefix + "-old", "trial_2")
    for family in (prefix, "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep"):
        _game(root, family + "-new", "trial_1")
    receipt = tmp_path / "history.json"
    receipt.write_text(json.dumps({"skills": {"clean": {
        "demonstration_paths": ["/previous/root/json_2.1.1/train/" + prefix + "-old/trial_1/traj_data.json"],
    }}, "episodes": [{"gamefile": str(old)}]}))
    first = plan_readiness(root, [receipt])
    second = plan_readiness(root, [receipt])
    assert first == second
    assert first["provider_calls"] == 0
    assert len(first["episodes"]) == 3
    assert len({row["task_template"] for row in first["episodes"]}) == 3
    assert all(row["task_template"] not in first["excluded_task_templates"] for row in first["episodes"])
    assert all("-new" in row["episode_id"] for row in first["episodes"])
    assert first["config"]["provider_call_cap"] == 180
    assert first["launch_blockers"]
    available = [row["gamefile"] for row in first["episodes"]]
    assert probe_game_files(available, first) == available
    with pytest.raises(ValueError, match="official train environment"):
        probe_game_files(available[:-1], first)


def test_missing_or_unrecognized_history_cannot_silently_mean_no_exclusions(tmp_path):
    with pytest.raises(ValueError, match="historical train receipts"):
        plan_readiness(tmp_path, [])
    receipt = tmp_path / "unknown.json"
    receipt.write_text('{"summary": {"successes": 0}}')
    with pytest.raises(ValueError, match="no recognized train episode"):
        plan_readiness(tmp_path, [receipt])


def test_heldout_episode_reference_is_not_accepted_as_train_history(tmp_path):
    receipt = tmp_path / "heldout.json"
    receipt.write_text(json.dumps({"episodes": [{"gamefile": "/data/json_2.1.1/valid_unseen/task/trial/game.tw-pddl"}]}))
    with pytest.raises(ValueError, match="official train split"):
        plan_readiness(tmp_path, [receipt])


def test_insufficient_fresh_families_does_not_duplicate_an_episode(tmp_path):
    old = _game(tmp_path, "pick_clean_then_place_in_recep-old", "trial_1")
    receipt = tmp_path / "history.json"
    receipt.write_text(json.dumps({"episodes": [{"gamefile": str(old)}]}))
    with pytest.raises(ValueError, match="no fresh compiled"):
        plan_readiness(tmp_path, [receipt])
