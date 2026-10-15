import json
from pathlib import Path

import pytest

from experiments.skill_evolution.readiness import plan_readiness
from experiments.skill_evolution.native_probe import probe_game_files


def _game(root, template, trial):
    path = root / "json_2.1.1" / "train" / template / trial / "game.tw-pddl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    task_type = template.rsplit("-", 1)[0]
    (path.parent / "traj_data.json").write_text(json.dumps({
        "task_type": task_type,
        "pddl_params": {
            "object_target": task_type, "parent_target": "cabinet",
            "toggle_target": "", "mrecep_target": "", "object_sliced": False,
        },
        "turk_annotations": {"anns": [{"task_desc": f"move object for {task_type}"}]},
        "scene": {"floor_plan": "FloorPlan1", "object_poses": [
            {"objectName": "Mug_abcd1234"}, {"objectName": "Cabinet_efgh5678"},
        ]},
    }))
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
    assert all(row["split"] == "train_inner" and row["template_digest"] for row in first["episodes"])
    assert all(row["identity_status"] == "IDENTIFIED" for row in first["episodes"])
    assert first["split_manifest"]["manifest_sha256"]
    assert first["split_manifest"]["untouched_eval"] == []
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


def test_native_probe_history_is_explicitly_exposed_and_eval_is_id_only(tmp_path):
    root = tmp_path / "data"
    families = ["pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                "pick_cool_then_place_in_recep"]
    probe_paths = [_game(root, family + "-probe", "trial_1") for family in families]
    planned_paths = [_game(root, family + "-new", "trial_1") for family in families]
    receipt = tmp_path / "native-probe" / "results.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "mode": "native_train_environment_probe",
        "episodes": [{"gamefile": str(path)} for path in probe_paths],
    }))
    plan = plan_readiness(root, [receipt], untouched_eval_ids=["opaque-task-1"])
    exposed = plan["split_manifest"]["exposure_ledger"]
    assert len(exposed) == 3
    assert {row["exposure"] for row in exposed} == {"native_probe_exposed"}
    assert {row["task_id"] for row in exposed} == {
        f"{path.parent.parent.name}/{path.parent.name}" for path in probe_paths
    }
    assert plan["split_manifest"]["untouched_eval"] == [{
        "task_id": "opaque-task-1", "split": "untouched_eval",
        "content_access": "ID_ONLY", "outcome_access": "PROHIBITED",
    }]
    assert all("gamefile" not in row for row in plan["split_manifest"]["untouched_eval"])
    assert {Path(row["gamefile"]).name for row in plan["episodes"]} == {"game.tw-pddl"}


def test_readiness_batch_is_derived_from_frozen_train_source_membership(tmp_path):
    from experiments.skill_evolution.readiness import plan_readiness
    from experiments.skill_evolution.split_manifest import build_episode_membership

    root = tmp_path / "data"
    families = ["pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                "pick_cool_then_place_in_recep"]
    paths = [_game(root, family + "-dev", "trial_1") for family in families]
    receipt = tmp_path / "history.json"
    receipt.write_text(json.dumps({"episodes": [{"gamefile": str(path)} for path in paths]}))
    split_manifest = {"revision": "frozen", "exposure_ledger": [], "task_membership": [
        build_episode_membership(path, data_root=root, family=family, split="train_inner",
                                 exposure="new_train_inner", exposure_sources=["fixture"])
        for family, path in zip(families, paths)]}
    plan = plan_readiness(root, [receipt], split_manifest=split_manifest)
    assert {row["split"] for row in plan["episodes"]} == {"train_inner"}
    assert {row["task_family"] for row in plan["episodes"]} == {"clean", "heat", "cool"}
    assert plan["split_manifest"] is split_manifest


def test_split_membership_rejects_template_repeated_across_seeds_and_possible_overlap():
    from experiments.skill_evolution.split_manifest import validate_split_membership

    repeated = [
        {"split": "train_outer", "task_id": "a/trial_1", "template_id": "a", "template_digest": "d1"},
        {"split": "train_outer", "task_id": "a/trial_2", "template_id": "a", "template_digest": "d2"},
    ]
    with pytest.raises(ValueError, match="template repeated"):
        validate_split_membership(repeated)
    possible = [
        {"split": "train_inner", "task_id": "a/trial_1", "template_id": "a",
         "template_digest": "d1", "possible_overlap_key": "same"},
        {"split": "train_outer", "task_id": "b/trial_1", "template_id": "b",
         "template_digest": "d2", "possible_overlap_key": "same"},
    ]
    with pytest.raises(ValueError, match="POSSIBLE_TEMPLATE_OVERLAP"):
        validate_split_membership(possible)


def test_unknown_template_identity_cannot_certify_train_outer_separation():
    from experiments.skill_evolution.split_manifest import validate_split_membership

    records = [
        {"split": "train_inner", "task_id": "a/trial_1", "template_id": "a"},
        {"split": "train_outer", "task_id": "b/trial_1", "template_id": "b"},
    ]
    with pytest.raises(ValueError, match="UNKNOWN_IDENTITY"):
        validate_split_membership(records)
