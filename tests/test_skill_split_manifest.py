import json

from experiments.skill_evolution.split_manifest import create_split_manifest, render_overlap_report


_TASK_TYPES = [
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]


def _write_game(data_root, template_id, *, task_type, scene, metadata=True):
    gamefile = data_root / "json_2.1.1" / "train" / template_id / "trial_1" / "game.tw-pddl"
    gamefile.parent.mkdir(parents=True, exist_ok=True)
    gamefile.touch()
    if metadata:
        (gamefile.parent / "traj_data.json").write_text(json.dumps({
            "task_type": task_type,
            "pddl_params": {
                "object_target": task_type, "parent_target": "cabinet",
                "toggle_target": "", "mrecep_target": "", "object_sliced": False,
            },
            "turk_annotations": {"anns": [{"task_desc": f"deliver an object for {task_type}"}]},
            "scene": {"floor_plan": scene, "object_poses": [
                {"objectName": "Mug_a1b2c3d4"}, {"objectName": "Cabinet_e5f6a7b8"},
            ]},
        }), encoding="utf-8")
    return gamefile


def _inputs(tmp_path, *, unknown_history=False):
    data_root = tmp_path / "data"
    history_paths = []
    for index in range(88):
        task_type = _TASK_TYPES[index % len(_TASK_TYPES)]
        path = _write_game(
            data_root, f"{task_type}-old-{index}", task_type=task_type,
            scene="OldLayout1", metadata=not (unknown_history and index == 0),
        )
        receipt = tmp_path / f"history-{index % 7}.json"
        if receipt.exists():
            rows = json.loads(receipt.read_text(encoding="utf-8"))["episodes"]
        else:
            rows = []
        rows.append({"gamefile": str(path)})
        receipt.write_text(json.dumps({"episodes": rows}), encoding="utf-8")
    history_paths = [tmp_path / f"history-{index}.json" for index in range(7)]

    probe_episodes = []
    for index, task_type in enumerate(_TASK_TYPES):
        path = _write_game(
            data_root, f"{task_type}-probe-{index}", task_type=task_type,
            scene="ProbeLayout1",
        )
        probe_episodes.append({"gamefile": str(path)})
    native_probe_results = tmp_path / "native-probe-results.json"
    native_probe_results.write_text(json.dumps({
        "mode": "native_train_environment_probe", "episodes": probe_episodes,
    }), encoding="utf-8")

    for task_type in _TASK_TYPES:
        for scene_index in range(3):
            _write_game(
                data_root, f"{task_type}-fresh-{scene_index}", task_type=task_type,
                scene=f"FreshLayout{scene_index}",
            )
    return data_root, history_paths, native_probe_results


def test_split_manifest_is_deterministic_and_keeps_eval_ids_only(tmp_path):
    data_root, history, probe = _inputs(tmp_path)
    kwargs = {
        "data_root": data_root, "history_paths": history,
        "native_probe_results": probe, "untouched_eval_ids": ["eval-id-1"],
        "revision": "abc123",
    }
    first = create_split_manifest(**kwargs)
    second = create_split_manifest(**kwargs)
    assert first == second
    assert first["revision"] == "abc123"
    assert first["exposure_counts"] == {
        "previous_88_templates": 88, "native_probe_exposed": 3,
    }
    assert {row["split"] for row in first["task_membership"]} >= {
        "train_dev", "train_inner", "train_outer",
    }
    assert all(row["confirmatory_eligible"] for row in first["task_membership"]
               if row["split"] in {"train_inner", "train_outer"})
    assert first["untouched_eval"] == [{
        "task_id": "eval-id-1", "split": "untouched_eval",
        "content_access": "ID_ONLY", "outcome_access": "PROHIBITED",
        "confirmatory_eligible": False,
    }]
    assert "gamefile" not in first["untouched_eval"][0]
    assert "Manifest SHA256" in render_overlap_report(first)
    report = render_overlap_report(first)
    assert "## Already-touched IDs and source trails" in report
    assert "history-" in report
    assert "## Unknown identities" in report


def test_unknown_historical_template_identity_blocks_confirmatory_qualification(tmp_path):
    data_root, history, probe = _inputs(tmp_path, unknown_history=True)
    manifest = create_split_manifest(
        data_root=data_root, history_paths=history, native_probe_results=probe,
        untouched_eval_ids=[], revision="abc123",
    )
    assert manifest["qualification_status"] == "BLOCKED_UNKNOWN_IDENTITY"
    assert not any(row["confirmatory_eligible"] for row in manifest["task_membership"])
    assert manifest["overlap_report"]["unknown_identity"]
    assert "UNKNOWN_IDENTITY" in render_overlap_report(manifest)


def test_template_grouping_uses_structured_target_relation_not_task_wording(tmp_path):
    from experiments.skill_evolution.split_manifest import _task_metadata

    data_root = tmp_path / "data"
    first = _write_game(data_root, "pick_and_place_simple-first", task_type="pick_and_place_simple",
                        scene="FloorPlan1")
    second = _write_game(data_root, "pick_and_place_simple-second", task_type="pick_and_place_simple",
                         scene="FloorPlan1")
    for gamefile, wording in ((first, "put the apple away"), (second, "store a red fruit")):
        metadata_path = gamefile.parent / "traj_data.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["turk_annotations"]["anns"][0]["task_desc"] = wording
        metadata["pddl_params"] = {
            "object_target": "Apple", "parent_target": "Cabinet",
            "toggle_target": "", "mrecep_target": "", "object_sliced": False,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert _task_metadata(first)["possible_overlap_key"] == _task_metadata(second)["possible_overlap_key"]

    metadata_path = second.parent / "traj_data.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["pddl_params"]["parent_target"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert _task_metadata(second)["identity_status"] == "UNKNOWN_IDENTITY"
