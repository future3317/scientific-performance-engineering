import json
import copy
from pathlib import Path

import pytest

from experiments.skill_evolution.split_manifest import (
    _OFFICIAL_FAMILIES, _canonical_manifest_hash, build_split_extension,
    create_split_manifest, render_overlap_report, sample_train_dev,
)


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


def test_split_cli_writes_revision_manifest_and_overlap_report(monkeypatch, tmp_path):
    import sys
    from experiments.skill_evolution import split_manifest as module
    data_root, history, probe = _inputs(tmp_path)
    output = tmp_path / "split.json"
    monkeypatch.setattr(sys, "argv", ["split_manifest", "--data-root", str(data_root),
                                     "--history", *(str(path) for path in history),
                                     "--native-probe-results", str(probe), "--output", str(output)])
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda args, **kwargs: "abc123\n" if args[1] == "rev-parse" else "")
    module.main()
    assert json.loads(output.read_text())["revision"] == "abc123"
    assert (tmp_path / "SPLIT_OVERLAP_REPORT.md").is_file()


def test_demonstration_provenance_does_not_require_a_compiled_game(tmp_path):
    data_root, history, probe = _inputs(tmp_path)
    first = json.loads(history[0].read_text())["episodes"][0]["gamefile"]
    Path(first).unlink()
    result = create_split_manifest(data_root=data_root, history_paths=history,
                                   native_probe_results=probe, untouched_eval_ids=[], revision="abc123")
    assert result["qualification_status"] == "PARTITIONS_READY_FOR_REVIEW"


def _phase3_parent(*, dev_per_family=4, outer_count=460):
    rows = []
    for family in _OFFICIAL_FAMILIES:
        for index in range(dev_per_family):
            rows.append({
                "split": "train_dev", "family": family, "task_type": family,
                "task_id": f"dev-{family}-{index}", "cluster_id": f"dev-cluster-{family}-{index}",
                "template_id": f"dev-template-{family}-{index}",
                "template_digest": f"dev-digest-{family}-{index}",
                "possible_overlap_key": f"dev-overlap-{family}-{index}",
                "gamefile": f"/metadata-only/{family}-{index}/game.tw-pddl",
                "gamefile_id": f"{family}-{index}/game.tw-pddl",
                "identity_status": "IDENTIFIED", "goal_relation": {"oracle": "must not project"},
                "exposure": "new_train_dev",
                "outcome": "success", "high_pddl": ["hidden expert plan"],
            })
    for index in range(outer_count):
        family = _OFFICIAL_FAMILIES[index % len(_OFFICIAL_FAMILIES)]
        rows.append({
            "split": "train_outer", "family": family, "task_type": family,
            "task_id": f"outer-{index}", "cluster_id": f"outer-cluster-{index}",
            "template_id": f"outer-template-{index}", "template_digest": f"outer-digest-{index}",
            "possible_overlap_key": f"outer-overlap-{index}",
            "identity_status": "IDENTIFIED", "confirmatory_eligible": True,
            "goal_relation": {"oracle": "not an input"}, "outcome": "success",
            "high_pddl": ["hidden expert plan"],
        })
    parent = {
        "schema_version": "2", "study_id": "parent-study", "revision": "parent-rev",
        "task_membership": rows, "exposure_ledger": [],
        "overlap_report": {"already_touched_task_ids": []},
    }
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    return parent


def test_phase3_outer_reservation_is_repeatable_disjoint_and_leaves_parent_unchanged():
    parent = _phase3_parent()
    original_parent = copy.deepcopy(parent)
    first = build_split_extension(parent, study_id="study-v3", seed=17)
    second = build_split_extension(parent, study_id="study-v3", seed=17)
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8") == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert parent == original_parent
    assert first["parent_manifest_sha256"] == parent["manifest_sha256"]
    assert first["child_weights"] == {
        "outer_skill_validation": "1/3",
        "outer_gate_calibration": "1/3",
        "untouched_eval_reserved": "1/3",
    }
    assert first["family_allocation"]["pick_and_place_simple"]["untouched_eval_reserved_groups"] == 25
    children = first["child_membership"]
    assert len(children) == 460
    assert {row["child_split"] for row in children} == {
        "outer_skill_validation", "outer_gate_calibration", "untouched_eval_reserved",
    }
    assert len({row["cluster_id"] for row in children}) == 460
    assert all(row["parent_split"] == "train_outer" for row in children)
    assert first["reservation_summary"]["parent_outer_clusters"] == 460
    assert first["reservation_summary"]["assigned_outer_clusters"] == 460
    assert "precision" in first["precision_note"]
    assert "powered" in first["precision_note"]


def test_phase3_reservation_allocates_460_single_family_groups_153_to_final():
    parent = _phase3_parent(outer_count=460)
    # Keep the test's outer population in one official family to make the
    # requested 460-cluster largest-remainder allocation directly observable.
    for row in parent["task_membership"]:
        if row["split"] == "train_outer":
            row["family"] = row["task_type"] = "pick_and_place_simple"
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    extension = build_split_extension(parent, study_id="study-v3", seed=17)
    counts = extension["family_allocation"]["pick_and_place_simple"]
    assert counts["available_clusters"] == 460
    assert counts["untouched_eval_reserved_groups"] == 153
    assert counts["outer_skill_validation_groups"] == 154
    assert counts["outer_gate_calibration_groups"] == 153


def test_exposure_and_unknown_identity_quarantine_entire_overlap_group():
    parent = _phase3_parent(outer_count=6)
    rows = [row for row in parent["task_membership"] if row["split"] == "train_outer"]
    rows[1]["template_digest"] = rows[0]["template_digest"]
    rows[1]["possible_overlap_key"] = rows[0]["possible_overlap_key"]
    rows[2]["identity_status"] = "UNKNOWN_IDENTITY"
    parent["exposure_ledger"] = [{
        "task_id": rows[4]["task_id"], "template_id": rows[4]["template_id"],
        "identity_status": "IDENTIFIED", "exposure_sources": ["prior-receipt"],
    }]
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    extension = build_split_extension(
        parent, study_id="study-v3", seed=1, exposed_cluster_ids=[rows[5]["cluster_id"]],
    )
    quarantined = {row["task_id"] for row in extension["child_membership"]
                   if row["child_split"] == "quarantine_exposed"}
    assert {rows[index]["task_id"] for index in (0, 1, 2, 4, 5)} <= quarantined
    assert all(row["task_id"] not in quarantined for row in extension["child_membership"]
               if row["child_split"] != "quarantine_exposed")


def test_train_dev_sampler_is_equal_family_metadata_projection_and_outcome_invariant():
    parent = _phase3_parent(dev_per_family=5, outer_count=6)
    extension = build_split_extension(parent, study_id="study-v3", seed=2026)
    first = sample_train_dev(parent, extension, study_id="study-v3", seed=2026)
    changed = copy.deepcopy(parent)
    for row in changed["task_membership"]:
        row["outcome"] = "changed failure"
        row["high_pddl"] = ["different hidden plan"]
        row["goal_relation"] = {"different": "oracle"}
    changed["manifest_sha256"] = _canonical_manifest_hash(changed)
    changed_extension = build_split_extension(changed, study_id="study-v3", seed=2026)
    second = sample_train_dev(changed, changed_extension, study_id="study-v3", seed=2026)
    assert extension["child_membership"] == changed_extension["child_membership"]
    assert [row["task_id"] for row in first["selected"]] == [row["task_id"] for row in second["selected"]]
    assert [row["task_id"] for row in first["reserve_order"]] == [row["task_id"] for row in second["reserve_order"]]
    assert len(first["selected"]) == 12
    assert len({row["cluster_id"] for row in first["selected"]}) == 12
    assert all(first["family_summary"][family]["selected_clusters"] == 2 for family in _OFFICIAL_FAMILIES)
    assert all("gamefile" in row for row in first["selected"])
    assert all(row["task_type"] == row["family"] and row["split"] == "train_dev"
               and row["exposure"] == "new_train_dev" and row["identity_status"] == "IDENTIFIED"
               for row in first["selected"] + first["reserve_order"])
    assert all(not {"goal_relation", "outcome", "high_pddl"} & row.keys()
               for row in first["selected"] + first["reserve_order"])


def test_sampler_excludes_extra_exposure_and_refuses_shortage():
    parent = _phase3_parent(dev_per_family=2, outer_count=6)
    extension = build_split_extension(
        parent, study_id="study-v3", seed=9,
        exposed_template_ids=[f"dev-template-{_OFFICIAL_FAMILIES[0]}-0"],
    )
    try:
        sample_train_dev(parent, extension, study_id="study-v3", seed=9)
    except ValueError as error:
        assert "insufficient unexposed train_dev clusters" in str(error)
    else:
        raise AssertionError("sampler silently fell back despite a family shortage")


def test_parent_hash_and_unknown_outer_metadata_fail_closed():
    parent = _phase3_parent(outer_count=6)
    parent["task_membership"][-1]["cluster_id"] = "tampered"
    try:
        build_split_extension(parent, study_id="study-v3", seed=1)
    except ValueError as error:
        assert "parent manifest hash" in str(error)
    else:
        raise AssertionError("tampered parent manifest was accepted")


def test_sampler_deduplicates_transitive_dev_cluster_and_template_groups():
    parent = _phase3_parent(dev_per_family=3, outer_count=6)
    original = parent["task_membership"][0]
    same_cluster = dict(original, task_id="duplicate-cluster-task", template_id="another-template",
                        template_digest="another-digest", possible_overlap_key="another-overlap")
    same_template = dict(original, task_id="duplicate-template-task", cluster_id="another-cluster",
                         template_id="third-template", possible_overlap_key="third-overlap")
    # This row joins the component through the first duplicate's source template.
    transitive = dict(original, task_id="transitive-task", cluster_id="third-cluster",
                      template_id="fourth-template", template_digest="fourth-digest",
                      possible_overlap_key="fourth-overlap", source_template_ids=["another-template"])
    parent["task_membership"].extend([same_cluster, same_template, transitive])
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    extension = build_split_extension(parent, study_id="study-v3", seed=2026)
    sample = sample_train_dev(parent, extension, study_id="study-v3", seed=2026)
    family = original["family"]
    assert sample["family_summary"][family]["eligible_clusters"] == 3
    assert sample["family_summary"][family]["inclusion_probability"] == 2 / 3
    family_reserve = [row for row in sample["reserve_order"] if row["family"] == family]
    assert len(family_reserve) == 3
    assert any(set(row["group_task_ids"]) == {
        original["task_id"], "duplicate-cluster-task", "duplicate-template-task", "transitive-task",
    } for row in family_reserve)
    assert len(sample["selected"]) == len({row["cluster_id"] for row in sample["selected"]}) == 12


@pytest.mark.parametrize("protected_split", ["train_inner", "train_outer"])
@pytest.mark.parametrize("overlap_field", [
    "cluster_id", "template_id", "template_digest", "possible_overlap_key", "source_template_ids",
])
def test_sampler_rejects_dev_overlap_with_inner_or_outer(protected_split, overlap_field):
    parent = _phase3_parent(outer_count=6)
    dev = parent["task_membership"][0]
    protected = dict(parent["task_membership"][-1], task_id="protected-task", split=protected_split,
                     cluster_id="protected-cluster", template_id="protected-template",
                     template_digest="protected-digest", possible_overlap_key="protected-overlap")
    if overlap_field == "source_template_ids":
        protected[overlap_field] = [dev["template_id"]]
    else:
        protected[overlap_field] = dev[overlap_field]
    parent["task_membership"].append(protected)
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    extension = build_split_extension(parent, study_id="study-v3", seed=2026)
    with pytest.raises(ValueError, match="parent semantic/template group overlaps"):
        sample_train_dev(parent, extension, study_id="study-v3", seed=2026)


def test_sampler_excludes_whole_dev_group_when_a_duplicate_is_exposed():
    parent = _phase3_parent(dev_per_family=3, outer_count=6)
    original = parent["task_membership"][0]
    duplicate = dict(original, task_id="exposed-duplicate", template_id="exposed-alias-template")
    parent["task_membership"].append(duplicate)
    parent["manifest_sha256"] = _canonical_manifest_hash(parent)
    extension = build_split_extension(parent, study_id="study-v3", seed=2026,
                                      exposed_template_ids=[duplicate["template_id"]])
    sample = sample_train_dev(parent, extension, study_id="study-v3", seed=2026)
    assert sample["family_summary"][original["family"]]["eligible_clusters"] == 2
    assert all(row["cluster_id"] != original["cluster_id"] for row in sample["reserve_order"])
