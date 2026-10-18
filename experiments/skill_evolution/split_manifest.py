"""Train-only task provenance and fail-closed template split checks."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


_TRAIN_MARKER = "/json_2.1.1/train/"
_EPISODE_FIELDS = {
    "gamefile", "extra.gamefile", "selected_game_files", "selected_traj_data",
    "demonstration_paths",
}


def _train_parts(path: str) -> tuple[str, str, str]:
    normalized = path.replace("\\", "/")
    if _TRAIN_MARKER not in normalized:
        raise ValueError("episode identity must refer to the official train split")
    parts = normalized.split(_TRAIN_MARKER, 1)[1].split("/")
    if len(parts) != 3 or parts[-1] not in {"game.tw-pddl", "traj_data.json"}:
        raise ValueError("expected ALFWorld train template/trial/game or trajectory path")
    return parts[0], "/".join(parts[:2]), parts[1]


def _metadata_path(path: Path) -> Path:
    return path.parent / "traj_data.json" if path.name == "game.tw-pddl" else path


def _normalized_target(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.casefold().split())


def _task_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _metadata_path(path)
    if not metadata_path.is_file():
        return {"identity_status": "UNKNOWN_IDENTITY"}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"identity_status": "UNKNOWN_IDENTITY"}

    task_type = data.get("task_type")
    pddl_params = data.get("pddl_params")
    target_fields = ("object_target", "parent_target", "toggle_target", "mrecep_target")
    target_relation = ({key: _normalized_target(pddl_params.get(key))
                        for key in target_fields if _normalized_target(pddl_params.get(key))}
                       if isinstance(pddl_params, dict) else {})
    if isinstance(pddl_params, dict) and isinstance(pddl_params.get("object_sliced"), bool):
        target_relation["object_sliced"] = pddl_params["object_sliced"]
    scene = data.get("scene")
    floor_plan = scene.get("floor_plan") if isinstance(scene, dict) else None
    object_poses = scene.get("object_poses") if isinstance(scene, dict) else None
    object_types = []
    if isinstance(object_poses, list):
        for row in object_poses:
            if isinstance(row, dict) and isinstance(row.get("objectName"), str):
                # ALFWorld object names append an instance identifier after the first underscore.
                object_types.append(row["objectName"].split("_", 1)[0].casefold())

    scene_group = floor_plan if isinstance(floor_plan, str) and floor_plan else None
    if (not isinstance(task_type, str) or not any(key in target_relation for key in target_fields)
            or not scene_group or not object_types
            or not isinstance(pddl_params, dict)
            or any(key not in pddl_params for key in target_fields)):
        return {
            "identity_status": "UNKNOWN_IDENTITY",
            "task_type": task_type if isinstance(task_type, str) else None,
            "goal_relation": target_relation or None,
            "scene_group": scene_group,
        }

    exact_key = {
        "task_type": task_type,
        "goal_relation": target_relation,
        "scene_group": scene_group,
        "object_layout": dict(sorted(Counter(object_types).items())),
    }
    possible_key = {
        "task_type": task_type,
        "goal_relation": target_relation,
        "scene_group": scene_group,
    }
    exact_json = json.dumps(exact_key, sort_keys=True, separators=(",", ":"))
    possible_json = json.dumps(possible_key, sort_keys=True, separators=(",", ":"))
    return {
        "identity_status": "IDENTIFIED",
        "task_type": task_type,
        "goal_relation": target_relation,
        "scene_group": scene_group,
        "template_digest": hashlib.sha256(exact_json.encode("utf-8")).hexdigest(),
        "possible_overlap_key": hashlib.sha256(possible_json.encode("utf-8")).hexdigest(),
    }


def build_episode_membership(
    gamefile: Path,
    *,
    data_root: Path,
    split: str,
    family: str,
    exposure: str,
    exposure_sources: list[str],
) -> dict[str, Any]:
    template_id, task_id, _ = _train_parts(str(gamefile))
    try:
        relative_gamefile = gamefile.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        relative_gamefile = None
    identity = _task_metadata(gamefile)
    return {
        "split": split,
        "family": family,
        "task_id": task_id,
        "template_id": template_id,
        "template_digest": identity.get("template_digest"),
        "identity_status": identity["identity_status"],
        "possible_overlap_key": identity.get("possible_overlap_key"),
        "task_type": identity.get("task_type"),
        "goal_relation": identity.get("goal_relation"),
        "scene_group": identity.get("scene_group"),
        "gamefile": str(gamefile),
        "gamefile_id": relative_gamefile,
        "gamefile_digest": None,
        "instance_seed": None,
        "exposure": exposure,
        "exposure_sources": list(exposure_sources),
        "selection_method": "predeclared_deterministic",
        "confirmatory_eligible": False,
    }


def history_membership(value: Any, source_id: str) -> list[dict[str, Any]]:
    """Extract only train IDs from receipts; classify probes as already exposed."""
    paths: list[str] = []
    has_native_probe_marker = "native-probe" in source_id.casefold()

    def visit(item: Any) -> None:
        nonlocal has_native_probe_marker
        if isinstance(item, dict):
            mode = str(item.get("mode", "")).casefold()
            has_native_probe_marker |= "native_train_environment_probe" in mode
            for key, nested in item.items():
                if key in _EPISODE_FIELDS:
                    values = nested if isinstance(nested, list) else [nested]
                    paths.extend(str(path) for path in values if path is not None)
                elif isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    exposure = "native_probe_exposed" if has_native_probe_marker else "prior_train"
    rows = []
    seen: set[str] = set()
    for path in paths:
        template_id, task_id, _ = _train_parts(path)
        if task_id in seen:
            continue
        seen.add(task_id)
        rows.append({
            "task_id": task_id,
            "template_id": template_id,
            "split": "train_dev",
            "exposure": exposure,
            "exposure_sources": [source_id],
            "identity_status": "UNKNOWN_IDENTITY",
            "template_digest": None,
            "confirmatory_eligible": False,
        })
    return rows


def _overlap_status(left: dict[str, Any], right: dict[str, Any]) -> str:
    if left.get("task_id") == right.get("task_id"):
        return "EXACT_ID_COLLISION"
    if left.get("template_id") and left.get("template_id") == right.get("template_id"):
        return "TEMPLATE_COLLISION"
    if left.get("template_digest") and left.get("template_digest") == right.get("template_digest"):
        return "TEMPLATE_COLLISION"
    if left.get("possible_overlap_key") and left.get("possible_overlap_key") == right.get("possible_overlap_key"):
        return "POSSIBLE_TEMPLATE_OVERLAP"
    if not left.get("template_digest") or not right.get("template_digest"):
        return "UNKNOWN_IDENTITY"
    return "NO_KNOWN_TEMPLATE_OVERLAP"


def validate_split_membership(records: Iterable[dict[str, Any]]) -> None:
    """Reject exact, possible, or unknown train_inner/train_outer overlap."""
    rows = list(records)
    seen_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    seen_by_template: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        split = row.get("split")
        if split not in {"train_inner", "train_outer", "untouched_eval"}:
            continue
        task_id = row.get("task_id")
        if not task_id:
            raise ValueError("split membership requires a stable task_id")
        key = (split, task_id)
        if key in seen_by_task:
            raise ValueError(f"task ID repeated within {split}: {task_id}")
        seen_by_task[key] = row
        template_identities = {value for value in (row.get("template_digest"), row.get("template_id")) if value}
        for template_identity in template_identities:
            template_key = (split, template_identity)
            if template_key in seen_by_template:
                previous = seen_by_template[template_key]
                raise ValueError(
                    f"template repeated within {split} across task IDs/seeds: "
                    f"{previous['task_id']} and {task_id}"
                )
            seen_by_template[template_key] = row

    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left.get("split") == right.get("split"):
                continue
            if {left.get("split"), right.get("split")} != {"train_inner", "train_outer"}:
                continue
            status = _overlap_status(left, right)
            if status != "NO_KNOWN_TEMPLATE_OVERLAP":
                raise ValueError(f"train_inner/train_outer overlap is not confirmatory: {status}")


def build_split_manifest(
    plan: dict[str, Any],
    *,
    episodes: list[dict[str, Any]],
    exposure_ledger: list[dict[str, Any]],
    untouched_eval: list[dict[str, Any]],
) -> dict[str, Any]:
    overlaps = []
    for index, left in enumerate(episodes):
        for right in episodes[index + 1:]:
            status = _overlap_status(left, right)
            overlaps.append({"task_ids": [left["task_id"], right["task_id"]], "status": status})
    for episode in episodes:
        for history in exposure_ledger:
            status = _overlap_status(episode, history)
            if status != "NO_KNOWN_TEMPLATE_OVERLAP":
                overlaps.append({"task_ids": [episode["task_id"], history["task_id"]], "status": status,
                                 "source_trail": history["exposure_sources"]})

    manifest = {
        "schema_version": "2",
        "study_id": "skill_evolution_real_v2",
        "revision": None,
        "source_population": "alfworld_json_2_1_1",
        "selection_seed": plan["selection_seed"],
        "task_membership": episodes,
        "exposure_ledger": exposure_ledger,
        # These entries deliberately contain opaque IDs only, never gamefiles or outcomes.
        "untouched_eval": untouched_eval,
        "overlap_report": {
            "exact_id_collisions": [row for row in overlaps if row["status"] == "EXACT_ID_COLLISION"],
            "template_collisions": [row for row in overlaps if row["status"] == "TEMPLATE_COLLISION"],
            "possible_template_overlaps": [row for row in overlaps if row["status"] == "POSSIBLE_TEMPLATE_OVERLAP"],
            "unknown_identity": [row for row in overlaps if row["status"] == "UNKNOWN_IDENTITY"],
            "no_known_overlap_pairs": [row for row in overlaps if row["status"] == "NO_KNOWN_TEMPLATE_OVERLAP"],
            "already_touched_task_ids": sorted(row["task_id"] for row in exposure_ledger),
            "source_trails": {row["task_id"]: row["exposure_sources"] for row in exposure_ledger},
        },
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return manifest


def _read_exposures(
    history_paths: list[Path], native_probe_results: Path, train_root: Path,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    historical: dict[str, dict[str, set[str]]] = {}
    for receipt in history_paths:
        rows = history_membership(json.loads(receipt.read_text(encoding="utf-8")), receipt.as_posix())
        for row in rows:
            record = historical.setdefault(row["template_id"], {"task_ids": set(), "sources": set()})
            record["task_ids"].add(row["task_id"])
            record["sources"].add(receipt.as_posix())
    if len(historical) != 88:
        raise ValueError(f"expected 88 historical template directories, found {len(historical)}")

    exposure_rows = []
    exposed_templates = set(historical)
    exposed_keys: set[str] = set()
    for template_id, source in sorted(historical.items()):
        metadata = []
        for metadata_file in sorted((train_root / template_id).glob("trial_*/traj_data.json")):
            metadata.append(_task_metadata(metadata_file))
        keys = {row["possible_overlap_key"] for row in metadata if row.get("possible_overlap_key")}
        identity_unknown = not metadata or any(row["identity_status"] != "IDENTIFIED" for row in metadata)
        exposed_keys.update(keys)
        if identity_unknown:
            task_type = next((name for name in (
                "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                "pick_two_obj_and_place",
            ) if template_id.startswith(name + "-")), None)
            if task_type:
                exposed_keys.add("UNKNOWN_FAMILY:" + task_type)
        exposure_rows.append({
            "split": "train_dev", "exposure": "previous_88_templates",
            "task_id": None, "source_task_ids": sorted(source["task_ids"]),
            "template_id": template_id, "possible_overlap_keys": sorted(keys),
            "identity_status": "UNKNOWN_IDENTITY" if identity_unknown else "IDENTIFIED",
            "exposure_sources": sorted(source["sources"]), "confirmatory_eligible": False,
        })

    probe = json.loads(native_probe_results.read_text(encoding="utf-8"))
    if probe.get("mode") != "native_train_environment_probe":
        raise ValueError("native-probe results must identify native_train_environment_probe mode")
    probe_rows = history_membership(probe, native_probe_results.as_posix())
    if len(probe_rows) != 3:
        raise ValueError(f"expected three native-probe episodes, found {len(probe_rows)}")
    for row in probe_rows:
        row["exposure"] = "native_probe_exposed"
        row["source_task_ids"] = [row["task_id"]]
        metadata = []
        for metadata_file in sorted((train_root / row["template_id"]).glob("trial_*/traj_data.json")):
            metadata.append(_task_metadata(metadata_file))
        keys = {item["possible_overlap_key"] for item in metadata if item.get("possible_overlap_key")}
        identity_unknown = not metadata or any(item["identity_status"] != "IDENTIFIED" for item in metadata)
        row["possible_overlap_keys"] = sorted(keys)
        row["identity_status"] = "UNKNOWN_IDENTITY" if identity_unknown else "IDENTIFIED"
        if identity_unknown:
            task_type = next((name for name in (
                "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                "pick_two_obj_and_place",
            ) if row["template_id"].startswith(name + "-")), None)
            if task_type:
                exposed_keys.add("UNKNOWN_FAMILY:" + task_type)
        exposed_templates.add(row["template_id"])
        exposed_keys.update(keys)
        exposure_rows.append(row)
    return exposure_rows, exposed_templates, exposed_keys


def _deterministic_task_membership(
    data_root: Path, train_root: Path, exposed_templates: set[str], exposed_keys: set[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    unknown = []
    for gamefile in sorted(train_root.glob("*/trial_*/game.tw-pddl")):
        template_id = gamefile.parent.parent.name
        if template_id in exposed_templates:
            continue
        identity = _task_metadata(gamefile)
        if identity["identity_status"] != "IDENTIFIED":
            unknown.append((gamefile, identity))
            continue
        groups.setdefault(identity["possible_overlap_key"], []).append((gamefile, identity))

    by_type: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {}
    for key, candidates in groups.items():
        gamefile, identity = candidates[0]
        by_type.setdefault(identity["task_type"], []).append((key, gamefile, identity))

    memberships = []
    split_names = ("train_dev", "train_inner", "train_outer")
    for task_type, task_groups in sorted(by_type.items()):
        eligible_index = 0
        for key, gamefile, identity in sorted(task_groups, key=lambda item: item[0]):
            exposed_overlap = key in exposed_keys or "UNKNOWN_FAMILY:" + task_type in exposed_keys
            if exposed_overlap:
                split = "train_dev"
                exposure = "possible_or_unknown_prior_overlap"
            else:
                split = split_names[eligible_index % len(split_names)]
                exposure = f"new_{split}"
                eligible_index += 1
            row = build_episode_membership(
                gamefile, data_root=data_root, split=split, family=task_type,
                exposure=exposure, exposure_sources=["deterministic-template-partition"],
            )
            row.update(identity)
            row["template_id"] = gamefile.parent.parent.name
            row["source_template_ids"] = sorted({path.parent.parent.name for path, _ in groups[key]})
            row["cluster_id"] = key
            row["confirmatory_eligible"] = split in {"train_inner", "train_outer"}
            memberships.append(row)

    for gamefile, identity in unknown:
        row = build_episode_membership(
            gamefile, data_root=data_root, split="train_dev", family="unknown",
            exposure="unknown_template_identity", exposure_sources=["deterministic-template-partition"],
        )
        row.update(identity)
        memberships.append(row)
    return memberships


def create_split_manifest(
    *, data_root: Path, history_paths: list[Path], native_probe_results: Path,
    untouched_eval_ids: list[str], revision: str,
) -> dict[str, Any]:
    """Create fixed train partitions while keeping held-out entries ID-only."""
    if len(history_paths) != 7:
        raise ValueError(f"expected seven historical train receipts, found {len(history_paths)}")
    if len(set(untouched_eval_ids)) != len(untouched_eval_ids) or any(not item for item in untouched_eval_ids):
        raise ValueError("untouched eval task IDs must be nonempty and unique")
    train_root = data_root / "json_2.1.1" / "train"
    if not train_root.is_dir():
        raise ValueError("official ALFWorld train data directory does not exist")
    exposures, exposed_templates, exposed_keys = _read_exposures(
        history_paths, native_probe_results, train_root,
    )
    tasks = _deterministic_task_membership(data_root, train_root, exposed_templates, exposed_keys)
    validate_split_membership(tasks)
    unknown_history = [row for row in exposures if row["identity_status"] == "UNKNOWN_IDENTITY"]
    unknown_task_identity = [row for row in tasks if row["identity_status"] != "IDENTIFIED"]
    if unknown_history or unknown_task_identity:
        for row in tasks:
            if row["split"] in {"train_inner", "train_outer"}:
                row["confirmatory_eligible"] = False
    untouched_eval = [{
        "task_id": task_id, "split": "untouched_eval",
        "content_access": "ID_ONLY", "outcome_access": "PROHIBITED",
        "confirmatory_eligible": False,
    } for task_id in untouched_eval_ids]

    overlap_rows = []
    for exposure in exposures:
        if exposure["identity_status"] == "UNKNOWN_IDENTITY":
            overlap_rows.append({
                "task_ids": exposure["source_task_ids"],
                "exposed_template_id": exposure["template_id"],
                "statuses": ["UNKNOWN_IDENTITY"],
                "source_trail": exposure["exposure_sources"],
            })
    for task in tasks:
        if task["identity_status"] != "IDENTIFIED":
            overlap_rows.append({
                "task_id": task["task_id"], "exposed_template_id": None,
                "statuses": ["UNKNOWN_IDENTITY"],
                "source_trail": task["exposure_sources"],
            })
        for exposure in exposures:
            statuses = []
            if task.get("task_id") in exposure.get("source_task_ids", []):
                statuses.append("EXACT_ID_COLLISION")
            if task.get("template_id") == exposure.get("template_id"):
                statuses.append("TEMPLATE_COLLISION")
            if task.get("cluster_id") in exposure.get("possible_overlap_keys", []):
                statuses.append("POSSIBLE_TEMPLATE_OVERLAP")
            if statuses:
                overlap_rows.append({
                    "task_id": task["task_id"], "exposed_template_id": exposure["template_id"],
                    "statuses": sorted(set(statuses)), "source_trail": exposure["exposure_sources"],
                })

    manifest = {
        "schema_version": "2", "study_id": "skill_evolution_real_v2", "revision": revision,
        "source_population": "alfworld_json_2_1_1",
        "qualification_status": (
            "BLOCKED_UNKNOWN_IDENTITY" if unknown_history or unknown_task_identity
            else "PARTITIONS_READY_FOR_REVIEW"
        ),
        "selection_method": "semantic template groups sorted per task_type, round-robin train_dev/inner/outer",
        "task_membership": tasks, "exposure_ledger": exposures,
        "untouched_eval": untouched_eval,
        "exposure_counts": {
            "previous_88_templates": len({row["template_id"] for row in exposures
                                           if row["exposure"] == "previous_88_templates"}),
            "native_probe_exposed": sum(row["exposure"] == "native_probe_exposed" for row in exposures),
        },
        "overlap_report": {
            "exact_id_collisions": [row for row in overlap_rows if "EXACT_ID_COLLISION" in row["statuses"]],
            "template_collisions": [row for row in overlap_rows if "TEMPLATE_COLLISION" in row["statuses"]],
            "possible_template_overlaps": [row for row in overlap_rows if "POSSIBLE_TEMPLATE_OVERLAP" in row["statuses"]],
            "unknown_identity": [row for row in overlap_rows if "UNKNOWN_IDENTITY" in row["statuses"]],
            "already_touched_task_ids": sorted(task_id for row in exposures for task_id in row["source_task_ids"]),
            "source_trails": {
                task_id: row["exposure_sources"]
                for row in exposures for task_id in row["source_task_ids"]
            },
        },
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return manifest


_OUTER_CHILDREN = (
    "outer_skill_validation", "outer_gate_calibration",
    "untouched_eval_reserved", "quarantine_exposed",
)
_OFFICIAL_FAMILIES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)


def _official_family(row: dict[str, Any]) -> str | None:
    """Use the parent's raw official task_type as the single family identity."""
    family, task_type = row.get("family"), row.get("task_type")
    if task_type not in _OFFICIAL_FAMILIES or family != task_type:
        return None
    return task_type


def _canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _row_group_keys(row: dict[str, Any]) -> set[str]:
    keys = set()
    for field in ("cluster_id", "possible_overlap_key", "template_id", "template_digest"):
        value = row.get(field)
        if isinstance(value, str) and value:
            keys.add(f"{field}:{value}")
    for field in ("source_template_ids",):
        values = row.get(field, ())
        if isinstance(values, list):
            keys.update(f"template_id:{value}" for value in values if isinstance(value, str) and value)
    return keys


def _parent_exposures(parent: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    task_ids: set[str] = set()
    group_keys: set[str] = set()
    unknown_families: set[str] = set()
    for row in parent.get("exposure_ledger", []):
        if not isinstance(row, dict):
            raise ValueError("exposure ledger rows must be objects")
        for field in ("task_id", "source_task_ids"):
            value = row.get(field)
            values = value if isinstance(value, list) else [value]
            task_ids.update(item for item in values if isinstance(item, str) and item)
        group_keys.update(_row_group_keys(row))
        for template in (row.get("template_id"), row.get("exposed_template_id")):
            if isinstance(template, str) and template:
                group_keys.add(f"template_id:{template}")
        for field in ("possible_overlap_keys",):
            values = row.get(field, ())
            if isinstance(values, list):
                group_keys.update(f"possible_overlap_key:{value}" for value in values if isinstance(value, str) and value)
        if row.get("identity_status") != "IDENTIFIED":
            family = row.get("family") or row.get("task_type")
            if family in _OFFICIAL_FAMILIES:
                unknown_families.add(family)
            else:
                unknown_families.update(_OFFICIAL_FAMILIES)
    return task_ids, group_keys, unknown_families


def build_split_extension(
    parent: dict[str, Any], *, study_id: str, seed: int,
    exposed_cluster_ids: Iterable[str] = (), exposed_template_ids: Iterable[str] = (),
    child_weights: dict[str, str | int | float] | None = None,
) -> dict[str, Any]:
    """Reserve disjoint outer children from parent metadata without reading outcomes."""
    if not isinstance(study_id, str) or not study_id or not isinstance(seed, int):
        raise ValueError("study_id and integer seed are required")
    parent_hash = parent.get("manifest_sha256")
    if not isinstance(parent_hash, str) or parent_hash != _canonical_manifest_hash(parent):
        raise ValueError("parent manifest hash is missing or does not match its contents")
    rows = parent.get("task_membership")
    if not isinstance(rows, list):
        raise ValueError("parent task_membership is required")

    supplied_weights = ({child: "1/3" for child in _OUTER_CHILDREN[:3]}
                        if child_weights is None else child_weights)
    try:
        weights = {child: Fraction(str(value)) for child, value in supplied_weights.items()}
    except (ValueError, ZeroDivisionError):
        raise ValueError("child_weights must be nonnegative fractions") from None
    if set(weights) != set(_OUTER_CHILDREN[:3]) or any(value < 0 for value in weights.values()) or sum(weights.values()) != 1:
        raise ValueError("child_weights must define the three reservation fractions summing to one")

    exposed_tasks, exposed_keys, unknown_families = _parent_exposures(parent)
    exposed_tasks.update(item for item in parent.get("overlap_report", {}).get("already_touched_task_ids", [])
                         if isinstance(item, str))
    exposed_cluster_ids = list(exposed_cluster_ids)
    exposed_template_ids = list(exposed_template_ids)
    if any(not isinstance(item, str) or not item for item in exposed_cluster_ids + exposed_template_ids):
        raise ValueError("additional exposed cluster/template IDs must be nonempty strings")
    exposed_keys.update(f"cluster_id:{item}" for item in exposed_cluster_ids)
    exposed_keys.update(f"template_id:{item}" for item in exposed_template_ids)

    outer_rows = [row for row in rows if isinstance(row, dict) and row.get("split") == "train_outer"]
    if not outer_rows:
        raise ValueError("parent has no train_outer task metadata")
    if len({row.get("task_id") for row in outer_rows}) != len(outer_rows):
        raise ValueError("parent train_outer task IDs must be unique")
    for row in rows:
        if not isinstance(row, dict) or row.get("split") == "train_outer":
            continue
        exposed_keys.update(_row_group_keys(row))
        if row.get("identity_status") != "IDENTIFIED":
            family = _official_family(row)
            unknown_families.update(_OFFICIAL_FAMILIES if family is None else (family,))
    # Union rows that share any known template/layout/semantic identity. Any
    # exposure or unknown member quarantines the whole connected group.
    groups: list[dict[str, Any]] = []
    for row in outer_rows:
        if not all(isinstance(row.get(field), str) and row[field]
                   for field in ("task_id", "cluster_id")) or _official_family(row) is None:
            raise ValueError("outer rows require task_id, family, and cluster_id")
        keys = _row_group_keys(row)
        matching = [group for group in groups if group["keys"] & keys]
        if not matching:
            groups.append({"keys": keys, "rows": [row]})
        else:
            target = matching[0]
            target["keys"].update(keys)
            target["rows"].append(row)
            for extra in matching[1:]:
                target["keys"].update(extra["keys"])
                target["rows"].extend(extra["rows"])
                groups.remove(extra)

    eligible_by_family: dict[str, list[dict[str, Any]]] = {}
    quarantined = []
    for group in groups:
        family_set = {_official_family(row) for row in group["rows"]}
        unknown_identity = any(row.get("identity_status") != "IDENTIFIED" for row in group["rows"])
        exposed = (unknown_identity or bool(group["keys"] & exposed_keys)
                   or any(row["task_id"] in exposed_tasks for row in group["rows"])
                   or bool(family_set & unknown_families))
        if len(family_set) != 1:
            exposed = True
        if exposed:
            quarantined.extend(group["rows"])
        else:
            group["family"] = next(iter(family_set))
            eligible_by_family.setdefault(group["family"], []).append(group)

    allocation = {}
    child_membership = []
    for family_index, family in enumerate(_OFFICIAL_FAMILIES):
        family_groups = eligible_by_family.get(family, [])
        ordered = sorted(family_groups, key=lambda group: hashlib.sha256(json.dumps(
            [study_id, family, sorted(row["cluster_id"] for row in group["rows"])[0], seed], ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest())
        n = len(ordered)
        raw = {child: n * weights[child] for child in _OUTER_CHILDREN[:3]}
        counts = {child: raw[child].numerator // raw[child].denominator for child in raw}
        remainder = n - sum(counts.values())
        tie_order = _OUTER_CHILDREN[:3][family_index % 3:] + _OUTER_CHILDREN[:3][:family_index % 3]
        priority = sorted(raw, key=lambda child: (-(raw[child] - counts[child]), tie_order.index(child)))
        for child in priority[:remainder]:
            counts[child] += 1
        allocation[family] = {
            "available_groups": n,
            "available_clusters": len({row["cluster_id"] for group in ordered for row in group["rows"]}),
            **{f"{child}_groups": count for child, count in counts.items()},
            "precision_note": f"{n} eligible overlap-closed groups; exploratory allocation, no power claim.",
        }
        offset = 0
        for child in _OUTER_CHILDREN[:3]:
            selected_groups = ordered[offset:offset + counts[child]]
            for group in selected_groups:
                for row in group["rows"]:
                    child_membership.append({
                        "task_id": row["task_id"], "family": family,
                        "cluster_id": row["cluster_id"], "template_id": row.get("template_id"),
                        "template_digest": row.get("template_digest"),
                        "possible_overlap_key": row.get("possible_overlap_key"),
                        "parent_split": "train_outer", "child_split": child,
                        "exposure_status": "unexposed_metadata_only",
                    })
            offset += counts[child]
    child_membership.extend({
        "task_id": row["task_id"], "family": row["family"],
        "cluster_id": row["cluster_id"], "template_id": row.get("template_id"),
        "template_digest": row.get("template_digest"),
        "possible_overlap_key": row.get("possible_overlap_key"),
        "parent_split": "train_outer", "child_split": "quarantine_exposed",
        "exposure_status": "unknown_or_previously_exposed",
    } for row in quarantined)
    assigned_clusters: dict[str, set[str]] = {child: set() for child in _OUTER_CHILDREN}
    for row in child_membership:
        assigned_clusters[row["child_split"]].add(row["cluster_id"])
    all_outer_clusters = {row["cluster_id"] for row in outer_rows}
    assigned_all = set().union(*assigned_clusters.values())
    if assigned_all != all_outer_clusters or sum(map(len, assigned_clusters.values())) != len(assigned_all):
        raise ValueError("outer cluster assignment is incomplete or overlaps across child partitions")
    result = {
        "schema_version": "3", "study_id": study_id, "seed": seed,
        "parent_manifest_sha256": parent_hash,
        "selection_method": "metadata_only_sha256_utf8_json_array_largest_remainder_v1",
        "child_weights": {child: f"{weight.numerator}/{weight.denominator}"
                           for child, weight in weights.items()},
        "family_allocation": allocation,
        "reservation_summary": {
            "parent_outer_clusters": len(all_outer_clusters),
            "assigned_outer_clusters": len(assigned_all),
            "child_cluster_counts": {child: len(assigned_clusters[child]) for child in _OUTER_CHILDREN},
        },
        "additional_exposure": {
            "cluster_ids": sorted(set(exposed_cluster_ids)),
            "template_ids": sorted(set(exposed_template_ids)),
        },
        "precision_note": "Exploratory reservation counts only; precision is not established and no allocation is claimed to be statistically powered.",
        "child_membership": sorted(child_membership, key=lambda row: (row["family"], row["task_id"])),
    }
    result["manifest_sha256"] = _canonical_manifest_hash(result)
    return result


def sample_train_dev(
    parent: dict[str, Any], extension: dict[str, Any], *,
    study_id: str, seed: int, per_family: int = 2,
) -> dict[str, Any]:
    """Return deterministic equal-family train_dev picks and complete reserve orders."""
    if parent.get("manifest_sha256") != _canonical_manifest_hash(parent):
        raise ValueError("parent manifest hash is missing or does not match its contents")
    if extension.get("parent_manifest_sha256") != parent.get("manifest_sha256"):
        raise ValueError("extension does not belong to this parent manifest")
    if extension.get("manifest_sha256") != _canonical_manifest_hash(extension):
        raise ValueError("extension manifest hash does not match its contents")
    if extension.get("study_id") != study_id or extension.get("seed") != seed:
        raise ValueError("sampler identity must match the frozen extension")
    if not isinstance(per_family, int) or per_family < 1:
        raise ValueError("per_family must be a positive integer")
    task_rows = parent.get("task_membership")
    if not isinstance(task_rows, list):
        raise ValueError("parent task_membership is required")
    if any(not isinstance(row, dict) for row in task_rows):
        raise ValueError("parent task_membership rows must be objects")
    if any(not isinstance(row.get("task_id"), str) or not row["task_id"] for row in task_rows):
        raise ValueError("parent task IDs must be nonempty strings")
    if len({row.get("task_id") for row in task_rows}) != len(task_rows):
        raise ValueError("parent task IDs must be unique across splits")

    membership_groups: list[dict[str, Any]] = []
    for row in task_rows:
        if row.get("split") not in {"train_dev", "train_inner", "train_outer"}:
            raise ValueError("parent task row has an unsupported split")
        if row.get("identity_status") == "IDENTIFIED" and not all(
            isinstance(row.get(field), str) and row[field]
            for field in ("cluster_id", "template_id", "template_digest", "possible_overlap_key")
        ):
            raise ValueError(f"identified parent row lacks group metadata: {row['task_id']}")
        keys = _row_group_keys(row)
        matching = [group for group in membership_groups if group["keys"] & keys]
        if not matching:
            membership_groups.append({"keys": keys, "rows": [row]})
        else:
            group = matching[0]
            group["keys"].update(keys)
            group["rows"].append(row)
            for other in matching[1:]:
                group["keys"].update(other["keys"])
                group["rows"].extend(other["rows"])
                membership_groups.remove(other)
    for group in membership_groups:
        splits = {row["split"] for row in group["rows"]}
        if len(splits) != 1:
            raise ValueError("parent semantic/template group overlaps train_dev, train_inner, or train_outer")

    exposed_tasks, exposed_keys, unknown_families = _parent_exposures(parent)
    exposed_tasks.update(item for item in parent.get("overlap_report", {}).get("already_touched_task_ids", [])
                         if isinstance(item, str))
    extra = extension.get("additional_exposure", {})
    exposed_keys.update(f"cluster_id:{item}" for item in extra.get("cluster_ids", []))
    exposed_keys.update(f"template_id:{item}" for item in extra.get("template_ids", []))
    candidates = {family: [] for family in _OFFICIAL_FAMILIES}
    for group in membership_groups:
        if group["rows"][0]["split"] != "train_dev":
            continue
        rows = group["rows"]
        family_ids = {_official_family(row) for row in rows}
        if len(family_ids) != 1 or None in family_ids:
            continue
        family = next(iter(family_ids))
        if family in unknown_families or any(row.get("identity_status") != "IDENTIFIED" for row in rows):
            continue
        if any(row["task_id"] in exposed_tasks for row in rows) or group["keys"] & exposed_keys:
            continue
        # A component is one sampling unit even if the parent contains repeated
        # rows or multiple metadata rows for the same semantic/template group.
        row = min(rows, key=lambda candidate: (candidate["cluster_id"], candidate["task_id"]))
        if not all(isinstance(row.get(field), str) and row[field]
                   for field in ("task_id", "cluster_id", "template_id", "gamefile")):
            raise ValueError(f"identified train_dev row lacks sampler metadata: {row.get('task_id')}")
        candidates[family].append((row, group))

    selected, reserve_order, family_summary = [], [], {}
    for family in _OFFICIAL_FAMILIES:
        ordered = sorted(candidates[family], key=lambda item: hashlib.sha256(json.dumps(
            [study_id, family, item[0]["cluster_id"], seed], ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest())
        if len(ordered) < per_family:
            raise ValueError(f"insufficient unexposed train_dev clusters for {family}: {len(ordered)} < {per_family}")
        probability = per_family / len(ordered)
        for rank, (row, group) in enumerate(ordered):
            projection = {key: row.get(key) for key in (
                "task_id", "family", "task_type", "split", "exposure", "identity_status",
                "cluster_id", "template_id", "template_digest", "possible_overlap_key",
                "gamefile", "gamefile_id",
            )}
            reserve_row = {
                **projection,
                "group_task_ids": sorted(item["task_id"] for item in group["rows"]),
                "family_rank": rank + 1, "inclusion_probability": probability,
            }
            reserve_order.append(reserve_row)
            if rank < per_family:
                selected.append(reserve_row)
        family_summary[family] = {
            "eligible_clusters": len(ordered), "selected_clusters": per_family,
            "inclusion_probability": probability,
        }
    return {
        "schema_version": "1", "study_id": study_id, "seed": seed,
        "parent_manifest_sha256": parent["manifest_sha256"],
        "extension_manifest_sha256": extension["manifest_sha256"],
        "selection_method": "sha256_utf8_json_array_study_family_cluster_seed_v1",
        "family_summary": family_summary, "selected": selected,
        "reserve_order": reserve_order,
        "outcome_access": "PROHIBITED_DURING_SELECTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, nargs=7, required=True,
                        help="the seven prior train receipts covering 88 template directories")
    parser.add_argument("--native-probe-results", type=Path, required=True)
    parser.add_argument("--untouched-eval-id", action="append", default=[],
                        help="opaque ID only; task content and outcomes are not read")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("write the split manifest from a clean, committed server revision")
    report_path = args.output.parent / "SPLIT_OVERLAP_REPORT.md"
    if args.output.exists() or report_path.exists():
        raise FileExistsError("manifest output and SPLIT_OVERLAP_REPORT.md must be fresh")
    manifest = create_split_manifest(
        data_root=args.data_root, history_paths=args.history,
        native_probe_results=args.native_probe_results,
        untouched_eval_ids=args.untouched_eval_id,
        revision=revision,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report = render_overlap_report(manifest)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": "SPLIT_MANIFEST_WRITTEN",
                      "manifest_sha256": manifest["manifest_sha256"],
                      "exposure_counts": manifest["exposure_counts"],
                      "qualification_status": manifest["qualification_status"],
                      "revision": revision, "task_count": len(manifest["task_membership"])}))


def render_overlap_report(manifest: dict[str, Any]) -> str:
    overlap = manifest["overlap_report"]
    counts = manifest["exposure_counts"]
    details = []
    for heading, key in (
        ("Exact task ID collisions", "exact_id_collisions"),
        ("Template collisions", "template_collisions"),
        ("Possible template overlaps", "possible_template_overlaps"),
        ("Unknown identities", "unknown_identity"),
    ):
        details.append(f"## {heading}\n")
        rows = overlap[key]
        if not rows:
            details.append("None.\n")
            continue
        for row in rows:
            task_ids = row.get("task_ids") or [row.get("task_id")]
            details.append(
                f"- Task IDs: `{', '.join(str(task_id) for task_id in task_ids if task_id)}`; "
                f"exposed template: `{row.get('exposed_template_id', 'n/a')}`; "
                f"classification: `{', '.join(row.get('statuses', [row.get('status', 'UNKNOWN')]))}`; "
                f"source trail: `{', '.join(row.get('source_trail', [])) or 'not recorded'}`.\n"
            )
    details.append("## Already-touched IDs and source trails\n")
    trails = overlap.get("source_trails", {})
    for task_id in overlap.get("already_touched_task_ids", []):
        details.append(f"- `{task_id}`: `{', '.join(trails.get(task_id, []))}`\n")
    return (
        "# Split overlap report\n\n"
        f"- Revision: `{manifest['revision']}`\n"
        f"- Manifest SHA256: `{manifest['manifest_sha256']}`\n"
        f"- Qualification status: `{manifest['qualification_status']}`\n"
        f"- Historical exposed template directories: {counts['previous_88_templates']}\n"
        f"- Native probe exposed episodes: {counts['native_probe_exposed']}\n"
        f"- Exact task ID collisions: {len(overlap['exact_id_collisions'])}\n"
        f"- Template collisions: {len(overlap['template_collisions'])}\n"
        f"- Possible template overlaps: {len(overlap['possible_template_overlaps'])}\n"
        f"- Unknown identities: {len(overlap['unknown_identity'])}\n\n"
        "Untouched evaluation rows contain IDs only; task content and outcomes were not read.\n\n"
        + "\n".join(details)
    )


if __name__ == "__main__":
    main()
