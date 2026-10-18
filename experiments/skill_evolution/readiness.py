"""Select a fresh train-only readiness batch; never construct a provider client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from .study import PHASE_I_CONFIG
from .split_manifest import build_episode_membership, build_split_manifest, history_membership


_FAMILIES = {
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
}


def plan_from_split(split: dict[str, Any], *, seed: int = 17) -> dict[str, Any]:
    """Select train-side source tasks; these may never be validation pairs."""
    rng = random.Random(seed)
    episodes = []
    for family in PHASE_I_CONFIG["families"]:
        candidates = sorted((row for row in split["task_membership"]
                             if row["split"] == "train_inner"
                             and row.get("exposure") == "new_train_inner"
                             and row.get("identity_status") == "IDENTIFIED"
                             and row.get("task_type") == _FAMILIES[family]), key=lambda row: row["task_id"])
        if not candidates:
            raise ValueError(f"no unexposed source template for {family}")
        row = dict(candidates[rng.randrange(len(candidates))])
        row.update(task_family=family, episode_id=row["task_id"], environment_seed=seed)
        episodes.append(row)
    return {"status": "DRY_RUN_NOT_EXECUTED", "provider_calls": 0,
            "config": dict(PHASE_I_CONFIG), "selection_seed": seed, "episodes": episodes}


def _train_identity(path: str) -> tuple[str, str]:
    marker = "/json_2.1.1/train/"
    normalized = path.replace("\\", "/")
    if marker not in normalized:
        raise ValueError("episode identity must refer to the official train split")
    parts = normalized.split(marker, 1)[1].split("/")
    if len(parts) != 3 or parts[-1] not in {"game.tw-pddl", "traj_data.json"}:
        raise ValueError("expected ALFWorld train template/trial/game or trajectory path")
    return parts[0], "/".join(parts[:2])


def _history_templates(value: Any) -> set[str]:
    """Retain the old directory-level exclusion helper for existing callers."""
    return {row["template_id"] for row in history_membership(value, "history")
            if row["template_id"] is not None}


def plan_readiness(
    data_root: Path,
    history_paths: list[Path],
    *,
    seed: int = 17,
    untouched_eval_ids: list[str] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not history_paths:
        raise ValueError("provide the historical train receipts, including synthesis and environment probes")
    excluded: set[str] = set()
    history = []
    exposure_ledger = []
    for path in history_paths:
        source_id = path.as_posix()
        rows = history_membership(json.loads(path.read_text(encoding="utf-8")), source_id)
        templates = {row["template_id"] for row in rows if row["template_id"] is not None}
        if not templates:
            raise ValueError(f"history receipt contains no recognized train episode identities: {path}")
        excluded.update(templates)
        history.append({"path": str(path), "source_id": source_id,
                        "task_template_count": len(templates)})
        exposure_ledger.extend(rows)
    train_root = data_root / "json_2.1.1" / "train"
    if not train_root.is_dir():
        raise ValueError("official ALFWorld train data directory does not exist")
    episodes = []
    if split_manifest is not None:
        frozen_tasks = split_manifest.get("task_membership")
        if not isinstance(frozen_tasks, list):
            raise ValueError("split manifest requires a task_membership list")
        episodes = plan_from_split(split_manifest, seed=seed)["episodes"]
        eligible_counts = {
            family: sum(row.get("split") == "train_inner" and row.get("exposure") == "new_train_inner"
                        and row.get("identity_status") == "IDENTIFIED" and row.get("family") == _FAMILIES[family]
                        for row in frozen_tasks)
            for family in PHASE_I_CONFIG["families"]
        }
    else:
        grouped: dict[str, dict[str, list[Path]]] = {family: {} for family in _FAMILIES}
        for path in sorted(train_root.glob("*/trial_*/game.tw-pddl")):
            template, _ = _train_identity(str(path))
            if template in excluded:
                continue
            family = next((key for key, prefix in _FAMILIES.items() if template.startswith(prefix + "-")), None)
            if family is not None:
                grouped[family].setdefault(template, []).append(path)
        rng = random.Random(seed)
        for family in PHASE_I_CONFIG["families"]:
            templates = sorted(grouped[family])
            if not templates:
                raise ValueError(f"no fresh compiled train task templates for {family}")
            template = templates[rng.randrange(len(templates))]
            paths = grouped[family][template]
            selected = paths[rng.randrange(len(paths))]
            _, episode_id = _train_identity(str(selected))
            episodes.append(build_episode_membership(
                selected, data_root=data_root, split="train_inner", family=family,
                exposure="new_train_inner", exposure_sources=["readiness-selection"],
            ) | {"task_family": family, "task_template": template,
                 "episode_id": episode_id, "environment_seed": seed})
        eligible_counts = {family: len(templates) for family, templates in grouped.items()}
    from .split_manifest import validate_split_membership
    validate_split_membership(episodes)
    untouched_ids = untouched_eval_ids or []
    if len(set(untouched_ids)) != len(untouched_ids) or any(not item for item in untouched_ids):
        raise ValueError("untouched eval task IDs must be nonempty and unique")
    result = {
        "status": "DRY_RUN_NOT_EXECUTED", "provider_calls": 0,
        "purpose": "three-episode engineering diagnostic, not efficacy",
        "config": dict(PHASE_I_CONFIG), "selection_seed": seed,
        "history": history, "excluded_task_templates": sorted(excluded),
        "exposure_ledger": (split_manifest["exposure_ledger"]
                             if split_manifest is not None else exposure_ledger),
        "exclusion_unit": "whole ALFWorld task-template directory, across all trials",
        "eligible_template_counts": eligible_counts,
        "episodes": episodes,
        "untouched_eval": [{"task_id": task_id, "split": "untouched_eval",
                            "content_access": "ID_ONLY", "outcome_access": "PROHIBITED"}
                           for task_id in untouched_ids],
        "launch_blockers": ["approved token and monetary caps", "per-call input reserve and price/resource contract"],
        "independence_scope": "template digests and overlap classifications are recorded; only NO_KNOWN_TEMPLATE_OVERLAP can be confirmatory",
    }
    result["split_manifest"] = split_manifest or build_split_manifest(
        result, episodes=episodes, exposure_ledger=exposure_ledger,
        untouched_eval=result["untouched_eval"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--untouched-eval-id", action="append", default=[],
                        help="opaque task ID only; no task content or outcome is read")
    parser.add_argument("--split-manifest", type=Path,
                        help="derive train-only source tasks from identified train_inner membership")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit the study before creating its task plan")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = plan_readiness(args.data_root, args.history, seed=args.seed,
                            untouched_eval_ids=args.untouched_eval_id,
                            split_manifest=(json.loads(args.split_manifest.read_text(encoding="utf-8"))
                                            if args.split_manifest else None))
    result["code_revision"] = revision
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]),
                      "excluded_templates": len(result["excluded_task_templates"]), "provider_calls": 0}))


if __name__ == "__main__":
    main()
