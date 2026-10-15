"""Select a fresh train-only readiness batch; never construct a provider client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from .study import PHASE_I_CONFIG


_FAMILIES = {
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
}
_EPISODE_FIELDS = {
    "gamefile", "extra.gamefile", "selected_game_files", "selected_traj_data",
    "demonstration_paths",
}


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
    templates: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _EPISODE_FIELDS:
                paths = item if isinstance(item, list) else [item]
                for path in paths:
                    if path is not None:
                        templates.add(_train_identity(str(path))[0])
            elif isinstance(item, (dict, list)):
                templates.update(_history_templates(item))
    elif isinstance(value, list):
        for item in value:
            templates.update(_history_templates(item))
    return templates


def plan_readiness(data_root: Path, history_paths: list[Path], *, seed: int = 17) -> dict[str, Any]:
    if not history_paths:
        raise ValueError("provide the historical train receipts, including synthesis and environment probes")
    excluded: set[str] = set()
    history = []
    for path in history_paths:
        templates = _history_templates(json.loads(path.read_text(encoding="utf-8")))
        if not templates:
            raise ValueError(f"history receipt contains no recognized train episode identities: {path}")
        excluded.update(templates)
        history.append({"path": str(path), "task_template_count": len(templates)})
    train_root = data_root / "json_2.1.1" / "train"
    if not train_root.is_dir():
        raise ValueError("official ALFWorld train data directory does not exist")
    grouped: dict[str, dict[str, list[Path]]] = {family: {} for family in _FAMILIES}
    for path in sorted(train_root.glob("*/trial_*/game.tw-pddl")):
        template, _ = _train_identity(str(path))
        if template in excluded:
            continue
        family = next((key for key, prefix in _FAMILIES.items() if template.startswith(prefix + "-")), None)
        if family is not None:
            grouped[family].setdefault(template, []).append(path)
    rng = random.Random(seed)
    episodes = []
    for family in PHASE_I_CONFIG["families"]:
        templates = sorted(grouped[family])
        if not templates:
            raise ValueError(f"no fresh compiled train task templates for {family}")
        template = templates[rng.randrange(len(templates))]
        paths = grouped[family][template]
        selected = paths[rng.randrange(len(paths))]
        _, episode_id = _train_identity(str(selected))
        episodes.append({"task_family": family, "task_template": template,
                         "episode_id": episode_id, "gamefile": str(selected),
                         "environment_seed": seed})
    return {
        "status": "DRY_RUN_NOT_EXECUTED", "provider_calls": 0,
        "purpose": "three-episode engineering diagnostic, not efficacy",
        "config": dict(PHASE_I_CONFIG), "selection_seed": seed,
        "history": history, "excluded_task_templates": sorted(excluded),
        "exclusion_unit": "whole ALFWorld task-template directory, across all trials",
        "eligible_template_counts": {family: len(templates) for family, templates in grouped.items()},
        "episodes": episodes,
        "launch_blockers": ["approved token and monetary caps", "per-call input reserve and price/resource contract"],
        "independence_scope": "directory-level separation; does not assert semantic template independence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit the study before creating its task plan")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = plan_readiness(args.data_root, args.history, seed=args.seed)
    result["code_revision"] = revision
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]),
                      "excluded_templates": len(result["excluded_task_templates"]), "provider_calls": 0}))


if __name__ == "__main__":
    main()
