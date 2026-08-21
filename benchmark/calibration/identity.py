"""Calibration identity and JSON digest primitives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from benchmark.provenance import json_digest

PACKAGE_DIRS = ("workspace", "public_tests", "hidden_verifier", "oracle")
PACKAGE_FILES = ("task.yaml", "metadata.json", "benchmark.py", "scientific_contract.py")


def task_package_digest(task_dir: str | Path) -> str:
    root = Path(task_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"task package not found: {root}")
    files: dict[str, str] = {}
    paths = [root / name for name in PACKAGE_FILES]
    for directory in PACKAGE_DIRS:
        base = root / directory
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"))
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"task package member missing: {path}")
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json_digest(files)


def taskset_digest(tasks_root: str | Path, task_ids: list[str]) -> str:
    root = Path(tasks_root)
    payload: dict[str, Any] = {}
    for task_id in task_ids:
        task_dir = root / task_id
        if not (task_dir / "task.yaml").is_file():
            raise FileNotFoundError(f"task manifest not found: {task_dir / 'task.yaml'}")
        payload[task_id] = {"package_digest": task_package_digest(task_dir), "task_id": task_id}
    return json_digest(payload)


def canonical_cell_identity(
    *, task_id: str, outer_trial_id: str, seed: int,
    measurement_family: str, task_package_digest: str,
    population_manifest_digest: str,
) -> dict[str, Any]:
    """Describe one cell at its raw-result and envelope identity layers."""
    family = str(measurement_family)
    if family not in {"atomic_performance", "evolution"}:
        raise ValueError(f"unsupported calibration measurement family: {family}")
    return {
        "task_id": str(task_id),
        "outer_trial_id": str(outer_trial_id),
        "seed": int(seed),
        "measurement_family": family,
        "raw_measurement_class": "episode_bounded_score" if family == "evolution" else family,
        "envelope_measurement_class": family,
        "task_package_digest": str(task_package_digest),
        "population_manifest_digest": str(population_manifest_digest),
    }
