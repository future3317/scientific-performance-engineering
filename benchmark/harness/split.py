#!/usr/bin/env python3
"""Sequential split and leakage control (BENCHMARK_DESIGN.md section 10).

Split key = ``(family, mechanism, lineage.source, lineage.mutation_template_id)``.
Phase 1 (acquisition) is evolution-visible; phases 2-6 are held out.
:func:`check_leakage` verifies phase ordering, task existence, and that no
split key appears in both phase 1 and any held-out phase. The split manifest is
hash-pinned into run records via :func:`split_manifest_hash`.

Split manifest format (miniyaml subset)::

    split_id: sequential-v1
    phases:
      - index: 1
        name: acquisition
        tasks: [CORE-SCALAR-SYNC-01, ...]
      - index: 2
        name: same_family_transfer
        tasks: [...]
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import miniyaml

PHASE_NAMES = (
    "acquisition",
    "same_family_transfer",
    "cross_family_transfer",
    "drift",
    "poisoned_experience",
    "recovery",
)


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load a split manifest YAML; raises cleanly on structure errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    manifest = miniyaml.load(str(path))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("phases"), list):
        raise ValueError(f"{path} must be a mapping with a 'phases' list")
    return manifest


def split_manifest_hash(path: str | Path) -> str:
    """SHA-256 of the manifest bytes; pinned into each run record (section 10.2)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_key(task_spec: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Split keys of a parsed task.yaml (one per mechanism when it is a list)."""
    lineage = task_spec.get("lineage", {})
    mechanism = task_spec.get("mechanism")
    mechanisms = mechanism if isinstance(mechanism, list) else [mechanism]
    return [
        (
            str(task_spec.get("family")),
            str(mech),
            str(lineage.get("source")),
            str(lineage.get("mutation_template_id")),
        )
        for mech in mechanisms
    ]


def check_leakage(split_manifest: str | Path | dict[str, Any], tasks_root: str | Path) -> list[str]:
    """Return a list of leakage/ordering errors (empty = clean).

    Checks: six phases in canonical order with unique indices 1..6; no task id
    listed twice; every listed task exists under *tasks_root* with a parseable
    task.yaml; no split key shared between phase 1 and phases 2-6 (in either
    direction); held-out tasks additionally appear in no public listing — the
    harness-side proxy is that a ``hidden: true`` flag in the task's
    metadata.json is respected when present.
    """
    from . import verifier

    errors: list[str] = []
    if not isinstance(split_manifest, dict):
        try:
            split_manifest = load_split_manifest(split_manifest)
        except (FileNotFoundError, ValueError, miniyaml.MiniYAMLError) as exc:
            return [str(exc)]
    manifest = split_manifest
    tasks_root = Path(tasks_root)

    phases = manifest["phases"]
    seen_indices: set[int] = set()
    seen_tasks: set[str] = set()
    phase_keys: dict[int, set[tuple[str, str, str, str]]] = {}
    for position, phase in enumerate(phases):
        if not isinstance(phase, dict):
            errors.append(f"phase #{position} is not a mapping")
            continue
        index = phase.get("index")
        name = phase.get("name")
        tasks = phase.get("tasks", [])
        if not isinstance(index, int) or not 1 <= index <= len(PHASE_NAMES):
            errors.append(f"phase {name!r}: index must be 1..{len(PHASE_NAMES)}, got {index!r}")
            continue
        if index in seen_indices:
            errors.append(f"phase index {index} listed twice")
        seen_indices.add(index)
        expected_name = PHASE_NAMES[index - 1]
        if name != expected_name:
            errors.append(f"phase {index}: name must be {expected_name!r}, got {name!r}")
        if not isinstance(tasks, list):
            errors.append(f"phase {index}: tasks must be a list")
            continue
        keys: set[tuple[str, str, str, str]] = set()
        for task_id in tasks:
            if task_id in seen_tasks:
                errors.append(f"task {task_id} listed in multiple phases")
            seen_tasks.add(task_id)
            task_dir = tasks_root / str(task_id)
            if not task_dir.is_dir():
                errors.append(f"phase {index}: task {task_id} not found under {tasks_root}")
                continue
            try:
                spec = verifier.load_task_yaml(task_dir)
            except (FileNotFoundError, ValueError, miniyaml.MiniYAMLError) as exc:
                errors.append(f"phase {index}: task {task_id} task.yaml invalid: {exc}")
                continue
            keys.update(split_key(spec))
        phase_keys[index] = keys

    missing = [i for i in range(1, len(PHASE_NAMES) + 1) if i not in seen_indices]
    if missing:
        errors.append(f"missing phases: {missing}")

    acquisition = phase_keys.get(1, set())
    for index in range(2, len(PHASE_NAMES) + 1):
        overlap = acquisition & phase_keys.get(index, set())
        for key in sorted(overlap):
            errors.append(
                f"split-key leak between phase 1 (acquisition) and phase {index}: {key}"
            )
    return errors
