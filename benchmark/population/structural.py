"""Structural checks for an explicitly authored task population."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_active_manifest(tasks_root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Load the explicit active population; never infer it from directories."""
    root = Path(tasks_root)
    path = Path(manifest_path) if manifest_path is not None else root.parent / "pilot_population.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"active population manifest is unreadable: {path}: {exc}") from exc
    task_ids = payload.get("task_ids") if isinstance(payload, dict) else None
    if payload.get("status") != "active_manifest" or not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise ValueError("active population manifest must be status=active_manifest with unique task_ids")
    return payload


def artifact_findings(task_dir: Path, spec: dict[str, Any]) -> list[str]:
    oracle = task_dir / "oracle"
    required = [
        "bottleneck.json", "expected_mechanism.json", "reference_patch.diff",
        "tempting_wrong_patch.md", "noise_floor.json", "validation.json",
    ]
    errors = [f"{task_dir.name}: missing oracle/{name}" for name in required if not (oracle / name).is_file()]
    if not (task_dir / "workspace" / str(spec["workspace"]["entrypoint"])).is_file():
        errors.append(f"{task_dir.name}: baseline workspace entrypoint missing")
    if not (task_dir / "benchmark.py").is_file():
        errors.append(f"{task_dir.name}: benchmark.py missing")
    validation_path = oracle / "validation.json"
    if validation_path.is_file():
        try:
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{task_dir.name}: validation.json is invalid: {exc}")
        else:
            for key in ("baseline_validation", "oracle_validation", "anti_cheat", "deterministic_fixture"):
                if key not in payload:
                    errors.append(f"{task_dir.name}: validation.json lacks {key}")
    noise_path = oracle / "noise_floor.json"
    if noise_path.is_file():
        try:
            noise = json.loads(noise_path.read_text(encoding="utf-8"))
            if not isinstance(noise.get("declared_percent"), (int, float)):
                errors.append(f"{task_dir.name}: noise_floor.json lacks numeric declared_percent")
        except json.JSONDecodeError as exc:
            errors.append(f"{task_dir.name}: noise_floor.json is invalid: {exc}")
    if not (task_dir / "hidden_verifier").is_dir():
        errors.append(f"{task_dir.name}: hidden_verifier directory missing")
    return errors


def metadata_findings(task_dir: Path, spec: dict[str, Any]) -> list[str]:
    path = task_dir / "metadata.json"
    if not path.is_file():
        return [f"{task_dir.name}: metadata.json missing"]
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{task_dir.name}: metadata.json is invalid: {exc}"]
    lineage = spec.get("lineage", {})
    errors: list[str] = []
    if metadata.get("task_id") != spec.get("task_id"):
        errors.append(f"{task_dir.name}: metadata task_id does not match task.yaml")
    if metadata.get("track") != spec.get("track") or metadata.get("family") != spec.get("family"):
        errors.append(f"{task_dir.name}: metadata track/family does not match task.yaml")
    metadata_lineage = metadata.get("lineage", {})
    for key in ("source", "mutation_template_id"):
        if metadata_lineage.get(key) != lineage.get(key):
            errors.append(f"{task_dir.name}: metadata lineage.{key} does not match task.yaml")
    if metadata.get("difficulty") != spec.get("difficulty_tier"):
        errors.append(f"{task_dir.name}: metadata difficulty does not match difficulty_tier")
    for key in ("family_id", "anchor_instance_id"):
        if spec.get(key) and metadata.get(key) != spec.get(key):
            errors.append(f"{task_dir.name}: metadata {key} does not match task.yaml")
    return errors


def isolated_validate_task(task_dir: Path) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.harness.cli", "validate-task", str(task_dir), "--no-fixture-check"],
        capture_output=True, text=True, timeout=120,
    )
    if completed.returncode == 0:
        return []
    output = (completed.stderr or completed.stdout).strip()
    return [output or "isolated validate-task failed"]
