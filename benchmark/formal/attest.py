"""Experiment and artifact attestation for formal evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmark.harness import anticheat, miniyaml
from scripts.render_skill_view import validate_skill_view_bundle


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_mapping(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def skill_view_digest(bundle: str | Path) -> str:
    bundle = Path(bundle)
    errors = validate_skill_view_bundle(bundle)
    if errors:
        raise ValueError("invalid skill-view bundle: " + "; ".join(errors))
    files = anticheat.hash_tree(bundle)
    files.pop("skill_view_manifest.json", None)
    return digest_mapping(files)


def task_manifest_digest(tasks_root: str | Path, task_ids: list[str]) -> str:
    root = Path(tasks_root)
    payload: dict[str, Any] = {}
    for task_id in task_ids:
        task_dir = root / task_id
        task_path = task_dir / "task.yaml"
        if not task_path.is_file():
            raise FileNotFoundError(f"task manifest not found: {task_path}")
        payload[task_id] = miniyaml.load(str(task_path))
    return digest_mapping(payload)


def benchmark_revision(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


REQUIRED_FIELDS = (
    "schema_version", "experiment_id", "benchmark_revision", "skill_view_digest",
    "task_manifest_digest", "agent_model_id", "agent_config", "condition",
    "context_mode", "worker_isolation", "task_order", "outer_trial_id", "budgets",
    "hardware_fingerprint", "software_fingerprint", "torch_version", "cuda_version",
)


def validate_experiment(manifest: dict[str, Any]) -> list[str]:
    errors = [f"missing {key}" for key in REQUIRED_FIELDS if key not in manifest]
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("condition") not in {"A", "B", "C", "C_STRESS", "D"}:
        errors.append("condition must be A, B, C, C_STRESS, or D")
    if manifest.get("context_mode") not in {"reset", "carry"}:
        errors.append("context_mode must be reset or carry")
    if not isinstance(manifest.get("task_order"), list) or not manifest.get("task_order"):
        errors.append("task_order must be a non-empty list")
    isolation = manifest.get("worker_isolation")
    if not isinstance(isolation, dict):
        errors.append("worker_isolation must be an object")
    else:
        if isolation.get("mode") != "external_namespace_executor":
            errors.append("worker_isolation.mode must be external_namespace_executor")
        if isolation.get("network_mode") != "none":
            errors.append("worker_isolation.network_mode must be none")
        if not isinstance(isolation.get("mount_allowlist"), list) or not isolation.get("mount_allowlist"):
            errors.append("worker_isolation.mount_allowlist must be non-empty")
        receipt = isolation.get("executor_receipt")
        if receipt is not None and (not isinstance(receipt, dict) or receipt.get("network_mode") != "none"):
            errors.append("worker_isolation.executor_receipt must attest network_mode=none")
    return errors


def write_experiment(path: str | Path, manifest: dict[str, Any]) -> None:
    errors = validate_experiment(manifest)
    if errors:
        raise ValueError("invalid experiment manifest: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
