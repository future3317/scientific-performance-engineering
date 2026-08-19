"""Formal release manifest and sealed-package materialization helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark.formal.attest import task_package_digest


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def materialize_frozen_slots(preregistered: dict[str, Any], sealed_root: str | Path) -> dict[str, Any]:
    """Create the executable slot index under a private sealed root.

    The package contents are supplied by the caller; the public repository only
    receives this opaque index and never receives sealed task files.
    """
    slots = list(preregistered.get("slots", []))
    if preregistered.get("status") not in {"preregistered_content_withheld", "materialized_frozen"}:
        raise ValueError("population manifest is not in a materializable state")
    root = Path(sealed_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for slot in slots:
        item = dict(slot)
        package_dir = root / str(item["slot_id"])
        task_dir = item.get("source_task_dir")
        if task_dir:
            source = Path(str(task_dir)).resolve()
            if not source.is_dir():
                raise FileNotFoundError(source)
            if package_dir.exists():
                shutil.rmtree(package_dir)
            shutil.copytree(source, package_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            item["task_id"] = str(item.get("task_id") or package_dir.name)
            item["package_digest"] = task_package_digest(package_dir)
        item["materialization"] = "materialized"
        materialized.append(item)
    return {
        "schema_version": 1,
        "population_id": preregistered.get("population_id"),
        "status": "materialized_frozen",
        "primary_population": preregistered.get("primary_population"),
        "sealed_root": str(root),
        "slots": materialized,
        "slot_count": len(materialized),
    }


def validate_materialized_manifest(manifest: dict[str, Any], sealed_root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if manifest.get("status") != "materialized_frozen":
        errors.append("formal population must be materialized_frozen")
    slots = manifest.get("slots")
    if not isinstance(slots, list) or len(slots) != 50:
        errors.append("materialized formal manifest must contain 50 slots")
    seen: set[str] = set()
    for slot in slots or []:
        if not isinstance(slot, dict):
            errors.append("slot must be an object")
            continue
        for key in ("slot_id", "task_id", "package_digest", "visibility", "track", "split"):
            if not slot.get(key):
                errors.append(f"slot missing {key}")
        slot_id = str(slot.get("slot_id"))
        if slot_id in seen:
            errors.append(f"duplicate slot_id: {slot_id}")
        seen.add(slot_id)
        if slot.get("visibility") not in {"sealed", "public_dev"}:
            errors.append(f"invalid visibility for {slot_id}")
        if sealed_root and slot.get("visibility") == "sealed":
            package = Path(sealed_root) / slot_id
            if not package.is_dir():
                errors.append(f"sealed package missing: {slot_id}")
            elif slot.get("package_digest") != task_package_digest(package):
                errors.append(f"sealed package digest mismatch: {slot_id}")
    return errors


def build_release_manifest(*, repo_root: str | Path, population: dict[str, Any], approval: dict[str, Any], contamination: dict[str, Any], claims_path: Path, protocol_path: Path, campaign_config: Path, executor_image: str) -> dict[str, Any]:
    root = Path(repo_root)
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "git_commit": str(approval.get("benchmark_revision", "")),
        "population_manifest_digest": digest_json(population),
        "approval_digest": approval.get("approval_digest"),
        "contamination_report_digest": digest_json(contamination),
        "claims_digest": hashlib.sha256(claims_path.read_bytes()).hexdigest(),
        "statistical_protocol_digest": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "campaign_config_digest": hashlib.sha256(campaign_config.read_bytes()).hexdigest(),
        "executor_image": executor_image,
    }
    manifest["release_digest"] = digest_json({k: v for k, v in manifest.items() if k != "release_digest"})
    return manifest

