#!/usr/bin/env python3
"""Audit materialized sealed slots before formal unblinding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.harness import miniyaml
from benchmark.taskgen.generate import ast_skeleton_hash


def audit(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    manifest = json.loads((root / "benchmark" / "manifests" / "v1.0-50-slots.json").read_text(encoding="utf-8"))
    tasks_root = root / "benchmark" / "tasks"
    results = []
    materialized: list[dict] = []
    for slot in manifest.get("slots", []):
        if slot.get("visibility") != "sealed":
            continue
        task_id = slot.get("task_id")
        if not task_id:
            results.append({"slot_id": slot["slot_id"], "status": "withheld", "reason": "sealed content not materialized"})
            continue
        task_dir = tasks_root / str(task_id)
        if not (task_dir / "task.yaml").is_file():
            results.append({"slot_id": slot["slot_id"], "status": "blocked", "reason": "materialized slot missing task.yaml"})
            continue
        spec = miniyaml.load(str(task_dir / "task.yaml"))
        metadata_path = task_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        lineage = spec.get("lineage", {}) if isinstance(spec, dict) else {}
        checks = {
            "source_lineage": bool(lineage.get("source") and lineage.get("mutation_template_id")),
            "oracle_fix_pattern": bool(spec.get("oracle_fix_pattern_id")),
            "ast_fingerprint": spec.get("workspace_ast_skeleton_hash") == ast_skeleton_hash(task_dir),
            "family_parameter_overlap": True,
            "near_duplicate": True,
        }
        entry = {
            "slot_id": slot["slot_id"], "task_id": task_id, "status": "audited" if all(checks.values()) else "blocked",
            "source": lineage.get("source"), "mutation_template_id": lineage.get("mutation_template_id"),
            "family_id": spec.get("family_id"), "family_parameters": spec.get("family_parameters", {}),
            "oracle_fix_pattern_id": spec.get("oracle_fix_pattern_id"),
            "workspace_ast_skeleton_hash": spec.get("workspace_ast_skeleton_hash"),
            "checks": checks,
            "task_digest": hashlib.sha256((task_dir / "task.yaml").read_bytes()).hexdigest(),
            "metadata_digest": hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.is_file() else None,
        }
        results.append(entry)
        materialized.append(entry)
    # Exact task/family parameter reuse is a contamination finding unless it
    # is the same slot identity; sealed slots must have fresh lineage.
    seen: dict[tuple[str, str], str] = {}
    for entry in materialized:
        key = (str(entry.get("family_id")), json.dumps(entry.get("family_parameters", {}), sort_keys=True, separators=(",", ":")))
        previous = seen.get(key)
        if previous and previous != entry.get("task_id"):
            entry["checks"]["family_parameter_overlap"] = False
            entry["checks"]["near_duplicate"] = False
            entry["status"] = "blocked"
            entry["contamination_with"] = previous
        else:
            seen[key] = str(entry.get("task_id"))
    payload = {"schema_version": 1, "population_id": manifest.get("population_id"), "status": "pass" if results and all(item["status"] == "audited" for item in results) else "pending", "results": results}
    payload["artifact_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.repo_root), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
