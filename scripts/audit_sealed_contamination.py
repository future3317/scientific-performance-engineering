#!/usr/bin/env python3
"""Audit materialized sealed slots before formal unblinding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.harness import miniyaml
from benchmark.harness.runner import call_benchmark_fn, fixture_hash, import_module_by_path
from benchmark.taskgen.generate import ast_skeleton_hash
from benchmark.formal.attest import task_package_digest


def _fixture_digest(task_dir: Path) -> str:
    module = import_module_by_path(task_dir / "benchmark.py")
    fixtures = call_benchmark_fn(module.make_fixtures, seed=0, device="cpu")
    return fixture_hash(fixtures)


def audit(
    repo_root: str | Path,
    sealed_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    root = Path(repo_root)
    manifest_file = Path(manifest_path) if manifest_path is not None else root / "benchmark" / "manifests" / "v1.0-50-slots.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    tasks_root = Path(sealed_root).resolve() if sealed_root else root / "benchmark" / "tasks"
    public_root = root / "benchmark" / "tasks"
    public_records: list[dict] = []
    for public_dir in sorted(public_root.iterdir()) if public_root.is_dir() else ():
        if not (public_dir / "task.yaml").is_file():
            continue
        try:
            public_spec = miniyaml.load(str(public_dir / "task.yaml"))
            public_records.append({
                "task_id": public_dir.name,
                "spec": public_spec,
                "package_digest": task_package_digest(public_dir),
                "fixture_digest": _fixture_digest(public_dir),
                "ast": ast_skeleton_hash(public_dir),
            })
        except (OSError, ValueError):
            continue
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
        sealed_ast = ast_skeleton_hash(task_dir)
        fixture_digest = _fixture_digest(task_dir)
        checks = {
            "source_lineage": bool(lineage.get("source") and lineage.get("mutation_template_id")),
            "oracle_fix_pattern": bool(spec.get("oracle_fix_pattern_id")),
            "ast_fingerprint": spec.get("workspace_ast_skeleton_hash") == sealed_ast,
            "formal_variant_seed": isinstance(spec.get("formal_variant_seed"), int),
            "fixture_fingerprint": bool(fixture_digest),
            "family_parameter_overlap": True,
            "near_duplicate": False,
        }
        package_digest = task_package_digest(task_dir)
        for public in public_records:
            if public["task_id"] == task_id:
                continue
            public_spec = public["spec"]
            if package_digest == public["package_digest"]:
                checks["near_duplicate"] = True
            if (
                public.get("fixture_digest")
                and public["fixture_digest"] == fixture_digest
                and str(public_spec.get("family_id")) == str(spec.get("family_id"))
            ):
                checks["near_duplicate"] = True
            if str(public_spec.get("oracle_fix_pattern_id")) == str(spec.get("oracle_fix_pattern_id")):
                checks["near_duplicate"] = True
            if public_spec.get("lineage", {}) == lineage:
                checks["near_duplicate"] = True
        checks_pass = all(value for key, value in checks.items() if key != "near_duplicate") and not checks["near_duplicate"]
        entry = {
            "slot_id": slot["slot_id"], "task_id": task_id, "status": "audited" if checks_pass else "blocked",
            "source": lineage.get("source"), "mutation_template_id": lineage.get("mutation_template_id"),
            "family_id": spec.get("family_id"), "family_parameters": spec.get("family_parameters", {}),
            "oracle_fix_pattern_id": spec.get("oracle_fix_pattern_id"),
            "workspace_ast_skeleton_hash": spec.get("workspace_ast_skeleton_hash"),
            "formal_variant_seed": spec.get("formal_variant_seed"),
            "fixture_digest": fixture_digest,
            "checks": checks,
            "task_digest": package_digest,
            "metadata_digest": hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.is_file() else None,
        }
        results.append(entry)
        materialized.append(entry)
    # Exact task/family parameter reuse is acceptable only with an independent
    # formal fixture seed.
    seen: dict[tuple[str, str, str], str] = {}
    seen_packages: dict[str, str] = {}
    seen_fixtures: dict[tuple[str, str], str] = {}
    seen_seeds: dict[int, str] = {}
    for entry in materialized:
        task_id = str(entry.get("task_id"))
        family = str(entry.get("family_id"))
        parameters = json.dumps(entry.get("family_parameters", {}), sort_keys=True, separators=(",", ":"))
        formal_seed = entry.get("formal_variant_seed")
        key = (family, parameters, str(formal_seed))
        previous = seen.get(key)
        if previous and previous != task_id:
            entry["checks"]["family_parameter_overlap"] = False
            entry["checks"]["near_duplicate"] = False
            entry["status"] = "blocked"
            entry["contamination_with"] = previous
        else:
            seen[key] = task_id
        package_digest = str(entry.get("task_digest"))
        previous = seen_packages.get(package_digest)
        if previous and previous != task_id:
            entry["checks"]["near_duplicate"] = True
            entry["status"] = "blocked"
            entry["contamination_with"] = previous
        else:
            seen_packages[package_digest] = task_id
        fixture_key = (family, str(entry.get("fixture_digest")))
        previous = seen_fixtures.get(fixture_key)
        if previous and previous != task_id:
            entry["checks"]["near_duplicate"] = True
            entry["status"] = "blocked"
            entry["contamination_with"] = previous
        else:
            seen_fixtures[fixture_key] = task_id
        previous = seen_seeds.get(formal_seed)
        if formal_seed is not None and previous and previous != task_id:
            entry["checks"]["near_duplicate"] = True
            entry["status"] = "blocked"
            entry["contamination_with"] = previous
        elif formal_seed is not None:
            seen_seeds[formal_seed] = task_id
    payload = {"schema_version": 1, "population_id": manifest.get("population_id"), "status": "PASS" if results and all(item["status"] == "audited" for item in results) else "BLOCKED", "results": results}
    payload["artifact_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sealed-tasks-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.repo_root, args.sealed_tasks_root, args.manifest)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
