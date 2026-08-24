from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from benchmark.formal.release_manifest import materialize_frozen_slots


ROOT = Path(__file__).resolve().parents[2]


def test_formal_slot_materializes_from_frozen_seed_and_passes_projection(tmp_path: Path):
    preregistered = json.loads(
        (ROOT / "benchmark" / "manifests" / "v1.0-50-slots.json").read_text(encoding="utf-8")
    )
    declared_slots = list(preregistered["slots"])
    assert len(declared_slots) == 50
    assert sum(slot.get("visibility") == "public_dev" for slot in declared_slots) == 15
    assert sum(slot.get("visibility") == "sealed" for slot in declared_slots) == 35
    probe_slot = next(slot for slot in declared_slots if slot["visibility"] == "sealed")
    probe_manifest = {**preregistered, "slots": [probe_slot]}

    materialized = materialize_frozen_slots(
        probe_manifest,
        tmp_path / "sealed",
        tasks_root=ROOT / "benchmark" / "tasks",
    )

    assert materialized["status"] == "materialized_frozen"
    assert len(materialized["slots"]) == 1
    assert materialized["public_dev_total"] == 0
    assert materialized["sealed_total"] == 1
    assert all(slot.get("family_instance") for slot in materialized["slots"])
    assert all(slot.get("source_task_id") for slot in materialized["slots"])
    assert materialized["slots"][0]["projection"]["status"] == "pass"

    # Exercise the contamination auditor on one public/sealed pair here. The
    # full 15/35 population remains a release-level CLI dry-run, without making
    # every pytest invocation spawn 65 isolated fixture workers.
    audit_path = ROOT / "scripts" / "audit_sealed_contamination.py"
    spec = importlib.util.spec_from_file_location("audit_sealed_contamination", audit_path)
    assert spec is not None and spec.loader is not None
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    audit_root = tmp_path / "audit-repo"
    public_root = audit_root / "benchmark" / "tasks"
    manifest_root = audit_root / "benchmark" / "manifests"
    public_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    sealed_slot = next(slot for slot in materialized["slots"] if slot["visibility"] == "sealed")
    shutil.copytree(
        ROOT / "benchmark" / "tasks" / sealed_slot["source_task_id"],
        public_root / sealed_slot["source_task_id"],
    )
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps({"population_id": materialized["population_id"], "slots": [sealed_slot]}), encoding="utf-8")
    contamination = audit_module.audit(audit_root, tmp_path / "sealed", manifest_path)
    assert contamination["status"] == "PASS"
    assert len(contamination["results"]) == 1
    assert all(item["status"] == "audited" for item in contamination["results"])

def test_task_root_contains_only_current_active_packages():
    manifest = json.loads((ROOT / "benchmark" / "pilot_population.json").read_text(encoding="utf-8"))
    active = set(manifest["task_ids"])
    packages = {
        path.name
        for path in (ROOT / "benchmark" / "tasks").iterdir()
        if path.is_dir() and (path / "task.yaml").is_file()
    }
    assert packages == active
