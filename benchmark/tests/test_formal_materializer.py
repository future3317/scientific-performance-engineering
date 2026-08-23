from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from benchmark.formal.release_manifest import materialize_frozen_slots, validate_materialized_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_formal_slots_materialize_from_frozen_seed_and_pass_projection(tmp_path: Path):
    preregistered = json.loads(
        (ROOT / "benchmark" / "manifests" / "v1.0-50-slots.json").read_text(encoding="utf-8")
    )

    materialized = materialize_frozen_slots(
        preregistered,
        tmp_path / "sealed",
        tasks_root=ROOT / "benchmark" / "tasks",
    )

    assert materialized["status"] == "materialized_frozen"
    assert len(materialized["slots"]) == 50
    assert materialized["public_dev_total"] == 15
    assert materialized["sealed_total"] == 35
    assert all(slot.get("family_instance") for slot in materialized["slots"])
    assert all(slot.get("source_task_id") for slot in materialized["slots"])
    assert not validate_materialized_manifest(materialized, tmp_path / "sealed")

    audit_path = ROOT / "scripts" / "audit_sealed_contamination.py"
    spec = importlib.util.spec_from_file_location("audit_sealed_contamination", audit_path)
    assert spec is not None and spec.loader is not None
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    manifest_path = tmp_path / "materialized.json"
    manifest_path.write_text(json.dumps(materialized), encoding="utf-8")
    contamination = audit_module.audit(ROOT, tmp_path / "sealed", manifest_path)
    assert contamination["status"] == "PASS"
    assert len(contamination["results"]) == 35
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
