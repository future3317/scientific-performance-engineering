from __future__ import annotations

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

def test_task_root_contains_only_current_active_packages():
    manifest = json.loads((ROOT / "benchmark" / "pilot_population.json").read_text(encoding="utf-8"))
    active = set(manifest["task_ids"])
    packages = {
        path.name
        for path in (ROOT / "benchmark" / "tasks").iterdir()
        if path.is_dir() and (path / "task.yaml").is_file()
    }
    assert packages == active
