#!/usr/bin/env python3
"""Create the content-free, digest-attested v1.0-50 slot preregistration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.harness import miniyaml


def build(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    task_root = root / "benchmark" / "tasks"
    tasks = []
    for path in sorted(task_root.iterdir()):
        if (path / "task.yaml").is_file():
            spec = miniyaml.load(str(path / "task.yaml"))
            if spec.get("track") != "evolution":
                tasks.append(spec)
    by_track = {track: [item for item in tasks if item.get("track") == track] for track in ("spe_core", "sciml")}
    by_track["evolution"] = [
        miniyaml.load(str(path / "task.yaml"))
        for path in sorted(task_root.iterdir())
        if (path / "task.yaml").is_file() and miniyaml.load(str(path / "task.yaml")).get("track") == "evolution"
    ]
    quotas = {"spe_core": (24, 8), "sciml": (20, 6), "evolution": (6, 1)}
    slots = []
    ordinal = 1
    for track, (total, public_count) in quotas.items():
        source = by_track[track]
        for index in range(total):
            template = source[index % len(source)] if source else {}
            visibility = "public_dev" if index < public_count else "sealed"
            slots.append({
                "slot_id": f"SPE-v1.0-50-{ordinal:02d}",
                "visibility": visibility,
                "track": track,
                "family_id": template.get("family_id", template.get("family", "pending")),
                "mechanism": template.get("mechanism", "pending"),
                "polarity": template.get("kind", "pending"),
                "difficulty": template.get("difficulty_tier", "medium"),
                "lineage_class": "pilot-derived-public" if visibility == "public_dev" else "sealed-independent-lineage",
                "generator_version": "SPE-EvoBench-generator-v1",
                "seed": 100000 + ordinal,
                "scientific_contract": template.get("scientific_contract_id", "pending"),
                "replacement_class": "same-family-mechanism-polarity-difficulty",
                "materialization": "withheld",
            })
            ordinal += 1
    payload = {
        "schema_version": 2,
        "population_id": "SPE-EvoBench-v1.0-50",
        "status": "preregistered_content_withheld",
        "primary_population": "sealed-35",
        "tracks": {"spe_core": {"total": 24, "public_dev": 8, "sealed": 16}, "sciml": {"total": 20, "public_dev": 6, "sealed": 14}, "evolution": {"total": 6, "public_dev": 1, "sealed": 5}},
        "public_dev_total": 15,
        "sealed_total": 35,
        "slots": slots,
        "generation_gate": "calibration_approval.approved == true and strict-formal passes",
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["artifact_digest"] = hashlib.sha256(body).hexdigest()
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.repo_root / "benchmark" / "manifests" / "v1.0-50-slots.json"
    out.write_text(json.dumps(build(args.repo_root), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
