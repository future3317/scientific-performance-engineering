"""Materialize a frozen formal slot index into a private sealed root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal.release_manifest import materialize_frozen_slots, validate_materialized_manifest
from benchmark.formal import eligibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark" / "tasks")
    parser.add_argument("--efficacy-eligibility", type=Path, default=None)
    args = parser.parse_args()
    preregistered = json.loads(args.manifest.read_text(encoding="utf-8"))
    materialized = materialize_frozen_slots(preregistered, args.sealed_root, tasks_root=args.tasks_root)
    errors = validate_materialized_manifest(materialized, args.sealed_root)
    if errors:
        args.out.write_text(json.dumps({"status": "BLOCKED", "errors": errors}, indent=2) + "\n", encoding="utf-8")
        return 1
    if args.efficacy_eligibility is not None:
        source_eligibility = eligibility.load_manifest(args.efficacy_eligibility)
        eligibility.build_formal_efficacy_projection(
            source_eligibility,
            materialized,
            out_path=args.out.with_name("formal_efficacy_projection.json"),
        )
    args.out.write_text(json.dumps(materialized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
