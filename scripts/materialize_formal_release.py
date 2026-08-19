"""Materialize a frozen formal slot index into a private sealed root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.formal.release_manifest import materialize_frozen_slots, validate_materialized_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preregistered = json.loads(args.manifest.read_text(encoding="utf-8"))
    materialized = materialize_frozen_slots(preregistered, args.sealed_root)
    errors = validate_materialized_manifest(materialized, args.sealed_root)
    if errors:
        args.out.write_text(json.dumps({"status": "BLOCKED", "errors": errors}, indent=2) + "\n", encoding="utf-8")
        return 1
    args.out.write_text(json.dumps(materialized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
