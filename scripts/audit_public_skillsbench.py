#!/usr/bin/env python3
"""Audit public SkillsBench aggregate availability without provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_STRATIFICATION_FIELDS = ("category", "difficulty")


def audit_public_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return {
            "schema_version": 1,
            "status": "invalid_snapshot",
            "provider_calls": 0,
            "claim_boundary": "not external validity evidence",
        }
    missing = [field for field in REQUIRED_STRATIFICATION_FIELDS if any(field not in row for row in payload)]
    if missing:
        return {
            "schema_version": 1,
            "status": "metadata_insufficient",
            "provider_calls": 0,
            "rows": len(payload),
            "missing_stratification_fields": missing,
            "source": str(path),
            "claim_boundary": "not external validity evidence",
            "reason": "aggregate registry lacks the preregistered category/difficulty fields; no derived split or policy result is reported",
        }
    return {
        "schema_version": 1,
        "status": "metadata_available_but_audit_not_run",
        "provider_calls": 0,
        "rows": len(payload),
        "stratification_fields": list(REQUIRED_STRATIFICATION_FIELDS),
        "claim_boundary": "external aggregate audit pending frozen implementation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_public_snapshot(args.snapshot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
