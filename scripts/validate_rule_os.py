#!/usr/bin/env python3
"""Validate the typed RuleSpec/RuleState envelope without promotion side effects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models import RuleSpec, RuleState


def validate(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["rule card must be an object"]
    errors: list[str] = []
    if value.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    try:
        RuleSpec.from_dict(value["spec"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"spec: {exc}")
    try:
        RuleState.from_dict(value["state"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"state: {exc}")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source_cases") or not provenance.get("independence_groups"):
        errors.append("provenance requires source_cases and independence_groups")
    return errors


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "rule_card.json"
    errors = validate(json.loads(path.read_text(encoding="utf-8")))
    if errors:
        raise SystemExit("invalid rule card: " + "; ".join(errors))
    print("valid typed rule card: 1")


if __name__ == "__main__":
    main()
