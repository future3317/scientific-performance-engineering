#!/usr/bin/env python3
"""Validate rule retrieval/use telemetry without inferring utility from retrieval alone."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def validate_record(record: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["usage record must be an object"]
    if record.get("schema_version") != 1 or not isinstance(record.get("usage_id"), str) or not record["usage_id"].startswith("USE-"):
        errors.append("schema_version=1 and usage_id=USE-* are required")
    if not isinstance(record.get("task_id"), str) or not record["task_id"]:
        errors.append("task_id must be non-empty")
    if not isinstance(record.get("context_features"), dict):
        errors.append("context_features must be an object")
    for key in ("retrieved_rule_ids", "triggered_rule_ids", "followed_rule_ids", "overridden_rule_ids"):
        values = record.get(key)
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            errors.append(f"{key} must be a list of rule ids")
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") not in {"accepted", "rejected", "inconclusive"} or not isinstance(outcome.get("utility"), (int, float)) or not isinstance(outcome.get("scientific_gates_passed"), bool):
        errors.append("outcome needs status, numeric utility, and scientific_gates_passed")
    return errors


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_rule_usage.py RECORD.json")
    path = Path(sys.argv[1])
    record = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_record(record)
    if errors:
        raise SystemExit("invalid usage record: " + "; ".join(errors))
    print("valid rule usage record: 1")


if __name__ == "__main__":
    main()
