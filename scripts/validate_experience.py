#!/usr/bin/env python3
"""Validate auditable experience records without promoting them to rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


CAPTURE_REASONS = {
    "falsified_hypothesis", "rule_failure", "better_diagnostic_order", "silent_fallback",
    "repeated_failure", "applicability_boundary", "negative_result", "uncategorized_bottleneck",
}
ATTEMPT_RESULTS = {"accepted", "rejected", "inconclusive"}
LESSON_TYPES = {"candidate", "negative_result", "boundary", "gap"}


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"invalid experience schema: {path}")
    return value


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def validate_record(record: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    if record.get("schema_version") != schema.get("properties", {}).get("schema_version", {}).get("const"):
        errors.append("schema_version must be 1")
    required = schema.get("required", [])
    for key in required:
        if _missing(record.get(key)):
            errors.append(f"missing required field {key}")
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id.startswith("EXP-"):
        errors.append("case_id must start with EXP-")
    if record.get("status") not in {"inbox", "case", "archived"}:
        errors.append("status must be inbox, case, or archived; canonical promotion is forbidden")
    reasons = record.get("capture_reasons")
    if not isinstance(reasons, list) or not reasons or any(item not in CAPTURE_REASONS for item in reasons):
        errors.append("capture_reasons must contain at least one supported reason")
    if not isinstance(record.get("symptom"), list) or not record["symptom"]:
        errors.append("symptom must be a non-empty list")
    workload = record.get("workload")
    if not isinstance(workload, dict) or any(not isinstance(workload.get(key), str) or not workload[key] for key in ("framework", "domain", "mode")):
        errors.append("workload.framework/domain/mode must be non-empty strings")
    evidence = record.get("observed_evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("observed_evidence must be a non-empty list")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, dict) or any(_missing(item.get(key)) for key in ("name", "value", "source")):
                errors.append(f"observed_evidence[{index}] needs name, value, and source")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append("attempts must be a non-empty list")
    else:
        for index, item in enumerate(attempts):
            if not isinstance(item, dict) or _missing(item.get("action")) or item.get("result") not in ATTEMPT_RESULTS:
                errors.append(f"attempts[{index}] needs action and a supported result")
                continue
            if item["result"] in {"rejected", "inconclusive"} and _missing(item.get("why_rejected")):
                errors.append(f"attempts[{index}] needs why_rejected for a non-accepted result")
            if item["result"] == "accepted" and _missing(item.get("evidence")):
                errors.append(f"attempts[{index}] needs evidence for an accepted result")
    lesson = record.get("lesson")
    if not isinstance(lesson, dict) or lesson.get("type") not in LESSON_TYPES or _missing(lesson.get("text")):
        errors.append("lesson needs a supported type and non-empty text")
    scope = record.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("requires"), list) or not isinstance(scope.get("excludes"), list):
        errors.append("scope requires and excludes must be lists")
    if record.get("confidence") not in {"low", "medium", "high"}:
        errors.append("confidence must be low, medium, or high")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or any(not isinstance(value, str) or not value for value in artifacts.values()):
        errors.append("artifacts must be an object of non-empty strings")
    return errors


def validate_file(path: Path, schema: dict[str, Any]) -> list[str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: {exc}"]
    return [f"{path}: {error}" for error in validate_record(record, schema)]


def main() -> None:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path(__file__).resolve().parents[1] / "assets" / "experience_record.json"
    root = Path(__file__).resolve().parents[1]
    schema = load_schema(root / "assets" / "experience_record.schema.json")
    files = sorted(target.rglob("*.json")) if target.is_dir() else [target]
    files = [path for path in files if path.name != "experience_record.schema.json"]
    errors = [error for path in files for error in validate_file(path, schema)]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"valid experience records: {len(files)}")


if __name__ == "__main__":
    main()
