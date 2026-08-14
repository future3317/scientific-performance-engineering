#!/usr/bin/env python3
"""Validate and copy one practitioner record into the experience inbox."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_experience.py"
    spec = importlib.util.spec_from_file_location("validate_experience", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: capture_experience.py RECORD.json")
    source = Path(sys.argv[1]).resolve()
    validator = load_validator()
    schema = validator.load_schema(ROOT / "assets" / "experience_record.schema.json")
    record = json.loads(source.read_text(encoding="utf-8"))
    errors = validator.validate_record(record, schema)
    errors.extend(validator.validate_artifact_files(record, ROOT))
    if errors:
        raise SystemExit("invalid experience: " + "; ".join(errors))
    if record.get("status") != "inbox":
        raise SystemExit("capture accepts only status=inbox; promotion is a separate maintenance action")
    case_id = record["case_id"]
    destination = ROOT / "experience" / "inbox" / f"{case_id}.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing experience: {destination}")
    destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
