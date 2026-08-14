#!/usr/bin/env python3
"""Copy one validated rule usage record into experience/usage without overwriting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: capture_rule_usage.py RECORD.json")
    source = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location("validate_rule_usage", ROOT / "scripts" / "validate_rule_usage.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load usage validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = json.loads(source.read_text(encoding="utf-8"))
    errors = module.validate_record(record)
    if errors:
        raise SystemExit("invalid usage record: " + "; ".join(errors))
    destination = ROOT / "experience" / "usage" / f"{record['usage_id']}.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing usage record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
