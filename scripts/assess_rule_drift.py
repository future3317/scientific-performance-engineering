#!/usr/bin/env python3
"""Classify a rule effect stream and emit a RuleState update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.evolution import classify_drift
from core.models import RuleState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with rule_id, version, baseline, recent")
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    drift_state = classify_drift(payload["baseline"], payload["recent"], args.threshold)
    state = RuleState(payload["rule_id"], int(payload.get("version", 1)), drift_state=drift_state, effect={"baseline_mean": sum(payload["baseline"]) / len(payload["baseline"]), "recent_mean": sum(payload["recent"]) / len(payload["recent"])})
    args.output.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rule_id": state.rule_id, "drift_state": state.drift_state}))


if __name__ == "__main__":
    main()
