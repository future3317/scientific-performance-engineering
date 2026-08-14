#!/usr/bin/env python3
"""Fixtures for anytime confidence, drift, and provenance gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.evolution import anytime_lower_bound, classify_drift, leave_one_source_out


def main() -> None:
    assert anytime_lower_bound(200, 200, 0.05) > 0.8
    assert classify_drift([1.0] * 5, [1.0] * 5) == "stable"
    assert classify_drift([1.0] * 5, [2.0] * 5) == "suspected_drift"
    assert leave_one_source_out({"a": 0.3, "b": 0.4}, 0.2)["passed"]
    print("evolution statistics fixtures: ok")


if __name__ == "__main__":
    main()
