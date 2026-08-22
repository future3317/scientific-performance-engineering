#!/usr/bin/env python3
"""CLI for the active-30 calibration campaign orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.calibration.campaign import run_calibration_campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="run when the selected GPU has other compute processes; records shared resource provenance",
    )
    return run_calibration_campaign(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
