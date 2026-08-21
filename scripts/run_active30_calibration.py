#!/usr/bin/env python3
"""CLI for the active-30 calibration campaign orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.calibration.campaign import run_calibration_campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--task-id", default=None)
    return run_calibration_campaign(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
