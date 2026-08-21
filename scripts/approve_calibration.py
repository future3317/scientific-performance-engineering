#!/usr/bin/env python3
"""CLI for canonical calibration approval construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.calibration.approval import issue_calibration_approval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--empirical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--approver", required=True)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    issue_calibration_approval(
        report_path=args.report, pilot_path=args.pilot, empirical_path=args.empirical,
        out_path=args.out, repo_root=args.repo_root.resolve(), approver=args.approver,
        timestamp=args.timestamp,
    )
    print(json.dumps({"approved": True, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
