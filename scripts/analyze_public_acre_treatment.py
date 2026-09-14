#!/usr/bin/env python3
"""Create paper-facing treatment metrics from completed server artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.external.treatment_analysis import analyze_treatment_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    artifacts = {
        condition: json.loads((args.campaign / condition / "campaign.json").read_text(encoding="utf-8"))
        for condition in ("A", "C", "D")
    }
    report = analyze_treatment_artifacts(artifacts, bootstrap_seed=args.seed, bootstrap_reps=args.bootstrap_reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

