#!/usr/bin/env python3
"""CLI for the active-30 calibration campaign orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.calibration.campaign import run_calibration_campaign
from benchmark.formal.resource_domain import load_resource_domain_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--resource-domain-manifest", type=Path, default=None, help="frozen baseline-only resource-domain manifest")
    args = parser.parse_args()
    if args.resource_domain_manifest is not None:
        args.resource_domain_manifest = load_resource_domain_manifest(
            args.resource_domain_manifest, repo_root=args.repo_root.resolve(),
        )
    return run_calibration_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
