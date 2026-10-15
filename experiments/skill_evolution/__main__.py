"""Run the synthetic-only Skill evolution qualification fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .study import run_fixture_study
from .qualification import qualification_config, run_qualification_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true", help="run the synthetic mechanism fixture")
    mode.add_argument("--qualification", action="store_true", help="run train-only synthetic learned-ACRE qualification")
    parser.add_argument("--task-budget", type=float, help="estimated task execution units; defaults to the full frozen schedule")
    parser.add_argument("--output", type=Path, help="create a new JSON output file; qualification also requires a new sibling preregistration JSON")
    args = parser.parse_args(argv)
    if args.fixture:
        report = run_fixture_study(
            task_execution_budget=args.task_budget,
        )
    else:
        if args.output is None:
            parser.error("--qualification requires --output so its preregistration can be written before evaluation")
        preregistration_path = args.output.with_name(f"{args.output.stem}.preregistration.json")
        if args.output.exists() or preregistration_path.exists():
            parser.error("qualification report and preregistration paths must both be new")
        preregistration = json.dumps(qualification_config(), indent=2, sort_keys=True) + "\n"
        with preregistration_path.open("x", encoding="utf-8") as stream:
            stream.write(preregistration)
        report = run_qualification_study(task_execution_budget=args.task_budget)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
