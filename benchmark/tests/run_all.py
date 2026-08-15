#!/usr/bin/env python3
"""Run every standalone test script in benchmark/tests/ and summarize.

Usage: python benchmark/tests/run_all.py
Exit code 0 when all pass, 1 otherwise.
"""

import subprocess
import sys
import time
from pathlib import Path

TESTS = (
    "test_miniyaml.py",
    "test_stats.py",
    "test_anticheat.py",
    "test_split_conditions.py",
    "test_scientific_gates.py",
    "test_episode_split.py",
    "test_runner.py",
)


def main() -> int:
    here = Path(__file__).resolve().parent
    failures = 0
    for name in TESTS:
        path = here / name
        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True
        )
        elapsed = time.perf_counter() - start
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"[{status}] {name} ({elapsed:.1f}s)")
        if completed.returncode != 0:
            failures += 1
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
    print(f"run_all: {len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
