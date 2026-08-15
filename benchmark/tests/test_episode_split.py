#!/usr/bin/env python3
"""Standalone test that the episode split manifest has no group leakage.

This test is intentionally kept separate from the harness unit tests so it can
be updated as new tasks are added to `benchmark/split/sequential.yaml`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import split


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "benchmark" / "split" / "sequential.yaml"
    tasks_root = repo_root / "benchmark" / "tasks"
    errors = split.check_leakage(manifest, tasks_root)
    assert errors == [], f"split leakage errors: {errors}"
    print(f"test_episode_split: {manifest} clean ({tasks_root})")


if __name__ == "__main__":
    main()
