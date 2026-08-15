#!/usr/bin/env python3
"""Poison identities stay in harness metadata, never in agent-visible data."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import evolution


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    episode = root / "benchmark" / "tasks" / "EVOL-EPISODE-POISON-10" / "episodes" / "poison_episode.yaml"
    with tempfile.TemporaryDirectory() as tmp:
        result = evolution.run_episode(episode, "D", Path(tmp) / "out", snapshot_dir=root, core_repo=root)
        out_dir = Path(tmp) / "out"
        for path in out_dir.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert not _contains_key(payload, "poisoned"), path
        assert not _contains_key(result, "poisoned")
        assert result["raw"]["poison_ids"]

    print("test_poison_label_never_visible: OK")


if __name__ == "__main__":
    main()
