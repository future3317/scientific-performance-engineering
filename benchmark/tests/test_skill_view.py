#!/usr/bin/env python3
"""Regression test for condition snapshot allowlisting."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import conditions
from scripts.render_skill_view import render_skill_view


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "condition"
        bundle = Path(tmp) / "bundle"
        render_skill_view(root, bundle)
        conditions.materialize_condition("D", bundle, out)
        files = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
        assert not any(path == "benchmark" or path.startswith("benchmark/") for path in files)
        assert not any("oracle/" in path or "hidden_verifier/" in path for path in files)
        assert "SKILL.md" in files
        assert "registry/rules.json" in files

    print("test_skill_view: OK")


if __name__ == "__main__":
    main()
