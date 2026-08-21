#!/usr/bin/env python3
"""CLI wrapper for the harness skill-view authority."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.harness.skill_view import main, render_skill_view, validate_skill_view_bundle

__all__ = ["main", "render_skill_view", "validate_skill_view_bundle"]


if __name__ == "__main__":
    main()
