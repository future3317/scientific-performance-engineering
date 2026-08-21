#!/usr/bin/env python3
"""Compatibility entrypoint for the single repository pytest suite.

The repository's test authority is pytest. This command remains only for
local callers that used the historical path; it does not maintain a second
test inventory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
