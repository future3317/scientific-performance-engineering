from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_resource_domain_census_direct_entrypoint_imports_from_repo_root() -> None:
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_resource_domain_census.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "baseline-only" in result.stdout
