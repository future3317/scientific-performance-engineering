from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmark.calibration.identity import task_package_digest


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


def test_resource_probe_uses_registered_solution_entrypoint(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_resource_domain_census as census

    repo_root = Path(__file__).parents[2]
    task_dir = repo_root / "benchmark" / "tasks" / "CORE-AUTOGRAD-BATCHED-VJP-15R2"
    captured: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, _: Path) -> None:
            pass

        def run_module(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"exit_code": 0, "timed_out": False, "cleanup": {}}

    monkeypatch.setattr(census, "CellExecutor", FakeExecutor)
    census._run_attempt(
        repo_root=repo_root,
        task_dir=task_dir,
        task_id=task_dir.name,
        task_digest=task_package_digest(task_dir),
        output_path=tmp_path / "probe.json",
        session_id="session",
        seed=0,
    )

    args = captured["args"]
    assert isinstance(args, tuple)
    assert str(task_dir / "workspace" / "solution.py") in args
