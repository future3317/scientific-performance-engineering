from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from benchmark.calibration.identity import task_package_digest
from benchmark.harness import runner


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


def test_resource_probe_uses_registered_measurement_window(monkeypatch, tmp_path: Path) -> None:
    from benchmark.formal import resource_domain

    repo_root = Path(__file__).parents[2]
    task_dir = repo_root / "benchmark" / "tasks" / "CORE-REPEATED-BACKBONE-02R2"
    calls: list[dict[str, object]] = []

    module = SimpleNamespace(
        load_solution=lambda **_: object(),
        make_fixtures=lambda **_: {},
        run_performance=lambda **_: {"step_ms_p50": 1.0},
    )

    def fake_call(function: object, **kwargs: object) -> object:
        if function is module.run_performance:
            calls.append(kwargs)
        return function(**kwargs)  # type: ignore[operator]

    monkeypatch.setattr(runner, "select_device", lambda _: ("cpu", True))
    monkeypatch.setattr(runner, "import_module_by_path", lambda _: module)
    monkeypatch.setattr(runner, "call_benchmark_fn", fake_call)
    resource_domain.run_baseline_probe(
        task_dir=task_dir,
        solution_dir=task_dir / "workspace" / "solution.py",
        out_path=tmp_path / "probe.json",
    )

    assert len(calls) == 1
    assert calls[0]["warmup"] == 40
    assert calls[0]["iterations"] == 800
