"""Shared subprocess boundary for calibration and formal cell execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.harness import runner


class CellExecutor:
    """Own subprocess isolation; callers select the scientific cell kind."""

    def __init__(self, cwd: str | Path):
        self.cwd = Path(cwd)

    def _run(self, *, module: str | None = None, snippet: str | None = None,
             args: tuple[str, ...] = (), timeout_s: float = 600.0) -> dict[str, Any]:
        return runner.run_python_subprocess(
            module=module, snippet=snippet, args=args, timeout=timeout_s, cwd=self.cwd,
        )

    def run_atomic(self, *, args: tuple[str, ...], timeout_s: float) -> dict[str, Any]:
        return self._run(module="benchmark.harness.cli", args=args, timeout_s=timeout_s)

    def run_module(self, *, module: str, args: tuple[str, ...], timeout_s: float) -> dict[str, Any]:
        return self._run(module=module, args=args, timeout_s=timeout_s)

    def run_noise_control(self, *, args: tuple[str, ...], timeout_s: float) -> dict[str, Any]:
        return self._run(module="benchmark.harness.cli", args=args, timeout_s=timeout_s)

    def run_episode(self, *, snippet: str, timeout_s: float) -> dict[str, Any]:
        return self._run(snippet=snippet, timeout_s=timeout_s)
