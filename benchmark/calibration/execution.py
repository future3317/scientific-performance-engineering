"""Shared subprocess boundary for calibration and formal cell execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.harness import runner
from benchmark.provenance import digest_mapping, file_digest


def executor_digest(repo_root: str | Path) -> str:
    """Digest only files that can change one calibration cell's execution."""
    root = Path(repo_root)
    paths = [
        root / "benchmark" / "harness" / name
        for name in ("anticheat.py", "api.py", "cli.py", "fingerprint.py", "miniyaml.py", "runner.py", "stats.py", "verifier.py")
    ]
    paths.extend(root / "benchmark" / "calibration" / name for name in ("calibration_protocol.json", "execution.py", "identity.py"))
    paths.extend(root / "benchmark" / "schema" / name for name in ("task.schema.json", "result.schema.json"))
    files = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(set(paths)) if path.is_file()
    }
    return digest_mapping(files)


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
