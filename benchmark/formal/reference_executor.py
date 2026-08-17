"""Reference namespace executor contract used by formal campaigns."""

from __future__ import annotations

import hashlib
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ExecutorReceipt:
    mode: str
    network_mode: str
    mount_allowlist: tuple[str, ...]
    executor_digest: str
    worker_uid: str
    skill_view_digest: str | None = None
    network_namespace_attested: bool = False
    mount_verified: bool = False
    isolation_canary: bool = False
    usage: dict[str, float | int] | None = None
    canary_checks: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "network_mode": self.network_mode,
            "mount_allowlist": list(self.mount_allowlist),
            "executor_digest": self.executor_digest,
            "worker_uid": self.worker_uid,
            "network_namespace_attested": self.network_namespace_attested,
            "mount_receipt": {"verified": self.mount_verified, "root": "/worker"},
            "isolation_canary": self.isolation_canary,
            "canary_checks": dict(self.canary_checks or {}),
            "usage": dict(self.usage or {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "wall_time_s": 0.0}),
            "usage_meter_source": "reference-executor-observed",
            "skill_view_digest": self.skill_view_digest,
        }


class ReferenceExecutor:
    """Run a worker with a fixed no-network contract when bwrap is available."""

    def __init__(self, executable: str = "bwrap") -> None:
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def receipt(self, *, worker_uid: str = "unknown", include_skill: bool = True) -> ExecutorReceipt:
        executable = shutil.which(self.executable)
        digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest() if executable else "unavailable"
        mounts = ["task", "solution", "retrieved_context", "context_state", "result", "executor_receipt"]
        if include_skill:
            mounts.insert(1, "skill_view")
        return ExecutorReceipt("external_namespace_executor", "none", tuple(mounts), digest, worker_uid, network_namespace_attested=False, mount_verified=False, isolation_canary=False)

    @staticmethod
    def _resolve_executable(command: Sequence[str]) -> Path:
        if not command:
            raise ValueError("external command must not be empty")
        resolved = shutil.which(str(command[0]))
        if resolved is None:
            raise FileNotFoundError(f"executor command not found: {command[0]}")
        path = Path(resolved).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"executor command is not a regular file: {path}")
        return path

    @staticmethod
    def _elf_dependencies(executable: Path) -> tuple[Path, ...]:
        """Resolve the executable's actual loader and shared-library closure."""
        completed = subprocess.run(
            ["ldd", str(executable)], text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            return ()
        paths: set[Path] = set()
        interpreter = subprocess.run(
            ["readelf", "-l", str(executable)], text=True, capture_output=True, check=False,
        )
        match = re.search(r"Requesting program interpreter:\s*(/[^\]]+)", interpreter.stdout)
        if match and Path(match.group(1)).is_file():
            paths.add(Path(match.group(1)))
            paths.add(Path(match.group(1)).resolve())
        for line in completed.stdout.splitlines():
            match = re.search(r"(?:=>\s+)?(\/[^\s(]+)", line)
            if match:
                candidate = Path(match.group(1))
                if candidate.is_file():
                    paths.add(candidate)
                    paths.add(candidate.resolve())
        return tuple(sorted(paths, key=str))

    @staticmethod
    def _python_runtime(executable: Path) -> tuple[Path, ...]:
        """Return only the interpreter's discovered stdlib/site-library roots."""
        name = executable.name.lower()
        if not name.startswith("python"):
            return ()
        probe = subprocess.run(
            [str(executable), "-c", (
                "import json,sysconfig; "
                "print(json.dumps([sysconfig.get_path(name) for name in "
                "('stdlib','platstdlib','purelib','platlib') if sysconfig.get_path(name)]))"
            )],
            text=True, capture_output=True, check=False,
        )
        if probe.returncode != 0:
            return ()
        try:
            values = json.loads(probe.stdout.strip())
        except (json.JSONDecodeError, TypeError):
            return ()
        roots: set[Path] = set()
        for value in values if isinstance(values, list) else ():
            path = Path(str(value)).resolve()
            if path.is_dir():
                roots.add(path)
        return tuple(sorted(roots, key=str))

    def _runtime_mounts(self, command: Sequence[str]) -> tuple[Path, ...]:
        executable = self._resolve_executable(command)
        paths: set[Path] = {executable}
        paths.update(self._elf_dependencies(executable))
        paths.update(self._python_runtime(executable))
        return tuple(sorted(paths, key=str))

    def command(self, command: Sequence[str], worker_root: Path) -> list[str]:
        if not self.available():
            raise RuntimeError(f"reference executor unavailable: {self.executable}")
        root = Path(worker_root).resolve()
        executable = self._resolve_executable(command)
        runtime_mounts = self._runtime_mounts(command)
        args = [self.executable, "--die-with-parent", "--unshare-all", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for path in runtime_mounts:
            args.extend(["--ro-bind", str(path), str(path)])
        readonly = ("task", "skill_view", "retrieved_context", "context_state")
        writable = ("solution", "result", "executor_receipt")
        for name in readonly:
            source = root / name
            if source.is_dir():
                args.extend(["--ro-bind", str(source), f"/worker/{name}"])
        for name in writable:
            source = root / name
            if source.exists():
                args.extend(["--bind", str(source), f"/worker/{name}"])
        args.extend(["--chdir", "/worker", "--", str(executable), *map(str, command[1:])])
        return args

    def _run_isolation_canary(self, worker_root: Path) -> dict[str, bool]:
        """Probe the same namespace contract used for the worker.

        The receipt is based on observed probe output, not on the presence of
        bwrap flags.  A failed or unavailable probe is therefore fail-closed.
        """
        host_sentinel = Path(tempfile.mkstemp(prefix="acre-host-sentinel-", suffix=".txt")[1])
        host_sentinel.write_text("host-only", encoding="utf-8")
        benchmark_root = Path(__file__).resolve().parents[2]
        try:
            probe = f"""
import json, pathlib, socket
checks = {{"python_started": True, "network_blocked": False, "readonly_enforced": False,
          "host_path_hidden": False, "benchmark_root_hidden": False,
          "nonallowlist_hidden": False, "writable_dirs": False}}
try:
    socket.create_connection(('1.1.1.1', 53), 1)
except Exception:
    checks["network_blocked"] = True
try:
    target = pathlib.Path('/worker/task/public_task.json')
    if not target.is_file():
        raise RuntimeError('readonly canary target missing')
    with target.open('a', encoding='utf-8') as handle:
        handle.write('x')
except Exception:
    checks["readonly_enforced"] = True
host_visible = pathlib.Path({str(host_sentinel)!r}).exists()
checks["host_path_hidden"] = not host_visible
checks["benchmark_root_hidden"] = not pathlib.Path({str(benchmark_root)!r}).exists()
checks["nonallowlist_hidden"] = not pathlib.Path('/benchmark').exists() and not pathlib.Path('/condition_store').exists()
try:
    pathlib.Path('/worker/solution/.canary-write').write_text('ok', encoding='utf-8')
    pathlib.Path('/worker/result/.canary-write').write_text('ok', encoding='utf-8')
    checks["writable_dirs"] = True
except Exception:
    pass
print(json.dumps(checks))
"""
            python = shutil.which("python3") or shutil.which("python") or sys.executable
            completed = subprocess.run(
                self.command([python, "-c", probe], worker_root),
                text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                return {"python_started": False, "network_blocked": False, "readonly_enforced": False, "host_path_hidden": False, "benchmark_root_hidden": False, "nonallowlist_hidden": False, "writable_dirs": False}
            value = json.loads(completed.stdout.strip().splitlines()[-1])
            return {key: bool(value.get(key, False)) for key in ("python_started", "network_blocked", "readonly_enforced", "host_path_hidden", "benchmark_root_hidden", "nonallowlist_hidden", "writable_dirs")}
        except (OSError, ValueError, json.JSONDecodeError, IndexError):
            return {"python_started": False, "network_blocked": False, "readonly_enforced": False, "host_path_hidden": False, "benchmark_root_hidden": False, "nonallowlist_hidden": False, "writable_dirs": False}
        finally:
            host_sentinel.unlink(missing_ok=True)
            for path in (worker_root / "solution" / ".canary-write", worker_root / "result" / ".canary-write"):
                path.unlink(missing_ok=True)

    def execute(self, command: Sequence[str], worker_root: Path, *, receipt_path: Path, worker_uid: str = "unknown", include_skill: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute the namespace command and attest only observed properties."""
        argv = self.command(command, worker_root)
        started = time.monotonic()
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
        canary = self._run_isolation_canary(Path(worker_root)) if completed.returncode == 0 else {}
        wall_time_s = time.monotonic() - started
        checks = {key: bool(canary.get(key, False)) for key in ("python_started", "network_blocked", "readonly_enforced", "host_path_hidden", "benchmark_root_hidden", "nonallowlist_hidden", "writable_dirs")}
        receipt = ExecutorReceipt(
            "external_namespace_executor", "none",
            tuple(name for name in ("task", "solution", "skill_view", "retrieved_context", "context_state", "result", "executor_receipt") if include_skill or name != "skill_view"),
            hashlib.sha256(Path(shutil.which(self.executable)).read_bytes()).hexdigest(),
            worker_uid,
            network_namespace_attested=checks["network_blocked"],
            mount_verified=all(checks[key] for key in ("readonly_enforced", "host_path_hidden", "benchmark_root_hidden", "nonallowlist_hidden", "writable_dirs")),
            isolation_canary=all(checks.values()),
            usage={"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "wall_time_s": wall_time_s},
            canary_checks=checks,
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return completed


__all__ = ["ExecutorReceipt", "ReferenceExecutor"]


def main() -> int:
    """Small CLI adapter for the existing executor used by dry-run drivers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--command", required=True, help="worker command as one shell-style string")
    parser.add_argument("--worker-uid", default="reference-executor")
    args = parser.parse_args()
    include_skill = (args.worker_root / "skill_view").is_dir()
    completed = ReferenceExecutor().execute(
        shlex.split(args.command), args.worker_root,
        receipt_path=args.receipt, worker_uid=args.worker_uid,
        include_skill=include_skill,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
