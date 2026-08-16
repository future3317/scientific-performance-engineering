"""Reference namespace executor contract used by formal campaigns."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
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
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "network_mode": self.network_mode,
            "mount_allowlist": list(self.mount_allowlist),
            "executor_digest": self.executor_digest,
            "worker_uid": self.worker_uid,
            "network_namespace_attested": self.network_namespace_attested,
            "mount_receipt": {"verified": self.mount_verified, "root": "/worker"},
            "isolation_canary": self.isolation_canary,
            "usage": dict(self.usage or {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0}),
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

    def command(self, command: Sequence[str], worker_root: Path) -> list[str]:
        if not self.available():
            raise RuntimeError(f"reference executor unavailable: {self.executable}")
        root = Path(worker_root).resolve()
        args = [self.executable, "--die-with-parent", "--unshare-all", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
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
        args.extend(["--chdir", "/worker", "--", *map(str, command)])
        return args

    def _run_isolation_canary(self, worker_root: Path) -> dict[str, bool]:
        """Probe the same namespace contract used for the worker.

        The receipt is based on observed probe output, not on the presence of
        bwrap flags.  A failed or unavailable probe is therefore fail-closed.
        """
        host_sentinel = Path(tempfile.mkstemp(prefix="acre-host-sentinel-", suffix=".txt")[1])
        host_sentinel.write_text("host-only", encoding="utf-8")
        try:
            probe = f"""
import json, pathlib, socket
network = False
try:
    socket.create_connection(('1.1.1.1', 53), 1)
    network = True
except Exception:
    pass
readonly = False
try:
    target = pathlib.Path('/worker/task/public_task.json')
    if not target.is_file():
        raise RuntimeError('readonly canary target missing')
    with target.open('a', encoding='utf-8') as handle:
        handle.write('x')
    readonly = True
except Exception:
    pass
host_visible = pathlib.Path({str(host_sentinel)!r}).exists()
print(json.dumps({{'network_blocked': not network, 'readonly_enforced': not readonly, 'host_path_hidden': not host_visible}}))
"""
            python = shutil.which("python3") or shutil.which("python") or sys.executable
            completed = subprocess.run(
                self.command([python, "-c", probe], worker_root),
                text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                return {"network_blocked": False, "readonly_enforced": False, "host_path_hidden": False}
            value = json.loads(completed.stdout.strip().splitlines()[-1])
            return {key: bool(value.get(key, False)) for key in ("network_blocked", "readonly_enforced", "host_path_hidden")}
        except (OSError, ValueError, json.JSONDecodeError, IndexError):
            return {"network_blocked": False, "readonly_enforced": False, "host_path_hidden": False}
        finally:
            host_sentinel.unlink(missing_ok=True)

    def execute(self, command: Sequence[str], worker_root: Path, *, receipt_path: Path, worker_uid: str = "unknown", include_skill: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute the namespace command and attest only observed properties."""
        argv = self.command(command, worker_root)
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
        canary = self._run_isolation_canary(Path(worker_root)) if completed.returncode == 0 else {}
        receipt = ExecutorReceipt(
            "external_namespace_executor", "none",
            tuple(name for name in ("task", "skill_view", "retrieved_context", "context_state", "result", "executor_receipt") if include_skill or name != "skill_view"),
            hashlib.sha256(Path(shutil.which(self.executable)).read_bytes()).hexdigest(),
            worker_uid,
            network_namespace_attested=bool(canary.get("network_blocked", False)),
            mount_verified=bool(canary.get("readonly_enforced", False) and canary.get("host_path_hidden", False)),
            isolation_canary=all(canary.get(key, False) for key in ("network_blocked", "readonly_enforced", "host_path_hidden")),
            usage={"input_tokens": 0, "output_tokens": 0, "tool_calls": 0},
        )
        receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return completed


__all__ = ["ExecutorReceipt", "ReferenceExecutor"]
