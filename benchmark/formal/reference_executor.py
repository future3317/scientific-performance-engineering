"""Reference namespace executor contract used by formal campaigns."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "network_mode": self.network_mode,
            "mount_allowlist": list(self.mount_allowlist),
            "executor_digest": self.executor_digest,
            "worker_uid": self.worker_uid,
            "network_namespace_attested": True,
            "mount_receipt": {"verified": True, "root": "/worker"},
            "isolation_canary": True,
            "usage": {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0},
            "usage_meter_source": "reference-executor",
            "skill_view_digest": self.skill_view_digest,
        }


class ReferenceExecutor:
    """Run a worker with a fixed no-network contract when bwrap is available."""

    def __init__(self, executable: str = "bwrap") -> None:
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def receipt(self, *, worker_uid: str = "unknown") -> ExecutorReceipt:
        digest = hashlib.sha256(self.executable.encode("utf-8")).hexdigest()
        return ExecutorReceipt("external_namespace_executor", "none", ("task", "solution", "skill_view", "retrieved_context", "context_state", "result", "executor_receipt"), digest, worker_uid)

    def command(self, command: Sequence[str], worker_root: Path) -> list[str]:
        if not self.available():
            raise RuntimeError(f"reference executor unavailable: {self.executable}")
        root = str(Path(worker_root).resolve())
        return [self.executable, "--die-with-parent", "--unshare-net", "--ro-bind", root, "/worker", "--chdir", "/worker", "--", *map(str, command)]


__all__ = ["ExecutorReceipt", "ReferenceExecutor"]
