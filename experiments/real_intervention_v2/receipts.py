from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def read_receipts(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_pair(root: str | Path, receipt: Mapping[str, Any], *, storage_cap: int,
               storage_scope: str | Path | None = None) -> Path:
    """Publish one complete paired record atomically without replacing a prior receipt."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    session_id = str(receipt.get("session_id", "session-1"))
    target = directory / f"{receipt['context_id']}.{session_id}.json"
    encoded = (json.dumps(dict(receipt), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    scope = Path(storage_scope) if storage_scope is not None else directory
    current_size = sum(path.stat().st_size for path in scope.rglob("*") if path.is_file())
    if current_size + len(encoded) > storage_cap:
        raise RuntimeError("receipt storage cap would be exceeded")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        os.unlink(temporary)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def resume_receipt(root: str | Path, context_id: str, expected_identity: str,
                   session_id: str = "session-1") -> dict[str, Any] | None:
    path = Path(root) / f"{context_id}.{session_id}.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("identity") != expected_identity
            or receipt.get("session_id", "session-1") != session_id
            or set(receipt.get("arms", {})) != {"eager", "compile"}):
        raise ValueError(f"existing receipt identity or paired arms mismatch: {context_id}")
    return receipt
