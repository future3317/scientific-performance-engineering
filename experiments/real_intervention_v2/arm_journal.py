from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def _path(root: Path, context_id: str, session_id: str, arm: str) -> Path:
    if (arm not in {"eager", "compile", "eager_control"} or Path(context_id).name != context_id
            or Path(session_id).name != session_id):
        raise ValueError("invalid context or arm")
    return root / f"{context_id}.{session_id}.{arm}.json"


def _publish(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    data = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
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


def start_arm(root: str | Path, context_id: str, session_id: str, arm: str, identity: str) -> dict[str, Any]:
    """Persist an intent before execution; unresolved intents are never replayed."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(directory, context_id, session_id, arm)
    record = {"context_id": context_id, "session_id": session_id, "arm": arm,
              "identity": identity, "status": "started"}
    try:
        _publish(path, record, replace=False)
    except FileExistsError:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("existing arm receipt belongs to a different run identity")
        if previous.get("status") == "started":
            return {**previous, "status": "unknown", "reason": "unresolved_inflight_not_replayed"}
        return previous
    return record


def complete_arm(root: str | Path, context_id: str, session_id: str, arm: str, identity: str,
                 result: Mapping[str, Any]) -> None:
    path = _path(Path(root), context_id, session_id, arm)
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("identity") != identity or current.get("status") != "started":
        raise ValueError("arm completion does not match a live intent")
    _publish(path, {**current, **dict(result)}, replace=True)
