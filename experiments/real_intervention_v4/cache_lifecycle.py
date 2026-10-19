"""Isolated cold caches and digest-bound cache materialization for each arm."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid
from typing import Any, Mapping
from fcntl import LOCK_EX, LOCK_NB, flock

from .outcomes import atomic_json


_CACHE_NAMES = {
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "TMPDIR": "tmp",
    "XDG_CACHE_HOME": "xdg",
}


def tree_digest(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            file_hash = hashlib.sha256(path.read_bytes()).digest()
            digest.update(file_hash)
    return digest.hexdigest()


def cache_artifact_digest(paths: Mapping[str, Path]) -> tuple[str, dict[str, str]]:
    parts = {name: tree_digest(Path(path)) for name, path in sorted(paths.items())}
    material = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest(), parts


def _owned_writable(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if info.st_uid != os.getuid():
        raise PermissionError(f"cache path is not owned by current user: {resolved}")
    if not resolved.is_dir() or not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"cache path is not readable/writable: {resolved}")
    return {"path": str(resolved), "owner_uid": info.st_uid,
            "mode_octal": oct(stat.S_IMODE(info.st_mode)), "empty": not any(resolved.iterdir())}


def _probe(path: Path) -> None:
    probe = path / f".evospe-cache-probe-{uuid.uuid4().hex}"
    temporary = probe.with_suffix(".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(b"cache-lifecycle-probe")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(probe)
        nested = path / "probe-child"
        nested.mkdir()
        probe.read_bytes()
        probe.unlink()
        nested.rmdir()
    finally:
        temporary.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)
        (path / "probe-child").rmdir() if (path / "probe-child").exists() else None


def _directories(root: Path) -> list[Path]:
    return [root, *(path for path in root.rglob("*") if path.is_dir())]


def create_cache_scope(attempt_dir: Path, *, mode: str = "EMPTY",
                       prefilled: Mapping[str, Path] | None = None,
                       expected_artifact_digest: str | None = None,
                       acquisition_cost_seconds: float | None = None,
                       lifecycle_lock_fd: int | None = None,
                       intent_recorded: bool = False) -> dict[str, Any]:
    """Create unique per-arm cache paths; setup shares a lock with worker launch."""
    if mode not in {"EMPTY", "PREFILLED"}:
        raise ValueError("cache mode must be EMPTY or PREFILLED")
    if mode == "PREFILLED" and (not prefilled or not expected_artifact_digest):
        raise ValueError("PREFILLED requires source cache paths and an artifact digest")
    artifact_manifest: dict[str, str] = {}
    if mode == "PREFILLED":
        actual_digest, artifact_manifest = cache_artifact_digest(prefilled or {})
        if actual_digest != expected_artifact_digest:
            raise ValueError("prefilled cache artifact digest mismatch")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    record_path = attempt_dir / "CACHE_SCOPE.json"
    lock_path = attempt_dir / "CACHE_SCOPE.lock"
    owns_lock = lifecycle_lock_fd is None
    lock_fd = (os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
               if owns_lock else lifecycle_lock_fd)
    try:
        if owns_lock:
            try:
                flock(lock_fd, LOCK_EX | LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another process owns this attempt's cache/runner lifecycle") from exc
        events = attempt_dir / "events"
        event_paths = sorted(events.glob("*.json")) if events.is_dir() else []
        has_events = bool(event_paths)
        if intent_recorded:
            from .outcomes import _read_events

            lifecycle_events = _read_events(attempt_dir)
            if [event["kind"] for event in lifecycle_events] != ["intent"]:
                raise RuntimeError("cache setup requires this attempt's sole durable intent")
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record.get("mode") != mode
                    or record.get("artifact_digest") != expected_artifact_digest
                    or record.get("source_cache_digests", {}) != artifact_manifest):
                raise ValueError("existing cache scope differs from the requested arm")
            for raw_path in record.get("environment", {}).values():
                path = Path(raw_path)
                if not path.resolve(strict=True).is_relative_to(Path(record["root"]).resolve(strict=True)):
                    raise ValueError(f"cache path escaped its attempt scope: {path}")
                _owned_writable(path)
            return record

        root = attempt_dir / "cache"
        staging = attempt_dir / ".cache.initializing"
        if has_events and not intent_recorded:
            raise RuntimeError("started arm has no durable cache-scope record; do not rebuild or replay")
        # A crash before the arm intent is safe to recover: no worker was launched.
        if staging.exists():
            shutil.rmtree(staging)
        if root.exists():
            shutil.rmtree(root)
        staging.mkdir(mode=0o700)
        paths: dict[str, str] = {}
        identities: dict[str, Any] = {}
        copied: dict[str, Path] = {}
        try:
            for env_name, child in _CACHE_NAMES.items():
                target = staging / child
                source = (prefilled or {}).get(env_name)
                if source is None:
                    target.mkdir(mode=0o700)
                else:
                    source = Path(source).resolve(strict=True)
                    shutil.copytree(source, target, copy_function=shutil.copy2)
                    copied[env_name] = target
                    for directory in _directories(target):
                        directory.chmod(0o700)
                target.chmod(0o700)
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError(f"cache path escaped attempt staging directory: {target}")
                _probe(target)
                if mode == "EMPTY" and any(target.iterdir()):
                    raise RuntimeError(f"cold cache is not empty after write probe: {target}")
                identities[env_name] = _owned_writable(target)
                identities[env_name]["writable_directory_count"] = len(_directories(target))

            if mode == "PREFILLED":
                copied_digest, copied_manifest = cache_artifact_digest(copied)
                if copied_digest != expected_artifact_digest or copied_manifest != artifact_manifest:
                    raise ValueError("materialized cache differs from its prefilled source artifact")
            else:
                copied_manifest = {}

            staging.rename(root)
            final_paths = {name: str((root / child).resolve(strict=True))
                           for name, child in _CACHE_NAMES.items()}
            for name, identity in identities.items():
                identity["path"] = final_paths[name]
            record = {
                "mode": mode, "root": str(root.resolve(strict=True)),
                "environment": final_paths, "directories": identities,
                "artifact_digest": expected_artifact_digest if mode == "PREFILLED" else None,
                "source_cache_digests": artifact_manifest,
                "materialized_cache_digests": copied_manifest,
                "acquisition_cost_seconds": acquisition_cost_seconds,
                "cache_state": "EMPTY" if mode == "EMPTY" else "PREFILLED",
            }
            atomic_json(record_path, record)
            return record
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    finally:
        if owns_lock:
            flock(lock_fd, 8)
            os.close(lock_fd)


def cache_usage(root: Path) -> dict[str, Any]:
    """Measure only a quiescent study tree; missing transient entries stay explicit."""
    total = 0
    missing_during_scan = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            missing_during_scan += 1
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                missing_during_scan += 1
    complete = missing_during_scan == 0
    return {"bytes": total if complete else None,
            "observed_bytes_lower_bound": total,
            "missing_entries_during_scan": missing_during_scan,
            "complete": complete,
            "status": "COMPLETE" if complete else "UNKNOWN"}
