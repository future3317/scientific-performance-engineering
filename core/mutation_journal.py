"""Typed, append-only mutation records for governed condition stores."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .models import validate_identifier


@dataclass(frozen=True)
class Mutation:
    operation: str
    subject_id: str
    version: int | None = None
    artifact_path: str | None = None
    digest: str | None = None
    old_digest: str | None = None
    operation_detail: str | None = None
    timestamp: str = ""
    previous_digest: str = ""
    entry_digest: str = ""

    def __post_init__(self) -> None:
        if self.operation not in {"add_evidence", "add_v2_spec", "update_state", "activate_registry", "retire_revision"}:
            raise ValueError("unsupported mutation operation")
        validate_identifier(self.subject_id, "mutation subject_id")
        if self.version is not None and self.version < 1:
            raise ValueError("mutation version must be positive")


class MutationJournal:
    """Append-only journal; existing versioned artifacts are never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, operation: str, subject_id: str, *, version: int | None = None, artifact_path: str | None = None, digest: str | None = None, old_digest: str | None = None, operation_detail: str | None = None) -> Mutation:
        previous = self.entries()
        previous_digest = previous[-1].entry_digest if previous else ""
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {"operation": operation, "subject_id": subject_id, "version": version, "artifact_path": artifact_path, "digest": digest, "old_digest": old_digest, "operation_detail": operation_detail, "previous_digest": previous_digest, "timestamp": timestamp}
        entry_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        mutation = Mutation(operation, subject_id, version, artifact_path, digest, old_digest, operation_detail, timestamp, previous_digest, entry_digest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(mutation), sort_keys=True, ensure_ascii=False) + "\n")
        return mutation

    def entries(self) -> tuple[Mutation, ...]:
        if not self.path.is_file():
            return ()
        result: list[Mutation] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                payload.setdefault("previous_digest", "")
                payload.setdefault("entry_digest", "")
                result.append(Mutation(**payload))
        return tuple(result)

    def verify(self) -> None:
        """Verify append order and subject/version consistency."""
        previous = ""
        for entry in self.entries():
            if entry.previous_digest != previous:
                raise ValueError("mutation journal chain is inconsistent")
            payload = {"operation": entry.operation, "subject_id": entry.subject_id, "version": entry.version, "artifact_path": entry.artifact_path, "digest": entry.digest, "old_digest": entry.old_digest, "operation_detail": entry.operation_detail, "previous_digest": entry.previous_digest, "timestamp": entry.timestamp}
            expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if entry.entry_digest not in {expected, ""}:
                raise ValueError("mutation journal digest is invalid")
            previous = entry.entry_digest

    def verify_against_store_diff(self, store: str | Path) -> None:
        """Verify that journaled artifact digests match the current store."""
        root = Path(store)
        self.verify()
        for entry in self.entries():
            if not entry.artifact_path or not entry.digest:
                continue
            path = root / entry.artifact_path
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("journal artifact path escapes store")
            if not path.is_file():
                raise ValueError(f"journaled artifact is missing: {entry.artifact_path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry.digest:
                raise ValueError(f"journaled artifact digest mismatch: {entry.artifact_path}")


__all__ = ["Mutation", "MutationJournal"]
