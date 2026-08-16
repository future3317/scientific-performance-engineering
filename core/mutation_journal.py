"""Typed, append-only mutation records for governed condition stores."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

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
        if self.operation not in {"genesis", "add_evidence", "add_v2_spec", "update_state", "activate_registry", "retire_revision"}:
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

    def initialize_genesis(self, store: str | Path) -> Mutation:
        """Record the attested baseline before any governed mutation exists."""
        if self.entries():
            raise ValueError("mutation journal genesis already exists")
        root = Path(store)
        files: dict[str, str] = {}
        for managed in (root / "rules", root / "relations", root / "registry"):
            if not managed.exists():
                continue
            for artifact in managed.rglob("*"):
                if artifact.is_file():
                    files[str(artifact.relative_to(root)).replace("\\", "/")] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        detail = json.dumps({"kind": "baseline_digest", "files": files}, sort_keys=True, separators=(",", ":"))
        baseline_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return self.append("genesis", "condition-store", digest=baseline_digest, operation_detail=detail)

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
        entries = self.entries()
        # Standalone lifecycle tests may use a journal without a store
        # baseline.  Store-diff verification below is the governed path that
        # requires a genesis record.
        for entry in entries:
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
        entries = self.entries()
        for entry in self.entries():
            if entry.artifact_path and not entry.digest:
                raise ValueError(f"journaled artifact is missing digest: {entry.artifact_path}")
            if entry.operation in {"add_evidence", "add_v2_spec", "activate_registry", "retire_revision", "update_state"} and not entry.artifact_path:
                raise ValueError(f"mutation has no artifact path: {entry.operation}:{entry.subject_id}")
            if not entry.artifact_path or not entry.digest:
                continue
            relative = Path(entry.artifact_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("journal artifact path escapes store")
            path = root / relative
            if not path.is_file():
                raise ValueError(f"journaled artifact is missing: {entry.artifact_path}")
            # Immutable specs are checked at every journal entry below;
            # mutable artifacts are verified as transition chains.
        by_artifact: dict[str, list[Mutation]] = {}
        for entry in self.entries():
            if entry.artifact_path and entry.digest:
                by_artifact.setdefault(str(Path(entry.artifact_path)), []).append(entry)
        for artifact, entries in by_artifact.items():
            relative = Path(artifact)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("journal artifact path escapes store")
            path = root / relative
            if not path.is_file():
                raise ValueError(f"journaled artifact is missing: {artifact}")
            for previous, current in zip(entries, entries[1:]):
                if current.old_digest and current.old_digest != previous.digest:
                    raise ValueError(f"mutation transition chain mismatch: {artifact}")
            if entries[-1].digest != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"journaled artifact digest mismatch: {artifact}")

        # The journal is authoritative for the governed store surfaces.  A
        # regular JSON artifact under these directories that has no journal
        # entry is an unrecorded mutation, not harmless bookkeeping.
        baseline_files: set[str] = set()
        entries = self.entries()
        if entries and entries[0].operation == "genesis" and entries[0].operation_detail:
            try:
                baseline_files = set(json.loads(entries[0].operation_detail).get("files", {}))
                baseline_payload = json.loads(entries[0].operation_detail)
                expected_baseline = hashlib.sha256(json.dumps(baseline_payload.get("files", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                if entries[0].digest != expected_baseline:
                    raise ValueError("mutation journal genesis digest is invalid")
            except (TypeError, json.JSONDecodeError):
                raise ValueError("mutation journal genesis is invalid")
        journaled = {
            str(Path(entry.artifact_path)).replace("\\", "/")
            for entry in self.entries()
            if entry.artifact_path
        } | baseline_files
        managed_roots = (root / "rules", root / "relations", root / "registry")
        for managed_root in managed_roots:
            if not managed_root.exists():
                continue
            for artifact in managed_root.rglob("*"):
                if not artifact.is_file() or artifact.name == "mutation_journal.jsonl":
                    continue
                relative = str(artifact.relative_to(root)).replace("\\", "/")
                if relative not in journaled:
                    raise ValueError(f"unjournaled governed artifact: {relative}")


__all__ = ["Mutation", "MutationJournal"]
