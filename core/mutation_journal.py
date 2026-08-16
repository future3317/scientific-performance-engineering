"""Typed, append-only mutation records for governed condition stores."""

from __future__ import annotations

import json
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
    timestamp: str = ""

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

    def append(self, operation: str, subject_id: str, *, version: int | None = None, artifact_path: str | None = None, digest: str | None = None) -> Mutation:
        mutation = Mutation(operation, subject_id, version, artifact_path, digest, datetime.now(timezone.utc).isoformat())
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
                result.append(Mutation(**json.loads(line)))
        return tuple(result)


__all__ = ["Mutation", "MutationJournal"]
