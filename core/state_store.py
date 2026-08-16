"""Persistent mutable state for governed rule and relation revisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import RelationState, RuleState, identifier_digest


class StateStore:
    """Write the mutable state sidecar without rewriting immutable specs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _state_payload(state: RuleState | RelationState) -> dict[str, Any]:
        return asdict(state)

    @staticmethod
    def digest(state: RuleState | RelationState) -> str:
        return hashlib.sha256(
            json.dumps(StateStore._state_payload(state), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def path_for(self, state: RuleState | RelationState) -> Path:
        subject_type = "rules" if isinstance(state, RuleState) else "relations"
        return self.root / subject_type / identifier_digest(state.rule_id if isinstance(state, RuleState) else state.relation_id) / f"v{int(state.version):04d}.state.json"

    def load(self, subject_type: str, subject_id: str, version: int, *, spec_path: Path | None = None) -> RuleState | RelationState:
        if spec_path is not None:
            path = spec_path if spec_path.name.endswith(".state.json") else spec_path.with_name(f"{spec_path.stem}.state.json")
        else:
            path = self.root / ("rules" if subject_type == "rule" else "relations") / identifier_digest(subject_id) / f"v{int(version):04d}.state.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("state"), dict):
            payload = payload["state"]
        if subject_type == "rule":
            return RuleState.from_dict(payload)
        return RelationState.from_dict(payload)

    def apply_transition(self, old_state: RuleState | RelationState, new_state: RuleState | RelationState, *, decision: Any, journal: Any | None = None) -> tuple[Path, str, str]:
        old_digest = self.digest(old_state)
        new_digest = self.digest(new_state)
        path = self.path_for(new_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous_artifact_digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        if path.exists():
            try:
                current_payload = json.loads(path.read_text(encoding="utf-8"))
                current = RuleState.from_dict(current_payload) if isinstance(old_state, RuleState) else RelationState.from_dict(current_payload)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("existing state artifact is invalid") from exc
            if self.digest(current) != old_digest:
                raise ValueError("state transition compare-and-swap failed")
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._state_payload(new_state), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if journal is not None:
            artifact_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            journal.append(
                "update_state",
                decision.subject_id,
                version=int(new_state.version),
                artifact_path=str(path.relative_to(self.root)),
                digest=artifact_digest,
                old_digest=previous_artifact_digest,
                operation_detail=str(decision.operation),
            )
        return path, old_digest, new_digest


__all__ = ["StateStore"]
