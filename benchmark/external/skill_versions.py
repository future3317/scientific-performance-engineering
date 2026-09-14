"""Append-only immutable skill-version records for online evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True)
class SkillVersion:
    version_id: str
    parent_id: str | None
    task_index: int
    content: str
    content_sha256: str
    evidence_ids: tuple[str, ...]
    certificate: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        value["certificate"] = dict(self.certificate)
        value["provenance"] = dict(self.provenance)
        return value


class SkillVersionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._versions: dict[str, SkillVersion] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                version = SkillVersion(
                    version_id=str(value["version_id"]),
                    parent_id=None if value.get("parent_id") is None else str(value["parent_id"]),
                    task_index=int(value["task_index"]),
                    content=str(value["content"]),
                    content_sha256=str(value["content_sha256"]),
                    evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
                    certificate=dict(value.get("certificate", {})),
                    provenance=dict(value.get("provenance", {})),
                )
                expected = hashlib.sha256(version.content.encode("utf-8")).hexdigest()
                if expected != version.content_sha256:
                    raise ValueError(f"skill version content digest invalid: {version.version_id}")
                if version.version_id in self._versions and self._versions[version.version_id] != version:
                    raise ValueError(f"skill version is not immutable: {version.version_id}")
                self._versions[version.version_id] = version

    def create(
        self,
        *,
        parent_id: str | None,
        task_index: int,
        content: str,
        evidence_ids: tuple[str, ...] | list[str],
        certificate: Mapping[str, Any],
        provenance: Mapping[str, Any],
        version_id: str | None = None,
    ) -> SkillVersion:
        content_sha256 = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
        resolved_id = version_id or f"skill-v{int(task_index)}-{content_sha256[:12]}"
        version = SkillVersion(
            version_id=resolved_id,
            parent_id=None if parent_id is None else str(parent_id),
            task_index=int(task_index),
            content=str(content),
            content_sha256=content_sha256,
            evidence_ids=tuple(str(item) for item in evidence_ids),
            certificate=dict(certificate),
            provenance=dict(provenance),
        )
        existing = self._versions.get(resolved_id)
        if existing is not None:
            if existing != version:
                raise ValueError(f"skill version content is immutable: {resolved_id}")
            return existing
        if parent_id is not None and str(parent_id) not in self._versions:
            raise ValueError(f"unknown parent skill version: {parent_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(version.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        self._versions[resolved_id] = version
        return version

    def get(self, version_id: str) -> SkillVersion:
        try:
            return self._versions[str(version_id)]
        except KeyError as exc:
            raise KeyError(f"unknown skill version: {version_id}") from exc

    def versions(self) -> list[SkillVersion]:
        return list(self._versions.values())

