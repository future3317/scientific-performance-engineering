"""Stateful coordinator for the ACRE evidence-to-decision loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.models import EvidenceEvent

from .evidence import EvidenceAssessment, assess


class AcreController:
    """Keep the observed stream in one place for the public engine façade."""

    def __init__(self) -> None:
        self._events: list[EvidenceEvent] = []
        self._events_by_subject: dict[str, list[EvidenceEvent]] = {}

    def observe(self, event: EvidenceEvent | Mapping[str, Any]) -> EvidenceEvent:
        canonical = event if isinstance(event, EvidenceEvent) else EvidenceEvent.from_dict(dict(event))
        self._events.append(canonical)
        for subject_id in canonical.assignment.get("interventions", {}):
            self._events_by_subject.setdefault(str(subject_id), []).append(canonical)
        return canonical

    @property
    def events(self) -> tuple[EvidenceEvent, ...]:
        return tuple(self._events)

    def assess(self, subject_id: str | None = None, version: int | None = None) -> EvidenceAssessment:
        events = self._events if subject_id is None else self._events_by_subject.get(str(subject_id), [])
        if version is not None:
            filtered = []
            for event in events:
                rule_versions = event.context.get("rule_versions", {})
                relation_versions = event.context.get("relation_versions", {})
                recorded = rule_versions.get(str(subject_id)) if isinstance(rule_versions, Mapping) else None
                if recorded is None and isinstance(relation_versions, Mapping):
                    recorded = relation_versions.get(str(subject_id))
                if recorded is not None and int(recorded) == int(version):
                    filtered.append(event)
            events = filtered
        return assess(events)
