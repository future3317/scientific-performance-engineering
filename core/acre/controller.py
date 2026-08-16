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
            events = [event for event in events if int(event.context.get("rule_versions", {}).get(str(subject_id), version)) == int(version)]
        return assess(events)
