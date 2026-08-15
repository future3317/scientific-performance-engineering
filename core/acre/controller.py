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

    def observe(self, event: EvidenceEvent | Mapping[str, Any]) -> EvidenceEvent:
        canonical = event if isinstance(event, EvidenceEvent) else EvidenceEvent.from_dict(dict(event))
        self._events.append(canonical)
        return canonical

    @property
    def events(self) -> tuple[EvidenceEvent, ...]:
        return tuple(self._events)

    def assess(self) -> EvidenceAssessment:
        return assess(self._events)
