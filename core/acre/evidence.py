"""Asymmetric evidence policy for the ACRE method core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from core.models import EvidenceEvent


def _canonical(events: Iterable[EvidenceEvent]) -> tuple[EvidenceEvent, ...]:
    return tuple(event if isinstance(event, EvidenceEvent) else EvidenceEvent.from_dict(dict(event)) for event in events)


def representative_events(events: Iterable[EvidenceEvent]) -> tuple[EvidenceEvent, ...]:
    """Return only evidence allowed to update effect/confidence/promotion state."""
    return tuple(event for event in _canonical(events) if event.evidence_stream == "representative")


def adversarial_events(events: Iterable[EvidenceEvent]) -> tuple[EvidenceEvent, ...]:
    """Return evidence allowed to falsify or quarantine, never to promote."""
    return tuple(event for event in _canonical(events) if event.evidence_stream == "adversarial")


def _utility(event: EvidenceEvent) -> float | None:
    value = event.outcome_vector.get("utility")
    if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
        return None
    return float(value)


@dataclass(frozen=True)
class EvidenceAssessment:
    representative_count: int
    adversarial_count: int
    utility_effect_lcb: float | None
    promotion_probability_lcb: float | None
    falsifying_event_ids: tuple[str, ...]
    specialization_event_ids: tuple[str, ...]

    @property
    def promotion_lcb(self) -> float | None:
        """Compatibility view; routing must use ``utility_effect_lcb``."""
        return self.utility_effect_lcb


def assess(events: Iterable[EvidenceEvent], *, delta: float = 0.05) -> EvidenceAssessment:
    """Summarize evidence with an asymmetric promotion statistic.

    The confidence bound is computed exclusively from representative utility
    outcomes. Adversarial events are inspected only for falsification signals.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    canonical = _canonical(events)
    reps = representative_events(canonical)
    adv = adversarial_events(canonical)
    # Replay writes one audit event per arm, but one independence group is one
    # statistical observation.  Prefer the explicit paired effect and fall
    # back to a single unpaired utility only when no paired effect exists.
    grouped: dict[str, EvidenceEvent] = {}
    values: list[float] = []
    for event in reps:
        group = event.independence_group
        paired = event.outcome_vector.get("paired_effect")
        if paired is not None:
            if group not in grouped:
                grouped[group] = event
                if not isinstance(paired, bool) and math.isfinite(float(paired)):
                    values.append(float(paired))
            continue
        if group not in grouped:
            grouped[group] = event
            value = _utility(event)
            if value is not None:
                values.append(value)
    if values:
        # Allocate confidence over all prefixes so callers may inspect the
        # stream after any stopping time without reusing a fixed-sample bound.
        lcb = None
        for count in range(1, len(values) + 1):
            prefix = values[:count]
            mean = sum(prefix) / count
            prefix_delta = delta / (count * (count + 1))
            radius = math.sqrt(2.0 * math.log(2.0 / prefix_delta) / count)
            bound = max(-1.0, mean - radius)
            lcb = bound if lcb is None else max(lcb, bound)
    else:
        lcb = None
    falsifying = tuple(event.event_id for event in adv if not all(event.scientific_gates.values()))
    specializing = tuple(
        event.event_id
        for event in adv
        if event.outcome_vector.get("counterexample") is True or event.event_id in falsifying
    )
    return EvidenceAssessment(len(grouped), len(adv), lcb, None, falsifying, specializing)
