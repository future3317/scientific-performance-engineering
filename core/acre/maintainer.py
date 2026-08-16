"""One maintenance reducer for the ACRE lifecycle.

The formal driver supplies environment callbacks; semantic state transitions
remain in core so calibration and evolution use the same sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from core.models import EvidenceEvent


@dataclass(frozen=True)
class MaintenanceResult:
    observed: int
    assessment: Mapping[str, Any]
    falsify: Any = None
    synthesis: Any = None
    acquisition: Any = None
    replay: Any = None
    validation: Any = None
    governance: Any = None
    relation_update: Any = None
    lifecycle: Any = None


class AcreMaintainer:
    """Reducer that consumes evidence before producing any lifecycle decision."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, events: Sequence[EvidenceEvent | Mapping[str, Any]]) -> tuple[EvidenceEvent, ...]:
        return tuple(self.engine.observe(event) for event in events)

    def update_effect_process(self) -> Any:
        return self.engine.assess()

    def falsify(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def synthesize(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def acquire(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def replay(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def validate(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def govern(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def relation_update(self, callback: Callable[[], Any] | None = None) -> Any:
        return callback() if callback else None

    def lifecycle_update(self, subject_id: str | None = None) -> Any:
        return self.engine.evolve(subject_id) if subject_id else None

    def step(
        self,
        events: Sequence[EvidenceEvent | Mapping[str, Any]] = (),
        *,
        falsify: Callable[[], Any] | None = None,
        synthesize: Callable[[], Any] | None = None,
        acquire: Callable[[], Any] | None = None,
        replay: Callable[[], Any] | None = None,
        validate: Callable[[], Any] | None = None,
        govern: Callable[[], Any] | None = None,
        relation_update: Callable[[], Any] | None = None,
        subject_id: str | None = None,
    ) -> MaintenanceResult:
        observed = self.observe(events)
        assessment = self.update_effect_process()
        return MaintenanceResult(
            observed=len(observed),
            assessment=assessment.to_dict() if hasattr(assessment, "__dict__") else assessment,
            falsify=self.falsify(falsify),
            synthesis=self.synthesize(synthesize),
            acquisition=self.acquire(acquire),
            replay=self.replay(replay),
            validation=self.validate(validate),
            governance=self.govern(govern),
            relation_update=self.relation_update(relation_update),
            lifecycle=self.lifecycle_update(subject_id),
        )


__all__ = ["AcreMaintainer", "MaintenanceResult"]
