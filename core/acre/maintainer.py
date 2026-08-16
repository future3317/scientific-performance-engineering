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
    lifecycle_decisions: tuple[Any, ...] = ()


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
        subject_ids: Sequence[str] = (),
    ) -> MaintenanceResult:
        observed = self.observe(events)
        assessment = self.update_effect_process()
        # Keep the lifecycle reducer ordered: no proposal, acquisition,
        # replay, validation, or governance callback may run before the
        # current evidence has been observed and assessed.  This is the
        # single workflow ordering shared by formal and episode callers.
        falsify_result = self.falsify(falsify)
        synthesis_result = self.synthesize(synthesize)
        acquisition_result = self.acquire(acquire)
        replay_result = self.replay(replay)
        validation_result = self.validate(validate)
        governance_result = self.govern(govern)
        relation_result = self.relation_update(relation_update)
        lifecycle_decisions = tuple(self.engine.evolve(item) for item in subject_ids)
        return MaintenanceResult(
            observed=len(observed),
            assessment=assessment.to_dict() if hasattr(assessment, "__dict__") else assessment,
            falsify=falsify_result,
            synthesis=synthesis_result,
            acquisition=acquisition_result,
            replay=replay_result,
            validation=validation_result,
            governance=governance_result,
            relation_update=relation_result,
            lifecycle=self.lifecycle_update(subject_id),
            lifecycle_decisions=lifecycle_decisions,
        )


__all__ = ["AcreMaintainer", "MaintenanceResult"]
