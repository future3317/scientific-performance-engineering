"""One maintenance reducer for the ACRE lifecycle.

The formal driver supplies environment callbacks; semantic state transitions
remain in core so calibration and evolution use the same sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from core.models import EvidenceEvent


@dataclass(frozen=True)
class MaintenanceInput:
    events: tuple[EvidenceEvent | Mapping[str, Any], ...] = ()
    subject_ids: tuple[str, ...] = ()
    version: int | None = None
    experiment_plans: tuple[Any, ...] = ()
    experiment_executor: Any = None
    record_case: Callable[[Mapping[str, Any]], None] | None = None
    update_certificate: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class MaintenanceTransition:
    observed: tuple[EvidenceEvent, ...]
    decisions: tuple[Any, ...]
    assessment: Mapping[str, Any]


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

    def run(self, maintenance_input: MaintenanceInput) -> MaintenanceTransition:
        """Execute evidence, planned experiments, and lifecycle transitions in Core."""
        observed = self.observe(maintenance_input.events)
        if maintenance_input.experiment_plans:
            if maintenance_input.experiment_executor is None or maintenance_input.record_case is None or maintenance_input.update_certificate is None:
                raise ValueError("planned experiments require executor, case recorder, and certificate updater")
            for plan in maintenance_input.experiment_plans:
                self.execute_node_experiment(
                    plan,
                    maintenance_input.experiment_executor,
                    record_case=maintenance_input.record_case,
                    update_certificate=maintenance_input.update_certificate,
                )
        decisions = tuple(self.engine.evolve(subject_id) for subject_id in maintenance_input.subject_ids)
        assessment = self.engine.assess()
        return MaintenanceTransition(observed, decisions, assessment.to_dict() if hasattr(assessment, "to_dict") else dict(assessment))

    def execute_node_experiment(
        self,
        plan: Any,
        executor: Any,
        *,
        record_case: Callable[[Mapping[str, Any]], None],
        update_certificate: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    ) -> Any:
        """Run a Core-owned paired plan and return immutable execution evidence."""
        from .experiments import execute_paired_plan
        return execute_paired_plan(
            plan,
            executor,
            record_case=record_case,
            update_certificate=update_certificate,
        )

    def execute_relation_experiment(self, context_blocks: Mapping[str, Sequence[Any]], *, delta: float = 0.05, practical_margin: float = 0.05) -> Any:
        """Execute scheduled factorial blocks and identify a relation in core."""
        from .factorial import FactorialEngine
        from .relation import RelationIdentifier
        estimates = {}
        for context_id, blocks in context_blocks.items():
            engine = FactorialEngine(delta=delta, practical_margin=practical_margin, look_count=max(1, len(blocks)))
            for block in blocks:
                engine.add_block(block)
            estimates[str(context_id)] = engine.estimate()
        return RelationIdentifier(practical_margin=practical_margin).identify(estimates)

    def relation_certificates(
        self,
        context_blocks: Mapping[str, Sequence[Any]],
        *,
        endpoint_versions: Mapping[str, int],
        delta: float = 0.05,
        practical_margin: float = 0.05,
    ) -> Mapping[str, Any]:
        """Build typed, context-specific certificates from executed blocks."""
        from .factorial import FactorialEngine, RelationEvidenceCertificate
        certificates: dict[str, Any] = {}
        for context_id, blocks in context_blocks.items():
            engine = FactorialEngine(delta=delta, practical_margin=practical_margin, look_count=max(1, len(blocks)))
            for block in blocks:
                engine.add_block(block)
            estimate = engine.estimate()
            certificates[str(context_id)] = RelationEvidenceCertificate(
                contrast_cs={
                    name: {"lcb": float(bounds[0]), "ucb": float(bounds[1])}
                    for name, bounds in estimate.contrast_intervals.items()
                },
                alpha_budget=float(delta),
                look_schedule=(len(blocks),),
                scientific_arm_gates={
                    "00": estimate.scientific_00,
                    "10": estimate.scientific_10,
                    "01": estimate.scientific_01,
                    "11": estimate.scientific_11,
                },
                applicability_provenance={"source": "core-factorial", "context_id": str(context_id)},
                endpoint_versions={str(key): int(value) for key, value in endpoint_versions.items()},
            )
        return certificates

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


__all__ = ["AcreMaintainer", "MaintenanceInput", "MaintenanceTransition", "MaintenanceResult"]
