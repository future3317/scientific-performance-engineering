"""One maintenance reducer for the ACRE lifecycle.

The formal driver supplies environment callbacks; semantic state transitions
remain in core so calibration and evolution use the same sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def lifecycle_update(self, subject_id: str | None = None) -> Any:
        return self.engine.evolve(subject_id) if subject_id else None

    def run(self, maintenance_input: MaintenanceInput) -> MaintenanceTransition:
        """Execute evidence, planned experiments, and lifecycle transitions in Core."""
        observed = self.observe(maintenance_input.events)
        if maintenance_input.experiment_plans:
            if maintenance_input.experiment_executor is None or maintenance_input.record_case is None or maintenance_input.update_certificate is None:
                raise ValueError("planned experiments require executor, case recorder, and certificate updater")
            for plan in maintenance_input.experiment_plans:
                execution = self.execute_node_experiment(
                    plan,
                    maintenance_input.experiment_executor,
                    record_case=maintenance_input.record_case,
                    update_certificate=maintenance_input.update_certificate,
                )
                if execution.evidence_events:
                    observed += self.observe(execution.evidence_events)
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

    def execute_higher_order_experiment(
        self,
        contexts: Sequence[Mapping[str, Any]],
        executor: Any,
        *,
        delta: float = 0.05,
        practical_margin: float = 0.05,
    ) -> Any:
        """Execute complete 2^3 bundles and return a coverage-safe certificate."""
        from .factorial import ThreeWayBlock, HigherOrderCertificate, estimate_higher_order
        blocks = []
        for index, context in enumerate(contexts):
            outcomes = executor(dict(context))
            if not isinstance(outcomes, Mapping):
                raise ValueError("higher-order executor must return arm outcomes")
            raw_gates = outcomes.get("scientific_gates")
            required_arms = {"000", "001", "010", "011", "100", "101", "110", "111"}
            if not isinstance(raw_gates, Mapping) or set(raw_gates) != required_arms:
                raise ValueError("higher-order executor must return scientific gates for all eight arms")
            gates = raw_gates
            cells = outcomes.get("outcomes", outcomes)
            if not isinstance(cells, Mapping) or set(cells) != required_arms:
                raise ValueError("higher-order executor must return all eight factorial outcomes")
            blocks.append(ThreeWayBlock(str(context.get("context_id", index)), {str(key): float(value) for key, value in cells.items()}, scientific_gates={str(key): bool(gates[str(key)]) for key in required_arms}))
        estimate = estimate_higher_order(blocks, delta=delta, look_count=max(1, len(blocks)), practical_margin=practical_margin)
        first_context = contexts[0] if contexts else {}
        context_root = first_context.get("context", first_context) if isinstance(first_context, Mapping) else {}
        bundle_versions = {str(key): int(value) for key, value in (context_root.get("rule_versions", {}) if isinstance(context_root, Mapping) and isinstance(context_root.get("rule_versions", {}), Mapping) else {}).items()}
        if len(bundle_versions) != 3:
            raise ValueError("higher-order execution requires three versioned bundle endpoints")
        status = "pairwise_certified" if estimate.status == "confirmed_negligible" else "hyperedge_required" if estimate.status == "confirmed_nonzero" else "unresolved"
        certificate = HigherOrderCertificate(
            bundle_versions=bundle_versions,
            context_predicate=dict(first_context.get("context_predicate", {"all": []}) if isinstance(first_context, Mapping) else {"all": []}),
            regime_digest=str(first_context.get("regime_digest", "unknown") if isinstance(first_context, Mapping) else "unknown"),
            residual_lcb=estimate.residual_lcb,
            residual_ucb=estimate.residual_ucb,
            normalized_residual=estimate.normalized_residual,
            raw_residual=estimate.raw_residual,
            status=status,
            scientific_arm_gates={str(key): all(block.scientific_gates[str(key)] for block in blocks) for key in ("000", "001", "010", "011", "100", "101", "110", "111")},
        )
        result = certificate.to_dict()
        self.engine.register_higher_order_certificate(result)
        return result

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
            if self.engine.state_store is not None:
                import hashlib, json
                directory = self.engine.state_store.root / "evolution" / "relation_certificates"
                directory.mkdir(parents=True, exist_ok=True)
                payload = certificates[str(context_id)].to_dict()
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                (directory / f"{digest}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return certificates

__all__ = ["AcreMaintainer", "MaintenanceInput", "MaintenanceTransition"]
