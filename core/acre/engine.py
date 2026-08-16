"""Single public orchestration façade for ACRE core semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from core.governance import EvolutionDecision
from core.models import EvidenceEvent, RelationSpec, RelationState, RuleSpec, RuleState, TaskContext, identifier_digest

from .router import ConservativeCausalRouter, RoutingDecision
from .controller import AcreController
from .maintainer import AcreMaintainer


class AcreEngine:
    def __init__(
        self,
        *,
        rule_specs: Sequence[RuleSpec] = (),
        rule_states: Mapping[str, RuleState] | None = None,
        relation_specs: Sequence[RelationSpec] = (),
        relation_states: Mapping[str, RelationState] | None = None,
        higher_order_certificates: Mapping[str, Any] | None = None,
        query_proposer: Callable[[TaskContext | Mapping[str, Any], Sequence[EvidenceEvent]], Any] | None = None,
    ) -> None:
        self.rule_specs = tuple(rule_specs)
        self.rule_states = dict(rule_states or {})
        self.relation_specs = tuple(relation_specs)
        self.relation_states = dict(relation_states or {})
        self.higher_order_certificates = dict(higher_order_certificates or {})
        self._query_proposer = query_proposer
        self._controller = AcreController()
        self.maintainer = AcreMaintainer(self)

    @classmethod
    def from_store(cls, store: str | Path) -> "AcreEngine":
        """Load one governed store without manufacturing lifecycle state."""
        root = Path(store)
        if not root.is_dir():
            raise FileNotFoundError(f"condition store not found: {root}")

        def read(path: Path) -> dict[str, Any]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid governed artifact: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"governed artifact must be an object: {path}")
            return value

        def load_state(directory: Path, identifier: str, spec_path: Path | None = None) -> dict[str, Any]:
            candidates = (
                (spec_path.with_name(f"{spec_path.stem}.state.json") if spec_path is not None else directory / "__missing__.state.json"),
                directory / f"{identifier}.state.json",
                directory / f"{identifier_digest(identifier)}.state.json",
                root / "states" / directory.name / f"{identifier}.json",
                root / "relation_states" / f"{identifier}.json",
                root / f"{directory.name}_states" / f"{identifier}.json",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return read(candidate)
            raise ValueError(f"missing materialized state for {identifier}")

        def latest_paths(paths: list[Path], id_field: str, registry_name: str) -> list[Path]:
            registry_path = root / "registry" / registry_name
            registry_dir = root / "registry"
            if registry_dir.is_dir() and paths and not registry_path.is_file():
                raise ValueError(f"active {id_field} registry is required: {registry_name}")
            if registry_path.is_file():
                registry = read(registry_path)
                entries = registry.get("rules" if id_field == "rule_id" else "relations", [])
                selected: list[Path] = []
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        raw_path = entry.get("spec_path") or entry.get("path")
                        if isinstance(raw_path, str):
                            path = root / raw_path
                            if path.is_file():
                                expected_digest = entry.get("spec_digest") or entry.get("digest")
                                if expected_digest:
                                    actual_digest = __import__("hashlib").sha256(
                                        path.read_bytes()
                                    ).hexdigest()
                                    if actual_digest != str(expected_digest):
                                        raise ValueError(f"registry spec digest mismatch: {raw_path}")
                                selected.append(path)
                if not isinstance(entries, list):
                    raise ValueError(f"active {id_field} registry is invalid: {registry_name}")
                if entries and len(selected) != len(entries):
                    raise ValueError(f"active {id_field} registry references missing specs: {registry_name}")
                return selected
            selected: dict[str, tuple[int, Path]] = {}
            for path in paths:
                value = read(path)
                identifier = str(value.get(id_field, ""))
                version = int(value.get("version", 0))
                previous = selected.get(identifier)
                if previous is None or version > previous[0]:
                    selected[identifier] = (version, path)
            return [item[1] for item in sorted(selected.values(), key=lambda value: str(value[1]))]

        rule_specs: list[RuleSpec] = []
        rule_states: dict[str, RuleState] = {}
        rules_dir = root / "rules"
        rule_paths = latest_paths(sorted(
            [path for path in rules_dir.glob("*.json") if not path.name.endswith(".state.json")]
            + [path for path in rules_dir.glob("*/v*.json") if not path.name.endswith(".state.json")]
        ), "rule_id", "rules.json") if rules_dir.is_dir() else []
        for path in rule_paths:
            if path.name.endswith(".state.json"):
                continue
            card = read(path)
            spec = RuleSpec.from_dict(card)
            raw_state = card.get("state") if isinstance(card.get("state"), dict) else load_state(rules_dir, spec.rule_id, path)
            state = RuleState.from_dict(raw_state.get("state", raw_state))
            if spec.rule_id != state.rule_id or spec.version != state.version:
                raise ValueError(f"rule spec/state version mismatch: {spec.rule_id}")
            rule_specs.append(spec)
            rule_states[spec.rule_id] = state

        relation_specs: list[RelationSpec] = []
        relation_states: dict[str, RelationState] = {}
        relations_dir = root / "relations"
        relation_paths = latest_paths(sorted(
            [path for path in relations_dir.glob("*.json") if not path.name.endswith(".state.json")]
            + [path for path in relations_dir.glob("*/v*.json") if not path.name.endswith(".state.json")]
        ), "relation_id", "relations.json") if relations_dir.is_dir() else []
        for path in relation_paths:
            if path.name.endswith(".state.json"):
                continue
            card = read(path)
            spec = RelationSpec.from_dict(card)
            raw_state = card.get("state") if isinstance(card.get("state"), dict) else load_state(relations_dir, spec.relation_id, path)
            state = RelationState.from_dict(raw_state.get("state", raw_state))
            if spec.relation_id != state.relation_id or spec.version != state.version:
                raise ValueError(f"relation spec/state version mismatch: {spec.relation_id}")
            relation_specs.append(spec)
            relation_states[spec.relation_id] = state

        certificates: dict[str, Any] = {}
        certificate_dir = root / "evolution" / "certificates"
        if certificate_dir.is_dir():
            for path in sorted(certificate_dir.glob("*.json")):
                certificate = read(path)
                bundle_ids = certificate.get("bundle_ids") or certificate.get("rule_ids")
                predicate = certificate.get("context_predicate") or certificate.get("applicability") or {}
                predicate_digest = identifier_digest(path.stem) if not predicate else __import__("hashlib").sha256(json.dumps(predicate, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                key = (":".join(sorted(str(item) for item in bundle_ids)) + ":" + predicate_digest) if isinstance(bundle_ids, list) else path.stem
                certificates[key] = certificate
        return cls(
            rule_specs=rule_specs,
            rule_states=rule_states,
            relation_specs=relation_specs,
            relation_states=relation_states,
            higher_order_certificates=certificates,
        )

    def observe(self, event: EvidenceEvent | Mapping[str, Any]) -> EvidenceEvent:
        return self._controller.observe(event)

    def assess(self):
        return self._controller.assess()

    def propose_experiment(self, context: TaskContext | Mapping[str, Any]) -> Any:
        if self._query_proposer is None:
            return None
        return self._query_proposer(context, self._controller.events)

    def maintain(self, events: Sequence[EvidenceEvent | Mapping[str, Any]] = (), **callbacks: Any):
        """Run the canonical observe→assess→evolve workflow.

        Callers supply only environment/verifier callbacks; ordering and
        lifecycle semantics remain owned by the core maintainer.
        """
        return self.maintainer.step(events, **callbacks)

    def update_rule(self, rule_id: str) -> EvolutionDecision:
        return self.evolve(rule_id)

    def update_relation(self, relation_id: str) -> EvolutionDecision:
        return self.evolve(relation_id)

    def evolve(self, subject_id: str) -> EvolutionDecision:
        if subject_id in self.rule_states:
            subject_type, state = "rule", self.rule_states[subject_id]
        elif subject_id in self.relation_states:
            subject_type, state = "relation", self.relation_states[subject_id]
        else:
            return EvolutionDecision("rule", subject_id, "NO_OP", "rejected", "none", "unknown subject")
        assessment = self._controller.assess(subject_id, getattr(state, "version", None))
        evidence_ids = tuple(event.event_id for event in self._controller.events)
        if assessment.specialization_event_ids:
            return EvolutionDecision(subject_type, subject_id, "SPECIALIZE", "review_required", "human-review", "adversarial evidence requires specialization or quarantine review", evidence_ids)
        if state.status == "retired":
            return EvolutionDecision(subject_type, subject_id, "RETIRE", "review_required", "human-review", "subject is retired", evidence_ids)
        if state.drift_state != "stable":
            return EvolutionDecision(subject_type, subject_id, "REVALIDATE", "review_required", "human-review", "subject drift requires revalidation", evidence_ids)
        return EvolutionDecision(subject_type, subject_id, "NO_OP", "approved", "bounded-auto", "subject state is stable", evidence_ids)

    def route(self, context: TaskContext | Mapping[str, Any], token_budget: int | None = None, higher_order_evidence: Mapping[str, float] | None = None) -> RoutingDecision:
        budget = token_budget if token_budget is not None else (context.token_budget if isinstance(context, TaskContext) else 4096)
        return ConservativeCausalRouter(token_budget=budget).route(
            self.rule_specs, self.rule_states, self.relation_specs, self.relation_states, context,
            higher_order_evidence, self.higher_order_certificates, True,
        )
