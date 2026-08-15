"""Canonical, dependency-free data model for governed rule evolution.

The JSON files in the repository are projections of these models.  Keeping the
semantic objects here prevents rule meaning, evidence, and mutable state from
drifting into three independently hand-written contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from copy import deepcopy
import math
from typing import Any


def _nonempty(value: Any, name: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    version: int
    parent: str | None
    applicability: dict[str, Any]
    intervention: dict[str, Any]
    expected_mechanism: str
    evidence_requirements: list[str]
    scientific_invariants: list[str]
    abstain_conditions: dict[str, Any]
    relations: dict[str, list[str]]
    runtime_cost: dict[str, float]
    provenance_policy: dict[str, Any]
    severity: str = "P2"
    domain: str = "runtime"
    text: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.rule_id, "rule_id")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.severity not in {"P0", "P1", "P2", "P3", "P4"}:
            raise ValueError("severity must be P0-P4")
        _nonempty(self.applicability, "applicability")
        _nonempty(self.intervention, "intervention")
        _nonempty(self.expected_mechanism, "expected_mechanism")
        if not self.evidence_requirements:
            raise ValueError("evidence_requirements must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuleSpec":
        if "spec" in value:
            value = value["spec"]
        # Normalize the former flat card at the boundary; all internal code
        # uses this typed representation.
        if "rule_id" in value and "trigger" in value:
            return cls(
                rule_id=value["rule_id"], version=int(value.get("version", 1)),
                parent=value.get("parent"), applicability=value["trigger"],
                intervention=value.get("intervention", {"action": value.get("rule", {}).get("text", "")}),
                expected_mechanism=value.get("expected_mechanism", value.get("rule", {}).get("text", "")),
                evidence_requirements=list(value.get("requires_evidence", [])),
                scientific_invariants=list(value.get("scientific_invariants", [])),
                abstain_conditions={"any": value.get("do_not_apply_when", [])},
                relations={"requires": value.get("requires", []), "conflicts": value.get("conflicts_with", []), "supersedes": value.get("supersedes", [])},
                runtime_cost=dict(value.get("runtime_cost", {})),
                provenance_policy=dict(value.get("provenance_policy", {"required": True})),
                severity=value.get("severity", "P2"), domain=value.get("domain", "runtime"),
                text=value.get("rule", {}).get("text", ""),
            )
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_evidence_event(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy single-rule evidence into the v2 multi-intervention form."""
    payload = deepcopy(value)
    if payload.get("schema_version", 1) == 2 and "interventions" in payload.get("assignment", {}):
        return payload
    assignment = payload.get("assignment")
    rule_id = payload.get("rule_id")
    if not isinstance(assignment, dict) or assignment.get("arm") not in {"on", "off"}:
        raise ValueError("legacy evidence requires assignment.arm=on or off")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("legacy evidence requires rule_id")
    context = deepcopy(payload.get("context") or {})
    versions = dict(context.get("rule_versions") or {})
    rule_version = int(payload.get("rule_version", 1))
    if rule_version < 1:
        raise ValueError("rule_version must be >= 1")
    if rule_id in versions and int(versions[rule_id]) != rule_version:
        raise ValueError("legacy rule version conflicts with context.rule_versions")
    versions[rule_id] = rule_version
    context["rule_versions"] = versions
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be non-empty")
    return {
        "schema_version": 2,
        "event_id": event_id,
        "context": context,
        "assignment": {
            "interventions": {rule_id: 1 if assignment["arm"] == "on" else 0},
            "propensity": float(assignment.get("propensity", 0.5)),
            "design_id": str(assignment.get("design_id", "single-rule-v1")),
        },
        "evidence_stream": str(payload.get("evidence_stream", "representative")),
        "query_id": str(payload.get("query_id", event_id)),
        "outcome_vector": dict(payload.get("outcome_vector") or {}),
        "scientific_gates": dict(payload.get("scientific_gates") or {}),
        "artifacts": dict(payload.get("artifacts") or {}),
        "versions": dict(payload.get("versions") or {}),
        "source_id": payload.get("source_id"),
        "independence_group": payload.get("independence_group"),
        "timestamp": payload.get("timestamp"),
        "trust_zone": payload.get("trust_zone", "local"),
        "attacker_controlled_fields": list(payload.get("attacker_controlled_fields") or []),
    }


@dataclass(frozen=True)
class EvidenceEvent:
    schema_version: int
    event_id: str
    context: dict[str, Any]
    assignment: dict[str, Any]
    outcome_vector: dict[str, float]
    scientific_gates: dict[str, bool]
    artifacts: dict[str, Any]
    versions: dict[str, str]
    source_id: str
    independence_group: str
    timestamp: str
    evidence_stream: str
    query_id: str
    trust_zone: str
    attacker_controlled_fields: list[str]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("EvidenceEvent schema_version must be 2")
        _nonempty(self.event_id, "event_id")
        _nonempty(self.context, "context")
        interventions = self.assignment.get("interventions")
        if not isinstance(interventions, dict) or not interventions:
            raise ValueError("assignment.interventions must be non-empty")
        if any(not isinstance(rule_id, str) or not rule_id or isinstance(value, bool) or value not in {0, 1} for rule_id, value in interventions.items()):
            raise ValueError("assignment.interventions values must be 0 or 1")
        propensity = self.assignment.get("propensity")
        if isinstance(propensity, bool) or not isinstance(propensity, (int, float)) or not 0.0 < float(propensity) <= 1.0:
            raise ValueError("assignment.propensity must be in (0, 1]")
        _nonempty(self.assignment.get("design_id"), "assignment.design_id")
        if self.evidence_stream not in {"representative", "adversarial"}:
            raise ValueError("evidence_stream must be representative or adversarial")
        _nonempty(self.query_id, "query_id")
        _nonempty(self.source_id, "source_id")
        _nonempty(self.independence_group, "independence_group")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceEvent":
        return cls(**normalize_evidence_event(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    version: int
    parent: str | None
    rule_ids: list[str]
    kind: str
    applicability: dict[str, Any]
    contrast_definition: dict[str, Any]
    practical_margin: float
    scientific_invariants: list[str]
    provenance_policy: dict[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.relation_id, "relation_id")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if len(self.rule_ids) < 2 or len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("rule_ids must contain at least two unique rules")
        if self.kind not in {"synergy", "antagonism", "independence", "prerequisite", "redundancy", "semantic_conflict", "context_dependent_interaction"}:
            raise ValueError("invalid relation kind")
        _nonempty(self.applicability, "applicability")
        _nonempty(self.contrast_definition, "contrast_definition")
        if not math.isfinite(self.practical_margin) or self.practical_margin < 0:
            raise ValueError("practical_margin must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationSpec":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationState:
    relation_id: str
    version: int
    estimate: float
    confidence_sequence: dict[str, float] = field(default_factory=dict)
    status: str = "candidate"
    drift_state: str = "stable"
    counterexample_count: int = 0
    last_confirmed: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.relation_id, "relation_id")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if not math.isfinite(self.estimate):
            raise ValueError("estimate must be finite")
        if self.status not in {"candidate", "canonical", "retired"}:
            raise ValueError("invalid relation state status")
        if self.drift_state not in {"stable", "suspected_drift", "stale", "revalidating"}:
            raise ValueError("invalid relation drift state")
        if self.counterexample_count < 0:
            raise ValueError("counterexample_count must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationState":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleState:
    rule_id: str
    version: int
    status: str = "candidate"
    drift_state: str = "stable"
    effect: dict[str, float] = field(default_factory=dict)
    confidence_sequence: dict[str, float] = field(default_factory=dict)
    applicability_calibration: dict[str, float] = field(default_factory=dict)
    retrieval_utility: float = 0.0
    override_rate: float = 0.0
    provenance_diversity: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "canonical", "retired"}:
            raise ValueError("invalid rule state status")
        if self.drift_state not in {"stable", "suspected_drift", "stale", "revalidating"}:
            raise ValueError("invalid drift state")
        if self.version < 1:
            raise ValueError("version must be >= 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuleState":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskContext:
    domain: str
    workload: dict[str, Any]
    hardware: dict[str, Any]
    software: dict[str, Any]
    evidence: dict[str, Any]
    token_budget: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
