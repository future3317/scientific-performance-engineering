"""Canonical, dependency-free data model for governed rule evolution.

The JSON files in the repository are projections of these models.  Keeping the
semantic objects here prevents rule meaning, evidence, and mutable state from
drifting into three independently hand-written contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    context: dict[str, Any]
    rule_id: str
    rule_version: int
    assignment: dict[str, Any]
    outcome_vector: dict[str, float]
    scientific_gates: dict[str, bool]
    artifacts: dict[str, Any]
    versions: dict[str, str]
    source_id: str
    independence_group: str
    timestamp: str
    trust_zone: str = "local"
    attacker_controlled_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _nonempty(self.event_id, "event_id")
        _nonempty(self.context, "context")
        _nonempty(self.rule_id, "rule_id")
        if self.rule_version < 1:
            raise ValueError("rule_version must be >= 1")
        if self.assignment.get("arm") not in {"on", "off"}:
            raise ValueError("assignment.arm must be on or off")
        _nonempty(self.source_id, "source_id")
        _nonempty(self.independence_group, "independence_group")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceEvent":
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

