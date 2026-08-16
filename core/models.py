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
import hashlib
import json
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def validate_identifier(value: Any, name: str = "identifier") -> str:
    """Validate IDs before they cross a filesystem or governance boundary."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must match [A-Za-z0-9][A-Za-z0-9._:-]*")
    return value


def identifier_digest(value: str) -> str:
    validate_identifier(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nonempty(value: Any, name: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class RevisionRef:
    """Immutable reference to the exact parent revision used for lineage."""

    subject_id: str
    version: int
    spec_digest: str

    def __post_init__(self) -> None:
        validate_identifier(self.subject_id, "revision subject_id")
        if self.version < 1 or not isinstance(self.spec_digest, str) or not self.spec_digest:
            raise ValueError("revision references require version and spec_digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "RevisionRef | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value, 1, "legacy")
        if isinstance(value, dict):
            return cls(str(value["subject_id"]), int(value["version"]), str(value["spec_digest"]))
        raise ValueError("invalid revision reference")


@dataclass(frozen=True)
class ActionSpec:
    """Reusable semantic intervention, independent of task source layout."""

    action_id: str
    family: str
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    preserves: list[str] = field(default_factory=list)
    risk_class: str = "bounded"

    def __post_init__(self) -> None:
        validate_identifier(self.action_id, "action_id")
        _nonempty(self.family, "action family")
        _nonempty(self.risk_class, "action risk_class")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificPolicySpec:
    """Shared scientific gate contract consumed by every benchmark view."""

    policy_id: str
    required_gates: tuple[str, ...] = ()
    tolerance: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.policy_id, "scientific policy_id")

    def evaluate(self, gates: Mapping[str, Any]) -> dict[str, bool]:
        return {name: bool(gates.get(name, False)) for name in self.required_gates}

    def to_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "required_gates": list(self.required_gates), "tolerance": dict(self.tolerance)}


@dataclass(frozen=True)
class RealizationRecord:
    """Task-specific execution record for a semantic action."""

    action_id: str
    task_id: str
    context_id: str
    baseline_digest: str
    patch: dict[str, Any]
    realized_digest: str
    verifier_digest: str

    def __post_init__(self) -> None:
        for name, value in (("action_id", self.action_id), ("task_id", self.task_id), ("context_id", self.context_id)):
            validate_identifier(value, name)
        for name, value in (("baseline_digest", self.baseline_digest), ("realized_digest", self.realized_digest), ("verifier_digest", self.verifier_digest)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finalized(self) -> bool:
        return self.verifier_digest != "unverified"

    def finalize(self, verifier_digest: str) -> "RealizationRecord":
        if not isinstance(verifier_digest, str) or not verifier_digest or verifier_digest == "unverified":
            raise ValueError("a finalized realization requires a verifier digest")
        return RealizationRecord(
            action_id=self.action_id,
            task_id=self.task_id,
            context_id=self.context_id,
            baseline_digest=self.baseline_digest,
            patch=dict(self.patch),
            realized_digest=self.realized_digest,
            verifier_digest=verifier_digest,
        )


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    version: int
    parent: RevisionRef | str | None
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
        validate_identifier(self.rule_id, "rule_id")
        object.__setattr__(self, "parent", RevisionRef.from_value(self.parent))
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
        normalized = dict(value)
        normalized["parent"] = RevisionRef.from_value(normalized.get("parent"))
        # relations is retained only as a read-side legacy projection.  New
        # serialized specs omit it; canonical relation meaning lives in
        # RelationSpec/RelationState.
        normalized.setdefault("relations", {})
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("relations", None)
        return value


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
        validate_identifier(self.event_id, "event_id")
        _nonempty(self.context, "context")
        interventions = self.assignment.get("interventions")
        if not isinstance(interventions, dict) or not interventions:
            raise ValueError("assignment.interventions must be non-empty")
        for rule_id, value in interventions.items():
            validate_identifier(rule_id, "intervention rule_id")
            if isinstance(value, bool) or value not in {0, 1}:
                raise ValueError("assignment.interventions values must be 0 or 1")
        propensity = self.assignment.get("propensity")
        if isinstance(propensity, bool) or not isinstance(propensity, (int, float)) or not 0.0 < float(propensity) <= 1.0:
            raise ValueError("assignment.propensity must be in (0, 1]")
        _nonempty(self.assignment.get("design_id"), "assignment.design_id")
        utility = self.outcome_vector.get("utility")
        if utility is not None and (
            isinstance(utility, bool) or not isinstance(utility, (int, float))
            or not math.isfinite(float(utility)) or not -1.0 <= float(utility) <= 1.0
        ):
            raise ValueError("outcome_vector.utility must be bounded in [-1, 1]")
        if self.evidence_stream not in {"representative", "adversarial"}:
            raise ValueError("evidence_stream must be representative or adversarial")
        validate_identifier(self.query_id, "query_id")
        validate_identifier(self.source_id, "source_id")
        validate_identifier(self.independence_group, "independence_group")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceEvent":
        return cls(**normalize_evidence_event(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    version: int
    parent: RevisionRef | str | None
    endpoints: dict[str, str]
    orientation: str
    kind: str
    applicability: dict[str, Any]
    contrast_definition: dict[str, Any]
    practical_margin: float
    scientific_invariants: list[str]
    provenance_policy: dict[str, Any]

    def __post_init__(self) -> None:
        validate_identifier(self.relation_id, "relation_id")
        object.__setattr__(self, "parent", RevisionRef.from_value(self.parent))
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if set(self.endpoints) != {"left", "right"} or any(not isinstance(value, str) or not value for value in self.endpoints.values()):
            raise ValueError("endpoints must contain non-empty left and right rule ids")
        validate_identifier(self.endpoints["left"], "relation endpoint")
        validate_identifier(self.endpoints["right"], "relation endpoint")
        if self.endpoints["left"] == self.endpoints["right"]:
            raise ValueError("relation endpoints must be distinct")
        if self.orientation not in {"symmetric", "left_to_right", "right_to_left"}:
            raise ValueError("invalid relation orientation")
        if self.kind not in {"synergy", "antagonism", "independence", "prerequisite", "redundancy", "semantic_conflict", "context_dependent_interaction"}:
            raise ValueError("invalid relation kind")
        if self.kind == "prerequisite" and self.orientation == "symmetric":
            raise ValueError("prerequisite relations must be directed")
        if self.kind != "prerequisite" and self.orientation != "symmetric":
            raise ValueError("only prerequisite relations may be directed")
        _nonempty(self.applicability, "applicability")
        _nonempty(self.contrast_definition, "contrast_definition")
        if not math.isfinite(self.practical_margin) or self.practical_margin < 0:
            raise ValueError("practical_margin must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationSpec":
        # Legacy cards are accepted only at the read boundary and are emitted
        # in the explicit endpoint/orientation form below.
        if "endpoints" not in value and "rule_ids" in value:
            ids = value["rule_ids"]
            if not isinstance(ids, list) or len(ids) != 2:
                raise ValueError("legacy rule_ids must contain exactly two rules")
            value = dict(value)
            value.pop("rule_ids")
            value["endpoints"] = {"left": ids[0], "right": ids[1]}
            value["orientation"] = "symmetric" if value.get("kind") != "prerequisite" else "left_to_right"
        normalized = dict(value)
        normalized["parent"] = RevisionRef.from_value(normalized.get("parent"))
        return cls(**normalized)

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
    contrast_bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    semantic_certificate: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.relation_id, "relation_id")
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
        validate_identifier(self.rule_id, "rule_id")
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
