"""JSON Schema projections generated from the canonical model contract."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

try:
    from .models import EvidenceEvent, RuleSpec, RuleState
except ImportError:  # validate_skill.py executes this module as a source projection
    from core.models import EvidenceEvent, RuleSpec, RuleState


BASE = "https://github.com/future3317/scientific-performance-engineering/"


def _required(model: type[Any]) -> list[str]:
    """Derive required top-level fields from dataclass defaults."""
    return [
        item.name
        for item in fields(model)
        if item.default is MISSING and item.default_factory is MISSING
    ]


def _object(model: type[Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Build a strict object schema from the model's canonical field set."""
    model_fields = {item.name for item in fields(model)}
    if set(properties) != model_fields:
        missing = sorted(model_fields - set(properties))
        extra = sorted(set(properties) - model_fields)
        raise ValueError(f"schema/model field drift for {model.__name__}: missing={missing}, extra={extra}")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": _required(model),
        "properties": properties,
    }


def schemas() -> dict[str, dict[str, Any]]:
    return {
        "rule_spec.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "rule_spec.schema.json", "title": "RuleSpec", "type": "object",
            **_object(RuleSpec, {"rule_id": {"type": "string", "pattern": "^[A-Z0-9-]+$"}, "version": {"type": "integer", "minimum": 1}, "parent": {"type": ["string", "null"]}, "applicability": {"type": "object", "minProperties": 1}, "intervention": {"type": "object", "minProperties": 1}, "expected_mechanism": {"type": "string", "minLength": 1}, "evidence_requirements": {"type": "array", "minItems": 1, "items": {"type": "string"}}, "scientific_invariants": {"type": "array", "items": {"type": "string"}}, "abstain_conditions": {"type": "object"}, "relations": {"type": "object"}, "runtime_cost": {"type": "object"}, "provenance_policy": {"type": "object"}, "severity": {"enum": ["P0", "P1", "P2", "P3", "P4"]}, "domain": {"type": "string"}, "text": {"type": "string"}}),
        },
        "evidence_event.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "evidence_event.schema.json", "title": "EvidenceEvent", "type": "object",
            **_object(EvidenceEvent, {"event_id": {"type": "string", "minLength": 1}, "context": {"type": "object"}, "rule_id": {"type": "string"}, "rule_version": {"type": "integer", "minimum": 1}, "assignment": {"type": "object", "required": ["arm"], "properties": {"arm": {"enum": ["on", "off"]}}}, "outcome_vector": {"type": "object"}, "scientific_gates": {"type": "object"}, "artifacts": {"type": "object"}, "versions": {"type": "object"}, "source_id": {"type": "string"}, "independence_group": {"type": "string"}, "timestamp": {"type": "string"}, "trust_zone": {"type": "string"}, "attacker_controlled_fields": {"type": "array"}}),
        },
        "rule_state.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "rule_state.schema.json", "title": "RuleState", "type": "object",
            **_object(RuleState, {"rule_id": {"type": "string"}, "version": {"type": "integer", "minimum": 1}, "status": {"enum": ["candidate", "canonical", "retired"]}, "drift_state": {"enum": ["stable", "suspected_drift", "stale", "revalidating"]}, "effect": {"type": "object"}, "confidence_sequence": {"type": "object"}, "applicability_calibration": {"type": "object"}, "retrieval_utility": {"type": "number"}, "override_rate": {"type": "number", "minimum": 0, "maximum": 1}, "provenance_diversity": {"type": "integer", "minimum": 0}, "last_updated": {"type": "string"}}),
        },
    }
