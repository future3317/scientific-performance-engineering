"""JSON Schema projections generated from the canonical model contract."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

try:
    from .models import EvidenceEvent, RelationSpec, RelationState, RuleSpec, RuleState
except ImportError:  # validate_skill.py executes this module as a source projection
    from core.models import EvidenceEvent, RelationSpec, RelationState, RuleSpec, RuleState


BASE = "https://github.com/future3317/scientific-performance-engineering/"


def canonical_serialized_fields(model: type[Any]) -> tuple[str, ...]:
    """Return the fields that belong to the canonical wire projection."""
    names = [item.name for item in fields(model)]
    if model is RuleSpec:
        names.remove("relations")
    return tuple(names)


def canonical_required_fields(model: type[Any]) -> tuple[str, ...]:
    """Return canonical required fields, excluding constructor-only projections."""
    return tuple(
        item.name
        for item in fields(model)
        if item.name in canonical_serialized_fields(model)
        and item.default is MISSING
        and item.default_factory is MISSING
    )


def _required(model: type[Any]) -> list[str]:
    """Derive required top-level fields from dataclass defaults."""
    return list(canonical_required_fields(model))


def _object(model: type[Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Build a strict object schema from the model's canonical field set."""
    model_fields = set(canonical_serialized_fields(model))
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
            **_object(RuleSpec, {"rule_id": {"type": "string", "pattern": "^[A-Z0-9-]+$"}, "version": {"type": "integer", "minimum": 1}, "parent": {"type": ["object", "null"]}, "applicability": {"type": "object", "minProperties": 1}, "intervention": {"type": "object", "minProperties": 1}, "expected_mechanism": {"type": "string", "minLength": 1}, "evidence_requirements": {"type": "array", "minItems": 1, "items": {"type": "string"}}, "scientific_invariants": {"type": "array", "items": {"type": "string"}}, "abstain_conditions": {"type": "object"}, "runtime_cost": {"type": "object"}, "provenance_policy": {"type": "object"}, "severity": {"enum": ["P0", "P1", "P2", "P3", "P4"]}, "domain": {"type": "string"}, "text": {"type": "string"}}),
        },
        "evidence_event.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "evidence_event.schema.json", "title": "EvidenceEvent", "type": "object",
            **_object(EvidenceEvent, {"schema_version": {"const": 2}, "event_id": {"type": "string", "minLength": 1}, "context": {"type": "object"}, "assignment": {"type": "object", "additionalProperties": False, "required": ["interventions", "propensity", "design_id"], "properties": {"interventions": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "integer", "enum": [0, 1]}}, "propensity": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "design_id": {"type": "string", "minLength": 1}}}, "outcome_vector": {"type": "object"}, "scientific_gates": {"type": "object"}, "artifacts": {"type": "object"}, "versions": {"type": "object"}, "source_id": {"type": "string", "minLength": 1}, "independence_group": {"type": "string", "minLength": 1}, "timestamp": {"type": "string", "minLength": 1}, "evidence_stream": {"enum": ["representative", "adversarial"]}, "evidence_role": {"enum": ["synthesis", "promotion_representative", "adversarial", "validation"]}, "query_id": {"type": "string", "minLength": 1}, "trust_zone": {"type": "string"}, "attacker_controlled_fields": {"type": "array", "items": {"type": "string"}}}),
        },
        "relation_spec.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "relation_spec.schema.json", "title": "RelationSpec", "type": "object",
            **_object(RelationSpec, {"relation_id": {"type": "string", "minLength": 1}, "version": {"type": "integer", "minimum": 1}, "parent": {"type": ["object", "null"]}, "endpoints": {"type": "object", "additionalProperties": False, "required": ["left", "right"], "properties": {"left": {"type": "string", "minLength": 1}, "right": {"type": "string", "minLength": 1}}}, "orientation": {"enum": ["symmetric", "left_to_right", "right_to_left"]}, "kind": {"enum": ["synergy", "antagonism", "independence", "prerequisite", "redundancy", "semantic_conflict", "context_dependent_interaction"]}, "applicability": {"type": "object", "minProperties": 1}, "contrast_definition": {"type": "object", "minProperties": 1}, "practical_margin": {"type": "number", "minimum": 0}, "scientific_invariants": {"type": "array", "items": {"type": "string"}}, "provenance_policy": {"type": "object"}}),
        },
        "relation_state.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "relation_state.schema.json", "title": "RelationState", "type": "object",
            **_object(RelationState, {"relation_id": {"type": "string", "minLength": 1}, "version": {"type": "integer", "minimum": 1}, "estimate": {"type": "number"}, "confidence_sequence": {"type": "object"}, "contrast_bounds": {"type": "object"}, "semantic_certificate": {"type": "object"}, "status": {"enum": ["candidate", "canonical", "retired"]}, "drift_state": {"enum": ["stable", "suspected_drift", "stale", "revalidating"]}, "counterexample_count": {"type": "integer", "minimum": 0}, "last_confirmed": {"type": ["string", "null"]}}),
        },
        "rule_state.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "rule_state.schema.json", "title": "RuleState", "type": "object",
            **_object(RuleState, {"rule_id": {"type": "string"}, "version": {"type": "integer", "minimum": 1}, "status": {"enum": ["candidate", "canonical", "retired"]}, "drift_state": {"enum": ["stable", "suspected_drift", "stale", "revalidating"]}, "effect": {"type": "object"}, "confidence_sequence": {"type": "object"}, "applicability_calibration": {"type": "object"}, "retrieval_utility": {"type": "number"}, "override_rate": {"type": "number", "minimum": 0, "maximum": 1}, "provenance_diversity": {"type": "integer", "minimum": 0}, "last_updated": {"type": "string"}}),
        },
    }
