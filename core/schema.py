"""JSON Schema projections generated from the canonical model contract."""

from __future__ import annotations

from typing import Any


BASE = "https://github.com/future3317/scientific-performance-engineering/"


def schemas() -> dict[str, dict[str, Any]]:
    return {
        "rule_spec.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "rule_spec.schema.json", "title": "RuleSpec", "type": "object",
            "required": ["rule_id", "version", "applicability", "intervention", "expected_mechanism", "evidence_requirements", "scientific_invariants", "abstain_conditions", "relations", "runtime_cost", "provenance_policy"],
            "properties": {"rule_id": {"type": "string", "pattern": "^[A-Z0-9-]+$"}, "version": {"type": "integer", "minimum": 1}, "parent": {"type": ["string", "null"]}, "applicability": {"type": "object"}, "intervention": {"type": "object"}, "expected_mechanism": {"type": "string", "minLength": 1}, "evidence_requirements": {"type": "array", "minItems": 1, "items": {"type": "string"}}, "scientific_invariants": {"type": "array", "items": {"type": "string"}}, "abstain_conditions": {"type": "object"}, "relations": {"type": "object"}, "runtime_cost": {"type": "object"}, "provenance_policy": {"type": "object"}, "severity": {"enum": ["P0", "P1", "P2", "P3", "P4"]}, "domain": {"type": "string"}, "text": {"type": "string"}},
        },
        "evidence_event.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "evidence_event.schema.json", "title": "EvidenceEvent", "type": "object",
            "required": ["event_id", "context", "rule_id", "rule_version", "assignment", "outcome_vector", "scientific_gates", "artifacts", "versions", "source_id", "independence_group", "timestamp"],
            "properties": {"event_id": {"type": "string", "minLength": 1}, "context": {"type": "object"}, "rule_id": {"type": "string"}, "rule_version": {"type": "integer", "minimum": 1}, "assignment": {"type": "object", "required": ["arm"], "properties": {"arm": {"enum": ["on", "off"]}}}, "outcome_vector": {"type": "object"}, "scientific_gates": {"type": "object"}, "artifacts": {"type": "object"}, "versions": {"type": "object"}, "source_id": {"type": "string"}, "independence_group": {"type": "string"}, "timestamp": {"type": "string"}, "trust_zone": {"type": "string"}, "attacker_controlled_fields": {"type": "array"}},
        },
        "rule_state.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": BASE + "rule_state.schema.json", "title": "RuleState", "type": "object",
            "required": ["rule_id", "version", "status", "drift_state", "effect", "confidence_sequence", "applicability_calibration", "retrieval_utility", "override_rate", "provenance_diversity", "last_updated"],
            "properties": {"rule_id": {"type": "string"}, "version": {"type": "integer", "minimum": 1}, "status": {"enum": ["candidate", "canonical", "retired"]}, "drift_state": {"enum": ["stable", "suspected_drift", "stale", "revalidating"]}, "effect": {"type": "object"}, "confidence_sequence": {"type": "object"}, "applicability_calibration": {"type": "object"}, "retrieval_utility": {"type": "number"}, "override_rate": {"type": "number", "minimum": 0, "maximum": 1}, "provenance_diversity": {"type": "integer", "minimum": 0}, "last_updated": {"type": "string"}},
        },
    }

