#!/usr/bin/env python3
"""Focused tests for typed predicates and budgeted retrieval."""

from __future__ import annotations

import sys
from dataclasses import MISSING, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models import EvidenceEvent, RuleSpec, RuleState, TaskContext
from core.predicates import match_predicate
from core.retriever import retrieve_candidates, select_rules
from core.schema import schemas


def main() -> None:
    context = TaskContext("runtime", {"loader_wait": 0.03}, {"gpu": {"duty_cycle": 0.2}}, {"pytorch": "2.7"}, {"scalar_sync_count": 4}, 32)
    assert match_predicate({"compare": {"hardware.gpu.duty_cycle": {"lt": 0.45}}}, context.to_dict())
    specs = [
        RuleSpec("PERF-SYNC-004", 1, None, {"compare": {"hardware.gpu.duty_cycle": {"lt": 0.45}}}, {"action": "audit"}, "launch gaps", ["scalar_sync_count"], [], {}, {"conflicts": []}, {"tokens": 10, "expected_utility": 0.8}, {"required": True}),
        RuleSpec("PERF-DATA-001", 1, None, {"compare": {"workload.loader_wait": {"lt": 0.1}}}, {"action": "prefetch"}, "host wait", ["loader_wait"], [], {}, {"conflicts": ["PERF-SYNC-004"]}, {"tokens": 10, "expected_utility": 0.6}, {"required": True}),
    ]
    candidates = retrieve_candidates(specs, context)
    assert len(candidates) == 2
    selected = select_rules(candidates, context)
    assert selected == [{"rule_id": "PERF-SYNC-004", "version": 1, "token_cost": 10, "marginal_gain": 1.8}]

    # Relation requirements contribute to coverage just like evidence
    # requirements.  Otherwise a redundant rule can win the next slot merely
    # because ``requires`` was omitted from the covered set.
    relation_specs = [
        RuleSpec("RULE-A", 1, None, {"always": True}, {"action": "a"}, "a", ["e1"], [], {}, {"requires": ["r1"]}, {"tokens": 1, "expected_utility": 0.0}, {"required": True}),
        RuleSpec("RULE-B", 1, None, {"always": True}, {"action": "b"}, "b", ["r1"], [], {}, {}, {"tokens": 1, "expected_utility": 0.0}, {"required": True}),
        RuleSpec("RULE-C", 1, None, {"always": True}, {"action": "c"}, "c", ["e2"], [], {}, {}, {"tokens": 1, "expected_utility": 0.0}, {"required": True}),
    ]
    relation_selected = select_rules(relation_specs, TaskContext("runtime", {}, {}, {}, {}, 2))
    assert [item["rule_id"] for item in relation_selected] == ["RULE-A", "RULE-C"]
    assert relation_selected[0]["marginal_gain"] == 2.0
    assert relation_selected[1]["marginal_gain"] == 1.0

    # The external schema and runtime model must agree on top-level fields.
    for model, schema_name in (
        (RuleSpec, "rule_spec.schema.json"),
        (EvidenceEvent, "evidence_event.schema.json"),
        (RuleState, "rule_state.schema.json"),
    ):
        model_schema = schemas()[schema_name]
        assert model_schema["additionalProperties"] is False
        expected_required = {
            item.name for item in fields(model)
            if item.default is MISSING and item.default_factory is MISSING
        }
        assert set(model_schema["required"]) == expected_required
        assert set(model_schema["properties"]) == {item.name for item in fields(model)}

    rule_schema = schemas()["rule_spec.schema.json"]
    payload = specs[0].to_dict()
    payload["unexpected_field"] = True
    try:
        RuleSpec.from_dict(payload)
    except TypeError:
        pass
    else:
        raise AssertionError("RuleSpec.from_dict must reject unknown fields")
    print("rule engine fixtures: ok")


if __name__ == "__main__":
    main()
