#!/usr/bin/env python3
"""Focused tests for typed predicates and budgeted retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models import RuleSpec, TaskContext
from core.predicates import match_predicate
from core.retriever import retrieve_candidates, select_rules


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
    assert selected == [{"rule_id": "PERF-SYNC-004", "version": 1, "token_cost": 10, "marginal_gain": 0.8}]
    print("rule engine fixtures: ok")


if __name__ == "__main__":
    main()
