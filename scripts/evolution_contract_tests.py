#!/usr/bin/env python3
"""Behavioral fixtures for candidate-rule promotion and registry checks."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_evolution.py"
    spec = importlib.util.spec_from_file_location("validate_evolution", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_card(status: str = "candidate") -> dict:
    return {
        "schema_version": 1,
        "rule_id": "PERF-SYNC-004",
        "status": status,
        "severity": "P2",
        "domain": "runtime",
        "trigger": {"all": ["low_gpu_duty_cycle", "scalar_sync_evidence"]},
        "requires_evidence": ["profiler_or_nsys", "synchronization_census"],
        "rule": {"text": "Audit scalar synchronization before increasing input-pipeline parallelism when loader wait is small."},
        "do_not_apply_when": ["loader_wait_is_material", "cpu_preprocessing_dominates"],
        "risk": "low",
        "source_cases": ["EXP-2026-08-0017"],
        "validated_cases": [],
        "regression_cases": [],
        "conflicts_with": [],
        "supersedes": [],
        "last_verified": {"pytorch": "2.7.1", "date": "2026-08-14"},
        "owner": "runtime",
        "promotion": {"replay_status": "pending", "human_review": False},
    }


def main() -> None:
    validator = load_validator()
    schema = validator.load_schema(ROOT / "assets" / "rule_candidate.schema.json")
    candidate = valid_card()
    assert validator.validate_rule(candidate, schema) == []

    canonical = copy.deepcopy(candidate)
    canonical["status"] = "canonical"
    assert any("replay_status" in error for error in validator.validate_rule(canonical, schema))

    canonical["validated_cases"] = ["REG-SYNC-001"]
    canonical["regression_cases"] = ["REG-SYNC-001"]
    canonical["promotion"] = {"replay_status": "passed", "replay_evidence": "replay.json", "human_review": True, "reviewed_by": "human", "reviewed_at": "2026-08-14"}
    assert validator.validate_rule(canonical, schema) == []

    regression_schema = validator.load_schema(ROOT / "assets" / "rule_regression_case.schema.json")
    regression = {
        "schema_version": 1, "case_id": "REG-SYNC-001", "rule_id": "PERF-SYNC-004", "kind": "positive", "status": "pass",
        "scope": {"requires": ["scalar_sync_evidence"], "excludes": []}, "expected": "audit sync", "observed": "matched", "evidence": "replay.json",
    }
    assert validator.validate_regression_case(regression, regression_schema) == []

    registry = {"schema_version": 1, "rules": [{"rule_id": "PERF-SYNC-004", "path": "rules/PERF-SYNC-004.json", "status": "canonical"}]}
    assert validator.validate_registry(registry, {"PERF-SYNC-004": canonical}) == []
    duplicate = copy.deepcopy(registry)
    duplicate["rules"].append(duplicate["rules"][0])
    assert any("duplicate" in error for error in validator.validate_registry(duplicate, {"PERF-SYNC-004": canonical}))

    print("evolution contract fixtures: ok")


if __name__ == "__main__":
    main()
