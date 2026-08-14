#!/usr/bin/env python3
"""Behavioral fixtures for candidate-rule promotion and registry checks."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
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
        "admission_cases": ["REG-SYNC-ADMISSION-001"],
        "regression_cases": [],
        "conflicts_with": [],
        "supersedes": [],
        "last_verified": {"pytorch": "2.7.1", "date": "2026-08-14"},
        "owner": "runtime",
        "collector_confidence": "medium",
        "confidence": {
            "method": "beta-binomial",
            "prior_alpha": 1,
            "prior_beta": 1,
            "successes": 0,
            "failures": 0,
            "p_min": 0.8,
            "delta": 0.05,
            "posterior_probability": 0.0,
            "effective_samples": 0.0,
        },
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

    canonical["admission_cases"] = ["REG-SYNC-ADMISSION-001"]
    canonical["regression_cases"] = ["REG-SYNC-001"]
    canonical["promotion"] = {
        "replay_status": "passed",
        "replay_manifest": "replay.json",
        "human_review": True,
        "review_commit": "a" * 40,
        "reviewer": "human",
        "reviewed_at": "2026-08-14T12:00:00Z",
        "review_diff_hash": "b" * 64,
    }
    canonical["confidence"]["successes"] = 8
    canonical["confidence"]["failures"] = 2
    canonical["confidence"]["posterior_probability"] = 0.99
    assert any("replay" in error for error in validator.validate_rule(canonical, schema))

    # A canonical card must carry a structured, auditable review and replay record.
    canonical["promotion"]["replay_manifest"] = {
        "path": "replay.json",
        "command": "python scripts/run_rule_replay.py input.json replay.json",
        "case_bundle_path": "input.json",
        "case_bundle_sha256": "c" * 64,
        "harness_revision": "d" * 40,
        "result_digest": "e" * 64,
        "outcome": "passed",
    }
    canonical["admission_cases"] = ["REG-SYNC-ADMISSION-001"]
    canonical["regression_cases"] = ["REG-SYNC-REGRESSION-001"]
    canonical["confidence"]["posterior_probability"] = 0.99
    assert validator.validate_rule(canonical, schema) == []

    regression_schema = validator.load_schema(ROOT / "assets" / "rule_regression_case.schema.json")
    regression = {
        "schema_version": 1, "case_id": "REG-SYNC-REGRESSION-001", "rule_id": "PERF-SYNC-004", "kind": "positive", "status": "pass",
        "scope": {"requires": ["scalar_sync_evidence"], "excludes": []}, "expected": "audit sync", "observed": "matched", "evidence": "replay.json",
        "lineage": {"derived_from_experience_ids": ["EXP-2026-08-0099"], "repository_revision": "f" * 40, "task_family": "held-out-runtime"},
    }
    assert validator.validate_regression_case(regression, regression_schema) == []

    registry = {"schema_version": 1, "rules": [{"rule_id": "PERF-SYNC-004", "path": "rules/PERF-SYNC-004.json", "status": "canonical"}]}
    assert validator.validate_registry(registry, {"PERF-SYNC-004": canonical}) == []
    duplicate = copy.deepcopy(registry)
    duplicate["rules"].append(duplicate["rules"][0])
    assert any("duplicate" in error for error in validator.validate_registry(duplicate, {"PERF-SYNC-004": canonical}))

    # Admission and regression evidence must be distinct and linked to real cases.
    errors = validator.validate_card_links(
        canonical,
        {"EXP-2026-08-0017": {"status": "case"}},
        {
            "REG-SYNC-ADMISSION-001": {"status": "pass", "lineage": {"derived_from_experience_ids": ["EXP-2026-08-0017"]}},
            "REG-SYNC-REGRESSION-001": regression,
        },
    )
    assert any("leak" in error or "disjoint" in error for error in errors)

    # Graph validation rejects dangling edges and cycles.
    graph_errors = validator.validate_rule_graph({
        "A": {"status": "canonical", "supersedes": ["B"], "conflicts_with": []},
        "B": {"status": "canonical", "supersedes": ["A"], "conflicts_with": ["A"]},
    })
    assert any("cycle" in error for error in graph_errors)
    assert any("conflict" in error for error in graph_errors)

    print("evolution contract fixtures: ok")


if __name__ == "__main__":
    main()
