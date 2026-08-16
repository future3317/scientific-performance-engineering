#!/usr/bin/env python3
"""Focused contract tests for the single promotion boundary and ledger."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.governance import apply_promotion, evaluate_candidate
from benchmark.harness.evolution_ledger import EvolutionDecisionLedger


def candidate(severity: str) -> dict:
    return {
        "rule_id": "PERF-TEST-001",
        "version": 1,
        "severity": severity,
        "trigger": {"all": ["evidence"]},
        "intervention": {"action": "measure"},
        "expected_mechanism": "measured bottleneck",
        "requires_evidence": ["trace"],
        "scientific_invariants": ["objective_unchanged"],
        "do_not_apply_when": [],
        "runtime_cost": {"tokens": 10, "expected_utility": 0.1},
        "provenance_policy": {"required": True},
    }


def relation_candidate() -> dict:
    return {
        "relation_id": "REL-TEST-001",
        "version": 1,
        "parent": None,
        "endpoints": {"left": "RULE-A", "right": "RULE-B"},
        "orientation": "symmetric",
        "kind": "synergy",
        "applicability": {"equals": {"workload": "graph"}},
        "contrast_definition": {"quantity": "gamma"},
        "practical_margin": 0.05,
        "scientific_invariants": [],
        "provenance_policy": {"required": True},
    }


def main() -> None:
    passed = {"outcome": "passed", "result_digest": "d" * 64, "result": {"mean_effect": 0.2, "utility_effect_lcb": 0.1, "utility_effect_ucb": 0.3, "promotion_probability_lower_bound": 0.9, "p_min": 0.8}, "promotion_record": {"representative_groups": ["g1", "g2"], "promotion_case_ids": ["CASE-1"], "heldout_regression_digest": "h", "poison_gate": {"passed": True}, "promotion_probability_lcb": 0.9, "utility_effect_cs": {"lcb": 0.1, "ucb": 0.3}, "replay_manifest_digest": "m"}}
    assert evaluate_candidate(candidate("P1"), passed).status == "review_required"
    assert evaluate_candidate(candidate("P2"), passed).allowed
    assert not evaluate_candidate(candidate("P2"), {"outcome": "failed"}).allowed

    ledger = EvolutionDecisionLedger()
    ledger.record("R", 1, "x", "candidate")
    ledger.record("R", 1, "x", "evaluated")
    ledger.record("R", 1, "x", "promoted")
    assert ledger.precision() == 1.0
    try:
        ledger.record("R", 1, "x", "candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("ledger accepted a backward transition")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "registry").mkdir()
        (root / "registry" / "rules.json").write_text(json.dumps({"schema_version": 1, "rules": []}), encoding="utf-8")
        validation = {"promotion_case_ids": ["CASE-1"], "heldout_regression_cases": [{"case_id": "HELDOUT-1", "executed": True, "execution_source": "verifier"}], "poison_probe_cases": [{"case_id": "POISON-1", "executed": True, "execution_source": "environment", "accepted": False}]}
        validation_path = root / "evolution" / "validation.json"
        validation_path.parent.mkdir(parents=True)
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        passed["promotion_record"].update({"validation_artifact_path": "evolution/validation.json", "validation_artifact_digest": hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        decision = apply_promotion(root, candidate("P2"), passed, replay_path="evolution/replay.json")
        assert decision.allowed
        rule_digest = hashlib.sha256(b"PERF-TEST-001").hexdigest()
        card = json.loads((root / "rules" / rule_digest / "v0001.json").read_text(encoding="utf-8"))
        promotion = json.loads((root / "evolution" / "promotions" / rule_digest / "v0001.json").read_text(encoding="utf-8"))
        assert card["rule_id"] == "PERF-TEST-001" and promotion["mode"] == "bounded-auto"
        relation_passed = dict(passed, evidence_type="factorial_contrast")
        for endpoint in ("RULE-A", "RULE-B"):
            endpoint_dir = root / "rules" / hashlib.sha256(endpoint.encode()).hexdigest()
            endpoint_dir.mkdir(parents=True)
            endpoint_dir.joinpath("v0001.json").write_text(json.dumps({**candidate("P2"), "rule_id": endpoint}), encoding="utf-8")
            endpoint_dir.joinpath("v0001.state.json").write_text(json.dumps({"rule_id": endpoint, "version": 1, "status": "canonical"}), encoding="utf-8")
        relation_passed["relation_evidence_certificate"] = {"contrast_cs": {"gamma": {"lcb": 0.1, "ucb": 0.3}}, "alpha_budget": 0.05, "look_schedule": [8, 16], "scientific_arm_gates": {"00": True, "01": True, "10": True, "11": True}, "applicability_provenance": {"source": "test"}, "endpoint_versions": {"RULE-A": 1, "RULE-B": 1}}
        relation_decision = apply_promotion(root, relation_candidate(), relation_passed, replay_path="evolution/relation.json")
        assert relation_decision.allowed and relation_decision.subject_type == "relation"
        relation_digest = hashlib.sha256(b"REL-TEST-001").hexdigest()
        relation_card = json.loads((root / "relations" / relation_digest / "v0001.json").read_text(encoding="utf-8"))
        relation_registry = json.loads((root / "registry" / "relations.json").read_text(encoding="utf-8"))
        assert relation_card["relation_id"] == "REL-TEST-001"
        assert relation_registry["relations"][0]["relation_id"] == "REL-TEST-001"

    print("governance tests: ok")


if __name__ == "__main__":
    main()
