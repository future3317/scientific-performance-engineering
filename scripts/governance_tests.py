#!/usr/bin/env python3
"""Focused contract tests for the single promotion boundary and ledger."""

from __future__ import annotations

import json
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


def main() -> None:
    passed = {"outcome": "passed", "result_digest": "d" * 64}
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
        decision = apply_promotion(root, candidate("P2"), passed, replay_path="evolution/replay.json")
        assert decision.allowed
        card = json.loads((root / "rules" / "PERF-TEST-001.json").read_text(encoding="utf-8"))
        assert card["status"] == "canonical" and card["promotion"]["mode"] == "bounded-auto"

    print("governance tests: ok")


if __name__ == "__main__":
    main()
