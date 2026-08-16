from __future__ import annotations

import json

import pytest

from benchmark.harness.evolution_ledger import CandidateEvidenceLedger, EvolutionDecisionLedger
from core.acre.factorial import FactorialBlock, FactorialEngine


def test_factorial_decision_contrasts_share_normalized_scale() -> None:
    engine = FactorialEngine(delta=0.05, practical_margin=0.05)
    for index in range(64):
        engine.add_block(FactorialBlock(str(index), {"00": -0.2, "10": 0.2, "01": 0.1, "11": 0.6}))
    estimate = engine.estimate()
    assert all(-1.0 <= lower <= upper <= 1.0 for lower, upper in estimate.contrast_intervals.values())
    assert estimate.raw_contrasts["delta_a_given_b0"] == pytest.approx(0.4)
    assert estimate.delta_a_given_b0 == pytest.approx(0.2)


def test_candidate_evidence_membership_rejects_mutation(tmp_path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "candidate-evidence.jsonl")
    case = {"case_id": "CASE-A", "context": {"workload": {"x": 1}}, "source_id": "s", "independence_group": "g"}
    ledger.append("RULE-A:v1:action", 1, {**case, "case_path": "experience/cases/a.json"}, action_digest="action-a")
    with pytest.raises(ValueError, match="immutable"):
        ledger.append("RULE-A:v1:action", 1, {**case, "case_path": "experience/cases/b.json"}, action_digest="action-a")


def test_evolution_decision_ledger_rejects_broken_digest_chain(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    ledger = EvolutionDecisionLedger(path)
    ledger.record("RULE-A", 1, "replay-a", "candidate")
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["status"] = "promoted"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        EvolutionDecisionLedger(path)
