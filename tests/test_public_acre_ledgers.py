from __future__ import annotations

import pytest

from benchmark.external.contracts import DecisionRecord
from benchmark.external.decision_ledger import DecisionLedger
from benchmark.external.skill_versions import SkillVersionStore


def _decision(task_index: int = 0) -> DecisionRecord:
    return DecisionRecord(
        run_id="run-1",
        condition="D",
        task_id=f"task-{task_index}",
        task_index=task_index,
        stage="acquisition",
        policy="acre",
        decision_id=f"decision-{task_index}",
        reuse=True,
        score=0.7,
        uncertainty=0.1,
        reason="certificate",
        certificate_id="cert-1",
        skill_version="skill-v0",
        public_context_digest=f"ctx-{task_index}",
        available_evidence_ids=(),
    )


def test_skill_version_content_is_immutable(tmp_path) -> None:
    store = SkillVersionStore(tmp_path / "skills.jsonl")
    version = store.create(
        parent_id=None,
        task_index=0,
        content="fixed skill body",
        evidence_ids=("ev-1",),
        certificate={"status": "certified"},
        provenance={"commit": "abc"},
    )
    assert store.get(version.version_id).content == "fixed skill body"
    with pytest.raises(ValueError, match="immutable"):
        store.create(
            parent_id=None,
            task_index=0,
            content="different body",
            evidence_ids=("ev-1",),
            certificate={"status": "certified"},
            provenance={"commit": "abc"},
            version_id=version.version_id,
        )


def test_skill_version_parent_and_digest_are_recorded(tmp_path) -> None:
    store = SkillVersionStore(tmp_path / "skills.jsonl")
    root = store.create(parent_id=None, task_index=0, content="v0", evidence_ids=(), certificate={}, provenance={})
    child = store.create(parent_id=root.version_id, task_index=1, content="v1", evidence_ids=("ev-2",), certificate={}, provenance={})
    assert child.parent_id == root.version_id
    assert child.content_sha256 != root.content_sha256


def test_decision_must_precede_outcome(tmp_path) -> None:
    ledger = DecisionLedger(tmp_path / "decisions.jsonl")
    decision = _decision()
    with pytest.raises(ValueError, match="decision"):
        ledger.append_outcome(decision.decision_id, {"official_score": 1.0})
    ledger.append(decision)
    ledger.append_outcome(decision.decision_id, {"official_score": 1.0, "correctness_pass": True})
    rows = ledger.rows()
    assert [row["event"] for row in rows] == ["decision", "outcome"]


def test_decision_ledger_rejects_current_and_future_outcomes(tmp_path) -> None:
    ledger = DecisionLedger(tmp_path / "decisions.jsonl")
    ledger.append(_decision(2))
    with pytest.raises(ValueError, match="future"):
        ledger.append_outcome("decision-2", {"task_index": 3, "official_score": 1.0})
    ledger.append_outcome("decision-2", {"task_index": 2, "official_score": 1.0})
