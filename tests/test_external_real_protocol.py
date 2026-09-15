from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.external_real.freeze import (
    assert_decision_before_scoring,
    freeze_decisions,
)
from experiments.external_real.models import FALLBACK_TO_BASE_AGENT, PolicyDecision
from experiments.external_real.protocol import (
    ProtocolManifest,
    assign_split,
    assert_information_flow,
    validate_acre_r_request,
)


def test_task_split_is_deterministic_and_uses_fixed_phase_weights() -> None:
    first = assign_split("task-001", "scienceworld-v1", (4, 2, 3, 1))
    second = assign_split("task-001", "scienceworld-v1", (4, 2, 3, 1))
    changed = assign_split("task-001", "scienceworld-v2", (4, 2, 3, 1))

    assert first == second
    assert first in {"construction", "verification", "synthesis", "calibration"}
    assert changed in {"construction", "verification", "synthesis", "calibration"}


def test_protocol_manifest_rejects_missing_identity() -> None:
    with pytest.raises(ValueError, match="benchmark_revision"):
        ProtocolManifest.from_dict({"benchmark": "scienceworld"})


def test_policy_input_rejects_post_execution_fields() -> None:
    with pytest.raises(ValueError, match="skill_score"):
        assert_information_flow({"category": "navigation", "skill_score": 1.0})


def test_decisions_are_hashed_and_frozen_before_scoring(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc).isoformat()
    decision = PolicyDecision.create(
        task_id="task-001",
        policy_name="acre",
        deploy=True,
        reason="support_certificate",
        decision_timestamp=timestamp,
    )
    path = tmp_path / "test_policy_decisions.jsonl"

    digest = freeze_decisions(path, [decision])
    raw = path.read_bytes()

    assert digest == hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    assert payload["decision_hash"]
    assert_decision_before_scoring(payload["decision_timestamp"], "2026-09-15T12:00:01+00:00")


def test_non_deployed_policy_uses_explicit_base_agent_fallback() -> None:
    decision = PolicyDecision.create(
        task_id="task-002",
        policy_name="acre",
        deploy=False,
        reason=FALLBACK_TO_BASE_AGENT,
        decision_timestamp="2026-09-15T12:00:00+00:00",
    )

    assert decision.reason == FALLBACK_TO_BASE_AGENT


def test_context_adapter_contract_is_explicit() -> None:
    from experiments.external_real.context.adapter import ContextAdapter

    adapter = ContextAdapter(mode="native_structured", public_fields=("task_family",))
    task = type("Task", (), {"native_payload": {"task_family": "navigation"}})()

    assert adapter.extract(task) == {"task_family": "navigation"}


def test_raw_context_adapter_requires_frozen_parser_metadata() -> None:
    from experiments.external_real.context.adapter import ContextAdapter

    with pytest.raises(ValueError, match="frozen parser metadata"):
        ContextAdapter(mode="raw_task_to_typed", public_fields=("task_family",), parser=lambda _: {})


def test_acre_r_is_deferred_until_verified_skill_and_low_coverage() -> None:
    with pytest.raises(ValueError, match="verified skill"):
        validate_acre_r_request(verified_skill=False, support_evidence_count=10, strict_coverage=0.05, alpha=0.10)

    validate_acre_r_request(verified_skill=True, support_evidence_count=10, strict_coverage=0.05, alpha=0.10)


def test_decision_after_scoring_is_rejected() -> None:
    with pytest.raises(ValueError, match="before scoring"):
        assert_decision_before_scoring("2026-09-15T12:00:01+00:00", "2026-09-15T12:00:00+00:00")
