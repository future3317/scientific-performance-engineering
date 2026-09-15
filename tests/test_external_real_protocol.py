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
from experiments.external_real.models import PolicyDecision
from experiments.external_real.protocol import (
    ProtocolManifest,
    assign_split,
    assert_information_flow,
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


def test_decision_after_scoring_is_rejected() -> None:
    with pytest.raises(ValueError, match="before scoring"):
        assert_decision_before_scoring("2026-09-15T12:00:01+00:00", "2026-09-15T12:00:00+00:00")
