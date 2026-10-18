from __future__ import annotations

import json
import subprocess

import pytest

import scripts.run_public_acre_suite as public_acre_suite
from scripts.run_public_acre_suite import _support_key


def test_support_key_uses_public_task_type_fields_not_scoring_outcomes() -> None:
    train = {"metadata": {"benchmark": "scienceworld", "task_id": "6", "key_actions": ["hidden"]}}
    test = {"metadata": {"benchmark": "scienceworld", "task_id": "6", "key_actions": ["different-hidden"]}}
    assert _support_key("scienceworld", train) == _support_key("scienceworld", test)


def test_support_key_separates_public_mcp_server_sets() -> None:
    left = {"metadata": {"combination_type": "single_server", "servers": ["NixOS"]}}
    right = {"metadata": {"combination_type": "single_server", "servers": ["Unit Converter"]}}
    assert _support_key("mcp_bench", left) != _support_key("mcp_bench", right)


def test_skillgen_revision_check_requires_expected_checkout_head(tmp_path) -> None:
    root = tmp_path / "skillgen"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert public_acre_suite._verify_skillgen_revision(root, head) == head
    with pytest.raises(ValueError, match="must resolve to a commit"):
        public_acre_suite._verify_skillgen_revision(root, "missing-revision")
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "new head"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="revision mismatch"):
        public_acre_suite._verify_skillgen_revision(root, head)


@pytest.mark.parametrize("apply_value", [{"yes": True}, [True], "true", 1, None])
def test_router_rejects_non_boolean_apply(monkeypatch, apply_value) -> None:
    monkeypatch.setattr(
        public_acre_suite,
        "_chat",
        lambda *_args, **_kwargs: ('{"apply":' + json.dumps(apply_value) + '}', {}),
    )

    applied, reason, _ = public_acre_suite._router_decision(
        object(), "model", {}, "skill"
    )

    assert applied is False
    assert reason == "router_invalid_apply"


@pytest.mark.parametrize("apply_value", [True, False])
def test_router_accepts_boolean_apply(monkeypatch, apply_value) -> None:
    monkeypatch.setattr(
        public_acre_suite,
        "_chat",
        lambda *_args, **_kwargs: ('{"apply":' + str(apply_value).lower() + '}', {}),
    )

    applied, _, _ = public_acre_suite._router_decision(
        object(), "model", {}, "skill"
    )

    assert applied is apply_value
