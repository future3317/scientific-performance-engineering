from __future__ import annotations

from scripts.run_public_acre_suite import _support_key


def test_support_key_uses_public_task_type_fields_not_scoring_outcomes() -> None:
    train = {"metadata": {"benchmark": "scienceworld", "task_id": "6", "key_actions": ["hidden"]}}
    test = {"metadata": {"benchmark": "scienceworld", "task_id": "6", "key_actions": ["different-hidden"]}}
    assert _support_key("scienceworld", train) == _support_key("scienceworld", test)


def test_support_key_separates_public_mcp_server_sets() -> None:
    left = {"metadata": {"combination_type": "single_server", "servers": ["NixOS"]}}
    right = {"metadata": {"combination_type": "single_server", "servers": ["Unit Converter"]}}
    assert _support_key("mcp_bench", left) != _support_key("mcp_bench", right)
