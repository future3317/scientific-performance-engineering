from __future__ import annotations

import json

import pytest

from benchmark.external.skillgen_adapter import (
    build_public_context,
    load_pinned_benchmark,
    validate_paired_instance_ids,
)


def test_public_context_excludes_scoring_only_fields() -> None:
    instance = {
        "instance_id": "x",
        "metadata": {
            "benchmark": "scienceworld",
            "task_id": "1",
            "task_name": "task",
            "fold": "test",
            "task_description": "public",
            "initial_observation": "scene",
            "key_actions": ["hidden"],
            "gold_actions": ["hidden"],
            "ground_truth": "hidden",
        },
    }
    context = build_public_context(instance)
    assert context["task_name"] == "task"
    assert "key_actions" not in context
    assert "gold_actions" not in context
    assert "ground_truth" not in context


def test_load_pinned_benchmark_validates_task_ids(tmp_path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"instances": [{"instance_id": "a", "metadata": {}}]}), encoding="utf-8")
    view = load_pinned_benchmark("scienceworld", path, expected_commit="sw-commit")
    assert view.benchmark == "scienceworld"
    assert view.instances[0]["instance_id"] == "a"
    assert view.claimed_commit == "sw-commit"
    assert view.commit_verified is False


def test_paired_instance_ids_must_match() -> None:
    validate_paired_instance_ids(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="paired"):
        validate_paired_instance_ids(["a"], ["b"])
