from __future__ import annotations

import json

import pytest

from scripts.audit_public_skillsbench import (
    audit_public_snapshot,
    build_task_split,
    join_registry_rows,
)


def test_public_aggregate_without_category_metadata_is_not_used_as_external_evidence(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps([{
        "task": "t1", "modelShort": "m1", "harness": "h1", "family": "f1",
        "condition": "With Skills", "score": 100.0, "trials": 3,
    }]))
    report = audit_public_snapshot(path)
    assert report["status"] == "metadata_insufficient"
    assert report["provider_calls"] == 0
    assert report["claim_boundary"] == "not external validity evidence"


def _manifest(*rows):
    return [{"title": task_id, "category": category, "difficulty": difficulty, "excluded": excluded}
            for task_id, category, difficulty, excluded in rows]


def _result(task_id):
    return {
        "task": task_id,
        "model": "Model A",
        "modelShort": "A",
        "harness": "Harness",
        "condition": "With Skills",
        "score": 1.0,
        "trials": 3,
    }


def test_join_requires_one_active_metadata_row_and_non_null_features() -> None:
    joined = join_registry_rows([_result("t1")], _manifest(("t1", "science", "easy", False)))
    assert joined[0]["task_id"] == "t1"
    assert joined[0]["category"] == "science"
    assert joined[0]["difficulty"] == "easy"


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(("t1", "science", "easy", False), ("t1", "science", "hard", False)), "duplicate"),
        (_manifest(("t2", "science", "easy", False)), "missing active task metadata"),
        (_manifest(("t1", None, "easy", False)), "category"),
        (_manifest(("t1", "science", None, False)), "difficulty"),
    ],
)
def test_join_rejects_invalid_task_metadata(manifest, message) -> None:
    with pytest.raises(ValueError, match=message):
        join_registry_rows([_result("t1")], manifest)


def test_task_split_keeps_all_rows_for_a_task_in_one_phase() -> None:
    rows = [_result("t1"), _result("t1"), _result("t2"), _result("t3")]
    split = build_task_split(rows, split_seed="skillsbench-v1.1")
    phases_by_task = {}
    for phase, members in split.items():
        for row in members:
            phases_by_task.setdefault(row["task_id"], set()).add(phase)
    assert all(len(phases) == 1 for phases in phases_by_task.values())


def test_task_split_is_deterministic() -> None:
    rows = [_result("t1"), _result("t2"), _result("t3"), _result("t4")]
    assert build_task_split(rows, split_seed="skillsbench-v1.1") == build_task_split(rows, split_seed="skillsbench-v1.1")
