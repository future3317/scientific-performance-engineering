from __future__ import annotations

from experiments.external_real.scienceworld_oracle import (
    oracle_gate,
    select_oracle_cases,
)


def test_oracle_case_selection_is_deterministic_and_nonempty() -> None:
    manifest = {
        "train": [("boil", 3), ("boil", 1)],
        "dev": [("grow-plant", 7), ("grow-plant", 2)],
    }
    assert select_oracle_cases(manifest, limit=3) == [
        ("boil", 1, "train"),
        ("boil", 3, "train"),
        ("grow-plant", 2, "dev"),
    ]


def test_oracle_gate_requires_high_score_and_completion() -> None:
    assert oracle_gate([{"completed": True, "score": 100.0}]) == (True, [])
    passed, reasons = oracle_gate([{"completed": False, "score": 100.0}])
    assert passed is False
    assert reasons == ["one or more gold paths did not complete"]
