"""Hidden package hook; benchmark.py owns the executable scientific gates."""

from __future__ import annotations


def no_op_gate(solution, fixtures: dict) -> tuple[bool, dict]:
    return True, {}
