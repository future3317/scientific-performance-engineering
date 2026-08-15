#!/usr/bin/env python3
"""Hidden verifier gates for CORE-REPEATED-BACKBONE-02.

Correctness is the fp64 live-recomputed reference inside benchmark.py, which
already tests both the fixed and changing regimes. This module is a placeholder
so the task package structure is complete.
"""

from __future__ import annotations


def no_op_gate(solution, fixtures: dict) -> tuple[bool, dict]:
    return True, {}
