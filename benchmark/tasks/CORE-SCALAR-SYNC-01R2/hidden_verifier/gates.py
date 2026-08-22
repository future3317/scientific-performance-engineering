#!/usr/bin/env python3
"""Hidden verifier gates for CORE-SCALAR-SYNC-01R2.

Currently the only correctness/scientific check is the fp64 live-recomputed
reference inside benchmark.py. This module is a placeholder so the task
package structure is complete; additional probes may be added here without
changing the workspace contract.
"""

from __future__ import annotations


def no_op_gate(solution, fixtures: dict) -> tuple[bool, dict]:
    return True, {}
