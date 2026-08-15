#!/usr/bin/env python3
"""Task-specific scientific gates for CORE-SCALAR-SYNC-01.

This task has no domain-specific scientific gates; correctness is handled by
benchmark.run_correctness (fp64 live-recomputed reference). The harness still
invokes run_scientific_gates, which returns an empty gate map.
"""

from __future__ import annotations

from benchmark.harness import scientific_gates as gates  # noqa: F401
