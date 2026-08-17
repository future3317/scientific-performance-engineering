"""Harness for tensor-only dynamic-shape specialization."""

from pathlib import Path

from benchmark.tasks._compile_benchmark import configure

_API = configure(
    Path(__file__).resolve().parent,
    {
        "num_samples": 128,
        "in_dim": 16,
        "hidden_dim": 32,
        "num_blocks": 1,
        "batch_sizes": [8, 12, 16, 20],
        "compile_threads": 2,
        "primary_scope": "full_schedule",
    },
)
globals().update(_API)
