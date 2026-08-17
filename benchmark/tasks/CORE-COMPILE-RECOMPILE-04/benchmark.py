"""Harness for the graph-break plus cold-shape-recompile anchor."""

from pathlib import Path

from benchmark.tasks._compile_benchmark import configure

_API = configure(
    Path(__file__).resolve().parent,
    {
        "num_samples": 512,
        "in_dim": 64,
        "hidden_dim": 256,
        "num_blocks": 4,
        "batch_sizes": [16, 24, 32, 40, 48, 56, 64, 72],
        "compile_threads": 2,
    },
)
globals().update(_API)
