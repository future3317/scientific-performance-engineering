"""Harness for tensor-only dynamic-shape specialization."""

from pathlib import Path

from benchmark.tasks._compile_benchmark import configure

_API = configure(
    Path(__file__).resolve().parent,
    {
        "num_samples": 128,
        "in_dim": 16,
        "logical_steps": 128,
        "measurement_iterations": 128,
        "graph_size": 128,
        "batch_sizes": [8, 10, 12, 14, 16],
        "dynamic_shape_rate": 0.3,
        "compile_threads": 2,
        "primary_scope": "full_schedule",
    },
)
globals().update(_API)
