"""Harness for the tiny short-lived compile counterexample."""

from pathlib import Path

from benchmark.tasks._compile_benchmark import configure

_API = configure(
    Path(__file__).resolve().parent,
    {
        "num_samples": 128,
        "in_dim": 16,
        "logical_steps": 8,
        "measurement_iterations": 8,
        "graph_size": 64,
        "batch_sizes": [8, 12],
        "dynamic_shape_rate": 0.8,
        "compile_threads": 2,
        "primary_scope": "full_schedule",
    },
)
globals().update(_API)
