from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


TASK_ROOT = Path(__file__).parents[1] / "tasks" / "CORE-MEM-RETAINED-GRAPH-13R3"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_activation_fixtures(benchmark):
    """Exercise the same activation path without the full memory-pressure fixture."""
    generator = torch.Generator().manual_seed(0)
    model = benchmark._FixtureModel(4, 16, 2)
    for parameter in model.parameters():
        parameter.data.normal_(0.0, 0.02, generator=generator)
    return {
        "device": "cpu",
        "model_config": {"in_dim": 4, "hidden_dim": 16, "segment_count": 2, "checkpointed_segments": 1},
        "optimizer_config": {"lr": 0.001},
        "inputs": torch.randn(8, 4, generator=generator),
        "targets": torch.randn(8, generator=generator),
        "init_state": model.state_dict(),
    }


def test_13r3_fixture_uses_declared_long_horizon_memory_instance():
    benchmark = _load("benchmark_13r3", TASK_ROOT / "benchmark.py")

    fixtures = benchmark.make_fixtures(0)

    assert fixtures["memory_pressure"] == 0.82
    assert fixtures["segment_count"] == 12
    assert fixtures["checkpointed_segments"] == 6
    assert fixtures["model_config"]["segment_count"] == 12


def test_13r3_activation_contrasts_retained_history_with_oracle_cleanup():
    benchmark = _load("benchmark_13r3_activation", TASK_ROOT / "benchmark.py")
    baseline = _load("solution_13r3_activation", TASK_ROOT / "workspace" / "solution.py")
    oracle = _load("oracle_13r3_activation", TASK_ROOT / "oracle" / "solution_oracle.py")

    fixtures = _small_activation_fixtures(benchmark)
    evidence = benchmark.run_activation_evidence(baseline, oracle, fixtures)

    assert evidence["candidate_metrics"]["retained_graph_edges"] > 0
    assert evidence["baseline_metrics"]["retained_graph_edges"] == 0
