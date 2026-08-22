#!/usr/bin/env python3
"""Checks that all benchmark views share deterministic family instances."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.boundary.families import family_cases
from benchmark.families import (
    all_anchor_instances,
    family_instances,
    family_views,
    poisoning_transformation,
    transformation,
    validate_cross_view_consistency,
)
from benchmark.formal.aggregate import performance_profile
from benchmark.harness.evolution import evolution_regret
from benchmark.interaction.factorial_bench import generate_family_interaction_surface
from benchmark.taskgen.generate import generate_family_slots
from benchmark.families.projection import _compare, project_fixture


def main() -> None:
    families = ("compile", "graph_cache", "h2d_pipeline", "checkpoint", "scalar_sync")
    for family in families:
        first = family_instances(family, count=12, seed=4)
        assert first == family_instances(family, count=12, seed=4)
        views = family_views(family, count=12, seed=4)
        ids = [item.instance_id for pool in views.values() for item in pool]
        assert len(ids) == len(set(ids)) == 12
        assert family_cases(family, surface_count=12)
    surfaces = generate_family_interaction_surface(("h2d_pipeline", "compile"), count=8, seed=2)
    assert len(surfaces) == 8 and all(len(item["outcomes"]) == 4 for item in surfaces)
    assert transformation("compile", "software", version="B").kind == "software"
    assert poisoning_transformation("compile", "duplicate_provenance").kind == "poison"
    slots = generate_family_slots("compile", count=3, seed=9)
    assert [item["instance_id"] for item in slots] == [item["instance_id"] for item in generate_family_slots("compile", count=3, seed=9)]
    regret = evolution_regret([{"oracle_utility": 1.0, "deployed_utility": 0.7, "experiment_cost": 2.0, "acquisition_regret": 0.1}])
    assert regret["total"] == 2.3 and regret["acquisition"] == 0.1
    profile = performance_profile([{"track": "spe_core", "verified": True, "verified_speedup": {"median_speedup": 1.2}}])
    assert profile["spe_core"]["verified_optimization_rate"] == 1.0
    anchors = all_anchor_instances()
    assert len(anchors) == 30 and all(item.anchor_task_id == item.instance_id for item in anchors)
    assert validate_cross_view_consistency(surface_count=12)["ok"]
    print("test_families: OK")


def test_projection_compare_is_fail_closed() -> None:
    assert _compare({"x": 1}, {"x": 1})["status"] == "pass"
    drift = _compare({"x": 1, "y": 2}, {"x": 3, "y": None})
    assert drift["status"] == "drift"
    assert drift["mismatches"] == {"x": {"declared": 1, "actual": 3}}
    assert drift["missing"] == ["y"]


def test_projection_reads_fixture_values_not_declared_metadata() -> None:
    fixture = {"num_heads": 4, "hidden_dim": 256, "batch_size": 32}
    assert project_fixture("repeated_compute", fixture) == {
        "repeat_count": 4, "backbone_width": 256, "batch_size": 32,
    }
    assert project_fixture(
        "repeated_compute",
        {"repeat_count": 2, "width": 128, "logical_batch_size": 16},
    ) == {"repeat_count": 2, "backbone_width": 128, "batch_size": 16}
    assert project_fixture(
        "checkpoint",
        {"checkpoint_config": {"segment_count": 3, "memory_pressure": 0.4, "recompute_ratio": 0.2}},
    ) == {"segment_count": 3, "logical_batch_size": None, "memory_pressure": 0.4, "recompute_ratio": 0.2}
    import torch
    assert project_fixture(
        "crystal_sampling",
        {"initial": torch.zeros(24, 3), "neighbor_count": 12, "geometry_variation": 0.7},
    ) == {"sample_count": 24, "neighbor_count": 12, "geometry_variation": 0.7}
    assert project_fixture(
        "compile",
        {"logical_steps": 192, "graph_size": 320, "dynamic_shape_rate": 0.2},
    ) == {"logical_steps": 192, "graph_size": 320, "dynamic_shape_rate": 0.2}


def test_new_materialized_task_requires_executable_projection(monkeypatch, tmp_path) -> None:
    from benchmark.taskgen import generate

    monkeypatch.setattr(
        generate,
        "audit_task",
        lambda task_dir, **_kwargs: {
            "task_id": task_dir.name,
            "status": "drift",
            "missing": ["worker_count"],
            "mismatches": {},
        },
    )
    with __import__("pytest").raises(ValueError, match="executable projection"):
        generate.assert_executable_projection(tmp_path / "TASK")


def test_materializer_rejects_cross_family_prototype(tmp_path) -> None:
    from benchmark.taskgen import generate

    source = tmp_path / "prototype"
    source.mkdir()
    (source / "task.yaml").write_text("family_id: checkpoint\n", encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="cross-family"):
        generate.assert_same_family_source(source, "h2d_pipeline")


if __name__ == "__main__":
    main()
