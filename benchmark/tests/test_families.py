#!/usr/bin/env python3
"""Checks that all benchmark views share deterministic family instances."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.boundary.families import family_cases
from benchmark.families import family_instances, family_views, poisoning_transformation, transformation
from benchmark.formal.aggregate import performance_profile
from benchmark.harness.evolution import evolution_regret
from benchmark.interaction.factorial_bench import generate_family_interaction_surface
from benchmark.taskgen.generate import generate_family_slots


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
    print("test_families: OK")


if __name__ == "__main__":
    main()
