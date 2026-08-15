#!/usr/bin/env python3
"""Standalone assert-script tests for harness/miniyaml.py (no pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import miniyaml


def main() -> None:
    # --- nested maps + scalars ----------------------------------------------
    doc = miniyaml.parse(
        """
        # comment line
        schema_version: 1
        task_id: CORE-SCALAR-SYNC-01
        requires_cuda: false
        time_budget_s: 600
        ratio: 1.0e-5
        negative: -3
        nothing: null
        tilde: ~
        empty:
        title: "Per-step scalar synchronization"
        lineage:
          source: synthetic
          mutation_template_id: MT-SCALAR-SYNC-V1
          mutation_seed: 1234
        """
    )
    assert doc["schema_version"] == 1
    assert doc["task_id"] == "CORE-SCALAR-SYNC-01"
    assert doc["requires_cuda"] is False
    assert doc["time_budget_s"] == 600
    assert abs(doc["ratio"] - 1.0e-5) < 1e-12
    assert doc["negative"] == -3
    assert doc["nothing"] is None and doc["tilde"] is None and doc["empty"] is None
    assert doc["title"] == "Per-step scalar synchronization"
    assert doc["lineage"] == {"source": "synthetic", "mutation_template_id": "MT-SCALAR-SYNC-V1", "mutation_seed": 1234}

    # --- flow collections ----------------------------------------------------
    doc = miniyaml.parse(
        """
        tolerance: {rtol: 1.0e-5, atol: 1.0e-6}
        choices: [scalar_sync, h2d_blocking, repeated_compute]
        expected_speedup_range: [1.2, 6.0]
        empty_list: []
        empty_map: {}
        nested: {a: [1, 2], b: {c: true}}
        """
    )
    assert doc["tolerance"] == {"rtol": 1.0e-5, "atol": 1.0e-6}
    assert doc["choices"] == ["scalar_sync", "h2d_blocking", "repeated_compute"]
    assert doc["expected_speedup_range"] == [1.2, 6.0]
    assert doc["empty_list"] == [] and doc["empty_map"] == {}
    assert doc["nested"] == {"a": [1, 2], "b": {"c": True}}

    # --- block lists ----------------------------------------------------------
    doc = miniyaml.parse(
        """
        phases:
          - index: 1
            name: acquisition
            tasks: [A-01, A-02]
          - index: 2
            name: drift
            tasks:
              - B-01
              - B-02
        scientific_gates: []
        mechanism:
          - scalar_sync
          - launch_fragmentation
        """
    )
    assert doc["phases"][0] == {"index": 1, "name": "acquisition", "tasks": ["A-01", "A-02"]}
    assert doc["phases"][1]["tasks"] == ["B-01", "B-02"]
    assert doc["mechanism"] == ["scalar_sync", "launch_fragmentation"]

    # --- inline comments, quoted #, document markers --------------------------
    doc = miniyaml.parse(
        """---
        a: 1  # trailing comment
        b: "quoted # not comment"
        c: 'single # also not'
        ...
        """
    )
    assert doc == {"a": 1, "b": "quoted # not comment", "c": "single # also not"}

    # --- full spec-shaped document --------------------------------------------
    spec = miniyaml.parse(
        """
        schema_version: 1
        task_id: CORE-SCALAR-SYNC-01
        track: spe_core
        family: training_loop_overhead
        mechanism: scalar_sync
        kind: positive
        lineage:
          source: synthetic
          mutation_template_id: MT-SCALAR-SYNC-V1
          mutation_seed: 1234
        title: "Per-step scalar synchronization in a materials GNN training loop"
        requires_cuda: false
        time_budget_s: 600
        workspace:
          entrypoint: solution.py
          api: train_loop_v1
        measurement:
          primary_metric: step_ms_p50
          higher_is_better: false
          warmup_iterations: 5
          measured_iterations: 30
          repetitions: 5
          min_improvement_percent: 5.0
          noise_floor_percent: 2.0
          speedup_tripwire: 20.0
        correctness:
          num_fresh_inputs: 3
          reference: fp64_recompute
          tolerance: {rtol: 1.0e-5, atol: 1.0e-6}
        scientific_gates: []
        diagnosis:
          enabled: true
          choices: [scalar_sync, h2d_blocking]
        oracle:
          expected_speedup_range: [1.2, 6.0]
        """
    )
    assert spec["workspace"] == {"entrypoint": "solution.py", "api": "train_loop_v1"}
    assert spec["measurement"]["speedup_tripwire"] == 20.0
    assert spec["correctness"]["tolerance"]["rtol"] == 1.0e-5
    assert spec["oracle"]["expected_speedup_range"] == [1.2, 6.0]

    # --- dump round-trip -------------------------------------------------------
    tree = {
        "a": 1,
        "b": [1, 2.5, "x", True, None],
        "c": {"nested": {"deep": "v"}, "list_of_maps": [{"k": 1}, {"k": 2}]},
        "d": "needs: quoting",
        "e": [],
        "f": {},
    }
    assert miniyaml.parse(miniyaml.dump(tree)) == tree

    # --- rejections -------------------------------------------------------------
    for bad in ("\ta: 1", "a:\n\tb: 2", "x: [1, 2", "y: {a: 1", 'z: "unterminated'):
        try:
            miniyaml.parse(bad)
        except miniyaml.MiniYAMLError:
            pass
        else:
            raise AssertionError(f"expected MiniYAMLError for {bad!r}")

    print("test_miniyaml: OK")


if __name__ == "__main__":
    main()
