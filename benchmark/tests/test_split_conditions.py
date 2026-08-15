#!/usr/bin/env python3
"""Standalone assert-script tests for harness/split.py + harness/conditions.py."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import conditions, miniyaml, split
from scripts.render_skill_view import render_skill_view


_TASK_TEMPLATE = """
schema_version: 1
task_id: {task_id}
track: spe_core
family: {family}
mechanism: {mechanism}
kind: positive
lineage:
  source: {source}
  mutation_template_id: {template}
  mutation_seed: 1
title: "test task"
requires_cuda: false
time_budget_s: 60
workspace:
  entrypoint: solution.py
  api: train_loop_v1
measurement:
  primary_metric: step_ms_p50
  higher_is_better: false
  warmup_iterations: 1
  measured_iterations: 2
  repetitions: 2
  min_improvement_percent: 5.0
  noise_floor_percent: 2.0
  speedup_tripwire: 20.0
correctness:
  num_fresh_inputs: 1
  reference: fp64_recompute
  tolerance: {{rtol: 1.0e-5, atol: 1.0e-6}}
scientific_gates: []
diagnosis:
  enabled: false
oracle:
  expected_speedup_range: [1.2, 6.0]
"""


def _make_task(root: Path, task_id: str, family: str, mechanism, source: str, template: str) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    mech_yaml = mechanism if isinstance(mechanism, str) else "[" + ", ".join(mechanism) + "]"
    (task_dir / "task.yaml").write_text(
        _TASK_TEMPLATE.format(
            task_id=task_id, family=family, mechanism=mech_yaml, source=source, template=template
        ),
        encoding="utf-8",
    )


def _make_split(root: Path, phases) -> Path:
    manifest = {"split_id": "test-v1", "phases": phases}
    path = root / "sequential.yaml"
    miniyaml.save(manifest, str(path))
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks = root / "tasks"
        # Phase 1 (acquisition) tasks.
        _make_task(tasks, "ACQ-01", "fam_a", "scalar_sync", "synthetic", "MT-A-V1")
        _make_task(tasks, "ACQ-02", "fam_a", "h2d_blocking", "synthetic", "MT-B-V1")
        # Held-out: same family, different template/lineage.
        _make_task(tasks, "HELD-01", "fam_a", "scalar_sync", "synthetic", "MT-A-V2")
        _make_task(tasks, "HELD-02", "fam_b", "ragged_loops", "synthetic", "MT-C-V1")

        phases = [
            {"index": 1, "name": "acquisition", "tasks": ["ACQ-01", "ACQ-02"]},
            {"index": 2, "name": "same_family_transfer", "tasks": ["HELD-01"]},
            {"index": 3, "name": "cross_family_transfer", "tasks": ["HELD-02"]},
            {"index": 4, "name": "drift", "tasks": []},
            {"index": 5, "name": "poisoned_experience", "tasks": []},
            {"index": 6, "name": "recovery", "tasks": []},
        ]
        manifest_path = _make_split(root, phases)
        errors = split.check_leakage(manifest_path, tasks)
        assert errors == [], errors
        assert len(split.split_manifest_hash(manifest_path)) == 64

        # Leak: phase 2 shares the (family, mechanism, source, template) key of ACQ-01.
        _make_task(tasks, "LEAK-01", "fam_a", "scalar_sync", "synthetic", "MT-A-V1")
        leaky = [dict(p) for p in phases]
        leaky[1] = dict(leaky[1], tasks=["HELD-01", "LEAK-01"])
        errors = split.check_leakage(_make_split(root, leaky), tasks)
        assert any("split-key leak" in e for e in errors), errors

        # Ordering: swapped phase names are rejected; duplicate task ids rejected.
        bad = [dict(p) for p in phases]
        bad[1] = dict(bad[1], name="cross_family_transfer")
        errors = split.check_leakage(_make_split(root, bad), tasks)
        assert any("name must be" in e for e in errors), errors
        bad = [dict(p) for p in phases]
        bad[2] = dict(bad[2], tasks=["ACQ-01"])
        errors = split.check_leakage(_make_split(root, bad), tasks)
        assert any("multiple phases" in e for e in errors), errors

        # Missing manifest / missing task give clean errors.
        errors = split.check_leakage(root / "nope.yaml", tasks)
        assert len(errors) == 1 and "not found" in errors[0]
        bad = [dict(p) for p in phases]
        bad[1] = dict(bad[1], tasks=["GHOST-99"])
        errors = split.check_leakage(_make_split(root, bad), tasks)
        assert any("not found" in e for e in errors), errors

        # --- conditions ----------------------------------------------------------
        snapshot = root / "snapshot"
        (snapshot / "rules").mkdir(parents=True)
        (snapshot / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        (snapshot / "rules" / "r1.json").write_text('{"rule_id": "r1"}\n', encoding="utf-8")
        snapshot_bundle = root / "snapshot-bundle"
        render_skill_view(snapshot, snapshot_bundle)

        out = root / "condA"
        manifest = conditions.materialize_condition("A", None, out)
        assert manifest["files"] == {} and manifest["injection_policy"]["mode"] == "none"

        for cond, mode in (("B", "frozen"), ("C", "inbox_any"), ("D", "canonical_only")):
            out = root / f"cond{cond}"
            manifest = conditions.materialize_condition(cond, snapshot_bundle, out)
            assert manifest["injection_policy"]["mode"] == mode
            assert "SKILL.md" in manifest["files"]
            ok, diffs = conditions.verify_attestation(out)
            assert ok, (cond, diffs)

        # C must have a writable experience/inbox; D must have the full pipeline dirs.
        assert (root / "condC" / "experience" / "inbox").is_dir()
        probe = root / "condC" / "experience" / "inbox" / "probe.json"
        probe.write_text("{}\n", encoding="utf-8")  # writable
        for rel in ("evolution/candidates", "rules", "registry", "tests/rule_cases"):
            assert (root / "condD" / rel).is_dir(), rel
        assert (root / "condD" / "registry" / "rules.json").is_file()

        # Tampering breaks attestation (re-enable write first: the attacker here is
        # the harness host itself, which can always chmod).
        target = root / "condB" / "SKILL.md"
        target.chmod(0o644)
        target.write_text("# tampered\n", encoding="utf-8")
        ok, diffs = conditions.verify_attestation(root / "condB")
        assert not ok and "SKILL.md" in diffs

        # Invalid condition and missing snapshot raise cleanly.
        for bad_call in (
            lambda: conditions.materialize_condition("E", snapshot, root / "x1"),
            lambda: conditions.materialize_condition("C", None, root / "x2"),
            lambda: conditions.materialize_condition("C", root / "ghost", root / "x3"),
        ):
            try:
                bad_call()
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("expected a clean error")

    print("test_split_conditions: OK")


if __name__ == "__main__":
    main()
