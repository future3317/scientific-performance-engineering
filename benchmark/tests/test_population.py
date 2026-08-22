#!/usr/bin/env python3
"""Population-validity and empirical-calibration report contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.calibration.report import EMPIRICAL_FLAGS, _empirical_flags, build_report
from benchmark.harness import miniyaml, runner


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report, errors = build_report(repo_root / "benchmark" / "tasks")
    assert errors == [], errors
    assert report["num_tasks"] == 30
    assert report["track_counts"] == {"spe_core": 16, "sciml": 11, "evolution": 3}
    assert report["empirical_calibration"]["status"] == "pending"
    assert report["empirical_calibration"]["calibration_gate"] == "blocked"
    assert report["task_calibration"]["CORE-COMPILE-DYNAMIC-11"]["status"] == "historical_calibration"
    assert report["task_calibration"]["CORE-COMPILE-DYNAMIC-11"]["stale_for_current_task_digest"] is True
    assert report["formal_50_task_results"] == "not_claimed"
    assert set(report["empirical_rejection_flags"]) == set(EMPIRICAL_FLAGS)
    print("test_population: OK")


def test_empirical_floor_uses_observed_control_noise() -> None:
    import json
    import tempfile

    specs = [{
        "task_id": "DYNAMIC",
        "track": "spe_core",
        "scientific_gates": ["finite_loss"],
        "measurement": {"noise_floor_percent": 2.0, "min_improvement_percent": 5.0},
    }]
    payload = {
        "tasks": [{
            "task_id": "DYNAMIC",
            "oracle_ci_low_percent": 61.2165,
            "oracle_ci_high_percent": 233.8057,
            "control_noise_percent": [51.8665, 61.2165, 58.4659],
            "baseline_speedups": [1.20, 1.30],
        }]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name
    flags, calibration = _empirical_flags(specs, __import__("pathlib").Path(path), [])
    assert flags["oracle_effect_too_small"] == []
    assert flags["oracle_effect_unstable"] == ["DYNAMIC"]
    assert flags["noise_too_high"] == ["DYNAMIC"]
    assert flags["baseline_already_optimal"] == ["DYNAMIC"]
    assert calibration["calibration_gate"] == "blocked"


def test_empirical_floor_can_clear_high_declared_noise() -> None:
    import json
    import tempfile

    specs = [{
        "task_id": "DYNAMIC",
        "track": "spe_core",
        "scientific_gates": ["finite_loss"],
        "measurement": {"noise_floor_percent": 2.0, "min_improvement_percent": 5.0},
    }]
    payload = {"tasks": [{
        "task_id": "DYNAMIC",
        "oracle_ci_low_percent": 156.0413,
        "oracle_ci_high_percent": 233.8057,
        "control_noise_percent": [51.8665, 61.2165, 58.4659],
    }]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name
    flags, calibration = _empirical_flags(specs, __import__("pathlib").Path(path), [])
    assert flags["oracle_effect_too_small"] == []
    assert flags["oracle_effect_unstable"] == []
    assert flags["noise_too_high"] == []
    assert calibration["calibration_gate"] == "ready_for_review"


def test_evolution_expected_delta_is_a_hard_calibration_gate() -> None:
    import json
    import tempfile
    from benchmark.formal.attest import task_package_digest

    repo_root = Path(__file__).resolve().parents[2]
    task_dir = repo_root / "benchmark" / "tasks" / "EVOL-EQUIVARIANT-SPECIALIZE-30"
    spec = {
        "task_id": "EVOL-EQUIVARIANT-SPECIALIZE-30", "track": "evolution",
        "workspace": {"api": "episode_v1"},
        "scientific_gates": ["state_transition_valid"], "measurement": {"repetitions": 3},
        "oracle": {"expected_delta_range": [0.1, 1.0]}, "_task_dir": task_dir,
    }
    base = {
        "task_digest": task_package_digest(task_dir), "revision": "a" * 40,
        "environment": {"python_version": "3.12"}, "outer_trials": [{}, {}, {}],
        "noise_control": {}, "oracle_ci": {}, "semantic_gates": {"state_transition_valid": True},
        "anti_cheat": {"status": "pass"}, "calibration_status": "eligible",
        "metric_class": "evolution",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump({"tasks": [{**base, "task_id": spec["task_id"], "episode_effect": {
            "outer_trial_deltas": [0.01, 0.02, 0.03], "mean_absolute_score_delta": 0.02,
        }}]}, handle)
        path = Path(handle.name)
    flags, calibration = _empirical_flags([spec], path, [])
    assert flags["evolution_delta_out_of_range"] == [spec["task_id"]]
    assert calibration["calibration_gate"] == "blocked"


def test_h2d_task_exposes_fixture_clone_contract() -> None:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    task_dir = repo_root / "benchmark" / "tasks" / "CORE-H2D-OVERFANOUT-23"
    spec = importlib.util.spec_from_file_location("h2d_overfanout_23", task_dir / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "clone_fixtures", None))


def test_evolution_regression_cases_use_regression_namespace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    episode = repo_root / "benchmark" / "tasks" / "EVOL-EQUIVARIANT-SPECIALIZE-30" / "episodes" / "equivariant_specialization_episode.yaml"
    payload = miniyaml.load(str(episode))
    candidate = payload["phases"][0]["inject_experiences"][0]
    cases = candidate["regression_cases"]
    assert cases and all(str(case).startswith("REG-") for case in cases)


def test_cross_view_consistency_uses_active_manifest_not_retired_tasks() -> None:
    from benchmark.families.consistency import validate_cross_view_consistency

    repo_root = Path(__file__).resolve().parents[2]
    report = validate_cross_view_consistency(
        tasks_root=repo_root / "benchmark" / "tasks", surface_count=6,
    )
    assert report["ok"], report["errors"]


def test_compile_boundary_grammar_uses_public_workload_features() -> None:
    from benchmark.families import family_predicate_grammar, family_decision_lattice
    grammar = family_predicate_grammar("compile")
    assert all(".evidence." not in feature["path"] for feature in grammar["features"])
    assert all(".evidence." not in path for path in grammar["threshold_universe"])
    assert family_decision_lattice("compile", count=8)


def test_episode_candidate_input_excludes_runtime_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    task_ids = (
        "EVOL-EPISODE-POISON-10",
        "EVOL-COMPILER-DRIFT-20",
        "EVOL-EQUIVARIANT-SPECIALIZE-30",
    )
    for index, task_id in enumerate(task_ids):
        task_dir = repo_root / "benchmark" / "tasks" / task_id
        module = runner.import_module_by_path(task_dir / "benchmark.py")
        observed: dict[str, object] = {}

        class SpySolution:
            @staticmethod
            def run_episode_task(task_workspace, skill_view, budget):
                observed.update({"task_workspace": task_workspace, "skill_view": skill_view, "budget": budget})
                return {"action": {"condition": "C"}}

        fixtures = module.make_fixtures(123 + index)
        module.run_performance(SpySolution(), fixtures)
        declared = miniyaml.load(str(task_dir / "task.yaml"))
        assert set(observed["skill_view"]) == {"public_context"}
        assert "seed" not in str(observed["skill_view"]).lower()
        assert "condition" not in str(observed["skill_view"]).lower()
        assert "episode_yaml" not in str(observed["skill_view"]).lower()
        assert set(observed["budget"]) == {"max_wall_time_s"}
        assert "seed" not in observed["budget"]
        assert observed["budget"]["max_wall_time_s"] == declared["time_budget_s"]


def test_episode_budget_uses_declared_task_budget() -> None:
    from benchmark.harness.verifier import _episode_arm_budget

    assert _episode_arm_budget({"time_budget_s": 600}) == 600.0
    assert _episode_arm_budget({"time_budget_s": 600}, 17.5) == 17.5


def test_poison_episode_control_exposes_retrieved_poison_regression(tmp_path: Path) -> None:
    from benchmark.harness.evolution import run_episode

    repo_root = Path(__file__).resolve().parents[2]
    episode = repo_root / "benchmark" / "tasks" / "EVOL-EPISODE-POISON-10" / "episodes" / "poison_episode.yaml"
    result = run_episode(
        episode,
        "C",
        tmp_path / "control",
        core_repo=repo_root,
        snapshot_dir=repo_root,
        seed=0,
        max_wall_time_s=600,
    )

    assert result["metrics"]["poisoning_survival_rate"] == 0.5


def test_compiler_drift_control_exposes_poison_regression(tmp_path: Path) -> None:
    from benchmark.harness.evolution import run_episode

    repo_root = Path(__file__).resolve().parents[2]
    episode = repo_root / "benchmark" / "tasks" / "EVOL-COMPILER-DRIFT-20" / "episodes" / "compiler_drift_episode.yaml"
    result = run_episode(
        episode,
        "C",
        tmp_path / "control",
        core_repo=repo_root,
        snapshot_dir=repo_root,
        seed=0,
        max_wall_time_s=600,
    )

    assert result["metrics"]["poisoning_survival_rate"] == 0.0


def test_evolution_specialization_gate_is_candidate_only() -> None:
    import importlib.util

    task_dir = Path(__file__).parents[1] / "tasks" / "EVOL-EQUIVARIANT-SPECIALIZE-30"
    module_spec = importlib.util.spec_from_file_location("evol30_contract", task_dir / "benchmark.py")
    assert module_spec is not None and module_spec.loader is not None
    benchmark_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(benchmark_module)
    result = {"metrics": {"rule_precision": None, "negative_transfer_rate": 0.0}, "condition": "C"}
    assert "specialization_applied" not in benchmark_module.gates_harness_episode(result)


def test_approval_rejects_tampered_derived_projection(tmp_path: Path, monkeypatch) -> None:
    import copy
    import json
    import benchmark.calibration.approval as approval_module
    from benchmark.calibration.identity import json_digest

    empirical = {"schema_version": 1, "tasks": []}
    digest = json_digest(empirical)
    canonical_report = {
        "active_task_ids": [],
        "empirical_calibration": {"calibration_gate": "ready_for_review", "empirical_digest": digest},
    }
    canonical_pilot = {
        "active_task_ids": [], "calibration_gate": "ready_for_review",
        "tasks": [], "empirical_digest": digest,
    }
    canonical_pilot["artifact_digest"] = json_digest(canonical_pilot)
    monkeypatch.setattr(
        approval_module,
        "rebuild_calibration_views",
        lambda **kwargs: (canonical_report, canonical_pilot, []),
    )
    report = copy.deepcopy(canonical_report)
    report["derived_edit"] = "not-authoritative"
    pilot = copy.deepcopy(canonical_pilot)
    report_path = tmp_path / "report.json"
    pilot_path = tmp_path / "pilot.json"
    empirical_path = tmp_path / "empirical.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
    empirical_path.write_text(json.dumps(empirical), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="derived calibration projections"):
        approval_module.issue_calibration_approval(
            report_path=report_path, pilot_path=pilot_path, empirical_path=empirical_path,
            out_path=tmp_path / "approval.json", repo_root=Path(__file__).parents[2],
            approver="test", timestamp="2026-08-21T00:00:00Z",
        )


if __name__ == "__main__":
    main()
