from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark.harness import runner, stats


def _artifact(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "task_id": "TASK-1",
        "outer_trial_id": "outer-0",
        "benchmark_revision": "abc123",
        "task_manifest_digest": "tasks-digest",
        "hardware_fingerprint": {"python_version": "3.11", "platform": "test", "torch_version": "2", "cuda_version": None, "gpu_name": None, "gpu_count": 0, "torch_geometric_version": None, "cuda_available": False},
        "software_fingerprint": {"python_version": "3.11", "platform": "test", "torch_version": "2", "cuda_version": None, "gpu_name": None, "gpu_count": 0, "torch_geometric_version": None, "cuda_available": False},
        "compile_threads": 2,
        "compiler_cache_policy": "arm-repetition-fresh",
        "expected_speedup_range": [0.9, 1.1],
        "higher_is_better": False,
        "primary_metric": "step_ms_p50",
        "control_a_runs": [10.0, 10.0, 10.0, 10.0, 10.0],
        "control_b_runs": [9.5, 10.0, 10.0, 10.0, 10.0],
        "observed_noise_floor_percent": 3.5,
        "declared_noise_floor_percent": 2.0,
        "effective_noise_floor_percent": 3.5,
    }
    value.update(overrides)
    return value


def test_noise_floor_separates_typical_log_noise_from_max_outlier() -> None:
    a = [100.0] * 5
    b = [99.0, 101.0, 100.5, 99.5, 1.0]
    floor = stats.estimate_noise_floor(a, b, False)
    assert floor["noise_estimator"] == "log_ratio_median_mad_v1"
    assert floor["noise_floor_percent_observed"] == floor["typical_noise_percent"]
    assert floor["max_deviation_percent"] > floor["typical_noise_percent"]
    assert floor["noise_spread_percent"] >= 0.0
    assert stats.effective_noise_floor(2.0, floor["noise_floor_percent_observed"]) == max(
        2.0, float(floor["typical_noise_percent"])
    )
    assert stats.effective_noise_floor(2.0, 0.8) == 2.0

def test_measurement_phase_distinguishes_compile_cold_start() -> None:
    from benchmark.harness.verifier import measurement_phase_for_task

    assert measurement_phase_for_task({"family": "compiler"}) == "cold_start_inclusive"
    assert measurement_phase_for_task({"family": "graph_energy_force"}) == "steady_state"



def test_runtime_provenance_keeps_process_and_host_state() -> None:
    from benchmark.harness.fingerprint import capture_runtime_provenance

    provenance = capture_runtime_provenance()
    assert provenance["pid"] > 0
    assert "cpu_affinity" in provenance
    assert "host" in provenance
    assert "timestamp_utc" in provenance


def test_paired_measurement_records_shared_session_boundaries(monkeypatch, tmp_path: Path) -> None:
    from benchmark.harness import runner

    fingerprint = {"gpu_uuid": "GPU-test", "cuda_visible_devices": "0"}
    telemetry = iter(
        {"status": "busy", "foreign_pids": [101], "gpu_uuid": "GPU-test"}
        for _ in range(5)
    )
    monkeypatch.setattr(runner, "capture_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(
        runner,
        "capture_runtime_provenance",
        lambda: {"gpu_shared_state": next(telemetry)},
    )
    monkeypatch.setattr(runner, "condition_cuda_device", lambda *_: 0.0)

    class Module:
        @staticmethod
        def load_solution(path: str, device: str) -> str:
            return path

        @staticmethod
        def make_fixtures(seed: int, device: str) -> dict[str, int]:
            return {"seed": seed}

        @staticmethod
        def run_performance(solution: str, fixtures: dict[str, int], warmup: int, iterations: int, device: str) -> float:
            return 1.0

    record = runner.run_paired_measurement(
        Module,
        baseline_path=tmp_path / "baseline.py",
        candidate_path=tmp_path / "candidate.py",
        measurement_cfg={"repetitions": 1, "warmup_iterations": 1, "measured_iterations": 1},
        seed=0,
        device="cuda:0",
    )

    telemetry_record = record["shared_state_telemetry"]
    assert telemetry_record["session_start"]["gpu_shared_state"]["foreign_pids"] == [101]
    assert len(telemetry_record["arm_boundaries"]) == 2
    assert telemetry_record["session_end"]["gpu_shared_state"]["foreign_pids"] == [101]
    assert record["runtime_contract"]["compatible"] is True
def test_cache_policy_matches_task_family() -> None:

    from benchmark.harness.verifier import cache_policy_for_task

    assert cache_policy_for_task({"family": "compiler"}) == "arm-repetition-fresh"
    assert cache_policy_for_task({"family": "scalar_sync"}) == "verifier-invocation-scoped"


def test_noise_floor_is_label_symmetric() -> None:
    a = [10.0, 8.0, 12.0, 10.0, 9.0]
    b = [9.0, 10.0, 10.0, 12.0, 10.0]
    left = stats.estimate_noise_floor(a, b, False)
    right = stats.estimate_noise_floor(b, a, False)
    for key in ("noise_floor_percent_observed", "noise_spread_percent", "max_deviation_percent"):
        assert left[key] == right[key]


def test_noise_control_artifact_round_trip_and_digest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        stats.write_noise_control(path, _artifact())
        loaded = stats.read_noise_control(
            path,
            {"task_id": "TASK-1", "outer_trial_id": "outer-0", "benchmark_revision": "abc123", "task_manifest_digest": "tasks-digest", "compile_threads": 2, "compiler_cache_policy": "arm-repetition-fresh", "expected_speedup_range": [0.9, 1.1], "hardware_fingerprint": _artifact()["hardware_fingerprint"]},
        )
        assert loaded["artifact_digest"]
        path.write_text(path.read_text(encoding="utf-8").replace("abc123", "tampered"), encoding="utf-8")
        try:
            stats.read_noise_control(path, {})
        except ValueError as exc:
            assert "digest" in str(exc)
        else:
            raise AssertionError("tampered noise artifact was accepted")


def test_noise_control_mismatch_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        stats.write_noise_control(path, _artifact())
        try:
            stats.read_noise_control(path, {"outer_trial_id": "outer-1"})
        except ValueError as exc:
            assert "outer_trial_id" in str(exc)
        else:
            raise AssertionError("mismatched outer trial was accepted")


def test_noise_control_range_change_invalidates_old_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        stats.write_noise_control(path, _artifact(expected_speedup_range=[1.15, 1.5]))
        try:
            stats.read_noise_control(path, {"expected_speedup_range": [2.56, 3.34]})
        except ValueError as exc:
            assert "expected_speedup_range" in str(exc)
        else:
            raise AssertionError("noise artifact from the stale oracle range was accepted")


def test_noise_control_requires_registered_baseline_implementation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        stats.write_noise_control(path, _artifact(control_implementation="oracle"))
        try:
            stats.read_noise_control(path, {})
        except ValueError as exc:
            assert "baseline" in str(exc)
        else:
            raise AssertionError("oracle control implementation was accepted")


def test_noise_control_environment_bindings_are_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        artifact = _artifact()
        stats.write_noise_control(path, artifact)
        for expected in (
            {"compile_threads": 4},
            {"compiler_cache_policy": "shared"},
            {"hardware_fingerprint": dict(artifact["hardware_fingerprint"], torch_version="other")},
            {"hardware_fingerprint": dict(artifact["hardware_fingerprint"], cpu_affinity=[99])},
        ):
            try:
                stats.read_noise_control(path, expected)
            except ValueError:
                pass
            else:
                raise AssertionError(f"incompatible environment was accepted: {expected}")


def test_paired_measurement_exposes_arm_wall_diagnostics() -> None:
    class Benchmark:
        @staticmethod
        def make_fixtures(seed: int, device: str):
            return {"seed": seed}

        @staticmethod
        def load_solution(path: str, device: str):
            return path

        @staticmethod
        def run_performance(solution, fixtures, warmup: int, iterations: int, device: str):
            return {"value": 1.0, "work_units": {"steps": iterations}, "output_checksums": {}, "timing": {}}

    record = runner.run_paired_measurement(
        Benchmark,
        "baseline.py",
        "candidate.py",
        {"repetitions": 1, "warmup_iterations": 0, "measured_iterations": 1},
        measurement_phase="steady_state",
        device_conditioning_prelude_ms=5.0,
    )
    assert record["measurement_phase"] == "steady_state"
    assert record["device_conditioning_prelude_ms"] == 5.0
    assert len(record["timing"]) == 2
    for item in record["timing"]:
        assert item["arm"] in {"baseline", "candidate"}
        assert item["repetition"] == item["rep"]
        assert item["arm_total_wall_s"] >= 0
        assert item["load_solution_wall_s"] >= 0
        assert item["run_performance_wall_s"] >= 0
        assert item["device_conditioning_prelude_ms"] == 5.0


def test_formal_verifier_consumes_artifact_without_rerunning_control(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from benchmark.harness import verifier

    task_dir = Path(__file__).resolve().parents[1] / "tasks" / "CORE-COMPILE-TINY-12"
    solution_dir = tmp_path / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.py").write_text("x = 1\n", encoding="utf-8")
    fake_module = SimpleNamespace(make_fixtures=lambda **kwargs: {}, load_solution=lambda **kwargs: {}, run_scientific_gates=lambda **kwargs: {"gate": {"passed": True, "details": {}}})
    monkeypatch.setattr(verifier.runner, "select_device", lambda requires_cuda: ("cpu", True))
    monkeypatch.setattr(verifier.runner, "import_module_by_path", lambda path: fake_module)
    monkeypatch.setattr(verifier, "_fresh_input_correctness", lambda *args, **kwargs: {"passed": True, "per_input": [], "output_checksums": {}})
    monkeypatch.setattr(verifier.runner, "call_benchmark_fn", lambda *args, **kwargs: {"gate": {"passed": True, "details": {}}})
    calls: list[tuple[str, str]] = []

    def paired(*args, **kwargs):
        calls.append((str(kwargs.get("baseline_path")), str(kwargs.get("candidate_path"))))
        return {"baseline_runs": [10.0] * 5, "candidate_runs": [9.0] * 5, "work_units": {f"baseline:{i}": {} for i in range(5)} | {f"candidate:{i}": {} for i in range(5)}, "timing": []}

    monkeypatch.setattr(verifier.runner, "run_paired_measurement", paired)
    fp = verifier.capture_fingerprint()
    artifact = _artifact(task_id="CORE-COMPILE-TINY-12", hardware_fingerprint=fp, software_fingerprint=fp, compile_threads=2, primary_metric="schedule_wall_ms", higher_is_better=False, expected_speedup_range=[0.9, 1.1])
    artifact_path = tmp_path / "noise.json"
    stats.write_noise_control(artifact_path, artifact)
    result = verifier.verify_task(
        task_dir,
        solution_dir,
        out_path=tmp_path / "result.json",
        noise_control_path=artifact_path,
        noise_control_required=True,
    )
    assert len(calls) == 1, result
    assert "control_runs" not in result.get("measurement", {})
    assert result["measurement"]["noise_floor_percent_effective"] == 3.5
    assert result["calibration_status"] == "eligible"


def test_required_noise_control_missing_is_resource_blocked(monkeypatch, tmp_path) -> None:
    from benchmark.harness import verifier

    task_dir = Path(__file__).resolve().parents[1] / "tasks" / "CORE-COMPILE-TINY-12"
    solution_dir = tmp_path / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.py").write_text("x = 1\n", encoding="utf-8")
    called = {"paired": False}
    monkeypatch.setattr(verifier.runner, "run_paired_measurement", lambda *args, **kwargs: called.__setitem__("paired", True))
    result = verifier.verify_task(task_dir, solution_dir, noise_control_required=True)
    assert result["execution_validity"] == "resource_blocked"
    assert result["verdict"] == "inconclusive"
    assert called["paired"] is False
