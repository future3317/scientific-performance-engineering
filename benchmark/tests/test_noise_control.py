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
        "compiler_cache_policy": "verifier-invocation-scoped",
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


def test_noise_control_effective_floor_uses_observed_maximum() -> None:
    assert stats.effective_noise_floor(2.0, 3.5) == 3.5
    assert stats.effective_noise_floor(2.0, 0.8) == 2.0


def test_noise_control_artifact_round_trip_and_digest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "noise_control.json"
        stats.write_noise_control(path, _artifact())
        loaded = stats.read_noise_control(
            path,
            {"task_id": "TASK-1", "outer_trial_id": "outer-0", "benchmark_revision": "abc123", "task_manifest_digest": "tasks-digest", "compile_threads": 2, "compiler_cache_policy": "verifier-invocation-scoped", "hardware_fingerprint": _artifact()["hardware_fingerprint"]},
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
    )
    assert len(record["timing"]) == 2
    for item in record["timing"]:
        assert item["arm"] in {"baseline", "candidate"}
        assert item["repetition"] == item["rep"]
        assert item["arm_total_wall_s"] >= 0
        assert item["load_solution_wall_s"] >= 0
        assert item["run_performance_wall_s"] >= 0


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
    artifact = _artifact(task_id="CORE-COMPILE-TINY-12", hardware_fingerprint=fp, software_fingerprint=fp, compile_threads=2, primary_metric="schedule_wall_ms", higher_is_better=False)
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
