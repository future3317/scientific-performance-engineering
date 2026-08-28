from argparse import Namespace
from pathlib import Path
import importlib.util
import json
import os
import signal
import shutil
import sys

import pytest

from benchmark.calibration.campaign import (
    _bounded_noise_control,
    _bounded_verifier_result,
    _calibration_record,
    _copy_oracle,
    _write_resource_blocked_cell,
)
from benchmark.calibration.protocol import outer_trial_count
from benchmark.formal.attest import calibration_envelope, validate_calibration_envelope
from benchmark.harness import stats


@pytest.mark.parametrize(
    "task_id",
    [
        "SCIML-CRYSTAL-DIFFUSION-07R2",  # replacement package with guidance scale
        "CORE-KERNEL-FUSION-09R2",  # R2 compile workspace
    ],
)
def test_copy_oracle_applies_mixed_reference_patch_headers(task_id, tmp_path):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id

    _copy_oracle(task_dir, tmp_path)

    assert (tmp_path / "solution.py").read_text(encoding="utf-8") != (
        task_dir / "workspace" / "solution.py"
    ).read_text(encoding="utf-8")


def test_resource_preflight_writes_a_complete_reusable_cell(tmp_path):
    task_id = "CORE-H2D-PIPELINE-INTERIOR-32"
    task_spec = {
        "workspace": {"api": "train_loop_v1"},
        "measurement": {"primary_metric": "step_ms_p50", "higher_is_better": False, "noise_floor_percent": 2.0},
        "oracle": {"expected_speedup_range": [1.0, 2.0]},
    }
    result, noise, envelope = _write_resource_blocked_cell(
        out=tmp_path, task_id=task_id, outer_id="outer-000", task_spec=task_spec,
        task_digest="t" * 64, population_digest="p" * 64, revision="r" * 40,
        harness_digest="h" * 64, runner_digest="c" * 64, protocol_digest="q" * 64,
        fingerprint={"cuda_available": True, "gpu_uuid": "GPU-test"},
        task_manifest_digest="m" * 64, error="GPU occupied",
    )
    assert result["execution_validity"] == "resource_blocked"
    assert noise["execution_validity"] == "resource_blocked"
    assert stats.read_noise_control(tmp_path / "noise-control" / "outer-000" / f"{task_id}.json")["task_id"] == task_id
    assert envelope["raw_result_digest"]
    assert (tmp_path / "raw" / "outer-000" / f"{task_id}.json").is_file()
    assert (tmp_path / "envelopes" / "outer-000" / f"{task_id}.json").is_file()


def test_copy_oracle_excludes_stale_bytecode_from_solution(tmp_path):
    task_dir = Path(__file__).parents[1] / "tasks" / "EVOL-EQUIVARIANT-SPECIALIZE-30"

    _copy_oracle(task_dir, tmp_path)

    assert not list(tmp_path.rglob("*.pyc"))
    module_spec = importlib.util.spec_from_file_location("evol30_oracle", tmp_path / "solution.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.run_episode_task("", {}, {})["action"]["condition"] == "D"


def test_harness_hash_ignores_derived_bytecode(tmp_path):
    from benchmark.harness.runner import hash_harness_files

    (tmp_path / "runner.py").write_text("value = 1\n", encoding="utf-8")
    before = hash_harness_files(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "runner.cpython-311.pyc").write_bytes(b"derived bytecode")

    assert hash_harness_files(tmp_path) == before


def test_dataloader_h2d_fixture_clone_is_mutation_isolated():
    task_dir = Path(__file__).parents[1] / "tasks" / "CORE-DATALOADER-FANOUT-16R2"
    spec = importlib.util.spec_from_file_location("dataloader_fanout_16r2", task_dir / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fixtures = module.make_fixtures(0)
    from benchmark.harness.runner import fixture_hash

    original_hash = fixture_hash(fixtures)
    cloned = module.clone_fixtures(fixtures)

    def tensors(value):
        import torch
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            return [item for child in value.values() for item in tensors(child)]
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in tensors(child)]
        return []

    original_leaves = tensors(fixtures)
    cloned_leaves = tensors(cloned)
    assert len(original_leaves) == len(cloned_leaves)
    assert all(left is not right for left, right in zip(original_leaves, cloned_leaves))
    cloned["inputs"][0, 0] += 1.0

    assert cloned["inputs"] is not fixtures["inputs"]
    assert cloned["inputs"][0, 0] != fixtures["inputs"][0, 0]
    assert fixture_hash(fixtures) == original_hash


def test_copy_oracle_patch_failure_leaves_no_partial_solution(tmp_path):
    source = Path(__file__).parents[1] / "tasks" / "SCIML-EQUIV-RECOMPUTE-06R2"
    task = tmp_path / source.name
    shutil.copytree(source, task)
    (task / "oracle" / "reference_patch.diff").write_text("not a patch\n", encoding="utf-8")
    destination = tmp_path / "solution"
    try:
        _copy_oracle(task, destination)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid oracle patch unexpectedly materialized")
    assert not destination.exists()


def test_calibration_record_aggregates_all_outer_trials():
    results = [
        {"fingerprint": {"trial": i}, "correctness_pass": True, "scientific_gates": {"gate": True},
         "execution_validity": "valid", "calibration_status": "eligible",
         "verified_speedup": {"median_speedup": 1.1 + i / 100, "ci_low": 1.0 + i / 100, "ci_high": 1.2 + i / 100, "verified": True, "inconclusive": False},
         "anticheat": {"findings": [], "hard_fail": False, "tripwired": False}}
        for i in range(3)
    ]
    record = _calibration_record(Path("."), "TASK", "rev", "digest", results, [
        {"observed_noise_floor_percent": 2.0}, {"observed_noise_floor_percent": 3.0}, {"observed_noise_floor_percent": 1.0}
    ])
    assert record["oracle_ci"]["ci_low"] == 1.0
    assert record["oracle_ci"]["ci_high"] == 1.22
    assert record["control_noise_percent"] == [2.0, 3.0, 1.0]


def test_outer_trial_count_uses_declared_episode_repetitions():
    assert outer_trial_count(
        {"workspace": {"api": "episode_v1"}, "measurement": {"repetitions": 3}},
        {"atomic_outer_trials": 1},
    ) == 3
    assert outer_trial_count(
        {"workspace": {"api": "train_loop_v1"}, "measurement": {"repetitions": 12}},
        {"atomic_outer_trials": 1},
    ) == 1


def test_calibration_cli_exposes_full_cell_identity():
    from benchmark.harness.cli import build_parser

    args = build_parser().parse_args([
        "run-task", "task", "--solution", "solution", "--out", "result",
        "--outer-trial-id", "outer-001", "--benchmark-revision", "rev",
        "--task-manifest-digest", "manifest", "--task-package-digest", "package",
        "--population-manifest-digest", "population",
    ])
    assert args.outer_trial_id == "outer-001"
    assert args.task_package_digest == "package"
    assert args.population_manifest_digest == "population"


def test_bounded_verifier_timeout_is_persisted_as_resource_block(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"

    def timed_out_runner(**kwargs):
        assert kwargs["timeout"] == 3.0
        return {
            "timed_out": True,
            "wall_time_s": 3.01,
            "exit_code": -1,
            "stdout": "",
            "stderr": "[harness] subprocess timed out after 3s",
        }

    monkeypatch.setattr(
        "benchmark.calibration.campaign.runner.run_python_subprocess",
        timed_out_runner,
    )

    result = _bounded_verifier_result(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        result_path=result_path,
        timeout_s=3.0,
        module="benchmark.harness.cli",
        args=("run-task",),
        cwd=tmp_path,
    )

    assert result["timeout"] is True
    assert result["execution_validity"] == "resource_blocked"
    assert result["failure_stage"] == "verifier"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_bounded_noise_control_timeout_is_reported(tmp_path, monkeypatch):
    def timed_out_runner(**kwargs):
        assert kwargs["timeout"] == 3.0
        return {
            "timed_out": True,
            "wall_time_s": 3.01,
            "exit_code": -1,
            "stdout": "",
            "stderr": "[harness] subprocess timed out after 3s",
        }

    monkeypatch.setattr(
        "benchmark.calibration.campaign.runner.run_python_subprocess",
        timed_out_runner,
    )

    noise, timed_out = _bounded_noise_control(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        noise_path=tmp_path / "noise.json",
        timeout_s=3.0,
        args=("calibrate-noise-control",),
        cwd=tmp_path,
        timeout_artifact={
            "measurement_session_id": "session",
            "benchmark_revision": "rev",
            "task_manifest_digest": "manifest",
            "task_package_digest": "package",
            "population_manifest_digest": "population",
            "hardware_fingerprint": {},
            "software_fingerprint": {},
            "compile_threads": 2,
            "compiler_cache_policy": "fresh",
            "primary_metric": "step_ms_p50",
            "higher_is_better": False,
            "control_a_runs": [1.0] * 5,
            "control_b_runs": [1.0] * 5,
            "declared_noise_floor_percent": 2.0,
            "expected_speedup_range": [1.0, 2.0],
            "execution_validity": "resource_blocked",
            "failure_stage": "noise_control",
        },
    )

    assert timed_out is True
    assert noise["timeout"] is True
    assert noise["failure_stage"] == "noise_control"

    persisted = stats.read_noise_control(tmp_path / "noise.json", {"measurement_session_id": "session"})
    assert persisted["execution_validity"] == "resource_blocked"
def test_noise_timeout_persists_shared_state_telemetry_in_raw_and_noise(tmp_path, monkeypatch):
    from benchmark.calibration import campaign

    repo_root = Path(__file__).parents[2]
    fingerprint = {
        "cuda_available": True,
        "gpu_uuid": "GPU-test",
        "cuda_visible_devices": "5",
        "thread_topology": {
            "omp_num_threads": "8",
            "mkl_num_threads": "8",
            "openblas_num_threads": "8",
            "numexpr_num_threads": "8",
            "torch_num_threads": 8,
            "torch_num_interop_threads": 1,
            "compiler_threads": "2",
        },
    }
    preflight = {"status": "busy", "gpu_uuid": "GPU-test", "foreign_pids": [101]}
    postflight = {"status": "busy", "gpu_uuid": "GPU-test", "foreign_pids": [101, 202]}
    preflights = iter([preflight, postflight] * 3)
    monkeypatch.setattr(campaign, "capture_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(campaign, "selected_gpu_preflight", lambda: next(preflights))
    monkeypatch.setattr(
        campaign.CellExecutor,
        "run_noise_control",
        lambda self, **_kwargs: {
            "timed_out": True,
            "wall_time_s": 3.01,
            "exit_code": -1,
            "stdout": "",
            "stderr": "[harness] subprocess timed out after 3s",
            "cleanup": {"quiescent": True},
        },
    )
    monkeypatch.setattr(campaign, "rebuild_calibration_views", lambda **_kwargs: ({}, {"tasks": []}, []))

    result = campaign.run_calibration_campaign(
        Namespace(repo_root=repo_root, out=tmp_path, task_id="CORE-H2D-OVERFANOUT-23R3", outer_trials=3)
    )

    assert result == 0
    for outer in range(3):
        outer_id = f"outer-{outer:03d}"
        raw = json.loads((tmp_path / "raw" / outer_id / "CORE-H2D-OVERFANOUT-23R3.json").read_text())
        noise = json.loads((tmp_path / "noise-control" / outer_id / "CORE-H2D-OVERFANOUT-23R3.json").read_text())
        assert raw["shared_state_telemetry"] == {"session_start": preflight, "session_end": postflight}
        assert noise["shared_state_telemetry"] == {"session_start": preflight, "session_end": postflight}


def test_noise_control_oom_without_artifact_is_resource_blocked(tmp_path, monkeypatch):
    from benchmark.calibration import campaign

    def oom_runner(self, **_kwargs):
        return {
            "timed_out": False,
            "wall_time_s": 1.2,
            "exit_code": 137,
            "stdout": "",
            "stderr": "CUDA out of memory",
            "cleanup": {"quiescent": True},
        }

    monkeypatch.setattr(campaign.CellExecutor, "run_noise_control", oom_runner)
    noise_path = tmp_path / "noise.json"
    noise, blocked = campaign._bounded_noise_control(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        noise_path=noise_path,
        timeout_s=3.0,
        args=("calibrate-noise-control",),
        cwd=tmp_path,
        timeout_artifact={
            "measurement_session_id": "session",
            "benchmark_revision": "rev",
            "task_manifest_digest": "manifest",
            "task_package_digest": "package",
            "population_manifest_digest": "population",
            "hardware_fingerprint": {},
            "software_fingerprint": {},
            "compile_threads": 2,
            "compiler_cache_policy": "fresh",
            "primary_metric": "step_ms_p50",
            "higher_is_better": False,
            "control_a_runs": [1.0] * 5,
            "control_b_runs": [1.0] * 5,
            "declared_noise_floor_percent": 2.0,
            "expected_speedup_range": [1.0, 2.0],
        },
    )

    assert blocked is True
    assert noise["execution_validity"] == "resource_blocked"
    assert noise["timeout"] is False
    persisted = stats.read_noise_control(noise_path, {"measurement_session_id": "session"})
    assert persisted["process_exit_code"] == 137
    assert persisted["error"] == "CUDA out of memory"



def test_bounded_noise_control_residual_group_is_resource_blocked(tmp_path, monkeypatch):
    noise_path = tmp_path / "noise.json"
    from benchmark.harness.stats import write_noise_control

    write_noise_control(noise_path, {
        "task_id": "CORE-COMPILE-DYNAMIC-11",
        "outer_trial_id": "outer-000",
        "benchmark_revision": "rev",
        "task_manifest_digest": "manifest",
        "task_package_digest": "package",
        "population_manifest_digest": "population",
        "hardware_fingerprint": {},
        "software_fingerprint": {},
        "compile_threads": 2,
        "compiler_cache_policy": "fresh",
        "primary_metric": "schedule_wall_ms",
        "higher_is_better": False,
        "control_a_runs": [1, 1, 1, 1, 1],
        "control_b_runs": [1, 1, 1, 1, 1],
        "observed_noise_floor_percent": 0.0,
        "declared_noise_floor_percent": 2.0,
        "expected_speedup_range": [1.0, 2.0],
    })

    def residual_runner(**kwargs):
        return {
            "timed_out": False,
            "wall_time_s": 12.0,
            "exit_code": 0,
            "stdout": "",
            "stderr": "[harness] subprocess exited with a residual process group",
            "cleanup": {
                "status": "failed",
                "residual_detected": True,
                "quiescent": True,
                "quiescent_after_cleanup": False,
                "residual_pids_after_cleanup": [123],
                "term_sent": True,
                "kill_sent": True,
            },
        }

    monkeypatch.setattr(
        "benchmark.calibration.campaign.runner.run_python_subprocess",
        residual_runner,
    )

    noise, blocked = _bounded_noise_control(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        noise_path=noise_path,
        timeout_s=600.0,
        args=("calibrate-noise-control",),
        cwd=tmp_path,
    )

    assert blocked is True
    assert noise["failure_stage"] == "executor_cleanup"
    assert noise["execution_validity"] == "resource_blocked"
    assert noise["executor_cleanup"]["kill_sent"] is True


def test_calibration_envelope_rejects_incompatible_fingerprint():
    fingerprint = {
        "python_version": "3.12", "platform": "linux", "torch_version": "2.0",
        "cuda_version": "12.1", "cuda_available": True, "gpu_name": "A",
        "gpu_count": 1, "driver_version": "550", "gpu_uuid": "GPU-A",
        "torch_geometric_version": None, "cpu_affinity": [0], "cuda_visible_devices": "0",
    }
    envelope = calibration_envelope(
        producer_revision="a" * 40, task_package_digest="b" * 64,
        population_manifest_digest="c" * 64, harness_digest_value="d" * 64,
        calibration_runner_digest="e" * 64, noise_digest="f" * 64,
        raw_result_digest="1" * 64, fingerprint=fingerprint,
        task_id="T", outer_trial_id="outer-000", seed=0, measurement_class="evolution",
    )
    expected = {"fingerprint": {**fingerprint, "gpu_uuid": "GPU-B"}}
    errors = validate_calibration_envelope(envelope, expected)
    assert any("fingerprint mismatch" in error for error in errors)


def test_subprocess_timeout_returns_cleanup_receipt():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(snippet="import time; time.sleep(1)", timeout=0.05)
    assert result["timed_out"] is True
    assert result["cleanup"]["process_group"]
    assert result["cleanup"]["quiescent"] is True


def test_evolution_cell_identity_separates_raw_class_from_measurement_family():
    from benchmark.calibration.identity import canonical_cell_identity

    identity = canonical_cell_identity(
        task_id="EVOL", outer_trial_id="outer-000", seed=0,
        measurement_family="evolution", task_package_digest="pkg",
        population_manifest_digest="population",
    )

    assert identity["measurement_family"] == "evolution"
    assert identity["raw_measurement_class"] == "episode_bounded_score"
    assert identity["envelope_measurement_class"] == "evolution"


@pytest.mark.skipif(__import__("os").name == "nt", reason="process-group membership is not queryable through the POSIX API on Windows")
def test_normal_exit_cleans_residual_process_group():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(
        snippet=(
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])"
        ),
        timeout=1.0,
    )

    assert result["timed_out"] is False
    assert result["cleanup"]["status"] == "cleaned"
    assert result["cleanup"]["residual_detected"] is True
    assert result["cleanup"]["cleanup_attempted"] is True
    assert result["cleanup"]["quiescent"] is True


def test_clean_exit_reports_clean_exit_without_cleanup():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(snippet="pass", timeout=5.0)

    assert result["cleanup"]["status"] == "clean_exit"
    assert result["cleanup"]["residual_detected_before_cleanup"] is False
    assert result["cleanup"]["quiescent_after_cleanup"] is True
    assert result["cleanup"]["residual_pids_after_cleanup"] == []


def test_forced_owned_worker_cleanup_is_reported_and_valid():
    from benchmark.harness.runner import run_python_subprocess

    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)"
    parent_code = (
        f"import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "import time; time.sleep(0.2); raise SystemExit(0)"
    )
    result = run_python_subprocess(snippet=parent_code, timeout=5.0)

    assert result["cleanup"]["status"] == "cleaned"
    assert result["cleanup"]["forced_cleanup"] is True
    assert result["cleanup"]["quiescent_after_cleanup"] is True
    assert result["cleanup"]["residual_pids_after_cleanup"] == []


def test_cleanup_failure_requires_nonquiescent_scope(monkeypatch):
    from benchmark.harness import runner

    def no_cleanup(process, *, process_group):
        return True, False

    monkeypatch.setattr(runner, "_terminate_process_group", no_cleanup)
    result = runner.run_python_subprocess(
        snippet=(
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        ),
        timeout=5.0,
    )
    process_group = result["cleanup"]["process_group"]
    try:
        assert result["cleanup"]["status"] == "failed"
        assert result["cleanup"]["quiescent_after_cleanup"] is False
        assert result["cleanup"]["residual_pids_after_cleanup"] == [process_group]
    finally:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_timeout_remains_invalid_even_when_scope_is_cleaned():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(snippet="import time; time.sleep(1)", timeout=0.05)

    assert result["timed_out"] is True
    assert result["cleanup"]["status"] == "cleaned"
    assert result["cleanup"]["quiescent_after_cleanup"] is True


def test_foreign_process_outside_scope_is_not_signalled():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(
        snippet=(
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(child.pid, flush=True)"
        ),
        timeout=5.0,
    )
    foreign_pid = int(result["stdout"].strip())
    try:
        os.kill(foreign_pid, 0)
        assert result["cleanup"]["residual_detected_before_cleanup"] is False
    finally:
        try:
            os.kill(foreign_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name == "nt", reason="process-group membership is POSIX-specific")
def test_dataloader_style_owned_child_tree_is_valid_after_cleanup():
    from benchmark.harness.runner import run_python_subprocess

    worker = "import subprocess,sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])"
    result = run_python_subprocess(
        snippet=f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {worker!r}])",
        timeout=5.0,
    )

    assert result["cleanup"]["status"] == "cleaned"
    assert result["cleanup"]["quiescent_after_cleanup"] is True


@pytest.mark.skipif(os.name == "nt", reason="process-group membership is POSIX-specific")
def test_compile_worker_style_owned_child_tree_is_valid_after_cleanup():
    from benchmark.harness.runner import run_python_subprocess

    worker = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']); "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])"
    )
    result = run_python_subprocess(
        snippet=f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {worker!r}])",
        timeout=5.0,
    )

    assert result["cleanup"]["status"] == "cleaned"
    assert result["cleanup"]["quiescent_after_cleanup"] is True


def test_cleaned_executor_residual_is_not_resource_blocked(tmp_path, monkeypatch):
    from benchmark.calibration import campaign

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({
            "execution_validity": "valid",
            "validity": "valid",
            "calibration_status": "eligible",
            "efficacy_eligible": True,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        campaign.CellExecutor,
        "run_module",
        lambda self, **kwargs: {
            "timed_out": False,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "wall_time_s": 0.1,
            "cleanup": {
                "status": "cleaned",
                "residual_detected": True,
                "residual_detected_before_cleanup": True,
                "quiescent": True,
                "quiescent_after_cleanup": True,
                "residual_pids": [],
                "residual_pids_after_cleanup": [],
            },
        },
    )

    result = campaign._bounded_verifier_result(
        task_id="T",
        outer_trial_id="outer-000",
        result_path=result_path,
        timeout_s=5.0,
        module="benchmark.harness.cli",
        args=(),
        cwd=tmp_path,
    )

    assert result["execution_validity"] == "valid"


def test_cleaned_noise_control_residual_is_valid(tmp_path, monkeypatch):
    noise_path = tmp_path / "noise.json"
    stats.write_noise_control(noise_path, {
        "task_id": "CORE-COMPILE-DYNAMIC-11",
        "outer_trial_id": "outer-000",
        "benchmark_revision": "rev",
        "task_manifest_digest": "manifest",
        "task_package_digest": "package",
        "population_manifest_digest": "population",
        "hardware_fingerprint": {},
        "software_fingerprint": {},
        "compile_threads": 2,
        "compiler_cache_policy": "fresh",
        "primary_metric": "schedule_wall_ms",
        "higher_is_better": False,
        "control_a_runs": [1, 1, 1, 1, 1],
        "control_b_runs": [1, 1, 1, 1, 1],
        "observed_noise_floor_percent": 0.0,
        "declared_noise_floor_percent": 2.0,
        "expected_speedup_range": [1.0, 2.0],
    })

    monkeypatch.setattr(
        "benchmark.calibration.campaign.CellExecutor.run_noise_control",
        lambda self, **kwargs: {
            "timed_out": False,
            "wall_time_s": 0.1,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "cleanup": {
                "status": "cleaned",
                "residual_detected": True,
                "residual_detected_before_cleanup": True,
                "cleanup_attempted": True,
                "term_sent": True,
                "kill_sent": False,
                "quiescent_after_cleanup": True,
                "residual_pids_after_cleanup": [],
                "quiescent": True,
                "residual_pids": [],
            },
        },
    )

    noise, blocked = _bounded_noise_control(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        noise_path=noise_path,
        timeout_s=600.0,
        args=("calibrate-noise-control",),
        cwd=tmp_path,
    )

    assert blocked is False
    assert noise["executor_cleanup"]["status"] == "cleaned"
    assert noise["executor_cleanup"]["residual_detected_before_cleanup"] is True

@pytest.mark.skipif(os.name == "nt", reason="process-group membership is POSIX-specific")
def test_sigkill_escalates_for_term_resistant_descendant():
    from benchmark.harness.runner import run_python_subprocess

    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)"
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(0.2); raise SystemExit(3)"
    result = run_python_subprocess(snippet=parent_code, timeout=5.0)

    assert result["exit_code"] == 3
    assert result["cleanup"]["term_sent"] is True
    assert result["cleanup"]["kill_sent"] is True
    assert result["cleanup"]["quiescent"] is True


def test_gpu_preflight_query_failure_is_not_clean(monkeypatch):
    from benchmark.harness import fingerprint

    def fail(*args, **kwargs):
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(fingerprint, "_selected_gpu_uuid", lambda: "GPU-test")
    monkeypatch.setattr(fingerprint.subprocess, "check_output", fail)

    result = fingerprint.selected_gpu_preflight()

    assert result["status"] == "unavailable"
    assert "nvidia-smi" in result["reason"]


def test_verifier_nonzero_exit_without_artifact_is_persisted_as_resource_blocked(monkeypatch, tmp_path):
    from benchmark.calibration import campaign

    def fail_without_artifact(self, **_kwargs):
        return {
            "timed_out": False,
            "exit_code": 1,
            "stderr": "CUDA out of memory",
            "stdout": "",
            "wall_time_s": 1.2,
            "cleanup": {"quiescent": True},
        }

    monkeypatch.setattr(campaign.CellExecutor, "run_module", fail_without_artifact)
    result_path = tmp_path / "raw.json"

    result = campaign._bounded_verifier_result(
        task_id="TASK",
        outer_trial_id="outer-000",
        result_path=result_path,
        timeout_s=10.0,
        module="benchmark.harness.cli",
        args=(),
        cwd=tmp_path,
    )

    assert result["execution_validity"] == "resource_blocked"
    assert result["calibration_status"] == "blocked"
    assert result_path.is_file()


def test_gpu_admission_accepts_shared_selected_gpu(monkeypatch):
    from benchmark.harness import fingerprint

    states = iter([
        {"status": "busy", "foreign_pids": [123]},
        {"status": "clean", "foreign_pids": []},
    ])
    monkeypatch.setattr(fingerprint, "selected_gpu_preflight", lambda: next(states))
    monkeypatch.setattr(fingerprint.time, "sleep", lambda _: None)

    result = fingerprint.wait_for_selected_gpu_admission(timeout_s=1.0, poll_interval_s=0.01)

    assert result["admission_status"] == "admitted"
    assert result["status"] == "busy"


def test_shared_gpu_mode_treats_busy_as_observed_not_invalid():
    from benchmark.calibration.campaign import _preflight_blocks

    assert _preflight_blocks({"status": "clean", "gpu_uuid": "GPU-test"}, expected_gpu_uuid="GPU-test") is False
    assert _preflight_blocks({"status": "busy", "gpu_uuid": "GPU-test"}, expected_gpu_uuid="GPU-test") is False
    assert _preflight_blocks({"status": "unavailable", "gpu_uuid": "GPU-test"}, expected_gpu_uuid="GPU-test") is True
    assert _preflight_blocks({"status": "clean", "gpu_uuid": "GPU-other"}, expected_gpu_uuid="GPU-test") is True


def test_shared_gpu_preflight_records_co_tenant_telemetry(monkeypatch):
    from benchmark.harness import fingerprint

    monkeypatch.setattr(fingerprint, "_selected_gpu_uuid", lambda: "GPU-test")

    def fake_check_output(command, **_):
        if command[1].startswith("--query-compute-apps"):
            return "101, GPU-test, 512 MiB\n"
        return "GPU-test, 12000, 12000, 37, 1800, 7000, P2, 95, 52\n"

    monkeypatch.setattr(fingerprint.subprocess, "check_output", fake_check_output)
    result = fingerprint.selected_gpu_preflight()

    assert result["status"] == "busy"
    assert result["gpu_uuid"] == "GPU-test"
    assert result["foreign_pids"] == [101]
    assert result["foreign_processes"] == [{"pid": 101, "used_memory": "512 MiB"}]
    assert result["gpu_state"] == {
        "uuid": "GPU-test",
        "memory_used_mib": 12000,
        "memory_free_mib": 12000,
        "utilization_gpu_percent": 37,
        "clock_sm_mhz": 1800,
        "clock_memory_mhz": 7000,
        "pstate": "P2",
        "power_w": 95,
        "temperature_c": 52,
    }


def test_paired_measurement_requires_both_arms_for_each_repetition():
    from benchmark.harness.runner import paired_measurement_complete

    complete = {"baseline_runs": [1.0, 1.0], "candidate_runs": [1.0, 1.0]}
    incomplete = {"baseline_runs": [1.0, 1.0], "candidate_runs": [1.0]}
    assert paired_measurement_complete(complete, repetitions=2) is True
    assert paired_measurement_complete(incomplete, repetitions=2) is False


def test_subprocess_applies_declared_thread_topology_before_fingerprint():
    from benchmark.harness.runner import run_python_subprocess

    result = run_python_subprocess(
        snippet=(
            "import json; from benchmark.harness.runner import configure_thread_topology_from_env; "
            "configure_thread_topology_from_env(); "
            "from benchmark.harness.fingerprint import capture_fingerprint; "
            "print(json.dumps(capture_fingerprint()['thread_topology']))"
        ),
        timeout=10.0,
        extra_env={
            "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "SPE_TORCH_NUM_THREADS": "8", "SPE_TORCH_NUM_INTEROP_THREADS": "1",
        },
    )

    assert result["exit_code"] == 0, result["stderr"]
    topology = json.loads(result["stdout"].strip())
    assert topology == {
        "omp_num_threads": "8", "mkl_num_threads": "8",
        "openblas_num_threads": "8", "numexpr_num_threads": "8",
        "torch_num_threads": 8, "torch_num_interop_threads": 1,
        "compiler_threads": "2",
    }
