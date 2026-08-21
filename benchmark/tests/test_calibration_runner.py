from pathlib import Path
import importlib.util
import json
import shutil

import pytest

from scripts.run_active30_calibration import (
    _bounded_noise_control,
    _bounded_verifier_result,
    _calibration_record,
    _copy_oracle,
)
from benchmark.formal.attest import calibration_envelope, validate_calibration_envelope


@pytest.mark.parametrize(
    "task_id",
    [
        "SCIML-CRYSTAL-DIFFUSION-07",  # a/solution.py
        "SCIML-EQUIV-RECOMPUTE-06",  # a/workspace/solution.py
        "CORE-KERNEL-FUSION-09",  # solution.py
    ],
)
def test_copy_oracle_applies_mixed_reference_patch_headers(task_id, tmp_path):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id

    _copy_oracle(task_dir, tmp_path)

    assert (tmp_path / "solution.py").read_text(encoding="utf-8") != (
        task_dir / "workspace" / "solution.py"
    ).read_text(encoding="utf-8")


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
    task_dir = Path(__file__).parents[1] / "tasks" / "CORE-DATALOADER-FANOUT-16"
    spec = importlib.util.spec_from_file_location("dataloader_fanout_16", task_dir / "benchmark.py")
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
    source = Path(__file__).parents[1] / "tasks" / "SCIML-EQUIV-RECOMPUTE-06"
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
        "scripts.run_active30_calibration.runner.run_python_subprocess",
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
        "scripts.run_active30_calibration.runner.run_python_subprocess",
        timed_out_runner,
    )

    noise, timed_out = _bounded_noise_control(
        task_id="CORE-COMPILE-DYNAMIC-11",
        outer_trial_id="outer-000",
        noise_path=tmp_path / "noise.json",
        timeout_s=3.0,
        args=("calibrate-noise-control",),
        cwd=tmp_path,
    )

    assert timed_out is True
    assert noise["timeout"] is True
    assert noise["failure_stage"] == "noise_control"


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
