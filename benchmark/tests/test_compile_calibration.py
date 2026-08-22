import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from benchmark.families.activation import classify_activation
from benchmark.families.catalog import FAMILY_SPECS
from benchmark.families.environment import FamilyEnvironment


def _load_compile_benchmark(task_id: str):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id
    spec = importlib.util.spec_from_file_location(
        f"compile_calibration_{task_id}", task_dir / "benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, task_dir


def test_compile_solution_loads_get_fresh_inductor_and_triton_caches():
    module, task_dir = _load_compile_benchmark("CORE-COMPILE-TINY-12")
    solution_path = task_dir / "workspace" / "solution.py"

    module.load_solution(str(solution_path))
    first = (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.environ.get("TRITON_CACHE_DIR"),
    )
    import torch._inductor.config as inductor_config

    assert inductor_config.compile_threads == 2
    module.load_solution(str(solution_path))
    second = (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.environ.get("TRITON_CACHE_DIR"),
    )

    assert first[0] and first[1]
    assert second[0] and second[1]
    assert first != second


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-COMPILE-RECOMPILE-04",
        "CORE-COMPILE-DYNAMIC-11",
        "CORE-COMPILE-TINY-12",
    ],
)
def test_reference_patch_reproduces_executable_oracle(task_id):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(task_dir / "workspace" / "solution.py", temp_path / "solution.py")
        subprocess.run(
            ["git", "apply", str(task_dir / "oracle" / "reference_patch.diff")],
            cwd=temp_path,
            check=True,
        )
        patched = (temp_path / "solution.py").read_text().splitlines()
    oracle = (task_dir / "oracle" / "solution_oracle.py").read_text().splitlines()
    assert patched == oracle


@pytest.mark.skipif(
    sys.platform == "win32" and shutil.which("cl") is None,
    reason="TorchInductor CPU compilation requires the MSVC compiler on Windows",
)
def test_dynamic_compile_activation_is_harness_owned_and_contrastive():
    module, task_dir = _load_compile_benchmark("CORE-COMPILE-DYNAMIC-11")
    fixtures = module.make_fixtures(7)
    candidate = module.load_solution(str(task_dir / "oracle" / "solution_oracle.py"))
    baseline = module.load_solution(str(task_dir / "workspace" / "solution.py"))

    evidence = module.run_activation_evidence(candidate, baseline, fixtures)

    assert set(evidence) == {"candidate_metrics", "baseline_metrics"}
    assert evidence["candidate_metrics"]["compile_cache"] != evidence["baseline_metrics"]["compile_cache"]
    assert evidence["candidate_metrics"]["dynamic_guard_stable"] is True
    assert evidence["baseline_metrics"]["dynamic_guard_stable"] is False


def test_recompile_activation_requires_graph_break_contrast():
    specs = {
        "remove_compile_graph_break": {
            "activation_validator": "compile_graph_break_removed"
        }
    }
    result = classify_activation(
        "compile",
        specs,
        {"graph_break_count": 0},
        {"graph_break_count": 1},
    )
    assert result["status"] == "passed"
    assert result["matched_actions"] == ["remove_compile_graph_break"]


def test_compile_anchors_have_mechanism_pure_oracles_and_frozen_horizons():
    task_root = Path(__file__).parents[1] / "tasks"
    expected = {
        "CORE-COMPILE-RECOMPILE-04": ("compile_graph_break", 128, 5),
        "CORE-COMPILE-DYNAMIC-11": ("compile_dynamic_shapes", 128, 5),
        "CORE-COMPILE-TINY-12": ("compile_tiny_graphs", 8, 3),
    }
    for task_id, (mechanism, horizon, repetitions) in expected.items():
        task_dir = task_root / task_id
        task_text = (task_dir / "task.yaml").read_text(encoding="utf-8")
        oracle_text = (task_dir / "oracle" / "solution_oracle.py").read_text(encoding="utf-8")
        assert f"mechanism: {mechanism}" in task_text
        assert f"logical_steps: {horizon}" in task_text
        assert f"measured_iterations: {horizon}" in task_text
        assert f"repetitions: {repetitions}" in task_text
        assert "primary_metric: schedule_wall_ms" in task_text
        if task_id == "CORE-COMPILE-RECOMPILE-04":
            assert "padded_inputs" not in oracle_text
            assert "mask =" not in oracle_text
            assert "dynamic=True" not in oracle_text
        if task_id == "CORE-COMPILE-DYNAMIC-11":
            assert "mark_dynamic" in oracle_text
            assert "dynamic=True" not in oracle_text


def test_compile_profile_projects_graph_size_and_horizon():
    for task_id in (
        "CORE-COMPILE-RECOMPILE-04",
        "CORE-COMPILE-DYNAMIC-11",
        "CORE-COMPILE-TINY-12",
    ):
        module, _ = _load_compile_benchmark(task_id)
        fixtures = module.make_fixtures(0)
        profile = fixtures["compile_profile"]
        assert profile["logical_steps"] == profile["measurement_iterations"]
        assert profile["graph_size"] == profile["model_config"]["hidden_dim"] * (profile["model_config"]["num_blocks"] + 1)
        assert profile["primary_scope"] == "full_schedule"


def test_compile_primary_uses_complete_schedule_bracket(monkeypatch):
    from benchmark.tasks import _compile_benchmark as module
    sync_calls = []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(module, "_set_compile_threads", lambda profile: None)
    fixtures = {
        "batch_sizes": [2, 2],
        "inputs": module.torch.randn(4, 2),
        "targets": module.torch.randn(4, 1),
        "optimizer_config": {"lr": 0.001},
        "compile_profile": {"compile_threads": 2, "primary_scope": "full_schedule"},
    }

    class Solution:
        @staticmethod
        def build_model(_fixtures):
            return module.torch.nn.Linear(2, 1)

        @staticmethod
        def train_step(model, batch, optimizer):
            optimizer.zero_grad()
            loss = (model(batch[0]) - batch[1]).pow(2).mean()
            loss.backward()
            optimizer.step()
            return {"loss": loss}

    result = module._run_performance(Solution, fixtures, warmup=1, iterations=2, device="cuda")
    assert result["value"] == result["timing"]["schedule_wall_ms"]
    assert result["timing"]["step_timing_scope"] == "host_dispatch_diagnostic"
    assert len(sync_calls) == 2


def test_compile_actions_use_mechanism_specific_applicability():
    family = FAMILY_SPECS["compile"]
    recompile = {"logical_steps": 128, "graph_size": 64, "dynamic_shape_rate": 0.0}
    graph_positive = {"logical_steps": 128, "graph_size": 128, "dynamic_shape_rate": 0.0}
    dynamic = {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.3}
    tiny = {"logical_steps": 8, "graph_size": 64, "dynamic_shape_rate": 0.8}
    assert not family.action_applicable("remove_compile_graph_break", recompile)
    assert family.action_applicable("remove_compile_graph_break", graph_positive)
    assert not family.action_applicable("stabilize_dynamic_guards", recompile)
    assert family.action_applicable("stabilize_dynamic_guards", dynamic)
    assert not family.action_applicable("remove_compile_graph_break", dynamic)
    assert family.action_applicable("bypass_compile", tiny)


def test_compile_deployable_actions_have_explicit_contract_and_anchor_routing():
    family = FAMILY_SPECS["compile"]
    assert "reuse_compile_cache" not in family.action_specs
    assert "revalidate_compile_cache" not in family.action_specs
    for action_id, metadata in family.action_specs.items():
        assert metadata.get("mechanism"), action_id
        assert metadata.get("applicability") is not None, action_id
        assert metadata.get("scientific_policy_ref"), action_id
        assert "activation_validator" in metadata, action_id

    environment = FamilyEnvironment("compile")
    dynamic = family.reconstruct_anchor("CORE-COMPILE-DYNAMIC-11").parameters
    recompile = family.reconstruct_anchor("CORE-COMPILE-RECOMPILE-04").parameters
    tiny = family.reconstruct_anchor("CORE-COMPILE-TINY-12").parameters
    fusion = family.reconstruct_anchor("CORE-KERNEL-FUSION-09R2").parameters
    assert environment.oracle(dynamic).oracle_bundle == ("stabilize_dynamic_guards",)
    assert environment.oracle(recompile).oracle_bundle == ()
    assert environment.oracle(tiny).oracle_bundle == ()
    assert environment.oracle(fusion).oracle_bundle == ("fuse_pointwise_chain",)
