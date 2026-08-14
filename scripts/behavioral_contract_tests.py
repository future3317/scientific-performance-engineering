#!/usr/bin/env python3
"""Small behavioral fixtures for the bundled performance-tool contracts."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    monitor = load("run_with_gpu_monitor")
    collect_env = load("collect_env")
    benchmark_validator = load("validate_benchmark")
    assert hasattr(monitor, "process_tree_memory"), "monitor must report child process-tree memory"
    assert "gpu_specs" in inspect.signature(monitor.sample_gpu).parameters
    assert "target_pid" in inspect.signature(monitor.host_sample).parameters
    assert hasattr(collect_env, "build_record"), "environment collection needs a privacy-safe record builder"
    record = collect_env.build_record(ROOT, include_sensitive=False)
    assert record["privacy"]["sensitive_host_metadata_included"] is False
    assert record["platform"]["hostname"] == "<redacted>"
    assert not str(record["python"]["executable"]).startswith(("/", "C:\\"))
    assert "PYTORCH_ALLOC_CONF" in collect_env.SAFE_ENV_KEYS
    assert "PYTORCH_CUDA_ALLOC_CONF" in collect_env.SAFE_ENV_KEYS

    template = json.loads((ROOT / "assets" / "benchmark_record.json").read_text(encoding="utf-8"))
    assert "preflight" in template, "benchmark record needs runtime compatibility preflight"
    assert "sync_census" in template["work"], "benchmark record needs a synchronization census"
    assert "logical_update_dag" in template["work"], "benchmark record needs a logical-update DAG"
    assert "campaign_lifecycle" in template["work"], "benchmark record needs the full campaign lifecycle"
    assert "cache_contract" in template["work"], "benchmark record needs cache correctness evidence"
    assert "h2d_proof" in template["work"], "benchmark record needs H2D overlap proof"
    assert "amortized_training_throughput" in template["metrics"]
    assert "time_to_quality_seconds" in template["metrics"]
    resume = template["contract"]["checkpoint_state_contract"]
    assert "dataloader_cursor" in resume, "resume contract must include dataloader position"
    assert "ema_swa_scheduler" in resume, "resume contract must include optimizer-adjacent state"
    schema = json.loads((ROOT / "assets" / "benchmark_record.schema.json").read_text(encoding="utf-8"))
    assert benchmark_validator.validate_record(template, schema) == []
    broken = json.loads(json.dumps(template))
    broken["work"]["h2d_proof"].pop("overlap_evidence")
    assert any("h2d_proof" in error for error in benchmark_validator.validate_record(broken, schema))
    broken_lifecycle = json.loads(json.dumps(template))
    broken_lifecycle["work"]["campaign_lifecycle"].pop("precompute")
    assert any("campaign_lifecycle" in error for error in benchmark_validator.validate_record(broken_lifecycle, schema))
    broken_lifecycle_evidence = json.loads(json.dumps(template))
    broken_lifecycle_evidence["work"]["campaign_lifecycle"]["startup"]["evidence"] = []
    assert any("evidence" in error for error in benchmark_validator.validate_record(broken_lifecycle_evidence, schema))
    print("behavioral contract fixtures: ok")


if __name__ == "__main__":
    main()
