#!/usr/bin/env python3
"""Small behavioral fixtures for the bundled performance-tool contracts."""

from __future__ import annotations

import importlib.util
import inspect
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
    assert hasattr(monitor, "process_tree_memory"), "monitor must report child process-tree memory"
    assert "gpu_specs" in inspect.signature(monitor.sample_gpu).parameters
    assert "target_pid" in inspect.signature(monitor.host_sample).parameters
    assert hasattr(collect_env, "build_record"), "environment collection needs a privacy-safe record builder"
    record = collect_env.build_record(ROOT, include_sensitive=False)
    assert record["privacy"]["sensitive_host_metadata_included"] is False
    assert record["platform"]["hostname"] == "<redacted>"
    assert not str(record["python"]["executable"]).startswith(("/", "C:\\"))
    print("behavioral contract fixtures: ok")


if __name__ == "__main__":
    main()
