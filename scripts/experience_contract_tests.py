#!/usr/bin/env python3
"""Behavioral fixtures for the experience-to-rule promotion boundary."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_experience.py"
    spec = importlib.util.spec_from_file_location("validate_experience", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_record() -> dict:
    return {
        "schema_version": 1,
        "case_id": "EXP-2026-08-0017",
        "status": "inbox",
        "created_at": "2026-08-14",
        "capture_reasons": ["silent_fallback"],
        "symptom": ["low_gpu_utilization", "cpu_launch_gaps"],
        "workload": {"framework": "pytorch", "domain": "materials_gnn", "mode": "training"},
        "observed_evidence": [{"name": "scalar_sync_events_per_step", "value": 37, "unit": "count", "source": "trace"}],
        "attempts": [
            {"action": "increase_num_workers", "result": "rejected", "why_rejected": "loader wait was already small"},
            {"action": "aggregate_metrics_on_device", "result": "accepted", "evidence": "benchmark_record.json"},
        ],
        "lesson": {
            "type": "candidate",
            "text": "Audit scalar synchronization before sweeping input-pipeline workers when loader wait is small.",
        },
        "scope": {"requires": ["low_data_wait", "scalar_sync_evidence"], "excludes": ["cpu_preprocessing_dominates"]},
        "collector_confidence": "medium",
        "artifacts": {
            "benchmark_record": {"path": "records/benchmark_record.json", "sha256": "a" * 64, "artifact_type": "benchmark_record", "producer": "compare_benchmarks.py", "benchmark_id": "BENCH-1"},
            "trace": {"path": "traces/step.nsys-rep", "sha256": "b" * 64, "artifact_type": "nsys_trace", "producer": "nsys", "benchmark_id": "BENCH-1"},
        },
    }


def main() -> None:
    validator = load_validator()
    schema = validator.load_schema(ROOT / "assets" / "experience_record.schema.json")
    record = valid_record()
    assert validator.validate_record(record, schema) == []

    missing_reason = copy.deepcopy(record)
    missing_reason["capture_reasons"] = []
    assert any("capture_reasons" in error for error in validator.validate_record(missing_reason, schema))

    canonical = copy.deepcopy(record)
    canonical["status"] = "canonical"
    assert any("status" in error for error in validator.validate_record(canonical, schema))

    rejected_without_reason = copy.deepcopy(record)
    rejected_without_reason["attempts"][0].pop("why_rejected")
    assert any("why_rejected" in error for error in validator.validate_record(rejected_without_reason, schema))

    print("experience contract fixtures: ok")


if __name__ == "__main__":
    main()
