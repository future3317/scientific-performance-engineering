from __future__ import annotations

from scripts.compare_benchmarks import gpu_state_status
from scripts.run_with_gpu_monitor import summarize


def _record(state: dict) -> dict:
    return {"hardware": {"gpu_state": state}}


def test_gpu_state_gate_accepts_matched_state() -> None:
    state = {
        "power_limit_w": 450.0,
        "avg_power_w": 180.0,
        "sm_clock_mhz": 2500.0,
        "mem_clock_mhz": 10000.0,
        "pstate": "P2",
        "temperature_c": 50.0,
        "throttle_reason": "0x0",
        "mig_mode": "[N/A]",
        "persistence_mode": "Disabled",
    }
    assert gpu_state_status(_record(state), _record(dict(state))) == []


def test_gpu_state_gate_withholds_material_clock_change() -> None:
    baseline = {
        "power_limit_w": 450.0,
        "avg_power_w": 180.0,
        "sm_clock_mhz": 2500.0,
        "mem_clock_mhz": 10000.0,
        "pstate": "P2",
        "temperature_c": 50.0,
        "throttle_reason": "0x0",
        "mig_mode": "[N/A]",
        "persistence_mode": "Disabled",
    }
    candidate = dict(baseline, sm_clock_mhz=1900.0, throttle_reason="thermal")
    reasons = gpu_state_status(_record(baseline), _record(candidate))
    assert "GPU state materially differs for sm_clock_mhz" in reasons
    assert "GPU state differs for throttle_reason" in reasons


def test_gpu_monitor_summary_uses_contract_power_name() -> None:
    samples = [{
        "physical_index": 0,
        "logical_index": 0,
        "name": "GPU",
        "uuid": "GPU-test",
        "gpu_util_percent": 0.0,
        "memory_util_percent": 0.0,
        "memory_used_mb": 1.0,
        "power_w": 42.0,
        "power_limit_w": 450.0,
        "sm_clock_mhz": 1000.0,
        "mem_clock_mhz": 5000.0,
        "pstate": "P2",
        "temperature_c": 40.0,
        "throttle_reason": "0x0",
        "mig_mode": "[N/A]",
        "persistence_mode": "Disabled",
    }]
    summary = summarize(samples)["0"]
    assert summary["avg_power_w"]["mean"] == 42.0
    assert "power_w" not in summary
