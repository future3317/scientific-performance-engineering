from experiments.real_intervention_v6.gpu_attestation import classify, device_uuid, idle_snapshot


def fixture(pids=(12,), begin=11):
    ready = {"pid": 12, "start_ticks": 45, "nspid": [120, 12], "gpu_uuid": "GPU-A", "ready_mono": 10, "cuda_event_ms": 2, "allocation_bytes": 100}
    parent = {"pid": 12, "start_ticks": 45}
    sample = {"capture_begin_mono": begin, "gpu_uuid": "GPU-A", "memory_used_bytes": 1000,
              "nvml": {"nvmlDeviceGetComputeRunningProcesses": {"status": "SUCCESS", "rows": [{"pid": p} for p in pids]}},
              "cli": {"returncode": 0, "stdout": ""}}
    return ready, parent, [sample]


def test_pre_ready_sample_cannot_attest_ready():
    assert classify(*fixture(begin=9), "GPU-A")["worker_cuda_provenance"] == "UNKNOWN"


def test_supported_torch_uuid_prefix_difference():
    assert device_uuid("520e1aa6-5eae-4984-b76a-724aecac1aae") == device_uuid("GPU-520e1aa6-5eae-4984-b76a-724aecac1aae")


def test_optional_v3_capability_does_not_block_idle_but_cli_foreign_does():
    row = fixture(pids=())[2][0]
    row["util_percent"] = 0
    row["nvml"]["nvmlDeviceGetComputeRunningProcesses_v3"] = {"status": "NVMLError_FunctionNotFound"}
    assert idle_snapshot(row)
    row["cli"]["stdout"] = "GPU-A,99,100\n"
    assert not idle_snapshot(row)


def test_empty_processes_not_quiet_even_with_cuda_event():
    result = classify(*fixture(pids=()), "GPU-A", baseline_memory=100)
    assert result["worker_cuda_provenance"] == "VERIFIED"
    assert result["co_tenancy_exposure"] == "UNKNOWN"


def test_namespace_mapping_and_external_competitor():
    assert classify(*fixture(pids=(120,)), "GPU-A")["worker_cuda_provenance"] == "VERIFIED"
    assert classify(*fixture(pids=(12, 99)), "GPU-A")["co_tenancy_exposure"] == "OBSERVED_FOREIGN"


def test_start_ticks_and_uuid_conflicts_are_contradictions():
    ready, parent, samples = fixture()
    parent["start_ticks"] += 1
    assert classify(ready, parent, samples, "GPU-A")["worker_cuda_provenance"] == "CONTRADICTED"
    assert classify(*fixture(), "GPU-WRONG")["worker_cuda_provenance"] == "CONTRADICTED"


def test_no_reservation_means_no_uncontended_claim():
    assert classify(*fixture(), "GPU-A")["co_tenancy_exposure"] == "UNKNOWN"
    assert classify(*fixture(), "GPU-A", reservation_verified=True)["co_tenancy_exposure"] == "VERIFIED_QUIET"


def test_snapshot_encodes_uuid_for_pynvml(monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from experiments.real_intervention_v6.gpu_attestation import snapshot

    calls = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda value: calls.append(value) or object(),
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [],
        nvmlDeviceGetComputeRunningProcesses_v3=lambda handle: [],
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=1024, total=8192),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=0),
        nvmlDeviceGetPowerUsage=lambda handle: 1,
        nvmlDeviceGetClockInfo=lambda handle, clock: 1,
        NVML_CLOCK_SM=0,
        nvmlDeviceGetTemperature=lambda handle, sensor: 1,
        NVML_TEMPERATURE_GPU=0,
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))

    snapshot("GPU-uuid")
    assert calls == [b"GPU-uuid"]
