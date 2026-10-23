"""Deterministically replay V5's observed pre-ready snapshot interleaving on CPU."""
import time

from experiments.real_intervention_v5 import runtime
from experiments.real_intervention_v5.tests.test_runtime import FakeMonitor, commands, GPU
from experiments.real_intervention_v6.gpu_attestation import classify


def test_v5_rejects_ready_written_during_sampling_but_v6_requires_fresh_sample(tmp_path, monkeypatch):
    folder = tmp_path / "interleaving"
    ready = folder / "arm-0.result.ready.json"

    class InterleavedMonitor(FakeMonitor):
        def sample(self, owner_pid, pgid):
            captured = super().sample(owner_pid, pgid)
            if pgid is not None:
                # Snapshot was taken before CUDA readiness. During the slower
                # OS census the real child publishes readiness. No retry occurs.
                captured["compute_processes"] = []
                end = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < end:
                    time.sleep(.01)
            return captured

    monkeypatch.setattr(runtime, "_GpuMonitor", InterleavedMonitor)
    result = runtime.run_session(commands(1), folder, GPU, 5, {})
    assert ready.exists()
    assert result["error_code"] == "WORKER_GPU_ATTRIBUTION_FAILED"
    assert result["execution_status"] == "UNKNOWN"

    provenance = {"pid": 12, "start_ticks": 45, "gpu_uuid": GPU, "ready_mono": 10, "cuda_event_ms": 2}
    parent = {"pid": 12, "start_ticks": 45}
    sample = {"capture_begin_mono": 9, "gpu_uuid": GPU, "memory_used_bytes": 1000,
              "nvml": {"nvmlDeviceGetComputeRunningProcesses": {"status": "SUCCESS", "rows": []}},
              "cli": {"returncode": 0, "stdout": ""}}
    assert classify(provenance, parent, [sample], GPU)["worker_cuda_provenance"] == "UNKNOWN"
    fresh = {**sample, "capture_begin_mono": 11,
             "nvml": {"nvmlDeviceGetComputeRunningProcesses": {"status": "SUCCESS", "rows": [{"pid": 12}]}}}
    assert classify(provenance, parent, [fresh], GPU)["worker_cuda_provenance"] == "VERIFIED"
