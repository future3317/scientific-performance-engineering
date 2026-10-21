from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
import os
import signal
import subprocess
import time

import pytest

from experiments.real_intervention_v4.outcomes import atomic_json
from experiments.real_intervention_v5 import runtime


GPU = "GPU-test-uuid"


class FakeMonitor:
    def __init__(self, gpu_uuid):
        self.gpu_uuid = gpu_uuid
        self.busy = False
        self.busy_for_samples = 0
        self.sample_count = 0
        self.contend_after_go = None

    def close(self):
        pass

    def sample(self, owner_pid, pgid):
        self.sample_count += 1
        pids = runtime._process_group_pids(pgid) if pgid is not None else set()
        self_pids = [pid for pid in sorted(pids) if pid != owner_pid]
        processes = [
            {"pid": pid, "memory_bytes": 1024, "origin": "STUDY_PROCESS_TREE"}
            for pid in self_pids
        ]
        if self.busy and (pgid is None or self.busy_for_samples > 0):
            processes.append({"pid": 987654, "memory_bytes": 1024, "origin": "EXTERNAL"})
            if pgid is None and self.busy_for_samples:
                self.busy_for_samples -= 1
        if pgid is not None and self.contend_after_go is not None and self.contend_after_go.exists():
            processes.append({"pid": 987654, "memory_bytes": 1024, "origin": "EXTERNAL"})
        return {
            "timestamp_unix": 1.0, "timestamp_monotonic": 1.0,
            "gpu_uuid": self.gpu_uuid, "memory_used_mib": 256,
            "memory_free_mib": 1024, "utilization_percent": 0,
            "compute_processes": processes, "process_query_error": None,
            "process_utilization": "NOT_SUPPORTED",
            "process_utilization_status": "NOT_SUPPORTED",
            "phase": None,
            "telemetry_complete": True,
        }


WORKER = textwrap.dedent("""
    import json, os, sys, time
    from pathlib import Path
    output, ack = Path(sys.argv[1]), Path(sys.argv[2])
    ready, go = output.with_suffix('.ready.json'), output.with_suffix('.go')
    ready.write_text(json.dumps({'pid': os.getpid(), 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}))
    while not go.exists(): time.sleep(.01)
    result = {'execution_status': 'COMPLETE', 'scientific_status': 'PASS',
              'correctness': {'passed': True}, 'timing': {'total_s': .1}}
    output.write_text(json.dumps(result))
    while not ack.exists(): time.sleep(.01)
""")


def commands(count=2):
    return [[sys.executable, "-c", WORKER, "{output}", "{ack}"] for _ in range(count)]


def patch_monitor(monkeypatch):
    monkeypatch.setattr(runtime, "_GpuMonitor", FakeMonitor)
    monkeypatch.setattr(runtime, "_enable_child_subreaper", lambda: None)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_worker_go_ack_handshake_keeps_session_matched(tmp_path, monkeypatch, count):
    patch_monitor(monkeypatch)

    result = runtime.run_session(commands(count), tmp_path / f"session-{count}", GPU, 8.0, {})

    assert result["execution_status"] == "COMPLETE"
    assert result["scientific_status"] == "PASS"
    assert result["exposure_status"] == "MATCHED"
    assert result["analysis_status"] == "VALID_PAIRED"
    assert len(result["arms"]) == count
    assert all(arm["worker_result"]["timing"]["total_s"] == 0.1 for arm in result["arms"])
    assert all((tmp_path / f"session-{count}" / f"arm-{index}.ack").exists()
               for index in range(count))
    assert all((tmp_path / f"session-{count}" / f"arm-{index}.result.stdout.log").is_file()
               and (tmp_path / f"session-{count}" / f"arm-{index}.result.stderr.log").is_file()
               for index in range(count))


def test_busy_gpu_does_not_start_either_arm(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    monitor = FakeMonitor(GPU)
    monitor.busy = True
    monkeypatch.setattr(runtime, "_GpuMonitor", lambda gpu_uuid: monitor)

    result = runtime.run_session(commands(), tmp_path / "busy", GPU, 0.5, {})

    assert result["execution_status"] == "NOT_STARTED"
    assert result["exposure_status"] == "INCOMPLETE"
    assert result["monitor"]["co_tenant_confirmed"] is False
    assert result["monitor"]["premeasurement_co_tenant_pids"] == [987654]
    assert all(arm["execution_status"] == "NOT_STARTED" for arm in result["arms"])
    assert not (tmp_path / "busy" / "intent.json").exists()


def test_started_session_identity_cannot_be_replayed(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    session = tmp_path / "already-started"
    session.mkdir()
    atomic_json(session / "intent.json", {"started": True})

    with pytest.raises(RuntimeError, match="refusing replay"):
        runtime.run_session(commands(), session, GPU, 1.0, {})


def test_worker_commands_require_both_path_placeholders():
    with pytest.raises(ValueError, match="placeholders"):
        runtime._substitute(["worker", "{output}"], Path("out"), Path("ack"))


def test_unstarted_nvml_unavailable_is_explicit(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    monitor = FakeMonitor(GPU)
    original = monitor.sample

    def incomplete(owner_pid, pgid):
        sample = original(owner_pid, pgid)
        sample["telemetry_complete"] = False
        sample["process_query_error"] = "permission denied"
        return sample

    monitor.sample = incomplete
    monkeypatch.setattr(runtime, "_GpuMonitor", lambda gpu_uuid: monitor)

    result = runtime.run_session(commands(), tmp_path / "unavailable", GPU, 1.0, {})

    assert result["execution_status"] == "NOT_STARTED"
    assert result["platform_status"] == "UNCERTAIN"
    assert result["error_code"] == "GPU_TELEMETRY_INCOMPLETE_BEFORE_START"


def test_sample_gpu_is_public_and_reports_process_attribution(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)

    sample = runtime.sample_gpu(GPU)

    assert sample["phase"] == "RESOURCE_SELECTION"
    assert sample["gpu_uuid"] == GPU
    assert sample["process_utilization_status"] == "NOT_SUPPORTED"


def test_delayed_process_utilization_uses_observed_study_pid_history():
    rows = [{"pid": 22}, {"pid": 11}, {"pid": 99}]
    runtime._attribute_process_rows(rows, {22}, {11})

    assert [row["origin"] for row in rows] == [
        "STUDY_PROCESS_TREE", "STUDY_PROCESS_HISTORY", "EXTERNAL",
    ]
    exposure = runtime._exposure([{
        "gpu_uuid": GPU, "telemetry_complete": True, "utilization_percent": 0,
        "compute_processes": [],
        "process_utilization": [{"pid": 11, "origin": "STUDY_PROCESS_HISTORY"}],
    }], GPU)
    assert exposure["status"] == "MATCHED"
    assert exposure["co_tenant_confirmed"] is False


def test_busy_sample_without_process_ownership_is_incomplete_not_inferred():
    exposure = runtime._exposure([{
        "gpu_uuid": GPU, "telemetry_complete": True, "utilization_percent": 40,
        "compute_processes": [], "process_utilization": [],
    }], GPU)

    assert exposure["status"] == "INCOMPLETE"
    assert exposure["co_tenant_confirmed"] is False


def test_session_rejects_existing_ready_output_and_go_artifacts(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    session = tmp_path / "stale-artifact"
    session.mkdir()
    (session / "arm-0.result.ready.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact already exists"):
        runtime.run_session(commands(), session, GPU, 1.0, {})


def test_session_rejects_existing_ack_artifact(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    session = tmp_path / "stale-ack"
    session.mkdir()
    (session / "arm-0.ack").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact already exists"):
        runtime.run_session(commands(), session, GPU, 1.0, {})


def test_idle_wait_contention_does_not_taint_a_later_measurement(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    monitor = FakeMonitor(GPU)
    monitor.busy = True
    monitor.busy_for_samples = 1
    monkeypatch.setattr(runtime, "_GpuMonitor", lambda gpu_uuid: monitor)

    result = runtime.run_session(commands(), tmp_path / "wait-then-run", GPU, 8.0, {})

    assert result["analysis_status"] == "VALID_PAIRED"
    assert result["monitor"]["premeasurement_co_tenant_pids"] == [987654]
    assert result["exposure_status"] == "MATCHED"


def test_exposure_ignores_idle_wait_samples_when_given_full_trace():
    waiting = {
        "measurement_active": False, "gpu_uuid": GPU, "telemetry_complete": True,
        "utilization_percent": 20,
        "compute_processes": [{"pid": 987654, "origin": "EXTERNAL"}],
        "process_utilization": [],
    }
    measured = {
        "measurement_active": True, "gpu_uuid": GPU, "telemetry_complete": True,
        "utilization_percent": 0,
        "compute_processes": [{"pid": 1234, "origin": "STUDY_PROCESS_TREE"}],
        "process_utilization": [],
    }

    exposure = runtime._exposure([waiting, measured], GPU)

    assert exposure["status"] == "MATCHED"
    assert exposure["co_tenant_pids"] == []


def test_external_pid_during_measurement_is_confirmed_contention(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    session = tmp_path / "contended-during-arm"
    monitor = FakeMonitor(GPU)
    monitor.contend_after_go = session / "arm-0.result.go"
    monkeypatch.setattr(runtime, "_GpuMonitor", lambda gpu_uuid: monitor)

    result = runtime.run_session(commands(), session, GPU, 8.0, {})

    assert result["exposure_status"] == "CONTENDED"
    assert result["monitor"]["co_tenant_confirmed"] is True
    assert result["monitor"]["co_tenant_pids"] == [987654]


def test_started_worker_interrupted_at_deadline_is_unknown_and_reaped(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    hanging_worker = textwrap.dedent("""
        import json, os, signal, sys, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        output = Path(sys.argv[1])
        ready, go = output.with_suffix('.ready.json'), output.with_suffix('.go')
        ready.write_text(json.dumps({'pid': os.getpid(),
                                     'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}))
        while not go.exists(): time.sleep(.01)
        while True: time.sleep(.02)
    """)
    worker_command = [sys.executable, "-c", hanging_worker, "{output}", "{ack}"]

    result = runtime.run_session([worker_command], tmp_path / "interrupted", GPU, 1.0, {})

    assert result["execution_status"] == "UNKNOWN"
    assert result["arms"][0]["execution_status"] == "UNKNOWN"
    assert result["arms"][0]["worker_pid"] not in {
        int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
    }


def test_parent_sigterm_reaps_ready_worker_and_restores_handler(tmp_path):
    session = tmp_path / "parent-interrupted"
    marker = tmp_path / "parent-result.json"
    parent = textwrap.dedent(r'''
        import json, os, signal, sys, time
        from pathlib import Path
        from experiments.real_intervention_v5 import runtime

        class Monitor:
            def __init__(self, gpu_uuid): self.gpu_uuid = gpu_uuid
            def close(self): pass
            def sample(self, owner_pid, pgid):
                pids = runtime._process_group_pids(pgid) if pgid is not None else set()
                processes = [{"pid": pid, "memory_bytes": 1, "origin": "STUDY_PROCESS_TREE"}
                             for pid in sorted(pids)]
                return {"timestamp_unix": time.time(), "timestamp_monotonic": time.monotonic(),
                        "gpu_uuid": self.gpu_uuid, "memory_used_mib": 0, "memory_free_mib": 1024,
                        "utilization_percent": 0, "compute_processes": processes,
                        "process_utilization": "NOT_SUPPORTED", "process_utilization_status": "NOT_SUPPORTED",
                        "telemetry_complete": True}

        runtime._GpuMonitor = Monitor
        runtime._enable_child_subreaper = lambda: None
        previous = lambda signum, frame: None
        signal.signal(signal.SIGTERM, previous)
        worker = r"""
        import json, os, sys, time
        from pathlib import Path
        output, ack = Path(sys.argv[1]), Path(sys.argv[2])
        ready = output.with_suffix('.ready.json')
        temporary = ready.with_suffix(ready.suffix + '.tmp')
        temporary.write_text(json.dumps({'pid': os.getpid(),
                                         'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}))
        temporary.replace(ready)
        while not output.with_suffix('.go').exists(): time.sleep(.01)
        while not ack.exists(): time.sleep(.01)
        """
        command = [sys.executable, "-c", worker, "{output}", "{ack}"]
        try:
            runtime.run_session([command], Path(sys.argv[1]), "GPU-test-uuid", 20.0, {})
        except InterruptedError:
            Path(sys.argv[2]).write_text(json.dumps({
                "handler_restored": signal.getsignal(signal.SIGTERM) is previous}))
    ''')
    process = subprocess.Popen(
        [sys.executable, "-c", parent, str(session), str(marker)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    worker_pid = None
    try:
        go = session / "arm-0.result.go"
        deadline = time.monotonic() + 8.0
        while not go.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert go.exists(), process.stderr.read() if process.poll() is not None else "worker never reached GO"
        worker_pid = json.loads(
            (session / "arm-0.result.ready.json").read_text(encoding="utf-8"))["pid"]

        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=8.0) == 0, process.stderr.read()
        assert json.loads(marker.read_text(encoding="utf-8"))["handler_restored"]
        with pytest.raises(ProcessLookupError):
            os.killpg(worker_pid, 0)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=8.0)
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.killpg(worker_pid, signal.SIGKILL)


def test_started_worker_exit_without_result_is_unknown_not_replayable(tmp_path, monkeypatch):
    patch_monitor(monkeypatch)
    exit_worker = textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path
        output = Path(sys.argv[1])
        ready, go = output.with_suffix('.ready.json'), output.with_suffix('.go')
        ready.write_text(json.dumps({'pid': os.getpid(),
                                     'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}))
        while not go.exists(): time.sleep(.01)
        sys.exit(17)
    """)
    worker_command = [sys.executable, "-c", exit_worker, "{output}", "{ack}"]

    result = runtime.run_session([worker_command], tmp_path / "exit-17", GPU, 5.0, {})

    assert result["execution_status"] == "UNKNOWN"
    assert result["arms"][0]["execution_status"] == "UNKNOWN"
    assert result["arms"][0]["error_code"] == "WORKER_EXITED_WITHOUT_RESULT"
