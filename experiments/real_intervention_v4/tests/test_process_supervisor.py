from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from experiments.real_intervention_v4 import cache_lifecycle
from experiments.real_intervention_v4.cache_lifecycle import create_cache_scope
from experiments.real_intervention_v4.outcomes import (
    ArmExecution, canonical_digest, record_event, recover_attempt,
)
from experiments.real_intervention_v4.process_supervisor import (
    _execute_arm_inner, _process_group_pids, _terminalize_watchdog_timeout, execute_arm,
)
from experiments.real_intervention_v4 import process_supervisor


def _worker_result(status: str = "valid", *, correctness: bool = True) -> str:
    return textwrap.dedent(
        f"""
        import json, os, time
        payload = {{
            "status": {status!r},
            "correctness": {{"passed": {correctness!r}}},
            "runtime_identity": {{"device_uuid": os.environ["CUDA_VISIBLE_DEVICES"]}},
            "finished_monotonic": time.monotonic(),
        }}
        with open(os.environ["V4_RESULT_PATH"], "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        """
    )


def _execute(root: Path, attempt_id: str, code: str, *, timeout: float = 3.0,
             cleanup: float = 0.8, term: float = 0.2, max_log_bytes: int = 8 * 1024 * 1024,
             environment: dict[str, str] | None = None,
             cache_setup: dict[str, str] | None = None):
    folder = root / "raw" / attempt_id
    return execute_arm(
        results_root=root,
        attempt_id=attempt_id,
        command=[sys.executable, "-c", code],
        spec={"arm": "S0", "context": "test"},
        identity={"source_commit": "test-source", "config_digest": "config",
                  "context_digest": "context"},
        gpu_uuid="GPU-test",
        timeout_seconds=timeout,
        cleanup_grace_seconds=cleanup,
        term_grace_seconds=term,
        heartbeat_seconds=0.1,
        sample_seconds=0.1,
        max_log_bytes=max_log_bytes,
        environment={"V4_RESULT_PATH": str(folder / "result.json"), **(environment or {})},
        cache_setup=cache_setup,
        monitor_fixture=[{"gpu_uuid": "GPU-test", "compute_processes": [],
                          "utilization_percent": 0}],
    )


def test_worker_completion_is_receipted_and_reused_without_relaunch(tmp_path):
    marker = tmp_path / "started-once"
    code = (f"import os\nfd = os.open({str(marker)!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
            "os.close(fd)\n" + _worker_result())
    first = _execute(tmp_path, "complete", code)
    second = _execute(tmp_path, "complete", code)

    assert first["arm_execution"] == ArmExecution.COMPLETE.value
    assert second["arm_execution"] == ArmExecution.COMPLETE.value
    assert marker.is_file()
    assert recover_attempt(tmp_path / "raw" / "complete", first["identity"])["arm_execution"] == "COMPLETE"


def test_intent_is_durable_before_spec_write_and_prestart_failure_is_not_replayed(tmp_path, monkeypatch):
    attempt_dir = tmp_path / "raw" / "spec-write-failure"
    spec = {"arm": "S0", "context": "test"}
    identity = {"source_commit": "test-source", "config_digest": "config",
                "context_digest": "context"}
    original_atomic_json = process_supervisor.atomic_json

    def fail_spec_write(path, value, *, exclusive=False):
        if path.name == "spec.json":
            events = [json.loads(item.read_text(encoding="utf-8"))
                      for item in sorted((attempt_dir / "events").glob("*.json"))]
            assert [event["kind"] for event in events] == ["intent"]
            raise OSError("injected spec publication failure")
        return original_atomic_json(path, value, exclusive=exclusive)

    monkeypatch.setattr(process_supervisor, "atomic_json", fail_spec_write)
    with pytest.raises(OSError, match="injected spec publication failure"):
        _execute_arm_inner(
            results_root=tmp_path, attempt_id="spec-write-failure",
            command=[sys.executable, "-c", "raise SystemExit('worker must not start')"],
            spec=spec, identity=identity, gpu_uuid="GPU-test",
            timeout_seconds=2.0, cleanup_grace_seconds=0.5, term_grace_seconds=0.1,
        )

    frozen_identity = {**identity, "attempt_id": "spec-write-failure", "gpu_uuid": "GPU-test",
                       "spec_digest": canonical_digest(spec)}
    recovered = recover_attempt(attempt_dir, frozen_identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert [event["kind"] for event in recovered["events"]] == ["intent"]


def test_timeout_kills_worker_that_ignores_sigterm_within_ceiling(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    timeout, cleanup = 0.8, 1.3
    started = time.monotonic()
    result = _execute(tmp_path, "stubborn", code, timeout=timeout, cleanup=cleanup, term=0.25)
    elapsed = time.monotonic() - started

    assert result["arm_execution"] == ArmExecution.TIMEOUT.value
    assert elapsed <= timeout + cleanup
    assert result["within_deadline_plus_grace"] == (
        result["host_supervisor_wall_seconds"] <= result["elapsed_ceiling_seconds"]
    )
    assert not result.get("remaining_process_groups")
    events = [json.loads(path.read_text()) for path in
              (tmp_path / "raw" / "stubborn" / "events").glob("*.json")]
    assert any(event.get("action") == "SIGKILL_GROUP" for event in events)


def test_timeout_does_not_signal_an_unrelated_process_group(tmp_path):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        result = _execute(
            tmp_path, "separate-owner", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            timeout=0.4, cleanup=0.6, term=0.1,
        )
        assert result["arm_execution"] == ArmExecution.TIMEOUT.value
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_cache_setup_time_consumes_arm_deadline(tmp_path, monkeypatch):
    attempt = tmp_path / "raw" / "slow-cache"
    marker = tmp_path / "worker-started"
    started = time.monotonic()

    def slow_cache(*_, **__):
        time.sleep(0.08)
        return {"environment": {}}

    monkeypatch.setattr(cache_lifecycle, "create_cache_scope", slow_cache)
    result = _execute_arm_inner(
        results_root=tmp_path, attempt_id="slow-cache",
        command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        spec={"arm": "S0"}, identity={"source_commit": "test"},
        gpu_uuid="GPU-test", timeout_seconds=0.03, cleanup_grace_seconds=0.2,
        term_grace_seconds=0.05, heartbeat_seconds=0.1, sample_seconds=0.1,
        cache_setup={"mode": "EMPTY"}, started_monotonic=started,
    )

    assert result["arm_execution"] == ArmExecution.TIMEOUT.value
    assert result["first_failure_stage"] == "CACHE_INITIALIZATION_DEADLINE"
    assert not marker.exists()
    assert (attempt / "events" / "0001-terminal.json").is_file()


def test_host_watchdog_bounds_cache_setup_and_does_not_replay(tmp_path, monkeypatch):
    python = sys.executable
    wrapper = tmp_path / "cache-setup-hang"
    marker = tmp_path / "cache-setup-start-count"
    wrapper.write_text(textwrap.dedent(f"""\
        #!{python}
        import hashlib, json, signal, sys, time
        from pathlib import Path

        request_path = Path(sys.argv[sys.argv.index('--supervisor-request') + 1])
        request = json.loads(request_path.read_text())
        folder = Path(request['results_root']) / 'raw' / request['attempt_id']
        identity = {{**request['identity'], 'attempt_id': request['attempt_id'],
                    'gpu_uuid': request['gpu_uuid'],
                    'spec_digest': hashlib.sha256(json.dumps(request['spec'], sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()}}
        with open({str(marker)!r}, 'a', encoding='utf-8') as stream:
            stream.write('setup\\n')
        event_dir = folder / 'events'
        event_dir.mkdir(parents=True, exist_ok=True)
        event = {{'sequence': 0, 'kind': 'intent', 'monotonic_seconds': time.monotonic(),
                 'unix_seconds': time.time(), 'identity': identity,
                 'command': request['command'], 'cache_setup_required': True}}
        (event_dir / '0000-intent.json').write_text(json.dumps(event))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
    """), encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(process_supervisor.sys, "executable", str(wrapper))

    started = time.monotonic()
    first = _execute(tmp_path, "cache-setup-hang", "pass", timeout=0.45,
                     cleanup=0.7, term=0.1, cache_setup={"mode": "EMPTY"})
    second = _execute(tmp_path, "cache-setup-hang", "pass", timeout=0.45,
                      cleanup=0.7, term=0.1, cache_setup={"mode": "EMPTY"})

    assert time.monotonic() - started <= 1.4
    assert first["arm_execution"] == ArmExecution.TIMEOUT.value
    assert first["first_failure_stage"] == "CACHE_INITIALIZATION_DEADLINE"
    assert second["arm_execution"] == ArmExecution.TIMEOUT.value
    assert marker.read_text(encoding="utf-8").splitlines() == ["setup"]
    assert not (tmp_path / "raw" / "cache-setup-hang" / "events" / "0001-started.json").exists()


@pytest.mark.parametrize("failure", [PermissionError("cache denied"), FileNotFoundError("cache disappeared")])
def test_cache_permission_and_missing_path_keep_traceback_as_infra(tmp_path, monkeypatch, failure):
    attempt = tmp_path / "raw" / "cache-error"
    attempt.mkdir(parents=True)
    if isinstance(failure, FileNotFoundError):
        record = create_cache_scope(attempt)
        Path(record["environment"]["TRITON_CACHE_DIR"]).rmdir()
    else:
        monkeypatch.setattr(cache_lifecycle, "_probe", lambda *_: (_ for _ in ()).throw(failure))
    marker = tmp_path / "worker-started"

    result = _execute_arm_inner(
        results_root=tmp_path, attempt_id="cache-error",
        command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        spec={"arm": "S0", "cache_scope_path": str((attempt / "cache").resolve())},
        identity={"source_commit": "test"},
        gpu_uuid="GPU-test", timeout_seconds=1.0, cleanup_grace_seconds=0.2,
        term_grace_seconds=0.05, cache_setup={"mode": "EMPTY"},
    )

    assert result["arm_execution"] == ArmExecution.FAILED.value
    assert result["infrastructure_classification"] == "INFRA"
    assert result["cache_error_type"] == type(failure).__name__
    assert type(failure).__name__ in result["traceback"]
    assert not marker.exists()


def test_worker_infrastructure_traceback_survives_supervisor_receipt(tmp_path):
    code = textwrap.dedent(
        """
        import json, os, time
        payload = {
            "status": "error", "arm_execution": "FAILED",
            "scientific_correctness": "UNASSESSED",
            "error_type": "FileNotFoundError", "error": "cache disappeared",
            "root_cause": "FileNotFoundError:cache disappeared",
            "traceback": "injected full traceback: FileNotFoundError\\n  cache path missing",
            "infrastructure_classification": "INFRA",
            "finished_monotonic": time.monotonic(),
        }
        with open(os.environ["V4_RESULT_PATH"], "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        """
    )
    result = _execute(tmp_path, "worker-cache-error", code)

    assert result["arm_execution"] == ArmExecution.FAILED.value
    assert result["infrastructure_classification"] == "INFRA"
    assert "FileNotFoundError" in result["traceback"]


def test_parent_exit_with_grandchild_holding_pipes_is_bounded(tmp_path):
    code = textwrap.dedent(
        """
        import subprocess, sys
        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        subprocess.Popen([sys.executable, "-c", child])
        """
    )
    timeout, cleanup = 1.5, 1.1
    started = time.monotonic()
    result = _execute(tmp_path, "pipe-holder", code, timeout=timeout, cleanup=cleanup, term=0.15)
    elapsed = time.monotonic() - started

    assert elapsed <= timeout + cleanup
    assert result["arm_execution"] == ArmExecution.FAILED.value
    assert result["root_cause"] == "WORKER_DESCENDANT_REQUIRED_CLEANUP"
    assert not result.get("remaining_process_group_pids")
    events = [json.loads(path.read_text()) for path in
              (tmp_path / "raw" / "pipe-holder" / "events").glob("*.json")]
    assert any(event.get("action") == "SIGKILL_GROUP" for event in events)


def test_detached_grandchild_holding_pipe_cannot_block_log_close(tmp_path):
    pid_path = tmp_path / "detached-grandchild.pid"
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child = (
        "import os,time; "
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    code = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)"
    timeout, cleanup = 1.5, 1.3
    try:
        started = time.monotonic()
        result = _execute(tmp_path, "detached-pipe-holder", code, timeout=timeout, cleanup=cleanup, term=0.15)
        elapsed = time.monotonic() - started
        assert elapsed <= timeout + cleanup
        assert result["arm_execution"] == ArmExecution.FAILED.value
        assert result["root_cause"] == "WORKER_DESCENDANT_REQUIRED_CLEANUP"
        assert pid_path.is_file()
        assert not result.get("remaining_process_group_pids")
        assert unrelated.poll() is None
    finally:
        if pid_path.is_file():
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=2)


def test_supervisor_exit_after_started_is_unknown_not_timeout_and_is_not_replayed(tmp_path, monkeypatch):
    python = sys.executable
    wrapper = tmp_path / "supervisor-crash"
    marker = tmp_path / "supervisor-start-count"
    wrapper.write_text(textwrap.dedent(f"""\
        #!{python}
        import hashlib, json, os, subprocess, sys, time
        from pathlib import Path

        args = sys.argv
        request_path = Path(args[args.index('--supervisor-request') + 1])
        request = json.loads(request_path.read_text())
        folder = Path(request['results_root']) / 'raw' / request['attempt_id']
        identity = {{**request['identity'], 'attempt_id': request['attempt_id'],
                    'gpu_uuid': request['gpu_uuid'],
                    'spec_digest': hashlib.sha256(json.dumps(request['spec'], sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()}}
        with open({str(marker)!r}, 'a', encoding='utf-8') as stream:
            stream.write('started\\n')
        child = subprocess.Popen(
            [sys.executable, '-c',
             'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        event_dir = folder / 'events'
        event_dir.mkdir(parents=True, exist_ok=True)
        events = [
            {{'sequence': 0, 'kind': 'intent', 'identity': identity,
             'monotonic_seconds': time.monotonic(), 'unix_seconds': time.time(),
             'command': request['command']}},
            {{'sequence': 1, 'kind': 'started', 'identity': identity,
             'monotonic_seconds': time.monotonic(), 'unix_seconds': time.time(),
             'pid': child.pid, 'process_group_id': child.pid}},
        ]
        for sequence, event in enumerate(events):
            (event_dir / f"{{sequence:04d}}-{{event['kind']}}.json").write_text(json.dumps(event))
        os._exit(23)
    """), encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(process_supervisor.sys, "executable", str(wrapper))

    first = _execute(tmp_path, "supervisor-crash", "pass", timeout=3.0, cleanup=0.8, term=0.2)
    second = _execute(tmp_path, "supervisor-crash", "pass", timeout=3.0, cleanup=0.8, term=0.2)

    assert first["arm_execution"] == ArmExecution.UNKNOWN.value
    assert first["first_failure_stage"] == "HOST_SUPERVISOR_EXIT"
    assert first["root_cause"] == "SUPERVISOR_EXIT_23_WITHOUT_TERMINAL"
    assert first["supervisor_returncode"] == 23
    assert first["cleanup_confirmed"] is True
    assert second["arm_execution"] == ArmExecution.UNKNOWN.value
    assert second["first_failure_stage"] == "HOST_SUPERVISOR_EXIT"
    assert marker.read_text(encoding="utf-8").splitlines() == ["started"]
    recovered = recover_attempt(tmp_path / "raw" / "supervisor-crash", first["identity"])
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert recovered["root_cause"] == "SUPERVISOR_EXIT_23_WITHOUT_TERMINAL"
    assert recovered["cleanup_confirmed"] is True


def test_worker_start_gate_keeps_child_idle_until_started_event_is_durable(tmp_path, monkeypatch):
    marker = tmp_path / "worker-ran"
    real_record_event = process_supervisor.record_event

    def fail_started_event(folder, kind, payload):
        if kind == "started":
            raise OSError("injected started-event publication failure")
        return real_record_event(folder, kind, payload)

    monkeypatch.setattr(process_supervisor, "record_event", fail_started_event)
    with pytest.raises(OSError, match="injected started-event publication failure"):
        _execute_arm_inner(
            results_root=tmp_path, attempt_id="start-gate",
            command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
            spec={"arm": "S0"}, identity={"source_commit": "test-source"},
            gpu_uuid="GPU-test", timeout_seconds=1.0, cleanup_grace_seconds=0.5,
            term_grace_seconds=0.1,
        )

    assert not marker.exists()
    events = [json.loads(path.read_text()) for path in
              (tmp_path / "raw" / "start-gate" / "events").glob("*.json")]
    assert [event["kind"] for event in events] == ["intent", "terminal"]


def test_supervisor_death_before_started_event_cannot_release_worker(tmp_path):
    marker = tmp_path / "worker-ran-before-started"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(textwrap.dedent("""\
        import os
        import sys

        if "--supervisor-request" in sys.argv:
            from experiments.real_intervention_v4 import outcomes
            record_event = outcomes.record_event

            def crash_before_started_is_durable(folder, kind, payload):
                if kind == "started":
                    os._exit(23)
                return record_event(folder, kind, payload)

            outcomes.record_event = crash_before_started_is_durable
    """), encoding="utf-8")
    pythonpath = os.pathsep.join(
        filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))
    )

    result = _execute(
        tmp_path, "supervisor-died-before-started",
        f"open({str(marker)!r}, 'w').close()",
        environment={"PYTHONPATH": pythonpath},
    )

    events = [json.loads(path.read_text()) for path in
              (tmp_path / "raw" / "supervisor-died-before-started" / "events").glob("*.json")]
    assert result["arm_execution"] == ArmExecution.UNKNOWN.value
    assert result["first_failure_stage"] == "HOST_SUPERVISOR_EXIT"
    assert result["supervisor_returncode"] == 23
    assert [event["kind"] for event in events] == ["intent"]
    assert not marker.exists()


def test_exception_cleanup_kills_worker_group_with_sigterm_ignoring_grandchild(tmp_path, monkeypatch):
    code = textwrap.dedent(
        """
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        subprocess.Popen([sys.executable, "-c", child])
        time.sleep(30)
        """
    )
    attempt_id = "exception-cleanup"
    folder = tmp_path / "raw" / attempt_id
    identity = {"source_commit": "test-source", "config_digest": "config",
                "context_digest": "context"}
    spec = {"arm": "S0", "context": "test"}
    timeout, cleanup = 2.0, 1.0
    started = time.monotonic()

    def fail_drain(*_args):
        raise RuntimeError("injected drain failure")

    with monkeypatch.context() as patch:
        patch.setattr("experiments.real_intervention_v4.process_supervisor._drain", fail_drain)
        with pytest.raises(RuntimeError, match="injected drain failure"):
            _execute_arm_inner(
                results_root=tmp_path, attempt_id=attempt_id,
                command=[sys.executable, "-c", code], spec=spec, identity=identity,
                gpu_uuid="GPU-test", timeout_seconds=timeout, cleanup_grace_seconds=cleanup,
                term_grace_seconds=0.15, heartbeat_seconds=0.1, sample_seconds=0.1,
                environment={"V4_RESULT_PATH": str(folder / "result.json")},
                monitor_fixture=[{"gpu_uuid": "GPU-test", "compute_processes": []}],
            )

    elapsed = time.monotonic() - started
    events = [json.loads(path.read_text()) for path in (folder / "events").glob("*.json")]
    group_id = next(event["process_group_id"] for event in events if event["kind"] == "started")
    assert elapsed <= timeout + cleanup
    assert not _process_group_pids(group_id)


def test_continuous_stdout_and_stderr_are_log_capped_and_deadline_bounded(tmp_path):
    code = textwrap.dedent(
        """
        import os
        chunk = b'x' * 65536
        while True:
            os.write(1, chunk)
            os.write(2, chunk)
        """
    )
    timeout, cleanup = 1.0, 1.4
    started = time.monotonic()
    result = _execute(tmp_path, "output-flood", code, timeout=timeout, cleanup=cleanup,
                      term=0.15, max_log_bytes=2048)

    assert time.monotonic() - started <= timeout + cleanup
    assert result["arm_execution"] == ArmExecution.TIMEOUT.value
    assert result["within_deadline_plus_grace"]
    for name in ("stdout", "stderr"):
        capture = result["log_capture"][name]
        assert capture["bytes_saved"] <= 2048
        assert capture["truncated"]


def test_corrupt_worker_json_is_terminal_failure_not_success(tmp_path):
    code = "import os; open(os.environ['V4_RESULT_PATH'], 'w').write('{')"
    result = _execute(tmp_path, "partial-json", code)

    assert result["arm_execution"] == ArmExecution.FAILED.value
    assert result["first_failure_stage"] == "WORKER_RESULT_RECEIPT"
    assert result["root_cause"] == "WORKER_RESULT_INVALID:JSONDecodeError"
    recovered = recover_attempt(tmp_path / "raw" / "partial-json", result["identity"])
    assert recovered["arm_execution"] == ArmExecution.FAILED.value
    assert recovered["receipt_verified"] is True
    assert recovered["worker_result"] is None


def test_memory_exhaustion_is_infrastructure_failure_not_candidate_harm(tmp_path):
    code = textwrap.dedent(
        f"""
        import json, os, resource, time
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
        try:
            bytearray(512 * 1024 * 1024)
        except MemoryError:
            payload = {{"status": "error", "error_type": "MemoryError",
                       "finished_monotonic": time.monotonic()}}
            with open(os.environ["V4_RESULT_PATH"], "w") as stream:
                json.dump(payload, stream)
        """
    )
    result = _execute(tmp_path, "memory-limit", code)

    assert result["arm_execution"] == ArmExecution.FAILED.value
    assert result["scientific_correctness"] == "UNASSESSED"
    assert result["root_cause"] == "MemoryError"


def test_duplicate_supervisor_does_not_launch_two_workers(tmp_path):
    marker = tmp_path / "single-worker"
    code = (f"import os\nfd = os.open({str(marker)!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
            "os.close(fd)\n" + _worker_result())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_execute, tmp_path, "duplicate", code) for _ in range(2)]
        results = [future.result(timeout=8) for future in futures]

    assert marker.is_file()
    assert all(result.get("arm_execution") in {
        ArmExecution.COMPLETE.value, ArmExecution.UNKNOWN.value, ArmExecution.NOT_STARTED.value,
    } for result in results)
    assert all(result.get("arm_execution") in {"COMPLETE", "UNKNOWN", "NOT_STARTED"}
               for result in results)


def test_runner_refuses_to_start_while_cache_initialization_holds_attempt_lock(tmp_path, monkeypatch):
    attempt = tmp_path / "raw" / "cache-race"
    attempt.mkdir(parents=True)
    initialization_entered = threading.Event()
    allow_initialization = threading.Event()
    original_probe = cache_lifecycle._probe

    def blocked_probe(path):
        if not initialization_entered.is_set():
            initialization_entered.set()
            if not allow_initialization.wait(timeout=5):
                raise TimeoutError("test did not release cache initialization")
        original_probe(path)

    marker = tmp_path / "worker-started-during-cache-setup"
    code = f"open({str(marker)!r}, 'w').close()\n" + _worker_result()
    monkeypatch.setattr(cache_lifecycle, "_probe", blocked_probe)
    with ThreadPoolExecutor(max_workers=1) as pool:
        cache_future = pool.submit(create_cache_scope, attempt)
        try:
            assert initialization_entered.wait(timeout=2)
            result = _execute(tmp_path, "cache-race", code)
            assert result["arm_execution"] == ArmExecution.NOT_STARTED.value
            assert not marker.exists()
        finally:
            allow_initialization.set()
        record = cache_future.result(timeout=5)

    assert record["cache_state"] == "EMPTY"


def test_restart_with_started_nonterminal_attempt_never_relaunches(tmp_path):
    attempt_id = "interrupted"
    spec = {"arm": "S0", "context": "test"}
    identity = {
        "source_commit": "test-source", "config_digest": "config",
        "context_digest": "context", "attempt_id": attempt_id,
        "gpu_uuid": "GPU-test", "spec_digest": canonical_digest(spec),
    }
    folder = tmp_path / "raw" / attempt_id
    record_event(folder, "intent", {"identity": identity, "command": ["must-not-run"]})
    record_event(folder, "started", {"identity": identity, "pid": 12345,
                                      "process_group_id": 12345})
    marker = tmp_path / "replayed"

    result = execute_arm(
        results_root=tmp_path, attempt_id=attempt_id,
        command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        spec=spec, identity={key: value for key, value in identity.items()
                             if key not in {"attempt_id", "gpu_uuid", "spec_digest"}},
        gpu_uuid="GPU-test", timeout_seconds=2.0, cleanup_grace_seconds=0.5,
        term_grace_seconds=0.1,
    )

    assert result["arm_execution"] == ArmExecution.UNKNOWN.value
    assert not marker.exists()


def test_host_timeout_preserves_unknown_after_durable_worker_start(tmp_path):
    attempt_id = "started-without-terminal"
    spec = {"arm": "S0"}
    identity = {
        "source_commit": "test-source", "config_digest": "config",
        "context_digest": "context", "attempt_id": attempt_id,
        "gpu_uuid": "GPU-test", "spec_digest": canonical_digest(spec),
    }
    folder = tmp_path / "raw" / attempt_id
    record_event(folder, "intent", {"identity": identity, "command": ["worker"]})
    record_event(folder, "started", {"identity": identity, "pid": 9002,
                                      "process_group_id": 9002})

    result = _terminalize_watchdog_timeout(
        folder, identity, supervisor_returncode=-9,
        elapsed_seconds=1.0, ceiling_seconds=1.0,
    )

    assert result["arm_execution"] == ArmExecution.UNKNOWN.value
    assert not (folder / "receipt.json").exists()
    events = [json.loads(path.read_text()) for path in (folder / "events").glob("*.json")]
    assert [event["kind"] for event in events] == ["intent", "started"]
