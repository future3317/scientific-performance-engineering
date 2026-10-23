import json
import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from experiments.real_intervention_v4.process_supervisor import (
    _enable_child_subreaper,
    _process_group_pids,
)
from experiments.real_intervention_v5.common import read_json, write_json
from experiments.real_intervention_v6 import study
from experiments.real_intervention_v6.execution_adapter import publish_cache, run_arm


def _reap_group(group_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-group_id, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.02)
    assert not _process_group_pids(group_id)


def test_crashed_supervisor_is_unknown_and_not_restarted(tmp_path):
    _enable_child_subreaper()
    folder = tmp_path / "arm"
    script = r'''
import subprocess, sys
from experiments.real_intervention_v6 import execution_adapter as adapter

real_popen = subprocess.Popen
def cpu_snapshot(_uuid):
    return {"memory_used_bytes": 0, "util_percent": 0,
            "nvml": {"cpu-fixture": {"rows": [], "status": "SUCCESS"}}}
adapter.snapshot = cpu_snapshot
def start_cpu_worker(*_args, **kwargs):
    child = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)']); time.sleep(600)"
    return real_popen([sys.executable, "-c", child], **kwargs)
adapter.subprocess.Popen = start_cpu_worker
adapter.run_arm({"gpu_uuid": "CPU-fixture", "source_sha": "fixture", "timeout_s": 60}, sys.argv[1])
'''
    supervisor = subprocess.Popen(
        [sys.executable, "-c", script, str(folder)],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker_group = None
    try:
        deadline = time.monotonic() + 10
        started_path = folder / "started.json"
        while time.monotonic() < deadline and not started_path.exists():
            if supervisor.poll() is not None:
                pytest.fail(f"supervisor exited before worker start: {supervisor.stderr.read()}")
            time.sleep(0.02)
        assert started_path.is_file()
        started_bytes = started_path.read_bytes()
        worker_identity = json.loads(started_bytes)
        worker_group = worker_identity["pgid"]
        assert worker_group == worker_identity["pid"]
        assert worker_group != supervisor.pid
        assert _process_group_pids(worker_group)

        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=3)
        os.killpg(worker_group, signal.SIGKILL)
        _reap_group(worker_group)
        assert not _process_group_pids(worker_group)

        result = run_arm({}, folder)
        assert result["execution_status"] == "UNKNOWN"
        assert result["primary_pair_status"] == "UNKNOWN"
        assert result["error"] == "STARTED_IDENTITY_NO_RETRY"
        assert started_path.read_bytes() == started_bytes
    finally:
        if supervisor.poll() is None:
            os.kill(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=3)
        if worker_group is not None and _process_group_pids(worker_group):
            os.killpg(worker_group, signal.SIGKILL)
            _reap_group(worker_group)


def test_published_cache_supports_concurrent_read_only_copies(tmp_path):
    source = tmp_path / "private-cache"
    source.mkdir()
    (source / "entry.bin").write_bytes(b"compiler-cache-fixture\x00\x01")
    published = Path(publish_cache(source, tmp_path / "published-cache", {"fixture": "CPU"}))
    manifest_path = published.with_suffix(".manifest.json")
    published_bytes = (published / "entry.bin").read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_path)
    assert manifest["compatibility"] == {"fixture": "CPU"}

    def copy(index):
        destination = tmp_path / f"worker-copy-{index}"
        import shutil
        shutil.copytree(published, destination)
        return (destination / "entry.bin").read_bytes()

    outputs = [None, None]
    threads = [threading.Thread(target=lambda i=i: outputs.__setitem__(i, copy(i))) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outputs == [published_bytes, published_bytes]
    assert (published / "entry.bin").read_bytes() == published_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert read_json(manifest_path)["members"] == manifest["members"]


@pytest.mark.parametrize("mismatch", ["manifest", "code", "environment", "qualification_hash"])
def test_freeze_rejects_changed_resume_identity(tmp_path, monkeypatch, mismatch):
    old = tmp_path / "old-qualification"
    old.mkdir()
    root = tmp_path / "study"
    root.mkdir()
    snapshot = [{
        "family_id": "family-1",
        "eligibility": "ELIGIBLE",
        "checks": [{"batch": 1, "payload": {
            "weights_sha256": "weights",
            "inputs_sha256": "inputs",
            "reference_sha256": "reference",
            "features": {"layers": 1, "conv2d": 0},
        }}],
    }]
    shakedown = {"pairs": [{"records": [{"family_id": "family-1"}]}]}
    real_read_json = study.read_json

    def fixture_read_json(path):
        path = Path(path)
        if path == old / "QUALIFICATION_SNAPSHOT.json":
            return snapshot
        if path == old / "ENGINEERING_SHAKEDOWN.json":
            return shakedown
        return real_read_json(path)

    state = {"code": "code-a", "version": "1.0", "qualification_hash": "hash-a"}
    monkeypatch.setattr(study, "read_json", fixture_read_json)
    monkeypatch.setattr(study, "old_root", lambda _study: old)
    monkeypatch.setattr(study, "build_cohort", lambda *_args: {
        "sample_feasibility": {"independent_pilot": {"feasible": True}},
        "rows": [{"family_id": "family-1", "split": "development"}],
    })
    monkeypatch.setattr(study, "preflight", lambda: state["code"])
    monkeypatch.setattr(study, "environment_identity", lambda: {"package": state["version"]})
    monkeypatch.setattr(study, "file_digest", lambda _path: state["qualification_hash"])

    config = {
        "study": "A",
        "study_id": "fixture-study",
        "split_seed": 7,
        "construction_seed": 11,
        "horizons": [1],
        "sessions_per_context": 1,
        "precision": "BF16",
        "correctness": {},
        "arm_timeout_s": 10,
        "reported_actual_reuse_counts": [1],
        "projected_reuse_counts": [1],
        "actual_deployment_jobs": 1,
        "idle_wait_s": 1,
    }
    gpu = {"uuid": "GPU-fixture"}
    study.freeze(root, config, gpu)

    if mismatch == "manifest":
        frozen_path = root / "FROZEN_MANIFEST.json"
        corrupted = read_json(frozen_path)
        corrupted["study_id"] = "different-study"
        write_json(frozen_path, corrupted)
    elif mismatch == "code":
        state["code"] = "code-b"
    elif mismatch == "environment":
        state["version"] = "2.0"
    else:
        state["qualification_hash"] = "hash-b"

    with pytest.raises(RuntimeError, match="resume code/env/config/qualification/GPU mismatch"):
        study.freeze(root, config, gpu)
