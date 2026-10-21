"""Real own-worker timeout, interrupt, and same-card contention acceptance."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time

from .common import read_json


def run_fault_injection(root, config, gpu, cell, row):
    from .runtime import run_session, _process_group_pids
    from .run_all import PYTHON, environment, execute, make_spec
    from .validity import classify_pair
    directory = root / "fault_injection"
    directory.mkdir(exist_ok=True)
    tests = {}
    for kind in ("ignore-term", "interrupt"):
        location = directory / kind
        terminal = location / "terminal.json"
        command = [PYTHON, "-u", "-m", "experiments.real_intervention_v5.fault_worker", "--kind", kind,
                   "--output", "{output}", "--ack-path", "{ack}"]
        result = read_json(terminal) if terminal.exists() else run_session([command, command], location, gpu["uuid"], 15.0, environment())
        groups = [a.get("worker_pgid") for a in result.get("arms", []) if a.get("worker_pgid")]
        tests[kind] = {"passed": result["analysis_status"] != "VALID_PAIRED" and all(not _process_group_pids(g) for g in groups),
                       "execution_status": result["execution_status"], "error_code": result.get("error_code"),
                       "session_wall_s": result.get("timings", {}).get("session_wall_seconds"), "remaining_owned_pids": [p for g in groups for p in _process_group_pids(g)]}
        if kind == "ignore-term":
            tests[kind]["passed"] &= tests[kind]["session_wall_s"] <= 18.0
        # Started sessions are refused, including UNKNOWN/no-result attempts.
        try:
            run_session([command, command], location, gpu["uuid"], 1.0, environment())
        except RuntimeError:
            tests[f"{kind}_no_implicit_retry"] = {"passed": True}
        else:
            tests[f"{kind}_no_implicit_retry"] = {"passed": False}

    q = next(c["payload"] for c in row["checks"] if c["batch"] == cell["batch"])
    scope = "controlled-contention"
    location = directory / scope
    specs = [make_spec(row, config, gpu, root, batch=cell["batch"], horizon=16, mode="engineering", arm=arm,
                       scope=scope, qualification=q) for arm in ("S0", "S1")]
    for spec in specs:
        spec["split"] = "engineering"
        spec["study"] = "A"  # same precision/layout engineering workload in either study
    contender = []
    def inject():
        trigger = location / "arm-0.result.go"
        # with_suffix('.go') on arm-0.result.json yields arm-0.result.go.
        limit = time.monotonic() + 30
        while not trigger.exists() and time.monotonic() < limit:
            time.sleep(.02)
        if not trigger.exists(): return
        env = os.environ.copy()
        env.update(environment())
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        process = subprocess.Popen([PYTHON, "-u", "-m", "experiments.real_intervention_v5.fault_worker", "--kind", "contender"], env=env, start_new_session=True)
        contender.append(process)
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            from .runtime import _terminate_group
            _terminate_group(process, process.pid)
    terminal = location / "terminal.json"
    if terminal.exists():
        result = read_json(terminal)
    else:
        thread = threading.Thread(target=inject)
        thread.start()
        result = execute(specs, location, config, gpu)
        thread.join(timeout=60)
    tests["controlled_same_gpu_process"] = {"passed": result["exposure_status"] == "CONTENDED",
                                           "exposure": result.get("exposure"), "analysis_status": result["analysis_status"]}
    good = {"execution_status": "COMPLETE", "scientific_status": "PASS"}
    mismatch = classify_pair([good, good], {"status": "MATCHED", "gpu_uuids": [gpu["uuid"], "GPU-deliberately-different"]})
    delayed = classify_pair([good, good], {"status": "INCOMPLETE", "telemetry_complete": False})
    unsafe = classify_pair([good, {**good, "scientific_status": "NUMERICAL_UNSAFE"}], {"status": "MATCHED"})
    tests["different_uuid"] = {"passed": mismatch["analysis_status"] == "NONCOMPARABLE"}
    tests["delayed_sampling_not_harm"] = {"passed": delayed["scientific_status"] == "PASS" and delayed["analysis_status"] != "VALID_PAIRED"}
    tests["unsafe_even_if_fast"] = {"passed": unsafe["scientific_status"] == "NUMERICAL_UNSAFE" and unsafe["analysis_status"] != "VALID_PAIRED"}
    return {"passed": all(x["passed"] for x in tests.values()), "tests": tests,
            "scope": "Controlled own-process faults on selected GPU; classification-only tests identified separately; no actual wedged GPU kernel claimed."}
