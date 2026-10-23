import os
import signal
import subprocess
import sys
import time

import pytest

from experiments.real_intervention_v5.common import write_json
from experiments.real_intervention_v5.runtime import _terminate_group
from experiments.real_intervention_v4.process_supervisor import _enable_child_subreaper, _process_group_pids
from experiments.real_intervention_v6.execution_adapter import primary_status, run_arm, publish_cache


def valid():
    return {"execution_status": "COMPLETE", "scientific_correctness": "PASS", "workload_identity": "MATCH",
            "worker_cuda_provenance": "VERIFIED", "co_tenancy_exposure": "VERIFIED_QUIET", "measurement_contract": "COMPARABLE"}


@pytest.mark.parametrize("field,value,status", [
    ("scientific_correctness", "FAIL", "SCIENTIFIC_UNSAFE"),
    ("worker_cuda_provenance", "UNKNOWN", "INFRA_INVALID"),
    ("worker_cuda_provenance", "CONTRADICTED", "INFRA_INVALID"),
    ("co_tenancy_exposure", "UNKNOWN", "EXPOSURE_UNCERTAIN"),
    ("co_tenancy_exposure", "OBSERVED_FOREIGN", "INFRA_INVALID"),
    ("measurement_contract", "NOT_COMPARABLE", "INFRA_INVALID"),
    ("workload_identity", "MISMATCH", "INFRA_INVALID"),
    ("execution_status", "UNKNOWN", "UNKNOWN"),
])
def test_orthogonal_invalidity(field, value, status):
    assert primary_status({**valid(), field: value}) == status


def test_started_without_terminal_is_unknown_and_never_replayed(tmp_path):
    write_json(tmp_path / "intent.json", {"started": True})
    assert run_arm({}, tmp_path)["primary_pair_status"] == "UNKNOWN"
    assert not (tmp_path / "started.json").exists()


def test_real_group_deadline_with_descendant():
    _enable_child_subreaper()
    code = "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)']); print('ready',flush=True); time.sleep(600)"
    process = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, start_new_session=True, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        start = time.monotonic()
        _terminate_group(process, process.pid)
        assert time.monotonic() - start < 3.5
        assert not _process_group_pids(process.pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_cache_publication_is_atomic_and_no_overwrite(tmp_path):
    source, published = tmp_path / "private", tmp_path / "published"
    source.mkdir()
    write_json(source / "compiler-entry.json", {"identity": "test-fixture"})
    publish_cache(source, published, {"gpu": "test"})
    assert published.is_dir() and not source.exists()
    assert published.with_suffix(".manifest.json").exists()
    with pytest.raises(FileExistsError):
        publish_cache(source, published, {})


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "value"])
def test_real_output_checks_reject_unsafe(bad):
    import torch
    from experiments.real_intervention_v5.worker import check
    reference = torch.ones((2, 2), dtype=torch.float32)
    output = {"shape": torch.ones(3), "dtype": torch.ones((2, 2), dtype=torch.float64),
              "nan": torch.full((2, 2), float("nan")), "value": torch.zeros((2, 2))}[bad]
    result = check(reference, output, {"rtol": .02, "atol": .002, "relative_l2_max": .02})
    assert not result["passed"]
