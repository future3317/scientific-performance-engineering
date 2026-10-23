"""Server-only targeted validation receipts; GPU fixtures own a fresh process."""
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time

from experiments.real_intervention_v5.common import write_json, utc
from experiments.real_intervention_v5.run_all import preflight
from .benchmark_timing_contract import run_microfixture
from .gpu_attestation import device_uuid


def micro(output, uuid):
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("microfixture device binding missing")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    actual = device_uuid(torch.cuda.get_device_properties(0).uuid)
    if actual != uuid:
        raise RuntimeError("microfixture physical UUID mismatch")
    cpu = run_microfixture(lambda: time.sleep(.002), clock="HOST_MONOTONIC", repeats=3)
    # Calibrate a bounded native CUDA sleep to about2ms on this device; no new kernel.
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    torch.cuda._sleep(2_000_000)
    end.record()
    torch.cuda.synchronize()
    cycles = int(2_000_000 * 2 / max(.01, begin.elapsed_time(end)))
    cuda = run_microfixture(lambda: torch.cuda._sleep(cycles), clock="CUDA_EVENT", repeats=3)
    records = [{**asdict(r), "host_to_benchmark_ratio": r.host_per_call_seconds / r.median_seconds_per_call} for r in (*cpu, *cuda)]
    passed = all(.5 <= r["host_to_benchmark_ratio"] <= 2 for r in records)
    write_json(output, {"source_sha": preflight(), "gpu_uuid": actual, "records": records, "passed": passed,
                       "cuda_native_sleep_cycles": cycles, "timestamp": utc()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-output", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--cpu-output", type=Path)
    args = parser.parse_args()
    sha = preflight()
    if args.micro_output:
        micro(args.micro_output, args.gpu_uuid)
    else:
        args.cpu_output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        with (args.cpu_output / "pytest.log").open("x") as stream:
            process = subprocess.run(["/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python", "-m", "pytest", "-q", "experiments/real_intervention_v6/tests", "experiments/real_intervention_v5/tests"], stdout=stream, stderr=subprocess.STDOUT)
        write_json(args.cpu_output / "CPU_TESTS.json", {"source_sha": sha, "exit_code": process.returncode, "wall_s": time.monotonic() - start,
            "scope": "V6 and V5 contracts, including direct tests of reused V4 cleanup/numerical helpers. V4 full supervisor is not called by V6; its separate regression run remains recorded, including deadline failures. CPU tests do not certify CUDA", "timestamp": utc()})
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
