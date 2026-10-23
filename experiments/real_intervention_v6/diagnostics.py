"""Read-only V5 audit and one bounded wait for a safe real CUDA sentinel."""
import argparse
from pathlib import Path
import time

from experiments.real_intervention_v5.common import append_jsonl, write_json, utc
from experiments.real_intervention_v5.run_all import preflight
from .audit_v5 import audit
from .gpu_attestation import idle_device, probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=3600)
    args = parser.parse_args()
    sha = preflight()
    args.root.mkdir(parents=True, exist_ok=False)
    write_json(args.root / "SOURCE.json", {"source_sha": sha, "timestamp": utc()})
    print(audit(args.root), flush=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        gpu, inventory = idle_device()
        append_jsonl(args.root / "RESOURCE_SCANS.jsonl", {"timestamp": utc(), "gpus": inventory})
        if gpu:
            report = probe(args.root / "sentinel", gpu["uuid"])
            print({k: report[k] for k in ("worker_cuda_provenance", "co_tenancy_exposure", "externally_visible_owner")}, flush=True)
            return
        if time.monotonic() >= deadline:
            write_json(args.root / "BLOCKER.json", {"status": "NO_IDLE_GPU", "timestamp": utc(), "gpus": inventory,
                       "next_condition": "One observed idle GPU (no compute process, <=512MiB, utilization0); no user process may be interrupted."})
            raise SystemExit(3)
        write_json(args.root / "PROGRESS.json", {"phase": "WAITING_FOR_IDLE_GPU", "timestamp": utc()})
        time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
