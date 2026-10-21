"""Controlled own-process faults; never targets another user's process."""
import argparse
import os
from pathlib import Path
import signal
import time

from .common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["ignore-term", "contender", "interrupt"], required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--ack-path", type=Path)
    p.add_argument("--trigger", type=Path)
    args = p.parse_args()
    if args.trigger:
        limit = time.monotonic() + 30
        while not args.trigger.exists() and time.monotonic() < limit:
            time.sleep(.05)
    import torch
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    if args.kind == "contender":
        end = time.monotonic() + 8
        while time.monotonic() < end:
            x = x.sin()
            torch.cuda.synchronize()
        return
    write_json(args.output.with_suffix(".ready.json"), {"pid": os.getpid(), "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"]})
    if args.kind == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(.1)
    while not args.output.with_suffix(".go").exists():
        time.sleep(.02)
    os._exit(17)


if __name__ == "__main__":
    main()
