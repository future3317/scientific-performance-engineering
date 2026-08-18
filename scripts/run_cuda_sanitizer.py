#!/usr/bin/env python3
"""Run a real CUDA command under Compute Sanitizer and emit a fail-closed gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def run_tool(tool: str, command: list[str], timeout: float) -> dict[str, Any]:
    argv = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "1", "--target-processes", "all", *command]
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        else:
            process.kill()
        process.wait()
    return {
        "tool": tool,
        "command": argv,
        "returncode": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout": stdout if isinstance(stdout, str) else stdout.decode(errors="replace"),
        "stderr": stderr if isinstance(stderr, str) else stderr.decode(errors="replace"),
        "status": "resource_blocked" if timed_out else ("passed" if process.returncode == 0 else "failed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", action="append", choices=TOOLS, help="tool to run; repeat for multiple tools (default: memcheck)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide a CUDA command after --")
    tools = tuple(args.tool or ("memcheck",))
    record: dict[str, Any] = {
        "schema_version": 1,
        "command": command,
        "tools": list(tools),
        "status": "blocked",
        "results": [],
    }
    if not shutil.which("compute-sanitizer"):
        record["reason"] = "compute-sanitizer is not installed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return 2
    for tool in tools:
        record["results"].append(run_tool(tool, command, args.timeout))
    statuses = [item["status"] for item in record["results"]]
    record["status"] = "passed" if all(item == "passed" for item in statuses) else ("resource_blocked" if "resource_blocked" in statuses else "failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
