"""Sequential two-study entry point under a single clean code identity."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import signal
import json

from .common import read_json, write_json, utc
from .run_all import preflight, PYTHON


def combine_results(parent, source_sha, results):
    from .report import bundle
    metrics, summaries = {}, []
    with (parent / "PAIRS.jsonl").open("w") as pairs:
        for result in results:
            root = Path(result["root"])
            key = result["study_id"]
            metrics[key] = read_json(root / "V5_METRICS.json")
            summaries.append((root / "V5_SUMMARY.md").read_text())
            for line in (root / "PAIRS.jsonl").open():
                row = json.loads(line)
                row["study_id"] = key
                pairs.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json(parent / "V5_METRICS.json", {"source_sha": source_sha, "studies": metrics, "study_effects_never_pooled": True})
    (parent / "V5_SUMMARY.md").write_text("\n\n".join(summaries))
    bundle(parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-parent", type=Path, required=True)
    args = parser.parse_args()
    sha = preflight()
    args.results_parent.mkdir(parents=True, exist_ok=True)
    studies = ["V5_LAYOUT_BF16_FROZEN.json", "V5_COMPILE_CACHE_FROZEN.json"]
    results = []
    current = None
    def cancel(signum, frame):
        if current is not None and current.poll() is None:
            current.terminate()
            try:
                current.wait(timeout=10)
            except subprocess.TimeoutExpired:
                current.kill()
                current.wait(timeout=5)
        raise KeyboardInterrupt("campaign canceled")
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    for name in studies:
        config_path = Path(__file__).parent / "config" / name
        config = read_json(config_path)
        root = args.results_parent / config["study_id"]
        command = [PYTHON, "-u", "-m", "experiments.real_intervention_v5.run_all", "--study-config", str(config_path), "--results-root", str(root), "--mode", "full", "--resume-safe"]
        with (args.results_parent / (config["study_id"] + ".log")).open("ab", buffering=0) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            current = process
            result = process.wait()
        results.append({"study_id": config["study_id"], "exit_code": result, "root": str(root), "complete": read_json(root / "RUN_COMPLETE.json") if (root / "RUN_COMPLETE.json").exists() else None})
        write_json(args.results_parent / "CAMPAIGN_PROGRESS.json", {"source_sha": sha, "timestamp": utc(), "studies": results})
        if result:
            # An unhandled implementation/platform failure is a real blocker;
            # preserve receipts and require diagnosis, never silently retry.
            raise SystemExit(result)
    write_json(args.results_parent / "CAMPAIGN_COMPLETE.json", {"source_sha": sha, "timestamp": utc(), "studies": results})
    combine_results(args.results_parent, sha, results)


if __name__ == "__main__":
    main()
