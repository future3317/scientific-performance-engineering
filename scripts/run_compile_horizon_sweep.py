#!/usr/bin/env python3
"""Calibration-only paired horizon sweep for the graph-break anchor."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Support both `python -m scripts.run_compile_horizon_sweep` and the
# documented direct-script form.  The latter sets `sys.path[0]` to
# `scripts/`; the repository root must be explicit for the benchmark
# package import below.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.harness import runner, stats


def _load_benchmark(task_dir: Path):
    spec = importlib.util.spec_from_file_location(f"compile_sweep_{task_dir.name}", task_dir / "benchmark.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {task_dir / 'benchmark.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task_dir = args.task_dir
    module = _load_benchmark(task_dir)
    config = {
        "warmup_iterations": 0,
        "repetitions": args.repetitions,
        "min_improvement_percent": 5.0,
        "noise_floor_percent": 2.0,
        "higher_is_better": False,
        "speedup_tripwire": 20.0,
    }
    rows = []
    for horizon in (64, 128, 256, 512, 1024):
        measurement = dict(config, measured_iterations=horizon)
        record = runner.run_paired_measurement(
            module,
            baseline_path=task_dir / "workspace" / "solution.py",
            candidate_path=task_dir / "oracle" / "solution_oracle.py",
            measurement_cfg=measurement,
            seed=1908234210 + horizon,
            device=args.device,
            l2_thrash_between=True,
        )
        verdict = stats.robust_speedup_verdict(
            record["baseline_runs"],
            record["candidate_runs"],
            False,
            5.0,
            2.0,
        )
        rows.append(
            {
                "logical_steps": horizon,
                "repetitions": args.repetitions,
                "baseline_runs": record["baseline_runs"],
                "candidate_runs": record["candidate_runs"],
                "noise_control": "calibrated artifact required for formal verifier; sweep remains diagnostic-only",
                "fixture_build_time_s": record.get("fixture_build_time_s"),
                "fixture_hash_time_s": record.get("fixture_hash_time_s"),
                "verified_speedup": verdict,
                "timing": record.get("timing", []),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "compile_graph_break_horizon_calibration",
                "task_id": task_dir.name,
                "horizons": [64, 128, 256, 512, 1024],
                "rows": rows,
                "formal_population": False,
                "efficacy_claim": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"task_id": task_dir.name, "rows": len(rows), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
