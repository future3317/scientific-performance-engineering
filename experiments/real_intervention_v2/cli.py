from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def census(output: str | Path, project_root: str | Path) -> dict:
    import torch

    root = Path(project_root).resolve()
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    gpus = []
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if query.returncode == 0:
            gpus = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_head": revision,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "packages": {name: _version(name) for name in ("torch", "torchvision", "torch-geometric", "torchbenchmark")},
        "torch": {"version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()},
        "gpus": gpus,
        "filesystem_available_bytes": {str(path): os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
                                       for path in ("/home/workspace", "/home/workspace/lrh/CODE") if Path(path).exists()},
        "torchbench_installed": _version("torchbenchmark") is not None,
        "provider_calls": 0,
        "performance_measurements": 0,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def analyze(results: str | Path, manifest_path: str | Path, policy_path: str | Path,
            output: str | Path) -> dict:
    from .replay import replay

    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((Path(results) / "pairs").glob("*.json"))]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    report = replay(records, manifest, policy)
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="experiments.real_intervention_v2.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    census_parser = commands.add_parser("census", help="read environment identity without running a model")
    census_parser.add_argument("--project-root", default=".")
    census_parser.add_argument("--out", required=True)
    analyze_parser = commands.add_parser("analyze", help="replay frozen policies from saved receipts only")
    analyze_parser.add_argument("--results", required=True)
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--policy", required=True)
    analyze_parser.add_argument("--out", required=True)
    analyze_parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    if args.command == "census":
        census(args.out, args.project_root)
    else:
        if not args.offline_only:
            parser.error("analysis requires --offline-only")
        analyze(args.results, args.manifest, args.policy, args.out)


if __name__ == "__main__":
    main()
