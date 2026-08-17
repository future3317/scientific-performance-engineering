"""Run the ReferenceExecutor contract and write CI attestation artifacts.

This is an attestation entry point, not a second executor implementation.  It
exercises the production ``ReferenceExecutor`` against a tiny real worker and
fails closed when bubblewrap or any observed isolation check is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmark.formal.reference_executor import ReferenceExecutor


def _digest_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root), text=True,
        capture_output=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _bwrap_version(executable: str) -> str | None:
    if shutil.which(executable) is None:
        return None
    completed = subprocess.run([executable, "--version"], text=True, capture_output=True, check=False)
    value = (completed.stdout or completed.stderr).strip()
    return value or None


def _environment_fingerprint(root: Path, *, executor_digest: str, skill_digest: str, receipt_schema: int) -> dict[str, Any]:
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_runtime = {
            "torch_cuda": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
        }
    except ImportError:
        torch_version = None
        cuda_runtime = {"torch_cuda": None, "available": False}
    task_files = sorted((root / "benchmark" / "tasks").glob("*/task.yaml"))
    population_path = root / "benchmark" / "population_report.json"
    statistical: dict[str, Any]
    try:
        from core.acre.budget import StatisticalBudget
        statistical = StatisticalBudget().to_dict()
    except (ImportError, AttributeError):
        statistical = {"status": "unavailable"}
    return {
        "git_commit_sha": _git_sha(root),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "pytorch": torch_version,
        "numpy": numpy_version,
        "cuda_runtime": cuda_runtime,
        "bubblewrap": _bwrap_version("bwrap"),
        "executor_digest": executor_digest,
        "skill_view_digest": skill_digest,
        "population_digest": hashlib.sha256(population_path.read_bytes()).hexdigest() if population_path.is_file() else None,
        "task_manifest_digest": _digest_files(task_files, root / "benchmark" / "tasks") if task_files else None,
        "statistical_configuration": statistical,
        "executor_receipt_schema": receipt_schema,
    }


def run(out_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = ReferenceExecutor()
    if not executor.available():
        result = {"status": "blocked", "reason": "bubblewrap unavailable", "execution_source": "reference_executor"}
        (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(result["reason"])

    with tempfile.TemporaryDirectory(prefix="acre-executor-contract-", dir=str(Path.home())) as raw_root:
        worker_root = Path(raw_root)
        for name in ("task", "skill_view", "retrieved_context", "context_state", "solution", "result", "executor_receipt"):
            (worker_root / name).mkdir()
        (worker_root / "task" / "public_task.json").write_text('{"task_id":"executor-contract"}\n', encoding="utf-8")
        (worker_root / "skill_view" / "README.md").write_text("read-only calibration skill view\n", encoding="utf-8")
        worker_code = (
            "from pathlib import Path; "
            "Path('/worker/solution/worker-output.json').write_text('{\\\"ok\\\":true}'); "
            "Path('/worker/result/worker_result.json').write_text('{\\\"status\\\":\\\"booted\\\"}'); "
            "Path('/worker/result/agent_usage.json').write_text('{\\\"input_tokens\\\":0,\\\"output_tokens\\\":0,\\\"tool_calls\\\":0,\\\"wall_time_s\\\":0.0}')"
        )
        receipt_path = worker_root / "executor_receipt" / "receipt.json"
        completed = executor.execute(
            [sys.executable, "-c", worker_code], worker_root,
            receipt_path=receipt_path, worker_uid="executor-contract",
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        worker_result = worker_root / "result" / "worker_result.json"
        solution_output = worker_root / "solution" / "worker-output.json"
        checks = receipt.get("canary_checks", {}) if isinstance(receipt, dict) else {}
        required_checks = ("python_started", "network_blocked", "readonly_enforced", "host_path_hidden", "benchmark_root_hidden", "nonallowlist_hidden", "writable_dirs")
        passed = (
            completed.returncode == 0
            and worker_result.is_file()
            and solution_output.is_file()
            and receipt.get("schema_version") == 1
            and receipt.get("network_mode") == "none"
            and receipt.get("isolation_canary") is True
            and receipt.get("mount_receipt", {}).get("verified") is True
            and all(checks.get(key) is True for key in required_checks)
        )
        receipt_copy = out_dir / "executor_receipt.json"
        receipt_copy.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        skill_digest = hashlib.sha256((worker_root / "skill_view" / "README.md").read_bytes()).hexdigest()
        fingerprint = _environment_fingerprint(
            root,
            executor_digest=str(receipt.get("executor_digest", "unknown")),
            skill_digest=skill_digest,
            receipt_schema=int(receipt.get("schema_version", 0)),
        )
        (out_dir / "environment_fingerprint.json").write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        calibration_manifest = {
            **fingerprint,
            "calibration_status": "pending_fail_closed",
            "formal_50": "not_generated",
            "formal_efficacy": "not_claimed",
            "executor_contract": "passed",
        }
        (out_dir / "calibration_environment_manifest.json").write_text(
            json.dumps(calibration_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        result = {
            "status": "pass" if passed else "fail",
            "execution_source": "reference_executor",
            "worker_returncode": completed.returncode,
            "worker_result_written": worker_result.is_file(),
            "solution_written": solution_output.is_file(),
            "receipt_verified": bool(receipt.get("isolation_canary") and receipt.get("mount_receipt", {}).get("verified")),
            "canary_checks": {key: bool(checks.get(key, False)) for key in required_checks},
        }
        (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("ReferenceExecutor contract failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.out_dir)
    except RuntimeError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
