from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

from .contracts import digest, load_manifest
from .paired_executor import run_pair
from .receipts import read_receipts
from .replay import aggregate_sessions, freeze_policy, replay


PYTHON = "/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python"
PER_ARM_TIMEOUT_SECONDS = 1800
WARMUPS = 2
WARM_REPETITIONS = 10


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _elapsed_since(iso: str) -> float:
    started = datetime.fromisoformat(iso)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _verify_source(project: Path, manifest: dict[str, Any], torchbench_root: Path,
                   binding: dict[str, Any]) -> str:
    source = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    benchmark = subprocess.run(["git", "-C", str(torchbench_root), "rev-parse", "HEAD"],
                               check=True, capture_output=True, text=True).stdout.strip()
    if source != manifest["source_commit"] or benchmark != manifest["torchbench_commit"]:
        raise ValueError("source or external benchmark commit differs from frozen manifest")
    for repository in (project, torchbench_root):
        dirty = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
                               check=True, capture_output=True, text=True).stdout.splitlines()
        non_bytecode = [line for line in dirty if "__pycache__/" not in line and not line.rstrip().endswith(".pyc")]
        if non_bytecode:
            raise ValueError(f"source worktree is not clean: {repository}: {non_bytecode[:5]}")
    if not binding.get("approved") or binding.get("run_identity") != manifest.get("run_identity"):
        raise ValueError("run has no matching explicit authorization binding")
    if binding.get("manifest_digest") != digest(manifest):
        raise ValueError("approval binding does not match frozen manifest")
    if binding.get("provider_calls") != 0:
        raise ValueError("this study has a zero-provider contract")
    return source


def _completed_pairs(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    pairs = {}
    for path in sorted((root / "pairs").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pairs[(str(row["context_id"]), str(row["session_id"]))] = row
    return pairs


def _resource_usage(root: Path, manifest: dict[str, Any], started_at: str) -> dict[str, float | int]:
    caps = manifest["resource_caps"]
    arm_records = []
    for directory in (root / "arms", root / "aa" / "arms"):
        if directory.exists():
            arm_records.extend(json.loads(path.read_text(encoding="utf-8"))
                               for path in directory.glob("*.json"))
    arms = arm_records
    gpu_seconds = sum(float(arm.get("parent_wall_s", 0.0)) for arm in arms)
    cpu_seconds = sum(float(arm.get("child_cpu_s", 0.0)) for arm in arms)
    return {
        "wall_seconds": _elapsed_since(started_at),
        "gpu_seconds": gpu_seconds if manifest["device"] == "cuda" else 0.0,
        "cpu_seconds": cpu_seconds,
        "storage_bytes": _bytes(root),
        "download_bytes": int(manifest.get("benchmark_source_bytes", 0))
                           + (_bytes(root / "model-cache") if (root / "model-cache").exists() else 0),
        "caps": caps,
    }


def _within_caps(usage: dict[str, Any]) -> bool:
    return all(float(usage[key]) < int(usage["caps"][key])
               for key in ("wall_seconds", "gpu_seconds", "cpu_seconds", "storage_bytes", "download_bytes"))


def _missing_sessions(contexts, completed):
    return [(context, session) for context in contexts for session in context["sessions"]
            if (context["context_id"], session["session_id"]) not in completed]


def _run_sessions(contexts, *, root: Path, project: Path, manifest: dict[str, Any],
                   started_at: str, state: dict[str, Any]) -> bool:
    completed = _completed_pairs(root)
    for context, session in _missing_sessions(contexts, completed):
        usage = _resource_usage(root, manifest, started_at)
        if not _within_caps(usage):
            state.update({"status": "resource_bounded_partial", "last_context_id": context["context_id"],
                          "last_session_id": session["session_id"], "resource_usage": usage})
            _atomic_json(root / "RUN_PROGRESS.json", state)
            return False

        def arm_timeout_budget() -> int:
            current = _resource_usage(root, manifest, started_at)
            if any(current[key] >= current["caps"][key]
                   for key in ("storage_bytes", "download_bytes")):
                return 0
            remaining = min(int(current["caps"][key] - current[key])
                            for key in ("wall_seconds", "gpu_seconds", "cpu_seconds"))
            return max(0, min(PER_ARM_TIMEOUT_SECONDS, remaining))

        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt = run_pair(
            context, session_id=session["session_id"], identity=manifest["run_identity"],
            project_root=project, python_executable=PYTHON, run_root=root,
            device=manifest["device"], order=session["order"],
            horizon=int(context["planned_horizon"]), warmups=WARMUPS,
            repetitions=WARM_REPETITIONS, rtol=float(manifest["rtol"]),
            atol=float(manifest["atol"]), timeout_seconds=PER_ARM_TIMEOUT_SECONDS,
            storage_cap_bytes=int(manifest["resource_caps"]["storage_bytes"]),
            arm_timeout_budget=arm_timeout_budget,
            storage_scope_root=root,
        )
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt["pair_child_cpu_s"] = ((after.ru_utime + after.ru_stime)
                                        - (before.ru_utime + before.ru_stime))
        completed[(context["context_id"], session["session_id"])] = receipt
        usage = _resource_usage(root, manifest, started_at)
        state.update({"status": "running", "completed_pair_sessions": len(completed),
                      "last_context_id": context["context_id"],
                      "last_session_id": session["session_id"], "last_classification": receipt["classification"],
                      "resource_usage": usage})
        _atomic_json(root / "RUN_PROGRESS.json", state)
        if not _within_caps(usage):
            state["status"] = "resource_bounded_partial"
            _atomic_json(root / "RUN_PROGRESS.json", state)
            return False
    return True


def _run_aa_controls(root: Path, project: Path, manifest: dict[str, Any],
                     started_at: str, state: dict[str, Any]) -> bool:
    contexts = {row["context_id"]: row for row in manifest["contexts"]}
    aa_contexts = [contexts[context_id] for context_id in manifest["aa_control_context_ids"]]
    completed = _completed_pairs(root / "aa")
    for context in aa_contexts:
        if (context["context_id"], "aa-control") in completed:
            continue
        usage = _resource_usage(root, manifest, started_at)
        if not _within_caps(usage):
            state.update({"status": "resource_bounded_in_aa", "resource_usage": usage})
            _atomic_json(root / "RUN_PROGRESS.json", state)
            return False

        def arm_timeout_budget() -> int:
            current = _resource_usage(root, manifest, started_at)
            if any(current[key] >= current["caps"][key]
                   for key in ("storage_bytes", "download_bytes")):
                return 0
            remaining = min(int(current["caps"][key] - current[key])
                            for key in ("wall_seconds", "gpu_seconds", "cpu_seconds"))
            return max(0, min(PER_ARM_TIMEOUT_SECONDS, remaining))

        receipt = run_pair(
            context, session_id="aa-control", identity=manifest["run_identity"],
            project_root=project, python_executable=PYTHON, run_root=root / "aa",
            device=manifest["device"], order="eager_first", horizon=int(context["planned_horizon"]),
            warmups=WARMUPS, repetitions=WARM_REPETITIONS, rtol=float(manifest["rtol"]),
            atol=float(manifest["atol"]), timeout_seconds=PER_ARM_TIMEOUT_SECONDS,
            storage_cap_bytes=int(manifest["resource_caps"]["storage_bytes"]),
            arm_timeout_budget=arm_timeout_budget, arm_names=("eager", "eager_control"),
            storage_scope_root=root,
        )
        completed[(context["context_id"], "aa-control")] = receipt
        usage = _resource_usage(root, manifest, started_at)
        state.update({"status": "running_aa", "completed_aa_controls": len(completed),
                      "last_aa_context_id": context["context_id"], "resource_usage": usage})
        _atomic_json(root / "RUN_PROGRESS.json", state)
        if not _within_caps(usage):
            state["status"] = "resource_bounded_in_aa"
            _atomic_json(root / "RUN_PROGRESS.json", state)
            return False
    return True


def _aa_noise_floor(root: Path) -> dict[str, Any]:
    rows = list(_completed_pairs(root / "aa").values())
    ratios = []
    for row in rows:
        first = row["arms"].get("eager", {})
        second = row["arms"].get("eager_control", {})
        if (first.get("status") == "valid" and second.get("status") == "valid"
                and row.get("correctness", {}).get("passed")):
            ratios.append(abs(math.log(float(first["total_job_s"]) / float(second["total_job_s"]))))
    ordered = sorted(ratios)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
    return {"scheduled": len(rows), "valid": len(ratios), "absolute_log_ratio_samples": ratios,
            "p95_absolute_log_ratio": p95,
            "effective_minimum_log_effect": max(float(p95 or 0.0), float(manifest_min_effect()))}


def manifest_min_effect() -> float:
    return 0.04879016416943205


def run_one_window(run_root: str | Path, *, project_root: str | Path,
                    torchbench_root: str | Path) -> dict[str, Any]:
    root, project, torchbench = Path(run_root).resolve(), Path(project_root).resolve(), Path(torchbench_root).resolve()
    manifest = load_manifest(root / "RUN_MANIFEST.json")
    binding = json.loads((root / "APPROVAL_BINDING.json").read_text(encoding="utf-8"))
    source_commit = _verify_source(project, manifest, torchbench, binding)
    os.environ["PYTHONPATH"] = str(torchbench) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["TORCH_HOME"] = str(root / "model-cache" / "torch")
    os.environ["HF_HOME"] = str(root / "model-cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(root / "model-cache" / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "model-cache" / "huggingface" / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    progress_path = root / "RUN_PROGRESS.json"
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != manifest["run_identity"]:
            raise ValueError("existing run progress belongs to another identity")
    else:
        state = {"run_identity": manifest["run_identity"], "source_commit": source_commit,
                 "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "planned_context_count": len(manifest["contexts"]),
                 "planned_pair_sessions": manifest["planned_pair_sessions"],
                 "completed_pair_sessions": 0}
        _atomic_json(progress_path, state, exclusive=True)
    started_at = state["started_at_utc"]
    training_contexts = [row for row in manifest["contexts"]
                         if row["split"] in {"development", "gate_calibration"}]
    final_contexts = [row for row in manifest["contexts"] if row["split"] == "final_untouched"]
    if not _run_aa_controls(root, project, manifest, started_at, state):
        return state
    aa_noise = _aa_noise_floor(root)
    noise_path = root / "AA_NOISE.json"
    if noise_path.exists():
        if json.loads(noise_path.read_text(encoding="utf-8")) != aa_noise:
            raise ValueError("saved A/A noise estimate differs from immutable control receipts")
    else:
        _atomic_json(noise_path, aa_noise, exclusive=True)
    if aa_noise["valid"] < 2:
        state.update({"status": "aa_control_insufficient_no_s1_started", "aa_noise": aa_noise})
        _atomic_json(progress_path, state)
        return state
    if not _run_sessions(training_contexts, root=root, project=project, manifest=manifest,
                         started_at=started_at, state=state):
        return state

    expected_train = {(row["context_id"], session["session_id"])
                      for row in training_contexts for session in row["sessions"]}
    completed = _completed_pairs(root)
    if not expected_train.issubset(completed):
        state.update({"status": "training_incomplete_no_final_open", "completed_training_sessions": len(expected_train & set(completed))})
        _atomic_json(progress_path, state)
        return state

    frozen_path = root / "POLICY_FROZEN.json"
    if not frozen_path.exists():
        training_rows = [completed[key] for key in sorted(expected_train)]
        train_context_records = aggregate_sessions(training_rows)
        policy = freeze_policy(train_context_records, final_contexts, manifest,
                               aa_noise_floor_log=float(aa_noise["effective_minimum_log_effect"]))
        policy.update({"run_identity": manifest["run_identity"],
                       "training_receipt_count": len(training_rows),
                       "training_context_count": len(train_context_records),
                       "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        policy["policy_digest"] = digest({key: value for key, value in policy.items() if key != "policy_digest"})
        _atomic_json(frozen_path, policy, exclusive=True)
    else:
        policy = json.loads(frozen_path.read_text(encoding="utf-8"))
        if policy.get("run_identity") != manifest["run_identity"]:
            raise ValueError("frozen policy belongs to another run identity")

    if not _run_sessions(final_contexts, root=root, project=project, manifest=manifest,
                         started_at=started_at, state=state):
        state["status"] = "resource_bounded_after_policy_freeze"
        _atomic_json(progress_path, state)
        return state

    all_pairs = list(_completed_pairs(root).values())
    result = replay(all_pairs, manifest, policy)
    _atomic_json(root / "POLICY_RESULTS.json", result, exclusive=True)
    state.update({"status": "completed", "resource_usage": _resource_usage(root, manifest, started_at),
                  "completed_pair_sessions": len(all_pairs), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    _atomic_json(progress_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen real S0/S1 study window")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--torchbench-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run_one_window(args.run_root, project_root=args.project_root,
                                    torchbench_root=args.torchbench_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
