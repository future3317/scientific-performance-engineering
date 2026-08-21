#!/usr/bin/env python3
"""Run or dry-run a complete SPE-EvoBench formal campaign.

Without ``--agent-command`` this writes only a frozen campaign plan and never
claims benchmark results. With a command, the driver gives the agent a fresh
solution workspace for each task and then invokes the immutable verifier.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shlex
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmark.harness import conditions, miniyaml, scoring, stats, verifier
from benchmark.harness.evolution_ledger import EvolutionDecisionLedger
from benchmark.harness.fingerprint import capture_fingerprint
from benchmark.formal import aggregate, attest, budget, schedule
from benchmark.formal.approval import validate_calibration_approval
from benchmark.formal.release_manifest import validate_formal_release, validate_materialized_manifest
from benchmark.formal.condition_adapter import FormalConditionAdapter
from core.public_context import build_public_context
from benchmark.harness.evolution import promote_via_replay
from benchmark.harness.evolution_ledger import CandidateEvidenceLedger
from benchmark.families import EpisodeEnvironmentState, FamilyEnvironment
from scripts.render_skill_view import render_skill_view, validate_skill_view_bundle
from core.models import identifier_digest, validate_identifier, ActionSpec, RawRealizationRecord, RealizationRecord
from core.utility import UTILITY_LOG_SCALE, practical_effect_threshold, utility_effect
from core.acre.cegis import synthesize_applicability, _case_effect_interval
from core.acre.budget import StatisticalBudget


def _workspace_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _trial_compiler_cache(trial_dir: Path, invocation_id: str):
    """Give one verifier invocation an isolated compile cache."""
    safe_id = validate_identifier(invocation_id, "compiler_invocation") or invocation_id
    cache_root = Path(trial_dir) / "compiler-cache" / safe_id
    torchinductor = cache_root / "torchinductor"
    triton = cache_root / "triton"
    torchinductor.mkdir(parents=True, exist_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    previous = {
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
    }
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    try:
        yield {
            "policy": "verifier-invocation-scoped",
            "torchinductor_cache_dir": str(torchinductor),
            "triton_cache_dir": str(triton),
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _timeout_result(task_dir: Path, condition: str, context_mode: str, timeout_s: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task_dir.name,
        "condition": condition,
        "context_mode": context_mode,
        "verdict": "inconclusive",
        "validity": "invalid",
        "correctness_pass": False,
        "scientific_gates": {},
        "task": {"track": "unknown", "family": "unknown", "kind": "positive", "expected_speedup_range": None},
        "diagnosis": {"enabled": False, "predicted": None, "expected": None, "diagnosis_correct": None},
        "anticheat": {"hard_fail": False, "findings": [], "tripwired": False, "canary_tripped": False},
        "verified_speedup": {"median_speedup": None, "verified": False, "inconclusive": True, "reason": "verifier time budget exceeded"},
        "measurement": {},
        "cost": {"wall_time_s": float(timeout_s) + 1e-6, "tokens": None, "tool_calls": None, "retries": 0},
        "errors": ["verifier subprocess exceeded task time budget"],
        "stage_times_s": {},
    }


def _trial_failure_record(manifest: dict[str, Any], task_spec: Mapping[str, Any], item: Mapping[str, Any], error: str, *, source: str, state_untrusted: bool = False) -> dict[str, Any]:
    """Serialize an infrastructure failure without pretending it is a task failure."""
    return {
        "experiment": manifest,
        "task_id": str(item["task_id"]),
        "slot_id": item.get("slot_id", item["task_id"]),
        "visibility": item.get("visibility"),
        "family": task_spec.get("family"),
        "family_id": task_spec.get("family_id", task_spec.get("family")),
        "condition": item["condition"],
        "context_mode": item["context_mode"],
        "outer_trial_id": item["outer_trial_id"],
        "phase": item["phase"],
        "agent": None,
        "agent_usage": {},
        "failure_stage": source,
        "failure_class": "infrastructure",
        "exception": {"type": "RuntimeError", "message": error},
        "receipt_valid": False if source == "executor" else None,
        "verifier_called": False,
        "activation": "not_evaluated",
        "cleanup_status": "not_observed",
        "surviving_pids": [],
        "store_mutated": False,
        "budget_errors": [f"{source}: {error}"],
        "attestation_ok": not state_untrusted,
        "execution_validity": "resource_blocked",
        "task_outcome": "error",
        "efficacy_eligible": False,
        "validity": "invalid",
        "transition": {"status": "state_mutation_error" if state_untrusted else f"{source}_error", "pre_store_digest": None, "post_store_digest": None},
        "score": {"task_id": str(item["task_id"]), "verdict": "error", "gates_passed": False, "task_score": 0.0},
    }


def _process_identity(pid: int) -> dict[str, int | None]:
    """Capture the verifier process identifiers before it exits."""
    identity: dict[str, int | None] = {"pid": int(pid), "pgid": None, "sid": None}
    if os.name != "posix":
        return identity
    for name, getter in (("pgid", os.getpgid), ("sid", os.getsid)):
        try:
            identity[name] = int(getter(pid))
        except (OSError, ProcessLookupError):
            pass
    return identity


def _compiler_process_snapshot() -> list[dict[str, Any]]:
    """Report live TorchInductor workers for failure diagnostics."""
    if os.name != "posix":
        return []
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,sid=,args="],
            text=True, capture_output=True, check=False,
        )
    except OSError:
        return []
    workers: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5 or "torch/_inductor/compile_worker" not in parts[4]:
            continue
        try:
            workers.append({"pid": int(parts[0]), "ppid": int(parts[1]), "pgid": int(parts[2]), "sid": int(parts[3]), "command": parts[4]})
        except ValueError:
            continue
    return workers


def _process_group_pids(pgid: int) -> list[int]:
    if os.name != "posix":
        return []
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,pgid="], text=True, capture_output=True, check=False,
        )
    except OSError:
        return []
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            if int(parts[1]) == int(pgid):
                pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def _cleanup_process_group(process: subprocess.Popen[str], grace_s: float = 1.0) -> list[int]:
    """Terminate the verifier session and reap every descendant on all exits."""
    if os.name != "posix":
        if process.poll() is None:
            process.kill()
        process.wait()
        return []
    pgid = process.pid
    try:
        if process.poll() is None or _process_group_pids(pgid):
            os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        survivors = _process_group_pids(pgid)
        if not survivors:
            break
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.05)

    survivors = _process_group_pids(pgid)
    if survivors:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        kill_deadline = time.monotonic() + 1.0
        while time.monotonic() < kill_deadline:
            survivors = _process_group_pids(pgid)
            if not survivors:
                break
            time.sleep(0.05)
    if process.poll() is None:
        process.wait()
    return _process_group_pids(pgid)


def _verify_task_with_cache(*args: Any, trial_dir: Path, invocation_id: str = "candidate", timeout_s: float | None = None, **kwargs: Any) -> dict[str, Any]:
    with _trial_compiler_cache(trial_dir, invocation_id):
        if timeout_s is None or os.name != "posix":
            try:
                return verifier.verify_task(*args, **kwargs)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                return _timeout_result(Path(args[0]), str(kwargs.get("condition", "standalone")), str(kwargs.get("context_mode", "reset")), float(timeout_s or 0.0)) | {"protocol_failure": True, "errors": [f"verifier raised: {exc}"], "failure_detail": {"exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc()}}
        task_dir, solution_dir = Path(args[0]), Path(args[1])
        out_path = Path(kwargs["out_path"])
        command = [
            sys.executable, "-m", "benchmark.harness.cli", "run-task", str(task_dir),
            "--solution", str(solution_dir), "--out", str(out_path),
            "--condition", str(kwargs.get("condition", "standalone")),
            "--context-mode", str(kwargs.get("context_mode", "reset")),
            "--seed", str(int(kwargs.get("seed", 0))),
        ]
        predicted = kwargs.get("predicted_mechanism")
        if isinstance(predicted, list) and predicted:
            command.extend(["--predict-mechanism", ",".join(str(item) for item in predicted)])
        noise_path = kwargs.get("noise_control_path")
        if noise_path is not None:
            command.extend(["--noise-control", str(noise_path)])
        if kwargs.get("noise_control_required"):
            command.append("--noise-control-required")
        for flag, key in (
            ("--outer-trial-id", "noise_outer_trial_id"),
            ("--benchmark-revision", "noise_benchmark_revision"),
            ("--task-manifest-digest", "noise_task_manifest_digest"),
            ("--task-package-digest", "noise_task_package_digest"),
            ("--population-manifest-digest", "noise_population_manifest_digest"),
        ):
            value = kwargs.get(key)
            if value is None and isinstance(kwargs.get("noise_control_expected"), Mapping):
                expected_key = {
                    "noise_task_package_digest": "task_package_digest",
                    "noise_population_manifest_digest": "population_manifest_digest",
                }.get(key, key.removeprefix("noise_"))
                value = kwargs["noise_control_expected"].get(expected_key)
            if value is not None:
                command.extend([flag, str(value)])
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        identity = _process_identity(process.pid)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_s))
            return_code = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return_code = 124
        finally:
            workers_before_cleanup = _compiler_process_snapshot()
            survivors = _cleanup_process_group(process)
        diagnostics = {
            "verifier_pid": identity["pid"],
            "verifier_pgid": identity["pgid"],
            "verifier_sid": identity["sid"],
            "known_compiler_workers": workers_before_cleanup,
            "surviving_pids": survivors,
            "stdout": stdout if "stdout" in locals() else "",
            "stderr": stderr if "stderr" in locals() else "",
            "cache_env": {key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
            "timeout": timed_out,
        }
        if timed_out:
            result = _timeout_result(task_dir, str(kwargs.get("condition", "standalone")), str(kwargs.get("context_mode", "reset")), float(timeout_s))
            result["errors"].append("verifier process group terminated after timeout")
            result["failure_detail"] = diagnostics
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return result
        if out_path.is_file():
            try:
                result = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(result, dict):
                    result.setdefault("failure_detail", {}).update(diagnostics)
                    if survivors:
                        result["protocol_failure"] = True
                        result["validity"] = "invalid"
                        result.setdefault("errors", []).append("verifier process group survivors remained after cleanup")
                    return result
            except (OSError, json.JSONDecodeError):
                pass
        result = _timeout_result(task_dir, str(kwargs.get("condition", "standalone")), str(kwargs.get("context_mode", "reset")), float(timeout_s))
        result["errors"] = [f"verifier subprocess failed with return code {return_code}"]
        if survivors:
            result["protocol_failure"] = True
            result["validity"] = "invalid"
            result["errors"].append("verifier process group survivors remained after cleanup")
        result["failure_detail"] = diagnostics
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result


def _calibrate_noise_control_with_cache(
    task_dir: Path,
    out_path: Path,
    *,
    calibration_dir: Path,
    task_id: str,
    outer_trial_id: str,
    benchmark_revision: str,
    task_manifest_digest: str,
    task_package_digest: str | None = None,
    population_manifest_digest: str | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    """Run one calibration subprocess under the normal cache/process boundary."""
    calibration_dir.mkdir(parents=True, exist_ok=True)
    solution_dir = task_dir / "workspace"
    with _trial_compiler_cache(calibration_dir, "noise-control"):
        command = [
            sys.executable, "-m", "benchmark.harness.cli", "calibrate-noise-control", str(task_dir),
            "--solution", str(solution_dir), "--out", str(out_path),
            "--task-id", task_id, "--outer-trial-id", outer_trial_id,
            "--benchmark-revision", benchmark_revision,
            "--task-manifest-digest", task_manifest_digest,
        ]
        if task_package_digest is not None:
            command.extend(["--task-package-digest", task_package_digest])
        if population_manifest_digest is not None:
            command.extend(["--population-manifest-digest", population_manifest_digest])
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_s))
        except subprocess.TimeoutExpired as exc:
            if os.name == "posix":
                _cleanup_process_group(process)
            else:
                process.kill()
                process.wait()
            return {"ok": False, "status": "resource_blocked", "error": "noise-control calibration timed out", "stdout": exc.stdout or "", "stderr": exc.stderr or ""}
        if process.returncode != 0 or not out_path.is_file():
            return {"ok": False, "status": "resource_blocked", "error": "noise-control calibration failed", "stdout": stdout, "stderr": stderr}
    return {"ok": True, "status": "calibrated", "path": str(out_path)}


class InterventionRealizer:
    """Materialize one explicit worker patch as an executable intervention."""

    @staticmethod
    def realize(baseline: Path, destination: Path, proposal: dict[str, Any]) -> Path:
        if not baseline.is_dir():
            raise ValueError("intervention baseline directory is missing")
        if not isinstance(proposal, dict):
            raise ValueError("intervention proposal must be an object")
        intervention = proposal.get("intervention")
        if not isinstance(intervention, dict) or not intervention:
            raise ValueError("intervention proposal needs an explicit patch")
        relative_file = intervention.get("file")
        replacements = intervention.get("replacements")
        if not isinstance(relative_file, str) or not relative_file or not isinstance(replacements, list) or not replacements:
            raise ValueError("intervention patch requires file and replacements")
        relative = Path(relative_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("intervention file must be baseline-relative")
        source = (baseline / relative).resolve()
        baseline_root = baseline.resolve()
        if baseline_root not in source.parents or not source.is_file() or source.is_symlink():
            raise ValueError("intervention file must be a regular baseline file")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(baseline, destination)
        target = destination / relative
        content = target.read_text(encoding="utf-8")
        for replacement in replacements:
            if not isinstance(replacement, dict) or not isinstance(replacement.get("old"), str) or not isinstance(replacement.get("new"), str):
                raise ValueError("intervention replacements must contain old and new text")
            old, new = replacement["old"], replacement["new"]
            if content.count(old) != 1:
                raise ValueError("intervention replacement must match exactly once")
            content = content.replace(old, new, 1)
        target.write_text(content, encoding="utf-8")
        return destination

    @staticmethod
    def realize_raw(
        baseline: Path,
        destination: Path,
        proposal: dict[str, Any],
        *,
        task_id: str,
        context_id: str,
    ) -> RawRealizationRecord:
        """Materialize only the worker patch; no semantic action is inferred."""
        output = InterventionRealizer.realize(baseline, destination, proposal)
        return RawRealizationRecord(
            task_id=task_id,
            context_id=context_id,
            baseline_digest=_workspace_digest(baseline),
            patch=dict(proposal.get("intervention") or {}),
            realized_digest=_workspace_digest(output),
        )

    @staticmethod
    def classify_after_verification(raw: RawRealizationRecord, action_spec: Mapping[str, Any], verifier_digest: str) -> RealizationRecord:
        action = ActionSpec(
            action_id=str(action_spec["action_id"]),
            family=str(action_spec["family"]),
            parameters=dict(action_spec.get("parameters", {})),
            preconditions=dict(action_spec.get("preconditions", {})),
            preserves=list(action_spec.get("preserves", [])),
            risk_class=str(action_spec.get("risk_class", "bounded")),
        )
        return RealizationRecord(
            action_id=action.action_id,
            task_id=raw.task_id,
            context_id=raw.context_id,
            baseline_digest=raw.baseline_digest,
            patch=dict(raw.patch),
            realized_digest=raw.realized_digest,
            verifier_digest=verifier_digest,
        )

    @staticmethod
    def realize_action(
        baseline: Path,
        destination: Path,
        proposal: dict[str, Any],
        *,
        family_id: str,
        task_id: str,
        context_id: str,
        verifier_digest: str = "unverified",
        action_spec: dict[str, Any] | None = None,
    ) -> RealizationRecord:
        """Materialize a reusable semantic action and record its realization.

        The source patch is a realization detail.  The candidate identity and
        governance path use the ActionSpec digest; this record links that
        semantic action to the task-local artifact without making source text
        part of the canonical rule meaning.
        """
        raw = InterventionRealizer.realize_raw(baseline, destination, proposal, task_id=task_id, context_id=context_id)
        if action_spec is None:
            raise ValueError("action_spec must be supplied after verifier activation")
        record = InterventionRealizer.classify_after_verification(raw, action_spec, verifier_digest)
        (destination / "realization_record.json").write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return record


def sanitize_submission(
    source: Path,
    destination: Path,
    allowed_files: set[str],
    *,
    baseline: Path | None = None,
    max_file_bytes: int = 8 * 1024 * 1024,
    max_total_bytes: int = 32 * 1024 * 1024,
) -> Path:
    """Copy only allowlisted regular files out of a worker namespace."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("worker solution root must be a regular directory")
    normalized_allowlist = {str(Path(item)) for item in allowed_files}
    if not normalized_allowlist:
        raise ValueError("submission allowlist must not be empty")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        relative_text = str(relative)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"submission contains symlink: {relative_text}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"submission contains non-regular file: {relative_text}")
        if relative_text not in normalized_allowlist:
            if baseline is None:
                raise ValueError(f"submission file is not allowlisted: {relative_text}")
            baseline_path = baseline / relative
            if not baseline_path.is_file() or baseline_path.read_bytes() != path.read_bytes():
                raise ValueError(f"submission changes outside public change surface: {relative_text}")
        if info.st_size > max_file_bytes:
            raise ValueError(f"submission file exceeds size limit: {relative_text}")
        total_bytes += info.st_size
        if total_bytes > max_total_bytes:
            raise ValueError("submission exceeds total size limit")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_files = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file() and not path.is_symlink()}
    required_files = normalized_allowlist
    if baseline is not None:
        required_files = {
            str(path.relative_to(baseline))
            for path in baseline.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
    missing = sorted(required_files - source_files)
    if missing:
        raise ValueError(f"submission is missing allowlisted files: {missing}")
    return destination


def representative_case_ids(predicate: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    """Return only certified representative cases covered by a learned predicate."""
    from core.predicates import match_predicate

    return sorted({
        str(case["case_id"])
        for case in cases
        if isinstance(case, dict)
        and isinstance(case.get("case_id"), str)
        and isinstance(case.get("context"), dict)
        and match_predicate(predicate, case["context"])
        and (_case_effect_interval(case) is not None)
        # Promotion consumes certified positive anchors only.  An interval
        # whose lower endpoint is at or below the practical threshold is
        # retained for CEGIS but cannot become a promotion Bernoulli trial.
        and _case_effect_interval(case)[1] > float(case.get("epsilon", 0.0))
        and bool(case.get("scientific_ok", False))
    })


def rewrite_validation_membership(
    store: Path,
    candidate: dict[str, Any],
    *,
    synthesis_case_ids: list[str],
    promotion_case_ids: list[str],
) -> None:
    """Bind validation membership after CEGIS has identified applicability."""
    reference = candidate.get("validation_artifacts")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("candidate validation artifact is missing")
    path = store / str(reference["path"])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate validation artifact is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("candidate validation artifact must be an object")
    promotion_ids = sorted(set(str(item) for item in promotion_case_ids))
    # A candidate projection may retain the full CEGIS evidence set while
    # promotion uses only certified representative cases.  Materialize the
    # validation artifact with disjoint memberships instead of relying on a
    # later audit to discover the overlap.
    promotion_set = set(promotion_ids)
    synthesis_ids = sorted({str(item) for item in synthesis_case_ids if str(item) not in promotion_set})
    value["synthesis_case_ids"] = synthesis_ids
    value["promotion_case_ids"] = promotion_ids
    case_dir = store / "experience" / "cases"
    def groups(ids: list[str]) -> list[str]:
        found: set[str] = set()
        for case_id in ids:
            path = case_dir / f"{identifier_digest(str(case_id))}.json"
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(item, dict) and item.get("independence_group") is not None:
                found.add(str(item["independence_group"]))
        return sorted(found)
    value["synthesis_independence_groups"] = groups(synthesis_ids)
    value["promotion_independence_groups"] = groups(promotion_ids)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    target = store / "evolution" / "validation" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate["validation_artifacts"] = {
        "path": str(target.relative_to(store)).replace("\\", "/"),
        "digest": digest,
        "heldout_count": len(value.get("heldout_regression_cases", [])),
        "poison_probe_count": len(value.get("poison_probe_cases", [])),
    }


def _case_independence_groups(store: Path, case_ids: list[str]) -> list[str]:
    """Read verifier-owned group labels for a validation membership list."""
    case_dir = store / "experience" / "cases"
    groups: set[str] = set()
    for case_id in case_ids:
        try:
            value = json.loads((case_dir / f"{identifier_digest(str(case_id))}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict) and value.get("independence_group") is not None:
            groups.add(str(value["independence_group"]))
    return sorted(groups)


def hydrate_candidate_cases(
    store: Path,
    candidate: dict[str, Any],
    ledger: CandidateEvidenceLedger,
) -> list[dict[str, Any]]:
    """Rebuild all immutable case payloads recorded for a candidate revision."""
    subject_id = str(candidate.get("candidate_identity") or candidate.get("rule_id") or candidate.get("relation_id") or candidate.get("id") or "")
    version = int(candidate.get("version", 1))
    action_digest = str(candidate.get("action_semantic_digest") or candidate.get("intervention_digest") or "") or None
    memberships = ledger.members(subject_id, version, action_digest=action_digest)
    paths: dict[str, Path] = {}
    for membership in memberships:
        case_id = membership.get("case_id")
        case_path = membership.get("case_path")
        if not isinstance(case_id, str) or not isinstance(case_path, str):
            continue
        relative = Path(case_path)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        paths[case_id] = store / relative
    values: list[dict[str, Any]] = []
    for case_id in sorted(paths):
        try:
            value = json.loads(paths[case_id].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            case_for_hash = {key: item for key, item in value.items() if key != "case_path"}
            expected = next((item.get("case_sha256") for item in memberships if item.get("case_id") == case_id), None)
            actual = hashlib.sha256(json.dumps(case_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            if expected and actual != expected:
                raise ValueError(f"candidate evidence digest mismatch: {case_id}")
            values.append(value)
    return values


def candidate_intervention_digest(intervention: dict[str, Any]) -> str:
    """Digest the semantic proposal identity used to partition evidence."""
    return hashlib.sha256(json.dumps(intervention, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def candidate_identity(rule_id: str, version: int, intervention: dict[str, Any]) -> str:
    return f"{rule_id}:v{int(version)}:{candidate_intervention_digest(intervention)[:24]}"


def semantic_action_spec(
    family_id: str | None,
    proposal: dict[str, Any],
    *,
    activation_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a task patch onto the reusable family action vocabulary."""
    from core.acre.actions import action_from_proposal
    resolved_family = str(family_id) if family_id is not None else ""
    try:
        from benchmark.families import resolve_family_id
        resolved_family = resolve_family_id(resolved_family)
    except (KeyError, ValueError):
        pass
    try:
        action = action_from_proposal(resolved_family or None, proposal)
    except ValueError:
        from core.models import ActionSpec
        from benchmark.families.catalog import FAMILY_SPECS
        family_spec = FAMILY_SPECS.get(resolved_family)
        if family_spec is None:
            raise
        if not isinstance(activation_certificate, Mapping) or activation_certificate.get("passed") is not True:
            raise ValueError("source patch requires a passed activation certificate")
        metrics = activation_certificate.get("activation_metrics")
        causal = metrics.get("causal") if isinstance(metrics, Mapping) else None
        activation = causal.get("activation") if isinstance(causal, Mapping) else None
        matched = activation.get("matched_actions") if isinstance(activation, Mapping) else None
        action_id = activation.get("action_id") if isinstance(activation, Mapping) else None
        if not isinstance(matched, list) or len(matched) != 1:
            raise ValueError("source patch activation must match exactly one ActionSpec")
        if action_id is None:
            action_id = matched[0]
        if str(action_id) != str(matched[0]) or str(action_id) not in family_spec.action_specs:
            raise ValueError("activation ActionSpec is not registered for this family")
        action = ActionSpec(
            action_id=str(action_id),
            family=resolved_family,
            # The source patch is recorded only by RawRealizationRecord.  A
            # reusable ActionSpec contains registry-owned semantic parameters.
            parameters=dict(family_spec.action_specs[str(action_id)].get("parameters") or {}),
            scientific_policy_ref=str(family_spec.action_specs[str(action_id)].get("scientific_policy_ref", family_spec.scientific_contract_id)),
            activation_validator=str(family_spec.action_specs[str(action_id)].get("activation_validator", "")),
            realization_interface=str(family_spec.action_specs[str(action_id)].get("realization_interface", "source_patch")),
        )
    if resolved_family:
        if not isinstance(activation_certificate, Mapping) or activation_certificate.get("passed") is not True:
            raise ValueError("family action requires a passed activation certificate")
        metrics = activation_certificate.get("activation_metrics")
        causal = metrics.get("causal") if isinstance(metrics, Mapping) else None
        activation = causal.get("activation") if isinstance(causal, Mapping) else None
        if not isinstance(activation, Mapping) or str(activation.get("status", "")) not in {"passed", "verified"}:
            raise ValueError("activation certificate requires action-specific verifier instrumentation")
        matched = activation.get("matched_actions")
        if not isinstance(matched, list) or len(matched) != 1 or str(matched[0]) != action.action_id:
            raise ValueError("activation must match exactly one registered ActionSpec")
        matched_id = activation.get("action_id") or activation.get("matched_action_id")
        if matched_id is not None and str(matched_id) != action.action_id:
            raise ValueError("activation ActionSpec does not match the classified action")
        from core.models import ActivationCertificate
        ActivationCertificate.from_dict(activation_certificate, action_id=action.action_id)
        from benchmark.families.catalog import FAMILY_SPECS
        family_spec = FAMILY_SPECS.get(resolved_family)
        if family_spec is not None and action.action_id not in family_spec.action_specs:
            raise ValueError(f"action {action.action_id} is not legal for family {resolved_family}")
        if family_spec is not None:
            metadata = family_spec.action_specs[action.action_id]
            action = ActionSpec(
                action_id=action.action_id,
                family=action.family,
                parameters=dict(metadata.get("parameters") or {}),
                preconditions=dict(action.preconditions),
                preserves=list(action.preserves),
                risk_class=action.risk_class,
                applicability=dict(metadata.get("applicability") or {}) if isinstance(metadata.get("applicability"), Mapping) else {},
                scientific_policy_ref=str(metadata.get("scientific_policy_ref", family_spec.scientific_contract_id)),
                activation_validator=str(metadata.get("activation_validator", "")),
                realization_interface=str(metadata.get("realization_interface", "source_patch")),
            )
    return action.canonical_dict()


def persist_collecting_proposals(
    store: Path,
    proposals: list[dict[str, Any]],
    case_ids: list[str],
    *,
    family_id: str | None = None,
) -> None:
    """Persist worker hypotheses before a boundary predicate is identifiable."""
    candidates_dir = Path(store) / "evolution" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.get("relation_id"):
            continue
        identifier = proposal.get("rule_id") or proposal.get("id")
        if not isinstance(identifier, str):
            continue
        try:
            validate_identifier(identifier, "candidate_id")
        except ValueError:
            continue
        version = int(proposal.get("version", 1))
        realization = proposal.get("intervention") if isinstance(proposal.get("intervention"), dict) else {"action": "measure"}
        try:
            intervention = semantic_action_spec(family_id, proposal)
        except ValueError:
            # Collect the worker hypothesis without granting it semantic
            # action identity.  A later verifier activation proof must replace
            # this patch projection before it can enter replay/governance.
            patch = dict(realization)
            patch_digest = candidate_intervention_digest(patch)
            intervention = {
                "action": f"patch-{patch_digest[:16]}",
                "action_id": f"patch-{patch_digest[:16]}",
                "family": str(family_id or "runtime"),
                "parameters": {},
                "preconditions": {}, "preserves": [], "risk_class": "review",
            }
        intervention_digest = candidate_intervention_digest(intervention)
        realization_digest = candidate_intervention_digest(realization)
        identity = candidate_identity(identifier, version, intervention)
        path = candidates_dir / f"{identifier_digest(identity)}.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        evidence_ids = sorted(set(str(item) for item in existing.get("synthesis_state", {}).get("evidence_ids", []) if isinstance(item, str)) | set(case_ids))
        candidate = {
            "rule_id": identifier,
            "family_id": family_id,
            "version": int(proposal.get("version", existing.get("version", 1))),
            "candidate_identity": identity,
            "intervention_digest": intervention_digest,
            "action_semantic_digest": intervention_digest,
            "realization_digest": realization_digest,
            "parent": None,
            "hypothesis": {
                "intervention": intervention,
                "realization": realization,
                "expected_mechanism": str(proposal.get("expected_mechanism") or proposal.get("hypothesis") or "task-local performance mechanism"),
            },
            "synthesis_state": existing.get("synthesis_state", {
                "status": "collecting_evidence",
                "predicate": None,
                "version_space_digest": None,
                "evidence_ids": evidence_ids,
            }),
            "intervention": intervention,
            "realization": realization,
            "expected_mechanism": str(proposal.get("expected_mechanism") or proposal.get("hypothesis") or "task-local performance mechanism"),
            "status": "collecting_evidence",
            "scope": str(existing.get("scope") or ("formal" if family_id else "calibration")),
            "p_min": 0.8,
            "delta": 0.05,
            "cases": sorted(set(str(item) for item in existing.get("cases", []) if isinstance(item, str)) | set(case_ids)),
        }
        candidate["synthesis_state"]["evidence_ids"] = evidence_ids
        payload = {**existing, **candidate}
        if payload.get("status") == "collecting_evidence":
            payload.pop("applicability", None)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute_poison_probe(
    task_spec: dict[str, Any],
    public_context: dict[str, Any],
    proposal: dict[str, Any],
    realized_solution: Path,
    baseline_solution: Path,
    task_dir: Path | None = None,
    verifier_out: Path | None = None,
    timeout_s: float = 900.0,
    noise_control_path: Path | None = None,
    noise_control_expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a poison probe against a materialized, non-empty intervention."""
    changed = False
    baseline_files = {path.relative_to(baseline_solution) for path in baseline_solution.rglob("*") if path.is_file()}
    realized_files = {path.relative_to(realized_solution) for path in realized_solution.rglob("*") if path.is_file()}
    for relative in baseline_files | realized_files:
        baseline_path, realized_path = baseline_solution / relative, realized_solution / relative
        if not baseline_path.is_file() or not realized_path.is_file() or baseline_path.read_bytes() != realized_path.read_bytes():
            changed = True
            break
    if not changed:
        raise ValueError("poison probe requires a realized intervention different from baseline")
    family_id = str(task_spec.get("family_id", task_spec.get("family", "compile")))
    # Poison validation executes the realized artifact itself.  An
    # unclassified source patch is intentionally not promoted into the
    # semantic action ledger, but it still receives a stable execution label.
    patch_digest = hashlib.sha256(
        json.dumps(proposal.get("intervention", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    try:
        from benchmark.families.catalog import FAMILY_SPECS, resolve_family_id
        intervention_id = str(FAMILY_SPECS[resolve_family_id(family_id)].action_policy.get("default") or f"unclassified-{patch_digest}")
    except (KeyError, ValueError):
        intervention_id = f"unclassified-{patch_digest}"
    deployed = [intervention_id]
    verifier_executed = False
    verifier_scientific_ok = True
    verifier_result_digest: str | None = None
    if task_dir is not None:
        result = _verify_task_with_cache(
            task_dir,
            realized_solution,
            out_path=verifier_out or realized_solution.parent / "poison-result.json",
            condition="D",
            context_mode="reset",
            seed=0,
            trial_dir=realized_solution.parent,
            invocation_id="poison",
            timeout_s=timeout_s,
            noise_control_path=noise_control_path,
            noise_control_required=task_dir is not None,
            noise_outer_trial_id=str(noise_control_expected.get("outer_trial_id")) if noise_control_expected else None,
            noise_benchmark_revision=str(noise_control_expected.get("benchmark_revision")) if noise_control_expected else None,
            noise_task_manifest_digest=str(noise_control_expected.get("task_manifest_digest")) if noise_control_expected else None,
            noise_control_expected=noise_control_expected,
        )
        verifier_executed = True
        verifier_scientific_ok = bool(scoring.score_task(result).get("gates_passed", False))
        verifier_result_digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    environment = FamilyEnvironment(family_id)
    state = EpisodeEnvironmentState(active_poison=("formal_validation_probe",))
    outcome = environment.evaluate(public_context.get("workload", {}), deployed, state)
    baseline = environment.evaluate(public_context.get("workload", {}), (), state)
    return {
        "case_id": f"POISON-PROBE-{task_spec.get('task_id', 'task')}",
        "executed": True,
        "execution_source": "verifier+family-environment" if verifier_executed else "family-environment",
        "validation_class": "hybrid_synthetic_adversarial" if verifier_executed else "synthetic_validation_only",
        "accepted": bool(
            outcome.utility > baseline.utility + 1e-9
            and all(outcome.scientific_gates.values())
            and verifier_scientific_ok
        ),
        "utility": outcome.utility,
        "baseline_utility": baseline.utility,
        "intervention_id": intervention_id,
        "realized_changed": True,
        "verifier_scientific_ok": verifier_scientific_ok,
        "verifier_result_digest": verifier_result_digest,
        "resource_blocked": bool(task_dir is not None and isinstance(result, Mapping) and result.get("protocol_failure")),
    }


def _copy_workspace(task_dir: Path, destination: Path) -> None:
    source = task_dir / "workspace"
    if not source.is_dir():
        raise FileNotFoundError(f"workspace missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def materialize_agent_task(task_dir: Path, destination: Path) -> None:
    """Create the public task view used by the external worker.

    Oracle and verifier code are harness-only.  The worker receives the task
    contract, workspace, and public tests, never the task package root.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    task = miniyaml.load(str(task_dir / "task.yaml"))
    measurement = task.get("measurement", {}) if isinstance(task, dict) else {}
    correctness = task.get("correctness", {}) if isinstance(task, dict) else {}
    declared_public_context = task.get("public_context") if isinstance(task.get("public_context"), dict) else None
    if declared_public_context is None:
        # Family parameters are observable workload facts, unlike the hidden
        # mechanism/oracle labels.  Existing anchors are projected through
        # this same public contract until their YAML declarations are rebuilt.
        family_parameters = task.get("family_parameters") if isinstance(task.get("family_parameters"), dict) else {}
        declared_public_context = {"workload": dict(family_parameters)}
    # Optional observable platform facts are part of the common public
    # context, never condition-specific metadata.
    if isinstance(task.get("public_hardware"), dict):
        declared_public_context.setdefault("hardware", dict(task["public_hardware"]))
    if isinstance(task.get("public_software"), dict):
        declared_public_context.setdefault("software", dict(task["public_software"]))
    if isinstance(task.get("pre_task_telemetry"), dict):
        declared_public_context.setdefault("evidence", dict(task["pre_task_telemetry"]))
    declared_public_context = build_public_context(declared_public_context)
    public_task = {
        "schema_version": 1,
        "task_id": task.get("task_id", task_dir.name),
        "title": task.get("title", ""),
        "requires_cuda": bool(task.get("requires_cuda", False)),
        "time_budget_s": int(task.get("time_budget_s", 0)),
        "workspace": dict(task.get("workspace", {})),
        "allowed_files": [str(task.get("workspace", {}).get("entrypoint", "solution.py"))],
        "scientific_gates": list(task.get("scientific_gates", [])),
        "scientific_contract": "scientific_contract.py",
        "measurement_contract": {
            "primary_metric": measurement.get("primary_metric"),
            "higher_is_better": measurement.get("higher_is_better"),
            "warmup_iterations": measurement.get("warmup_iterations"),
            "measured_iterations": measurement.get("measured_iterations"),
            "repetitions": measurement.get("repetitions"),
        },
        "correctness_contract": correctness,
        # Only fields explicitly declared public may drive retrieval/routing.
        "routing_context": declared_public_context,
    }
    (destination / "public_task.json").write_text(
        json.dumps(public_task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source = task_dir / "scientific_contract.py"
    if source.is_file():
        shutil.copy2(source, destination / "scientific_contract.py")
    for name in ("workspace", "public_tests"):
        source = task_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)


def _run_agent(command_template: str, env: dict[str, str], cwd: Path, timeout: float) -> dict[str, Any]:
    command = command_template.format(**env)
    try:
        inherited = {key: value for key, value in os.environ.items() if not key.startswith("SPE_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env={**inherited, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "timeout",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_isolated_agent(
    command_template: str,
    executor_command: str,
    env: dict[str, str],
    worker_root: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run through an externally supplied namespace/container executor.

    The executor owns the filesystem boundary.  The campaign never exposes the
    condition store, verifier, or benchmark root as worker paths.
    """
    executor = executor_command.format(
        agent_command=shlex.quote(command_template),
        worker_root=shlex.quote(str(worker_root)),
        task_dir=shlex.quote(env["SPE_TASK_DIR"]),
        solution_dir=shlex.quote(env["SPE_SOLUTION_DIR"]),
        retrieved_context=shlex.quote(env["SPE_RETRIEVED_CONTEXT"]),
        skill_view=shlex.quote(env.get("SPE_SKILL_VIEW_DIR", "")),
        executor_receipt=shlex.quote(env["SPE_EXECUTOR_RECEIPT_PATH"]),
    )
    # The namespace launcher is a host-side harness command.  Run it from the
    # repository root so a module-based ReferenceExecutor adapter resolves
    # without exposing that root to the worker namespace; all worker paths are
    # still passed explicitly and mounted by the executor.
    return _run_agent(executor, env, Path(__file__).resolve().parents[2], timeout)


def _prepare_worker_root(
    trial_dir: Path,
    agent_task_dir: Path,
    solution_dir: Path,
    skill_view: Path | None,
) -> Path:
    """Create the only host directory visible to an external executor."""
    worker_root = trial_dir / "worker"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    for name in ("retrieved_context", "context_state", "result", "executor_receipt"):
        (worker_root / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(agent_task_dir, worker_root / "task")
    shutil.copytree(solution_dir, worker_root / "solution")
    if skill_view is not None:
        shutil.copytree(skill_view, worker_root / "skill_view")
    return worker_root


def _build_required_experiment_executor(command_template: str | None, root: Path, timeout: float = 900.0):
    """Create the explicit external experiment callback used by formal D."""
    if not command_template:
        return None
    def execute(request: Mapping[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=str(root)) as directory:
            work = Path(directory)
            request_path, result_path = work / "request.json", work / "result.json"
            request_path.write_text(json.dumps(dict(request), ensure_ascii=False), encoding="utf-8")
            command = command_template.format(request_json=shlex.quote(str(request_path)), result_json=shlex.quote(str(result_path)), work_root=shlex.quote(str(work)))
            process = subprocess.Popen(command, shell=True, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=(os.name == "posix"))
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
            except subprocess.TimeoutExpired as exc:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                timeout_stderr = exc.stderr or ""
                if isinstance(timeout_stderr, bytes):
                    timeout_stderr = timeout_stderr.decode(errors="replace")
                return {
                    "status": "resource_blocked",
                    "reason": "external required-experiment timeout",
                    "execution_source": "external_executor",
                    "timeout_s": timeout,
                    "stderr": timeout_stderr[-2000:],
                }
            if return_code != 0 or not result_path.is_file():
                return {"status": "blocked", "reason": "external experiment executor failed", "execution_source": "external_executor", "stderr": (stderr or "")[-2000:]}
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {"status": "blocked", "reason": "external experiment result must be an object", "execution_source": "external_executor"}
            return payload
    return execute


def _verify_baseline(
    task_dir: Path,
    solution_dir: Path,
    out_path: Path,
    *,
    condition: str,
    context_mode: str,
    seed: int,
    trial_dir: Path | None = None,
    invocation_id: str = "baseline",
    timeout_s: float | None = None,
    noise_control_path: Path | None = None,
    noise_control_expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score the untouched arm on the same fixture before candidate scoring."""
    call = _verify_task_with_cache if trial_dir is not None else verifier.verify_task
    kwargs = dict(
        out_path=out_path, condition=condition, context_mode=context_mode, seed=seed,
        noise_control_path=noise_control_path,
        noise_control_required=trial_dir is not None,
        noise_control_expected=noise_control_expected,
    )
    baseline = call(task_dir, solution_dir, trial_dir=trial_dir, invocation_id=invocation_id, timeout_s=timeout_s, **kwargs) if trial_dir is not None else call(task_dir, solution_dir, **kwargs)
    return baseline, scoring.score_task(baseline)


def _check_verifier_budget(result: Mapping[str, Any], task_spec: Mapping[str, Any], budget_errors: list[str], label: str) -> None:
    """Turn a verifier overrun into a protocol/resource failure."""
    cost = result.get("cost") if isinstance(result, Mapping) else None
    elapsed = cost.get("wall_time_s") if isinstance(cost, Mapping) else None
    limit = task_spec.get("time_budget_s")
    if isinstance(elapsed, (int, float)) and isinstance(limit, (int, float)) and float(elapsed) > float(limit):
        budget_errors.append(f"{label} time_budget_s exceeded: {elapsed} > {limit}")
    if result.get("protocol_failure") is True:
        budget_errors.append(f"{label} failed at verifier boundary")


def _read_executor_receipt(
    path: Path,
    skill_digest: str | None,
    context_mode: str = "reset",
    expected_executor_digest: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["external executor receipt missing or invalid"]
    if not isinstance(receipt, dict):
        return {}, ["external executor receipt must be an object"]
    errors: list[str] = []
    required = (
        "mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid", "usage",
        "network_namespace_attested", "mount_receipt", "isolation_canary", "usage_meter_source",
        "canary_executed_this_invocation", "canary_mode", "executor_attested",
    )
    errors.extend(f"executor receipt missing {key}" for key in required if key not in receipt)
    if receipt.get("mode") != "external_namespace_executor":
        errors.append("executor receipt mode mismatch")
    if receipt.get("network_mode") != "none":
        errors.append("external executor must declare network_mode=none")
    if receipt.get("network_namespace_attested") is not True:
        errors.append("external executor network namespace is not attested")
    if not isinstance(receipt.get("mount_receipt"), dict) or receipt.get("mount_receipt", {}).get("verified") is not True:
        errors.append("external executor mount receipt is not verified")
    if receipt.get("isolation_canary") is not True:
        errors.append("external executor isolation canary did not pass")
    if receipt.get("canary_executed_this_invocation") is not True:
        errors.append("external executor did not execute an invocation canary")
    if receipt.get("canary_mode") not in {"executed", "referenced_attestation"}:
        errors.append("external executor canary_mode is not attested")
    if receipt.get("executor_attested") is not True:
        errors.append("external executor attestation is not valid")
    if not isinstance(receipt.get("attestation_digest"), str) or not receipt.get("attestation_digest"):
        errors.append("external executor attestation_digest is required")
    if receipt.get("attested_executor_digest") != receipt.get("executor_digest"):
        errors.append("external executor attested digest mismatch")
    if not isinstance(receipt.get("attested_environment_digest"), str) or not receipt.get("attested_environment_digest"):
        errors.append("external executor environment attestation is required")
    if not isinstance(receipt.get("usage_meter_source"), str) or not receipt.get("usage_meter_source"):
        errors.append("external executor usage meter source is required")
    mounts = receipt.get("mount_allowlist")
    if not isinstance(mounts, list) or not mounts:
        errors.append("executor receipt mount_allowlist must be non-empty")
    else:
        allowed_mounts = {"task", "solution", "skill_view", "retrieved_context", "context_state", "result", "executor_receipt"}
        unexpected = sorted(set(str(item) for item in mounts) - allowed_mounts)
        if unexpected:
            errors.append(f"executor receipt contains disallowed mounts: {unexpected}")
        required_mounts = {"task", "solution", "retrieved_context", "result", "executor_receipt"}
        if skill_digest is not None:
            required_mounts.add("skill_view")
        elif "skill_view" in set(str(item) for item in mounts):
            errors.append("condition A must not mount skill_view")
        if context_mode == "carry":
            required_mounts.add("context_state")
        missing_mounts = sorted(required_mounts - set(str(item) for item in mounts))
        if missing_mounts:
            errors.append(f"executor receipt missing required mounts: {missing_mounts}")
    for key in ("executor_digest", "worker_uid"):
        if not isinstance(receipt.get(key), str) or not receipt.get(key):
            errors.append(f"executor receipt {key} must be non-empty")
    if expected_executor_digest is not None and receipt.get("executor_digest") != expected_executor_digest:
        errors.append("external executor digest is not allowlisted")
    if skill_digest is not None and receipt.get("skill_view_digest") != skill_digest:
        errors.append("executor receipt skill_view_digest mismatch")
    if skill_digest is None and receipt.get("skill_view_digest") not in {None, ""}:
        errors.append("condition A must not attest a skill_view_digest")
    usage = receipt.get("usage")
    if not isinstance(usage, dict):
        errors.append("executor receipt usage must be an object")
    return receipt, errors


def _manifest_executor_receipt(receipt: Mapping[str, Any], errors: Sequence[str]) -> dict[str, Any] | None:
    """Return only an attested receipt for the experiment manifest.

    A missing or invalid receipt is itself an executor failure.  Keeping the
    invalid payload out of the manifest lets the failure record be persisted
    instead of making manifest validation terminate the campaign.
    """
    if errors:
        return None
    return {
        key: receipt.get(key)
        for key in (
            "mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid",
            "skill_view_digest", "canary_executed_this_invocation", "canary_mode",
            "executor_attested", "attestation_digest", "attested_environment_digest",
        )
        if key in receipt
    }


def _read_agent_extensions(path: Path) -> dict[str, Any]:
    """Read only worker-supplied lesson/proposal extensions before verification.

    The verifier remains authoritative for scores, gates, and verdicts.  A
    worker may contribute a typed proposal, but cannot author evidence,
    replay cases, confidence, or any scored field.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    extensions: dict[str, Any] = {}
    for key in ("lesson", "acre_proposals", "predicted_mechanisms", "abstain", "abstain_reason"):
        if key in payload:
            extensions[key] = payload[key]
    if not isinstance(extensions.get("lesson", {}), dict):
        extensions["lesson"] = {}
    mechanisms = extensions.get("predicted_mechanisms", [])
    if not isinstance(mechanisms, list) or any(not isinstance(item, str) or not item for item in mechanisms):
        extensions["predicted_mechanisms"] = []
    if not isinstance(extensions.get("abstain", False), bool):
        extensions["abstain"] = False
    if not isinstance(extensions.get("abstain_reason", ""), str):
        extensions["abstain_reason"] = ""
    proposals = extensions.get("acre_proposals", [])
    if not isinstance(proposals, list):
        extensions["acre_proposals"] = []
    else:
        clean: list[dict[str, Any]] = []
        proposal_fields = {"rule_id", "id", "expected_mechanism", "intervention", "text", "hypothesis", "query"}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            if any(key in proposal for key in ("cases", "confidence", "promotion", "p_min", "delta", "epsilon", "effect", "scientific_gates")):
                continue
            if "relation_id" in proposal or "endpoints" in proposal or "endpoint_versions" in proposal or "endpoint_families" in proposal:
                continue
            identifier = proposal.get("rule_id") or proposal.get("id")
            if not isinstance(identifier, str):
                continue
            try:
                validate_identifier(identifier, "proposal_id")
            except ValueError:
                continue
            normalized = {key: value for key, value in proposal.items() if key in proposal_fields}
            # Semantic action labels are harness-owned.  A worker may submit
            # a source realization proposal, but cannot choose the registered
            # ActionSpec that will be used for attribution.
            intervention = normalized.get("intervention")
            if isinstance(intervention, dict):
                normalized["intervention"] = {
                    key: value
                    for key, value in intervention.items()
                    if key not in {"action", "action_id", "action_spec", "family"}
                }
            clean.append(normalized)
        extensions["acre_proposals"] = clean
    return extensions


def post_task_update(
    *,
    condition: str,
    store: Path,
    task_id: str,
    result: dict[str, Any],
    scored: dict[str, Any],
    core_repo: Path,
    out_dir: Path,
    context_mode: str = "reset",
    family_id: str | None = None,
    ledger: EvolutionDecisionLedger | None = None,
    allow_maintenance: bool = True,
    control_result: dict[str, Any] | None = None,
    control_scored: dict[str, Any] | None = None,
    public_context: dict[str, Any] | None = None,
    causal_result: dict[str, Any] | None = None,
    causal_scored: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
    practical_epsilon: float = 0.0,
    seed: int = 0,
    execution_validity: str = "valid",
    slot_id: str | None = None,
    visibility: str | None = None,
) -> dict[str, Any]:
    """Run the explicit execute -> evidence -> maintenance -> attest transition."""
    condition = condition.upper()
    statistical_budget = StatisticalBudget()
    pre_digest = conditions.store_digest(store)
    experience_id = f"EXP-{task_id}"
    added_experience_ids: list[str] = []
    added_replay_case_ids: list[str] = []
    maintenance_decisions: list[dict[str, Any]] = []
    promoted_rule_ids: list[str] = []

    def mutation_result(status: str) -> dict[str, Any]:
        """Finalize a mutating early return with a verified store attestation."""
        if condition in {"C", "C_STRESS", "D"}:
            conditions.refresh_attestation(store)
            valid, errors = conditions.verify_attestation(store)
            if not valid:
                raise ValueError("store attestation failed after mutation: " + "; ".join(errors))
        return {
            "status": status,
            "pre_store_digest": pre_digest,
            "post_store_digest": conditions.store_digest(store),
            "added_experience_ids": added_experience_ids,
            "added_replay_case_ids": added_replay_case_ids,
            "maintenance_decisions": maintenance_decisions,
            "promoted_rule_ids": promoted_rule_ids,
        }
    if execution_validity != "valid":
        # Protocol-invalid trials are recorded by the caller but never enter
        # a sequential C/D store.  An inconclusive scientific outcome remains
        # execution-valid and therefore takes the normal path.
        return {
            "transition": "protocol_invalid_no_mutation",
            "execution_validity": execution_validity,
            "pre_store_digest": pre_digest,
            "post_store_digest": pre_digest,
            "added_experience_ids": [],
            "added_replay_case_ids": [],
            "maintenance_decisions": [],
            "promoted_rule_ids": [],
        }
    if condition in {"C", "C_STRESS", "D"}:
        from core.mutation_journal import MutationJournal
        mutation_journal = MutationJournal(store / "evolution" / "mutation_journal.jsonl") if condition == "D" else None
        experience = {
            "schema_version": 1,
            "record_type": "task_experience",
            "id": experience_id,
            "experience_id": experience_id,
            "task_id": task_id,
            "slot_id": slot_id or task_id,
            "visibility": visibility,
            "condition": condition,
            "context_mode": context_mode,
            "public_context": build_public_context(public_context),
            "retrieval_query": json.dumps(build_public_context(public_context).get("workload", {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "observation": {
                "verdict": result.get("verdict"),
                "task_score": scored.get("task_score"),
                "scientific_gates": result.get("scientific_gates", {}),
                "measurement": result.get("measurement", {}),
            },
            "lesson": result.get("lesson", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        inbox = store / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        validate_identifier(experience_id, "experience_id")
        experience_path = inbox / f"{identifier_digest(experience_id)}.json"
        if not experience_path.exists():
            experience_path.write_text(json.dumps(experience, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_experience_ids.append(experience_id)
            if mutation_journal is not None:
                mutation_journal.append("add_evidence", experience_id, artifact_path=str(experience_path.relative_to(store)).replace("\\", "/"), digest=hashlib.sha256(experience_path.read_bytes()).hexdigest())

        if condition == "D" and causal_result is not None and causal_scored is not None and control_result is not None and control_scored is not None:
            # The worker cannot author evidence.  The harness derives paired
            # case records from the verifier-owned outcome below.
            evidence_dir = store / "experience" / "cases"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            case_id = f"CASE-{task_id}"
            validate_identifier(case_id, "case_id")
            measurement = causal_result.get("measurement", {}) if isinstance(causal_result.get("measurement"), dict) else {}
            control_measurement = control_result.get("measurement", {}) if isinstance(control_result.get("measurement"), dict) else {}
            intervention_runs = list(measurement.get("candidate_runs", []))
            baseline_runs = list(measurement.get("baseline_runs", []))
            fixture_hashes = measurement.get("fixture_hashes", {})
            control_fixture_hashes = control_measurement.get("fixture_hashes", {})
            if not intervention_runs or len(intervention_runs) != len(baseline_runs) or not fixture_hashes or fixture_hashes != control_fixture_hashes:
                # The verifier's paired record is the only accepted source of
                # causal measurements; no task-score fallback is fabricated.
                persist_collecting_proposals(store, [item for item in result.get("acre_proposals", []) if isinstance(item, dict)], added_replay_case_ids, family_id=family_id)
                return mutation_result("no_replay_measurement")
            case = {
                "schema_version": 1,
                "record_type": "paired_replay_case",
                "case_id": case_id,
                "utility_on": float(causal_scored.get("task_score", 0.0)),
                "utility_off": float(control_scored.get("task_score", 0.0)),
                "control_measured": bool(control_measurement.get("candidate_runs")),
                "baseline_measurements": baseline_runs,
                "intervention_measurements": intervention_runs,
                "same_fixture_digest": measurement.get("fixture_hashes", {}),
                "same_seed": causal_result.get("seed"),
                "same_work_units": measurement.get("work_units", {}),
                "higher_is_better": bool(measurement.get("higher_is_better", False)),
                "utility_scale": UTILITY_LOG_SCALE,
                "scientific_ok": bool(causal_scored.get("gates_passed", False)),
                "quality_ok": bool(causal_scored.get("gates_passed", False)) and bool(control_scored.get("gates_passed", False)),
                "paired_replay": True,
                "same_fixture_id": task_id,
                "candidate_result_digest": hashlib.sha256(json.dumps(causal_result, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                "control_result_digest": hashlib.sha256(json.dumps(control_result, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                "scientific_gates_on": dict(causal_result.get("scientific_gates", {})),
                "scientific_gates_off": dict(control_result.get("scientific_gates", {})),
                "source_id": f"verifier-{task_id}",
                "independence_group": f"task-{task_id}",
                "context": build_public_context(public_context),
                "context_mode": context_mode,
            }
            evidence_path = evidence_dir / f"{identifier_digest(case_id)}.json"
            if not evidence_path.exists():
                evidence_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                if mutation_journal is not None:
                    mutation_journal.append("add_evidence", case_id, artifact_path=str(evidence_path.relative_to(store)).replace("\\", "/"), digest=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
            added_replay_case_ids.append(case_id)
            validation_dir = store / "evolution" / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(validation_evidence, dict):
                persist_collecting_proposals(store, [item for item in result.get("acre_proposals", []) if isinstance(item, dict)], added_replay_case_ids, family_id=family_id)
                return mutation_result("no_independent_validation")
            validation = {
                "schema_version": 1,
                "scope": "formal",
                "subject_context": build_public_context(public_context),
                "synthesis_case_ids": [case_id],
                "promotion_case_ids": [],
                "heldout_regression_cases": list(validation_evidence.get("heldout_regression_cases", [])),
                "poison_probe_cases": list(validation_evidence.get("poison_probe_cases", [])),
                "independence_groups": [case["independence_group"]],
                "promotion_independence_groups": [],
                "synthesis_independence_groups": [case["independence_group"]],
            }
            validation_digest = hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            validation_path = validation_dir / f"{validation_digest}.json"
            if not validation_path.exists():
                validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if condition == "D" and allow_maintenance:
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from dataclasses import asdict
        from core.acre.engine import AcreEngine
        engine = AcreEngine.from_store(store)
        # Core owns the evidence-to-lifecycle reducer.  The formal driver
        # only supplies this task's verifier-produced events and persistence
        # callbacks; it does not make a second lifecycle decision.
        maintenance_step = engine.maintain(
            events=[item for item in result.get("evidence_events", []) if isinstance(item, dict)],
            subject_ids=(*engine.rule_states, *engine.relation_states),
        )
        maintenance_decisions.append({"operation": "OBSERVE", "observed": maintenance_step.observed, "assessment": maintenance_step.assessment})
        maintenance_decisions.extend(asdict(item) for item in maintenance_step.decisions)
        active_ledger = ledger or EvolutionDecisionLedger()
        candidates_dir = store / "evolution" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        from benchmark.harness.evolution_ledger import CandidateEvidenceLedger
        candidate_evidence = CandidateEvidenceLedger(store / "evolution" / "candidate_evidence.jsonl")
        candidates_by_id: dict[str, dict[str, Any]] = {}
        proposals = [item for item in result.get("acre_proposals", []) if isinstance(item, dict)]
        if len(proposals) != 1:
            proposals = []
        # Collection is a Core-owned continuation: a collecting candidate is
        # revisited even when the next worker submits no new proposal.
        seen_candidate_keys = {
            str(item.get("candidate_identity") or item.get("rule_id") or item.get("id") or "")
            for item in proposals
        }
        for candidate_path in sorted((store / "evolution" / "candidates").glob("*.json")):
            try:
                pending_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pending_key = str(pending_candidate.get("candidate_identity") or pending_candidate.get("rule_id") or pending_candidate.get("id") or "")
            if pending_candidate.get("status") == "collecting_evidence" and pending_key and pending_key not in seen_candidate_keys:
                proposals.append(pending_candidate)
                seen_candidate_keys.add(pending_key)
        for candidate in proposals:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("relation_id"):
                # Relations use an explicit factorial experiment and never
                # enter the node utility replay path.
                from benchmark.formal.schedule import RelationExperimentScheduler
                relation_id = str(candidate.get("relation_id") or candidate.get("id") or "")
                validate_identifier(relation_id, "relation_id")
                endpoints = candidate.get("endpoints")
                endpoint_versions = candidate.get("endpoint_versions")
                endpoint_families = candidate.get("endpoint_families")
                if not isinstance(endpoints, dict) or set(endpoints) != {"left", "right"} or not isinstance(endpoint_versions, dict) or set(endpoint_versions) != {"left", "right"} or not isinstance(endpoint_families, dict) or set(endpoint_families) != {"left", "right"}:
                    relation_dir = store / "evolution" / "relation_experiments"
                    relation_dir.mkdir(parents=True, exist_ok=True)
                    (relation_dir / f"{identifier_digest(relation_id)}.json").write_text(json.dumps({
                        "evidence_type": "factorial_contrast", "relation_id": relation_id,
                        "status": "rejected", "error": "relation proposals require two canonical endpoint revisions and families",
                    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    continue
                relation_scheduler = RelationExperimentScheduler()
                relation_family = str(endpoint_families["left"])
                try:
                    from benchmark.formal.schedule import FamilyPairReplayExecutor
                    endpoint_specs = {spec.rule_id: spec for spec in engine.rule_specs}
                    left_spec = endpoint_specs.get(str(endpoints["left"]))
                    right_spec = endpoint_specs.get(str(endpoints["right"]))
                    if left_spec is None or right_spec is None:
                        raise ValueError("relation endpoints are not active canonical rules")
                    left_action = str(left_spec.intervention.get("action_id") or left_spec.intervention.get("action") or "")
                    right_action = str(right_spec.intervention.get("action_id") or right_spec.intervention.get("action") or "")
                    if not left_action or not right_action:
                        raise ValueError("relation endpoints lack canonical ActionSpec")
                    pair_executor = FamilyPairReplayExecutor(relation_family, left_action, right_action, right_family=str(endpoint_families["right"]))
                    def block_executor(_context: Mapping[str, Any], *, context_id: str) -> list[Any]:
                        from core.acre.factorial import FactorialBlock
                        return [FactorialBlock(f"{context_id}-{index}", item.get("outcomes", item), scientific_gates=item["scientific_gates"]) for index, item in enumerate(pair_executor.execute(_context.get("workload", _context), context_id=context_id))]
                    relation_schedule = relation_scheduler.execute(
                        candidate, relation_family,
                        block_executor=block_executor,
                        maintainer=engine.maintainer,
                        seed=int(seed),
                    )
                except (KeyError, ValueError) as exc:
                    relation_schedule = {
                        "evidence_type": "factorial_contrast",
                        "relation_id": relation_id,
                        "family_id": relation_family,
                        "status": "rejected",
                        "error": str(exc),
                    }
                relation_dir = store / "evolution" / "relation_experiments"
                relation_dir.mkdir(parents=True, exist_ok=True)
                (relation_dir / f"{identifier_digest(relation_id)}.json").write_text(
                    json.dumps(relation_schedule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                identification = relation_schedule.get("identification", {}) if isinstance(relation_schedule, dict) else {}
                decision_label = str(identification.get("decision", "unresolved"))
                if decision_label not in {"unresolved", "underidentified_context_relation", "context_dependent_relation"}:
                    from core.acre.relation import RelationIdentifier, RelationIdentification
                    try:
                        relation_spec = RelationIdentifier(practical_margin=0.05).to_spec(
                            relation_id,
                            str(endpoints["left"]),
                            str(endpoints["right"]),
                            RelationIdentification(
                                decision=decision_label,
                                context_decisions=dict(identification.get("context_decisions", {})),
                                applicability_predicate=identification.get("applicability_predicate"),
                            ),
                        )
                        candidate_path = store / "evolution" / "relation_candidates" / f"{identifier_digest(relation_id)}.json"
                        candidate_path.parent.mkdir(parents=True, exist_ok=True)
                        candidate_path.write_text(json.dumps({
                            **relation_spec.to_dict(),
                            "status": "candidate",
                            "relation_evidence_certificates": relation_schedule.get("relation_evidence_certificates", {}),
                            "endpoint_versions": {str(key): int(value) for key, value in endpoint_versions.items()},
                        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    except (TypeError, ValueError):
                        pass
                continue
            identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or "")
            if not identifier:
                continue
            candidate = dict(candidate)
            validate_identifier(identifier, "candidate_id")
            task_patch = candidate.get("intervention") if isinstance(candidate.get("intervention"), dict) else {"action": "measure"}
            version = int(candidate.get("version", 1))
            try:
                persisted_activation = candidate.get("activation_certificate")
                if causal_result is None and isinstance(persisted_activation, dict) and isinstance(candidate.get("intervention"), dict):
                    # A collecting candidate is a persisted hypothesis.  It
                    # continues from its harness-owned ActionSpec/certificate
                    # and never asks the next worker to re-prove activation.
                    action_spec = dict(candidate["intervention"])
                    activation_certificate = dict(persisted_activation)
                else:
                    activation_proof = hashlib.sha256(json.dumps({
                        "task_id": task_id,
                        "causal_result": causal_result,
                        "control_result": control_result,
                        "patch": task_patch,
                    }, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest() if causal_result is not None and control_result is not None else None
                    activation_certificate = {
                        "action_id": str((candidate.get("action_spec") or candidate.get("intervention") or {}).get("action", "")),
                        "activation_metrics": {"causal": causal_result, "control": control_result},
                        "expected_signature": "verifier-paired",
                        "observed_signature": activation_proof,
                        "verifier_artifacts": {"task_id": task_id},
                        "realization_digest": candidate_intervention_digest(task_patch),
                        "passed": causal_result is not None and control_result is not None,
                    }
                    action_spec = semantic_action_spec(family_id, candidate, activation_certificate=activation_certificate)
            except ValueError:
                # An unclassified source patch cannot enter the causal
                # evidence ledger.  It remains a worker hypothesis until the
                # harness can prove a unique registered ActionSpec.
                continue
            action_semantic_digest = candidate_intervention_digest(action_spec)
            realization_digest = candidate_intervention_digest(task_patch)
            identity = candidate_identity(identifier, version, action_spec)
            try:
                from benchmark.families.catalog import FAMILY_SPECS, resolve_family_id
                family_spec = FAMILY_SPECS.get(resolve_family_id(str(family_id)))
            except (ImportError, AttributeError):
                family_spec = None
            scientific_invariants = list(getattr(family_spec, "scientific_invariants", ()) or ()) or ["task_scientific_gates"]
            severity = str(getattr(family_spec, "default_severity", "P2"))
            from core.cost import PromptCostModel
            runtime_tokens = PromptCostModel().cost({
                "rule_id": identifier,
                "version": version,
                "action": action_spec,
                "mechanism": str(candidate.get("expected_mechanism") or candidate.get("hypothesis") or "task-local performance mechanism"),
                "applicability": {"all": []},
                "invariants": scientific_invariants,
            })
            # Proposal fields are hypotheses only.  Applicability is generated
            # below by the harness-owned CEGIS pass.
            candidate = {
                "rule_id": identifier,
                "family_id": family_id,
                "version": version,
                "candidate_identity": identity,
                "intervention_digest": realization_digest,
                "action_semantic_digest": action_semantic_digest,
                "realization_digest": realization_digest,
                "parent": None,
                "applicability": {"all": []},
                "hypothesis": {
                    "intervention": action_spec,
                    "expected_mechanism": str(candidate.get("expected_mechanism") or candidate.get("hypothesis") or "task-local performance mechanism"),
                },
                "synthesis_state": {
                    "status": "collecting_evidence",
                    "predicate": None,
                    "version_space_digest": None,
                    "evidence_ids": [],
                },
                "intervention": action_spec,
                "realization": task_patch,
                "expected_mechanism": str(candidate.get("expected_mechanism") or candidate.get("hypothesis") or "task-local performance mechanism"),
                "evidence_requirements": ["paired_replay"],
                "scientific_invariants": scientific_invariants,
                "abstain_conditions": {},
                "relations": {},
                "runtime_cost": {"tokens": runtime_tokens},
                "provenance_policy": {"required": True},
                "severity": severity,
                "domain": "runtime",
                "text": str(candidate.get("text", "")),
                "scope": "formal",
                "status": "candidate",
                "epsilon": float(practical_epsilon),
                "p_min": 0.8,
                "delta": 0.05,
                "promotion_case_ids": [],
                "activation_certificate": activation_certificate,
            }
            candidate["cases"] = list(added_replay_case_ids)
            candidate["validation_artifacts"] = {
                "path": str(validation_path.relative_to(store)).replace("\\", "/"),
                "digest": validation_digest,
                "heldout_count": 1,
                "poison_probe_count": 1,
            }
            path = candidates_dir / f"{identifier_digest(identity)}.json"
            existing: dict[str, Any] = {}
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            if existing:
                immutable = ("rule_id", "relation_id", "version", "intervention", "expected_mechanism")
                for field in immutable:
                    if field in existing and field in candidate and existing[field] != candidate[field]:
                        raise ValueError(f"candidate immutable field changed for {identifier}: {field}")
                merged = dict(existing)
                merged.update({key: value for key, value in candidate.items() if key not in {"cases", "applicability", "synthesis_state"}})
                candidate = merged
                candidate["applicability"] = existing.get("applicability", {"all": []})
                candidate["synthesis_state"] = existing.get("synthesis_state", {
                    "status": "collecting_evidence",
                    "predicate": None,
                    "version_space_digest": None,
                    "evidence_ids": [],
                })
            # Membership is authoritative in CandidateEvidenceLedger.  The
            # candidate projection may retain case ids for display only; it
            # must never union mutable worker data into the evidence set.
            candidate["cases"] = sorted(set(candidate.get("cases", [])))
            # A candidate may accumulate independent replay cases over several
            # tasks.  Keep the validation artifact aligned with the complete
            # promotion bundle instead of leaving only the latest case id.
            validation_sources: list[dict[str, Any]] = []
            for artifact_owner in (existing, candidate):
                artifact_ref = artifact_owner.get("validation_artifacts") if isinstance(artifact_owner, dict) else None
                if not isinstance(artifact_ref, dict):
                    continue
                artifact_path = artifact_ref.get("path")
                if not isinstance(artifact_path, str):
                    continue
                try:
                    artifact_value = json.loads((store / artifact_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(artifact_value, dict):
                    validation_sources.append(artifact_value)
            if validation_sources:
                heldout_cases: dict[str, dict[str, Any]] = {}
                poison_cases: dict[str, dict[str, Any]] = {}
                independence_groups: set[str] = set()
                for artifact_value in validation_sources:
                    groups = artifact_value.get("independence_groups", [])
                    if isinstance(groups, list):
                        independence_groups.update(str(group) for group in groups if isinstance(group, str))
                    for key, target in (("heldout_regression_cases", heldout_cases), ("poison_probe_cases", poison_cases)):
                        entries = artifact_value.get(key, [])
                        if not isinstance(entries, list):
                            continue
                        for entry in entries:
                            if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
                                target.setdefault(entry["case_id"], entry)
                merged_validation = dict(validation_sources[-1])
                promotion_ids = candidate.get("promotion_case_ids")
                if not isinstance(promotion_ids, list):
                    promotion_ids = merged_validation.get("promotion_case_ids", [])
                promotion_ids = sorted({str(item) for item in promotion_ids if isinstance(item, str)})
                promotion_set = set(promotion_ids)
                # Boundary counterexamples remain synthesis evidence; only
                # cases outside the certified representative promotion set
                # may be listed as synthesis cases.
                merged_validation["synthesis_case_ids"] = sorted({
                    str(item) for item in candidate["cases"] if str(item) not in promotion_set
                })
                merged_validation["promotion_case_ids"] = promotion_ids
                merged_validation["heldout_regression_cases"] = list(heldout_cases.values())
                merged_validation["poison_probe_cases"] = list(poison_cases.values())
                merged_validation["independence_groups"] = sorted(independence_groups)
                merged_validation["synthesis_independence_groups"] = _case_independence_groups(
                    store, merged_validation["synthesis_case_ids"]
                )
                merged_validation["promotion_independence_groups"] = _case_independence_groups(
                    store, merged_validation["promotion_case_ids"]
                )
                validation_digest = hashlib.sha256(json.dumps(
                    merged_validation, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")).hexdigest()
                validation_path = store / "evolution" / "validation" / f"{validation_digest}.json"
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                if not validation_path.exists():
                    validation_path.write_text(json.dumps(merged_validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                candidate["validation_artifacts"] = {
                    "path": str(validation_path.relative_to(store)).replace("\\", "/"),
                    "digest": validation_digest,
                    "heldout_count": len(merged_validation["heldout_regression_cases"]),
                    "poison_probe_count": len(merged_validation["poison_probe_cases"]),
                }
            case_values_for_cegis: list[dict[str, Any]] = []
            # The candidate card is only a hypothesis projection.  Existing
            # membership is rehydrated from the append-only ledger; only the
            # verifier-owned cases produced in this task may be appended now.
            for case_id in added_replay_case_ids:
                try:
                    case_value = json.loads((store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(case_value, dict):
                    case_values_for_cegis.append(case_value)
                    case_value = dict(case_value)
                    case_value["case_path"] = str((store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json").relative_to(store)).replace("\\", "/")
                    candidate_evidence.append(
                        str(candidate.get("candidate_identity") or identifier),
                        int(candidate.get("version", 1)),
                        case_value,
                        action_digest=str(candidate.get("action_semantic_digest") or candidate.get("intervention_digest", "")),
                    )
            case_values_for_cegis = hydrate_candidate_cases(store, candidate, candidate_evidence)
            candidate["cases"] = sorted({str(case.get("case_id")) for case in case_values_for_cegis if case.get("case_id")})
            from benchmark.formal.schedule import PromotionReplayScheduler
            scheduler = PromotionReplayScheduler()
            seen_groups = {str(case.get("independence_group")) for case in case_values_for_cegis if case.get("independence_group")}
            from benchmark.formal.schedule import SynthesisAcquisitionScheduler
            active_scheduler = SynthesisAcquisitionScheduler()
            seen_contexts = {str(case.get("context_id")) for case in case_values_for_cegis if case.get("context_id")}
            existing_version_space = candidate.get("synthesis_state", {}).get("version_space", []) if isinstance(candidate.get("synthesis_state"), dict) else []
            active_pending = active_scheduler.plan(
                str(family_id or "compile"),
                seen_context_ids=seen_contexts,
                version_space=existing_version_space if isinstance(existing_version_space, list) else None,
            )
            active_pending = active_pending[: max(1, min(4, len(active_pending)))]
            representative_pending = scheduler.pending_contexts(
                str(family_id or "compile"), seen_group_ids=seen_groups,
            )
            candidate["replay_schedule"] = {
                "minimum_groups": scheduler.minimum_groups,
                "max_groups": scheduler.max_groups,
                "synthesis_contexts": active_pending,
                "promotion_contexts": representative_pending,
                # Kept as a derived display field; execution below never
                # treats active acquisition observations as promotion trials.
                "pending_contexts": active_pending + representative_pending,
                "acquisition_context_count": len(active_pending),
            }
            # Formal replay is fail-closed here.  Node promotion evidence must
            # come from the external experiment executor and verifier; the
            # calibration-only FamilyEnvironment path is intentionally not
            # available in this lifecycle transition.
            candidate["replay_schedule"]["experiment_cost"] = float(candidate["replay_schedule"].get("experiment_cost", 0.0))
            candidate["status"] = "collecting_evidence"
            candidate["synthesis_state"] = {
                **dict(candidate.get("synthesis_state") or {}),
                "status": "collecting_evidence",
                "evidence_ids": list(candidate["cases"]),
            }
            candidate.pop("applicability", None)
            candidate.pop("applicability_provenance", None)
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            synthesized = synthesize_applicability(
                case_values_for_cegis,
                family_id=family_id,
                statistical_budget=statistical_budget,
                epsilon_true=float(practical_epsilon),
                epsilon_false=0.0,
                require_identified=True,
            )
            # Version-space state is durable evidence even before a deployable
            # predicate is identified.  Persist it immediately so the next
            # task can hydrate the same candidate and active acquisition can
            # target the remaining decision-equivalence classes.
            synthesis_certificate = synthesized.certificate.to_dict() if synthesized.certificate else None
            candidate["synthesis_state"] = {
                **dict(candidate.get("synthesis_state") or {}),
                "status": str(synthesized.status),
                "predicate": synthesized.predicate,
                "version_space_digest": (synthesized.provenance or {}).get("version_space_digest"),
                "version_space": list(synthesized.version_space),
                "evidence_ids": list(candidate.get("cases", [])),
                "certificate": synthesis_certificate,
            }
            candidate["applicability_provenance"] = dict(synthesized.provenance or {})
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if synthesized.status not in {"identified", "underidentified"} or synthesized.predicate is None:
                continue
            intervention = candidate.get("realization") or candidate.get("intervention")
            if not isinstance(intervention, dict) or not isinstance(intervention.get("file"), str) or not isinstance(intervention.get("replacements"), list):
                continue
            predicate = synthesized.predicate
            provenance = dict(synthesized.provenance or {})
            certificate = synthesis_certificate
            promotion_ids = [
                str(item) for item in (certificate or {}).get("positive_anchor_ids", [])
                if isinstance(item, str)
            ]
            case_by_id = {str(item.get("case_id")): item for item in case_values_for_cegis if isinstance(item, dict)}
            promotion_ids = [
                case_id for case_id in promotion_ids
                if case_by_id.get(case_id, {}).get("query_type", "representative") == "representative"
            ]
            if not promotion_ids or synthesized.status != "identified":
                continue
            if not isinstance(certificate, dict):
                raise ValueError("synthesis certificate is required")
            candidate["applicability"], candidate["applicability_provenance"] = predicate, provenance
            validation_ref = candidate.get("validation_artifacts")
            if isinstance(validation_ref, dict) and isinstance(validation_ref.get("path"), str):
                validation_file = store / validation_ref["path"]
                try:
                    validation_value = json.loads(validation_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    validation_value = None
                if isinstance(validation_value, dict):
                    from core.models import RuleSpec, RuleState
                    from core.acre.router import ConservativeCausalRouter
                    action_for_validation = candidate.get("intervention") if isinstance(candidate.get("intervention"), dict) else {"action": "validated-action"}
                    validation_rule = RuleSpec(
                        rule_id=str(candidate.get("rule_id") or identifier), version=int(candidate.get("version", 1)),
                        parent=None, applicability=predicate, intervention=action_for_validation,
                        expected_mechanism=str(candidate.get("expected_mechanism", "validated mechanism")),
                        evidence_requirements=["paired_replay"], scientific_invariants=list(candidate.get("scientific_invariants", [])),
                        abstain_conditions={}, relations={}, runtime_cost={"tokens": float(candidate.get("runtime_cost", {}).get("tokens", 1)) if isinstance(candidate.get("runtime_cost"), dict) else 1.0},
                        provenance_policy={"required": True}, severity=str(candidate.get("severity", "P2")),
                    )
                    validation_state = RuleState(validation_rule.rule_id, validation_rule.version, "canonical", "stable", effect={"lower_utility": 0.1}, confidence_sequence={"utility_effect_lcb": 0.1})
                    for entry in validation_value.get("heldout_regression_cases", []):
                        if not isinstance(entry, dict) or not isinstance(entry.get("context"), dict):
                            continue
                        routed = ConservativeCausalRouter(token_budget=4096).route(
                            (validation_rule,), {validation_rule.rule_id: validation_state}, (), {}, entry["context"],
                        )
                        inside = validation_rule.rule_id in routed.selected_rule_ids
                        entry["routed_rule_ids"] = list(routed.selected_rule_ids)
                        if entry.get("holdout_class") == "boundary":
                            entry["abstained"] = not inside
                        elif entry.get("holdout_class") in {"replication", "transfer"} and not inside:
                            entry["scientific_ok"] = False
                    validation_digest = hashlib.sha256(json.dumps(validation_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                    validation_target = store / "evolution" / "validation" / f"{validation_digest}.json"
                    validation_target.parent.mkdir(parents=True, exist_ok=True)
                    validation_target.write_text(json.dumps(validation_value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    candidate["validation_artifacts"] = {
                        "path": str(validation_target.relative_to(store)).replace("\\", "/"),
                        "digest": validation_digest,
                        "heldout_count": len(validation_value.get("heldout_regression_cases", [])),
                        "poison_probe_count": len(validation_value.get("poison_probe_cases", [])),
                    }
            candidate["synthesis_state"] = {
                "status": "identified",
                "predicate": predicate,
                "version_space_digest": provenance.get("version_space_digest"),
                "version_space": provenance.get("version_space", []),
                "evidence_ids": list(candidate["cases"]),
                "certificate": certificate,
            }
            candidate["promotion_case_ids"] = sorted(set(promotion_ids))
            rewrite_validation_membership(
                store,
                candidate,
                synthesis_case_ids=list(candidate["cases"]),
                promotion_case_ids=promotion_ids,
            )
            candidate["status"] = "candidate"
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            candidates_by_id[str(candidate.get("candidate_identity") or identifier)] = candidate
        for path in sorted(candidates_dir.glob("*.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            synthesis_state = candidate.get("synthesis_state") if isinstance(candidate.get("synthesis_state"), dict) else {}
            provenance = candidate.get("applicability_provenance") if isinstance(candidate.get("applicability_provenance"), dict) else {}
            if (
                candidate.get("cases")
                and candidate.get("status") == "candidate"
                and synthesis_state.get("status") == "identified"
                and int(provenance.get("decision_context_count", 0)) > 0
            ):
                identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or path.stem)
                candidates_by_id[str(candidate.get("candidate_identity") or identifier)] = candidate
        candidates = list(candidates_by_id.values())
        for candidate in candidates:
            hydrated_cases: list[dict[str, Any]] = []
            subject_id = str(candidate.get("candidate_identity") or candidate.get("rule_id") or candidate.get("id") or "")
            memberships = candidate_evidence.members(subject_id, int(candidate.get("version", 1)), action_digest=str(candidate.get("action_semantic_digest") or candidate.get("intervention_digest", "")) or None)
            membership_by_id = {str(item.get("case_id")): item for item in memberships}
            case_ids = list(membership_by_id)
            for case_id in case_ids:
                try:
                    case_path = store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json"
                    case_value = json.loads(case_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(case_value, dict):
                    membership = membership_by_id.get(str(case_id))
                    if membership and membership.get("case_sha256"):
                        actual_case_digest = hashlib.sha256(json.dumps(case_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                        if actual_case_digest != membership["case_sha256"]:
                            continue
                    hydrated_cases.append(case_value)
            candidate["cases"] = hydrated_cases
        if candidates:
            promoted_rule_ids = promote_via_replay(store, candidates, core_repo, out_dir, active_ledger)
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "governed_maintenance"
    elif condition == "D":
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "maintenance_blocked"
    elif condition in {"C", "C_STRESS"}:
        valid, errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("raw-experience transition failed policy validation: " + "; ".join(errors))
        transition = "raw_experience_capture"
    else:
        transition = "no_op"

    if condition in {"C", "C_STRESS", "D"}:
        conditions.refresh_attestation(store)
        valid, errors = conditions.verify_attestation(store)
        if not valid:
            raise ValueError("store attestation failed after mutation: " + "; ".join(errors))
    post_digest = conditions.store_digest(store)
    return {
        "status": transition,
        "pre_store_digest": pre_digest,
        "post_store_digest": post_digest,
        "added_experience_ids": added_experience_ids,
        "added_replay_case_ids": added_replay_case_ids,
        "maintenance_decisions": maintenance_decisions,
        "promoted_rule_ids": promoted_rule_ids,
    }


def _safe_post_task_update(*, budget_errors: list[str], **kwargs: Any) -> dict[str, Any]:
    """Record a lifecycle/audit failure without aborting other streams."""
    try:
        return post_task_update(**kwargs)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        budget_errors.append(f"state mutation/audit error: {exc}")
        store = Path(kwargs["store"])
        digest = conditions.store_digest(store)
        return {
            "status": "state_mutation_error",
            "error": repr(exc),
            "pre_store_digest": digest,
            "post_store_digest": digest,
            "added_experience_ids": [],
            "added_replay_case_ids": [],
            "maintenance_decisions": [],
            "promoted_rule_ids": [],
        }


def _experiment_manifest(
    *,
    repo_root: Path,
    tasks_root: Path,
    skill_digest: str,
    task_digest: str,
    task_ids: list[str],
    item: dict[str, Any],
    model_id: str,
    agent_config: dict[str, Any],
    budgets: budget.Budget,
    fingerprint: dict[str, Any],
    trial_dir: Path,
    population_id: str = "SPE-EvoBench-v1.0-30-pilot",
) -> dict[str, Any]:
    task_spec = miniyaml.load(str(tasks_root / item["task_id"] / "task.yaml"))
    lineage = task_spec.get("lineage", {}) if isinstance(task_spec, dict) else {}
    return {
        "schema_version": 1,
        "experiment_id": f"{population_id}-{item['stream_id']}-{item['task_id']}",
        "population_id": population_id,
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "agent_model_id": model_id,
        "agent_config": {
            **agent_config,
            "provider": agent_config.get("provider", "unknown"),
            "model_snapshot": agent_config.get("model_snapshot", model_id),
            "temperature": agent_config.get("temperature"),
            "top_p": agent_config.get("top_p"),
            "max_output_tokens": agent_config.get("max_output_tokens"),
            "system_prompt_digest": agent_config.get("system_prompt_digest"),
            "agent_code_commit": agent_config.get("agent_code_commit", attest.benchmark_revision(repo_root)),
            "tool_versions": agent_config.get("tool_versions", {}),
            "tool_allowlist": agent_config.get("tool_allowlist", []),
            "retry_policy": agent_config.get("retry_policy", {}),
            "timeout_s": agent_config.get("timeout_s", budgets.wall_time_s),
            "container_digest": agent_config.get("container_digest"),
            "pricing_revision": agent_config.get("pricing_revision"),
        },
        "condition": item["condition"],
        "context_mode": item["context_mode"],
        "worker_isolation": {
            "mode": "external_namespace_executor",
            "network_mode": "none",
            "mount_allowlist": ["task", "solution", "skill_view", "retrieved_context", "result", "executor_receipt"],
            "executor_receipt": None,
        },
        "task_order": task_ids,
        "family_id": task_spec.get("family_id", task_spec.get("family", "unknown")),
        "anchor_instance_id": task_spec.get("anchor_instance_id", item["task_id"]),
        "family_instance_digest": task_spec.get("family_instance_digest"),
        "lineage_id": lineage.get("mutation_template_id", item["task_id"]),
        "outer_trial_id": item["outer_trial_id"],
        "budgets": budgets.as_dict(),
        "hardware_fingerprint": fingerprint,
        "software_fingerprint": {
            "python_version": fingerprint.get("python_version"),
            "platform": fingerprint.get("platform"),
            "torch_geometric_version": fingerprint.get("torch_geometric_version"),
        },
        "torch_version": fingerprint.get("torch_version"),
        "cuda_version": fingerprint.get("cuda_version"),
        "compiler_cache": {
            "policy": "verifier-invocation-scoped",
            "root": str(trial_dir / "compiler-cache"),
            "non_shared_between_verifier_invocations": True,
        },
    }


def _formal_claim_gate(campaign: dict[str, Any], records: list[dict[str, Any]], report_path: Path) -> bool:
    """Allow a claim only after an attested calibration approval."""
    if campaign.get("status") != "complete" or not records:
        return False
    if any(record.get("validity") != "valid" for record in records):
        return False
    expected = int(campaign.get("schedule_size", 0))
    if len(records) != expected:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    calibration = report.get("empirical_calibration", {})
    if calibration.get("calibration_gate") != "ready_for_review":
        return False
    approval_path = report_path.parent / "calibration" / "calibration_approval.json"
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pilot_path = report_path.parent / "calibration" / "pilot_calibration.json"
    try:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(pilot, dict) or pilot.get("calibration_gate") != "ready_for_review":
        return False
    repo_root = report_path.parents[1]
    if validate_calibration_approval(report, pilot, approval, repo_root=repo_root):
        return False
    release_path = repo_root / "benchmark" / "formal_release_manifest.json"
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not validate_formal_release(
        release, repo_root=repo_root,
        population_manifest=repo_root / "benchmark" / "manifests" / "v1.0-50-slots.json",
        approval_path=approval_path, claims_path=repo_root / "CLAIMS.yaml",
        protocol_path=repo_root / "references" / "STATISTICAL_PROTOCOL.md",
        campaign_config=repo_root / "benchmark" / "formal" / "campaign_config.yaml",
    )


def _resume_stream_prefix(
    stream_dir: Path,
    task_items: list[tuple[str, str]],
    task_position: Mapping[str, int],
    stream_id: str,
    blocked_streams: set[str],
) -> tuple[int, dict[tuple[str, str], dict[str, Any]], Path]:
    """Load one contiguous stream prefix and validate only its final store."""
    store = stream_dir / "condition-store"
    records: dict[tuple[str, str], dict[str, Any]] = {}
    expected_prefix = 0
    last_post_digest: str | None = None
    for _phase, task_id in task_items:
        trial_path = stream_dir / task_id / "trial.json"
        if not trial_path.is_file():
            break
        try:
            record = json.loads(trial_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid resume trial {trial_path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"resume trial must be an object: {trial_path}")
        transition = record.get("transition") if isinstance(record.get("transition"), dict) else {}
        if transition.get("post_store_digest") is not None:
            last_post_digest = str(transition["post_store_digest"])
        if transition.get("status") == "state_mutation_error" or record.get("attestation_ok") is False:
            blocked_streams.add(stream_id)
        records[(stream_id, task_id)] = record
        expected_prefix += 1
    if expected_prefix and last_post_digest is not None:
        if not store.is_dir() or conditions.store_digest(store) != last_post_digest:
            raise ValueError(f"resume final store digest mismatch for {stream_id}")
    for candidate in stream_dir.iterdir() if stream_dir.is_dir() else ():
        if candidate.name == "condition-store" or not candidate.is_dir():
            continue
        if candidate.name in task_position and task_position[candidate.name] >= expected_prefix and (candidate / "trial.json").is_file():
            raise ValueError(f"resume requires a contiguous completed prefix for {stream_id}")
    return expected_prefix, records, store


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    tasks_root = Path(args.tasks_root).resolve()
    split_path = Path(args.split).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    population_manifest = Path(getattr(args, "population_manifest", None) or (repo_root / "benchmark" / "manifests" / "v1.0-50-slots.json")).resolve()
    campaign_config_path = getattr(args, "campaign_config", None)
    claims_path = Path(getattr(args, "claims", None) or (repo_root / "CLAIMS.yaml")).resolve()
    analysis_plan_path = Path(getattr(args, "analysis_plan", None) or (repo_root / "references" / "STATISTICAL_PROTOCOL.md")).resolve()
    if getattr(args, "formal", False):
        _validate_formal_entry(repo_root, population_manifest, campaign_config_path, claims_path, analysis_plan_path, getattr(args, "sealed_tasks_root", None))
        sealed_root = getattr(args, "sealed_tasks_root", None) or json.loads(population_manifest.read_text(encoding="utf-8")).get("sealed_root")
        if not sealed_root:
            raise ValueError("formal mode requires --sealed-tasks-root or manifest sealed_root")
        tasks_root = Path(sealed_root).resolve()
    population_id = "SPE-EvoBench-v1.0-50" if getattr(args, "formal", False) else "SPE-EvoBench-v1.0-30-pilot"
    resume = bool(getattr(args, "resume", False))
    existing_campaign: dict[str, Any] | None = None
    if resume:
        campaign_path = out_dir / "campaign.json"
        if not campaign_path.is_file():
            raise ValueError("--resume requires an existing campaign.json")
        try:
            loaded_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid campaign manifest for resume: {exc}") from exc
        if not isinstance(loaded_campaign, dict):
            raise ValueError("campaign manifest for resume must be an object")
        if loaded_campaign.get("status") == "complete":
            return loaded_campaign
        existing_campaign = loaded_campaign
    if getattr(args, "formal", False):
        formal_manifest = json.loads(population_manifest.read_text(encoding="utf-8"))
        task_items = schedule.task_order_from_materialized(formal_manifest)
    else:
        task_items = schedule.task_order(split_path)
    task_ids = [task_id for _, task_id in task_items]
    if args.skill_view:
        skill_view = Path(args.skill_view).resolve()
        if not (skill_view / "skill_view_manifest.json").is_file():
            raise ValueError("--skill-view must point to a rendered skill bundle")
        if validate_skill_view_bundle(skill_view):
            raise ValueError("--skill-view failed bundle validation")
    else:
        skill_view = out_dir / "skill-view"
        if not resume or not (skill_view / "skill_view_manifest.json").is_file():
            render_skill_view(args.skill_source, skill_view)
    skill_digest = attest.skill_view_digest(skill_view)
    task_digest = attest.task_manifest_digest(tasks_root, task_ids)
    population_digest = hashlib.sha256(population_manifest.read_bytes()).hexdigest() if population_manifest.is_file() else "unknown"
    conditions_list = tuple(item.strip().upper() for item in args.conditions.split(",") if item.strip())
    context_modes = tuple(item.strip() for item in args.context_modes.split(",") if item.strip())
    if getattr(args, "formal", False):
        _validate_campaign_config(
            Path(campaign_config_path), conditions=conditions_list, context_modes=context_modes,
            outer_trials=int(args.outer_trials), schedule_seed=int(getattr(args, "schedule_seed", 0)),
            population_manifest=population_manifest, repo_root=repo_root,
        )
    if getattr(args, "formal", False):
        plan = schedule.build_task_block_schedule(
            task_items, conditions=conditions_list, context_modes=context_modes,
            outer_trials=args.outer_trials, schedule_seed=int(getattr(args, "schedule_seed", 0)),
            task_visibility={str(slot["task_id"]): str(slot["visibility"]) for slot in formal_manifest.get("slots", [])},
        )
    else:
        plan = schedule.build_schedule(
            split_path, conditions=conditions_list, context_modes=context_modes,
            outer_trials=args.outer_trials, schedule_seed=int(getattr(args, "schedule_seed", 0)),
        )
    budgets = budget.parse_budget(json.loads(args.budgets) if args.budgets else None)
    statistical_budget = StatisticalBudget()
    experiment_executor = _build_required_experiment_executor(
        getattr(args, "experiment_executor_command", None), out_dir,
    )
    fingerprint = capture_fingerprint()
    campaign = {
        "schema_version": 1,
        "population_id": population_id,
        "status": "planned" if not args.agent_command else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "conditions": list(conditions_list),
        "context_modes": list(context_modes),
        "outer_trials": args.outer_trials,
        "task_order": task_ids,
        "budgets": budgets.as_dict(),
        "agent_model_id": args.model_id,
        "agent_config": json.loads(args.agent_config),
        "executor_digest": getattr(args, "executor_digest", None),
        "schedule_size": len(plan),
        "results_claimed": False,
        "population_manifest": str(population_manifest),
        "claims_path": str(claims_path),
        "analysis_plan_path": str(analysis_plan_path),
        "evolution_compute_budget": budget.EvolutionComputeBudget().as_dict(),
    }
    if existing_campaign is not None:
        immutable_keys = (
            "population_id", "benchmark_revision", "skill_view_digest", "task_manifest_digest",
            "conditions", "context_modes", "outer_trials", "task_order", "budgets",
            "agent_model_id", "agent_config", "schedule_size",
            "executor_digest",
        )
        mismatches = [key for key in immutable_keys if existing_campaign.get(key) != campaign.get(key)]
        if mismatches:
            raise ValueError("resume manifest mismatch: " + ", ".join(mismatches))
        campaign["created_utc"] = existing_campaign.get("created_utc", campaign["created_utc"])
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.agent_command:
        (out_dir / "schedule.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return campaign
    if not getattr(args, "executor_digest", None):
        raise ValueError("formal agent runs require an allowlisted --executor-digest")

    # Calibrate each task×outer-trial once.  All A/B/C/D verifier invocations
    # below consume this immutable artifact; no invocation reruns controls.
    noise_control_root = out_dir / "noise-control"
    noise_control_paths: dict[tuple[str, str], Path] = {}
    noise_control_failures: dict[tuple[str, str], dict[str, Any]] = {}
    unique_noise_cells = sorted({(str(item["task_id"]), str(item["outer_trial_id"])) for item in plan})
    for noise_task_id, noise_outer_trial_id in unique_noise_cells:
        artifact_path = noise_control_root / noise_outer_trial_id / f"{noise_task_id}.json"
        noise_spec = miniyaml.load(str(tasks_root / noise_task_id / "task.yaml"))
        expected_noise = {
            "task_id": noise_task_id,
            "outer_trial_id": noise_outer_trial_id,
            "benchmark_revision": campaign["benchmark_revision"],
            "task_manifest_digest": task_digest,
            "task_package_digest": attest.task_package_digest(tasks_root / noise_task_id),
            "population_manifest_digest": population_digest,
            "control_implementation": "baseline",
            "hardware_fingerprint": fingerprint,
            "software_fingerprint": fingerprint,
            "compiler_cache_policy": verifier.cache_policy_for_task(noise_spec),
            "expected_speedup_range": noise_spec.get("oracle", {}).get("expected_speedup_range"),
        }
        if resume and artifact_path.is_file():
            try:
                stats.read_noise_control(artifact_path, expected_noise)
                noise_control_paths[(noise_task_id, noise_outer_trial_id)] = artifact_path
                continue
            except ValueError as exc:
                noise_control_failures[(noise_task_id, noise_outer_trial_id)] = {"status": "resource_blocked", "error": str(exc)}
                continue
        calibration = _calibrate_noise_control_with_cache(
            tasks_root / noise_task_id,
            artifact_path,
            calibration_dir=out_dir / "noise-control" / noise_outer_trial_id / f"{noise_task_id}-runtime",
            task_id=noise_task_id,
            outer_trial_id=noise_outer_trial_id,
            benchmark_revision=campaign["benchmark_revision"],
            task_manifest_digest=task_digest,
            task_package_digest=expected_noise["task_package_digest"],
            population_manifest_digest=population_digest,
            timeout_s=float(min(300.0, budgets.wall_time_s)),
        )
        if calibration.get("ok"):
            try:
                stats.read_noise_control(artifact_path, expected_noise)
                noise_control_paths[(noise_task_id, noise_outer_trial_id)] = artifact_path
            except ValueError as exc:
                noise_control_failures[(noise_task_id, noise_outer_trial_id)] = {"status": "resource_blocked", "error": str(exc)}
        else:
            noise_control_failures[(noise_task_id, noise_outer_trial_id)] = calibration
    campaign["noise_control"] = {
        "policy": "task-outer-trial-preregistered-shared",
        "root": str(noise_control_root),
        "cells": len(unique_noise_cells),
        "calibrated": len(noise_control_paths),
        "resource_blocked": len(noise_control_failures),
    }
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records: list[dict[str, Any]] = []
    existing_budget = (existing_campaign or {}).get("evolution_compute_budget", {}) if isinstance(existing_campaign, dict) else {}
    evolution_budget = budget.EvolutionComputeBudget(**{key: existing_budget.get(key, 0) for key in budget.EvolutionComputeBudget.__dataclass_fields__})
    stores: dict[str, Path] = {}
    context_paths: dict[str, Path] = {}
    ledgers: dict[str, EvolutionDecisionLedger] = {}
    task_position = {task_id: index for index, (_phase, task_id) in enumerate(task_items)}
    resume_records: dict[tuple[str, str], dict[str, Any]] = {}
    resume_prefix: dict[str, int] = {}
    blocked_streams: set[str] = set()
    if resume:
        for item in plan:
            stream_id = str(item["stream_id"])
            resume_prefix.setdefault(stream_id, 0)
        for stream_id in resume_prefix:
            stream_dir = out_dir / "trials" / stream_id
            expected_prefix, stream_records, store = _resume_stream_prefix(
                stream_dir, task_items, task_position, stream_id, blocked_streams,
            )
            resume_records.update(stream_records)
            if expected_prefix:
                stores[stream_id] = store
            resume_prefix[stream_id] = expected_prefix
    for item in plan:
        stream_id = str(item["stream_id"])
        task_id = str(item["task_id"])
        if resume and task_position[task_id] < resume_prefix.get(stream_id, 0):
            records.append(resume_records[(stream_id, task_id)])
            continue
        task_spec = miniyaml.load(str(tasks_root / task_id / "task.yaml"))
        noise_control_path = noise_control_paths.get((task_id, str(item["outer_trial_id"])))
        noise_expected = {
            "task_id": task_id,
            "outer_trial_id": str(item["outer_trial_id"]),
            "benchmark_revision": campaign["benchmark_revision"],
            "task_manifest_digest": task_digest,
            "task_package_digest": attest.task_package_digest(tasks_root / task_id),
            "population_manifest_digest": population_digest,
            "control_implementation": "baseline",
            "hardware_fingerprint": fingerprint,
            "software_fingerprint": fingerprint,
            "compiler_cache_policy": verifier.cache_policy_for_task(task_spec),
            "expected_speedup_range": task_spec.get("oracle", {}).get("expected_speedup_range"),
        }
        trial_dir = out_dir / "trials" / stream_id / task_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        manifest = _experiment_manifest(
            repo_root=repo_root,
            tasks_root=tasks_root,
            skill_digest=skill_digest,
            task_digest=task_digest,
            task_ids=task_ids,
            item=item,
            model_id=args.model_id,
            agent_config=json.loads(args.agent_config),
            budgets=budgets,
            fingerprint=fingerprint,
            trial_dir=trial_dir,
            population_id=population_id,
        )
        manifest["schedule_digest"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        manifest["analysis_plan_digest"] = hashlib.sha256(analysis_plan_path.read_bytes()).hexdigest() if analysis_plan_path.is_file() else None
        manifest["population_manifest_digest"] = hashlib.sha256(population_manifest.read_bytes()).hexdigest() if population_manifest.is_file() else None
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        if stream_id in blocked_streams:
            record = {
                "experiment": manifest,
                "task_id": task_id,
                "family": task_spec.get("family"),
                "family_id": task_spec.get("family_id", task_spec.get("family")),
                "condition": item["condition"],
                "context_mode": item["context_mode"],
                "outer_trial_id": item["outer_trial_id"],
                "phase": item["phase"],
                "agent": None,
                "agent_usage": {},
                "budget_errors": ["stream blocked after prior state mutation/audit error"],
                "attestation_ok": False,
                "execution_validity": "resource_blocked",
                "task_outcome": "error",
                "efficacy_eligible": False,
                "validity": "invalid",
                "transition": {"status": "stream_blocked"},
                "score": {"task_id": task_id, "verdict": "error", "gates_passed": False, "task_score": 0.0},
            }
            (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            records.append(record)
            continue
        state_path = context_paths.setdefault(stream_id, trial_dir.parent / "context.json")
        if item["context_mode"] == "reset" and state_path.exists():
            state_path.unlink()
        store = stores.get(stream_id)
        if store is None:
            store = trial_dir.parent / "condition-store"
            conditions.materialize_condition(item["condition"], skill_view if item["condition"] != "A" else None, store, item["context_mode"])
            stores[stream_id] = store
        elif resume:
            valid_store, store_errors = conditions.verify_attestation(store)
            if not valid_store:
                raise ValueError("resume condition store failed attestation: " + "; ".join(store_errors))
        solution_dir = trial_dir / "solution"
        _copy_workspace(tasks_root / task_id, solution_dir)
        agent_task_dir = trial_dir / "agent-task"
        materialize_agent_task(tasks_root / task_id, agent_task_dir)
        public_task = json.loads((agent_task_dir / "public_task.json").read_text(encoding="utf-8"))
        public_routing = public_task.get("routing_context", {}) if isinstance(public_task, dict) else {}
        if not isinstance(public_routing, dict):
            public_routing = {}
        retrieved_context_path = agent_task_dir / "retrieved_context.json"
        if str(item["condition"]) in {"B", "C", "C_STRESS", "D"}:
            try:
                adapter = FormalConditionAdapter(
                    str(item["condition"]),
                    store,
                    token_budget=budgets.context_tokens,
                    family_id=str(task_spec.get("family_id", task_spec.get("family", ""))) or None,
                )
                retrieval_input = {
                    "domain": public_routing.get("domain", "scientific-performance"),
                    "workload": dict(public_routing.get("workload", {})),
                    "hardware": dict(public_routing.get("hardware", {})),
                    "software": dict(public_routing.get("software", {})),
                    "evidence": dict(public_routing.get("evidence", {})),
                    "token_budget": budgets.context_tokens,
                }
                retrieved_context = adapter.retrieved_context(retrieval_input)
                exposed_context = retrieved_context.get("context", {})
                for key in ("domain", "workload", "hardware", "software", "evidence"):
                    if exposed_context.get(key) != retrieval_input.get(key):
                        raise ValueError(f"retrieval context escaped public task context at {key}")
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                record = _trial_failure_record(manifest, task_spec, item, str(exc), source="context")
                (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                records.append(record)
                continue
        elif item["condition"] == "A_CTX":
            retrieved_context = {
                "schema_version": 1,
                "context": {"placebo": True, "context_mode": str(item["context_mode"]), "token_budget": budgets.context_tokens},
                "proposed_interventions": [],
            }
        else:
            retrieved_context = {
                "schema_version": 1,
                "context": {"context_mode": str(item["context_mode"])},
                "proposed_interventions": [],
            }
        retrieved_context_path.write_text(json.dumps(retrieved_context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        worker_root = _prepare_worker_root(trial_dir, agent_task_dir, solution_dir, skill_view if item["condition"] in {"B", "C", "D", "C_STRESS"} else None)
        worker_context_state = worker_root / "context_state" / "context_state.json"
        if item["context_mode"] == "carry" and state_path.is_file():
            shutil.copy2(state_path, worker_context_state)
        else:
            worker_context_state.write_text(json.dumps({"context_mode": item["context_mode"], "trajectory": []}) + "\n", encoding="utf-8")
        worker_task_dir = worker_root / "task"
        worker_solution_dir = worker_root / "solution"
        worker_retrieved_context_path = worker_root / "retrieved_context" / "retrieved_context.json"
        worker_retrieved_context_path.write_text(retrieved_context_path.read_text(encoding="utf-8"), encoding="utf-8")
        worker_result_path = worker_root / "result" / "worker_result.json"
        # The receipt is written by the executor outside the worker namespace;
        # placing it under worker/ would let the worker author its own trust
        # metadata.
        receipt_path = trial_dir / "executor_receipt.json"
        env = {
            "SPE_TASK_ID": task_id,
            "SPE_TASK_DIR": "/worker/task",
            "SPE_SOLUTION_DIR": "/worker/solution",
            "SPE_CONTEXT_MODE": str(item["context_mode"]),
            "SPE_RETRIEVED_CONTEXT": "/worker/retrieved_context/retrieved_context.json",
            "SPE_RESULT_PATH": "/worker/result/worker_result.json",
            "SPE_AGENT_USAGE_PATH": "/worker/result/agent_usage.json",
            "SPE_EXECUTOR_RECEIPT_PATH": str(receipt_path),
            "SPE_SKILL_VIEW_DIR": "/worker/skill_view" if item["condition"] in {"B", "C", "D", "C_STRESS"} else "",
            "SPE_BUDGET_JSON": json.dumps(budgets.as_dict(), sort_keys=True),
            "SPE_OUTER_TRIAL_ID": str(item["outer_trial_id"]),
            "SPE_CONTEXT_STATE_PATH": "/worker/context_state/context_state.json",
        }
        if not getattr(args, "executor_command", None):
            raise ValueError("formal agent runs require --executor-command with a namespace/container executor")
        try:
            agent = _run_isolated_agent(args.agent_command, args.executor_command, env, worker_root, budgets.wall_time_s)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            record = _trial_failure_record(manifest, task_spec, item, str(exc), source="executor")
            (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
            continue
        receipt, receipt_errors = _read_executor_receipt(
            receipt_path,
            None if item["condition"] == "A" else skill_digest,
            str(item["context_mode"]),
            getattr(args, "executor_digest", None),
        )
        usage = receipt.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        usage_errors = budgets.validate_usage(usage)
        budget_errors = list(usage_errors) + list(receipt_errors)
        failure_policy = budget.classify_failure(
            failure_stage="executor" if receipt_errors else ("agent" if usage_errors else None),
            protocol_failure=bool(receipt_errors),
            budget_exhausted=bool(usage_errors),
        )
        execution_valid = bool(failure_policy["efficacy_eligible"])
        manifest["worker_isolation"]["executor_receipt"] = _manifest_executor_receipt(receipt, receipt_errors)
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        if receipt_errors:
            failure = _trial_failure_record(
                manifest,
                task_spec,
                item,
                "; ".join(receipt_errors),
                source="executor",
            )
            failure.update({
                "agent": agent,
                "receipt_valid": False,
                "attestation_ok": False,
                "execution_validity": "invalid",
                "executor_receipt": receipt,
                "failure_stage": "executor",
                "failure_class": "infrastructure",
                "verifier_called": False,
                "activation": "not_evaluated",
                "cleanup_status": "executor_returned",
                "store_mutated": False,
            })
            (trial_dir / "trial.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            records.append(failure)
            continue
        if agent["returncode"] != 0:
            worker_budget_errors = list(budget_errors)
            failure = _trial_failure_record(
                manifest,
                task_spec,
                item,
                f"agent command failed with return code {agent['returncode']}",
                source="worker",
            )
            # Agent timeout/crash/budget exhaustion is a protocol-valid
            # outcome failure: retain the cell with score zero.  Only receipt
            # or verifier infrastructure failures are execution-invalid.
            worker_valid = True
            failure.update({
                "agent": agent,
                "receipt_valid": True,
                "attestation_ok": True,
                "execution_validity": "valid" if worker_valid else "invalid",
                "executor_receipt": receipt,
                "failure_stage": "worker",
                "failure_class": "candidate",
                "verifier_called": False,
                "activation": "not_evaluated",
                "cleanup_status": "executor_returned",
                "store_mutated": False,
                "budget_errors": worker_budget_errors,
                "task_outcome": "fail",
                "efficacy_eligible": worker_valid,
                "validity": "valid" if worker_valid else "invalid",
                "score": {"task_id": str(task_id), "verdict": "fail", "gates_passed": False, "task_score": 0.0},
                "transition": {"status": "worker_failed", "pre_store_digest": None, "post_store_digest": None},
            })
            (trial_dir / "trial.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            records.append(failure)
            continue
        agent_extensions = _read_agent_extensions(worker_result_path)
        submitted_solution_dir = trial_dir / "submitted_solution"
        try:
            sanitize_submission(
                worker_solution_dir,
                submitted_solution_dir,
                {str(item) for item in public_task.get("allowed_files", []) if isinstance(item, str)},
                baseline=solution_dir,
            )
        except ValueError as exc:
            budget_errors.append(f"submission sanitization failed: {exc}")
            submitted_solution_dir.mkdir(parents=True, exist_ok=True)
        control_result: dict[str, Any] | None = None
        control_scored: dict[str, Any] | None = None
        causal_result: dict[str, Any] | None = None
        causal_scored: dict[str, Any] | None = None
        validation_evidence: dict[str, Any] | None = None
        if str(item["condition"]) == "D":
            control_result, control_scored = _verify_baseline(
                tasks_root / task_id,
                solution_dir,
                trial_dir / "control-result.json",
                condition=str(item["condition"]),
                context_mode=str(item["context_mode"]),
                seed=int(item["outer_trial_index"]),
                trial_dir=trial_dir,
                invocation_id="control",
                timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
                noise_control_path=noise_control_path,
                noise_control_expected=noise_expected,
            )
            _check_verifier_budget(control_result, task_spec, budget_errors, "control verifier")
        result = _verify_task_with_cache(
            tasks_root / task_id,
            submitted_solution_dir,
            out_path=trial_dir / "result.json",
            condition=str(item["condition"]),
            context_mode=str(item["context_mode"]),
            seed=int(item["outer_trial_index"]),
            predicted_mechanism=[str(value) for value in agent_extensions.get("predicted_mechanisms", []) if isinstance(value, str)],
            trial_dir=trial_dir,
            invocation_id="candidate",
            timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
            noise_control_path=noise_control_path,
            noise_control_required=True,
            noise_outer_trial_id=str(item["outer_trial_id"]),
            noise_benchmark_revision=campaign["benchmark_revision"],
            noise_task_manifest_digest=task_digest,
            noise_control_expected=noise_expected,
        )
        _check_verifier_budget(result, task_spec, budget_errors, "candidate verifier")
        result["seed"] = int(item["outer_trial_index"])
        result.update(agent_extensions)
        predicted = [str(value) for value in result.get("predicted_mechanisms", []) if isinstance(value, str)]
        # Diagnosis correctness is computed inside the immutable verifier;
        # the driver only preserves the worker's submitted prediction.
        if isinstance(result.get("diagnosis"), dict):
            result["diagnosis"]["predicted_mechanisms"] = predicted
        result["abstained"] = bool(result.get("abstain", False))
        if result["abstained"]:
            proposals = [item for item in agent_extensions.get("acre_proposals", []) if isinstance(item, dict)]
            if _workspace_digest(submitted_solution_dir) != _workspace_digest(solution_dir) or proposals:
                budget_errors.append("abstain declaration conflicts with submitted artifact or executable proposal")
                result["validity"] = "invalid"
        result["condition_adapter"] = retrieved_context
        routing_payload = retrieved_context.get("routing", {}) if isinstance(retrieved_context, Mapping) else {}
        required_requests = routing_payload.get("required_experiments", []) if isinstance(routing_payload, Mapping) else []
        if str(item["condition"]) == "D" and isinstance(required_requests, list) and required_requests:
            # Router-owned requests are consumed by the harness.  A formal
            # campaign without an external executable callback remains
            # explicitly blocked; it is never replaced by synthetic replay.
            try:
                result["required_experiments"] = schedule.execute_required_experiments(required_requests, executor=experiment_executor)
                if any(isinstance(item, Mapping) and item.get("status") == "resource_blocked" for item in result["required_experiments"]):
                    budget_errors.append("required experiment resource_blocked")
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                budget_errors.append(f"required experiment protocol failure: {exc}")
                result["required_experiments"] = [
                    {**dict(request), "status": "resource_blocked", "reason": "required experiment execution failed"}
                    for request in required_requests if isinstance(request, Mapping)
                ]
        result.setdefault("cost", {}).update({
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "tool_calls": usage.get("tool_calls"),
        })
        scored = scoring.score_task(result)
        if str(item["condition"]) == "D":
            proposals = [item for item in agent_extensions.get("acre_proposals", []) if isinstance(item, dict)]
            if len(proposals) == 1:
                try:
                    realized_solution = trial_dir / "realized_solution"
                    # Source patch realization is intentionally semantic-free.
                    # ActionSpec classification is deferred until the
                    # realized artifact has passed the causal verifier.
                    raw_realization = InterventionRealizer.realize_raw(
                        solution_dir,
                        realized_solution,
                        proposals[0],
                        task_id=task_id,
                        context_id=task_id,
                    )
                    causal_result = _verify_task_with_cache(
                        tasks_root / task_id,
                        realized_solution,
                        out_path=trial_dir / "causal-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]),
                        trial_dir=trial_dir,
                        invocation_id="causal",
                        timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
                        noise_control_path=noise_control_path,
                        noise_control_required=True,
                        noise_outer_trial_id=str(item["outer_trial_id"]),
                        noise_benchmark_revision=campaign["benchmark_revision"],
                        noise_task_manifest_digest=task_digest,
                        noise_control_expected=noise_expected,
                    )
                    _check_verifier_budget(causal_result, task_spec, budget_errors, "causal verifier")
                    causal_result["seed"] = int(item["outer_trial_index"])
                    verifier_digest = hashlib.sha256(json.dumps(causal_result, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                    family_name = str(task_spec.get("family_id", task_spec.get("family", "runtime")))
                    activation = causal_result.get("activation") if isinstance(causal_result, Mapping) else None
                    activation_passed = isinstance(activation, Mapping) and str(activation.get("status", "")) in {"passed", "verified"} and isinstance(activation.get("matched_actions"), list) and len(activation.get("matched_actions")) == 1
                    activation_certificate = {
                        "action_id": str(activation.get("action_id", activation.get("matched_actions", [""])[0])) if isinstance(activation, Mapping) else "",
                        "activation_metrics": {"causal": causal_result, "control": control_result},
                        "expected_signature": "verifier-paired",
                        "observed_signature": verifier_digest,
                        "verifier_artifacts": {"task_id": task_id},
                        "realization_digest": raw_realization.realized_digest,
                        "passed": causal_result is not None and control_result is not None and activation_passed,
                    }
                    action_spec = semantic_action_spec(
                        family_name,
                        proposals[0],
                        activation_certificate=activation_certificate,
                    )
                    activation_certificate["action_id"] = str(action_spec["action_id"])
                    finalized = InterventionRealizer.classify_after_verification(raw_realization, action_spec, verifier_digest)
                    (realized_solution / "realization_record.json").write_text(
                        json.dumps(finalized.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                    causal_scored = scoring.score_task(causal_result)
                    heldout_result = _verify_task_with_cache(
                        tasks_root / task_id,
                        realized_solution,
                        out_path=trial_dir / "heldout-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]) + 1000003,
                        trial_dir=trial_dir,
                        invocation_id="heldout",
                        timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
                        noise_control_path=noise_control_path,
                        noise_control_required=True,
                        noise_outer_trial_id=str(item["outer_trial_id"]),
                        noise_benchmark_revision=campaign["benchmark_revision"],
                        noise_task_manifest_digest=task_digest,
                        noise_control_expected=noise_expected,
                    )
                    _check_verifier_budget(heldout_result, task_spec, budget_errors, "held-out verifier")
                    heldout_scored = scoring.score_task(heldout_result)
                    heldout_control, heldout_control_scored = _verify_baseline(
                        tasks_root / task_id,
                        solution_dir,
                        trial_dir / "heldout-control-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]) + 1000003,
                        trial_dir=trial_dir,
                        invocation_id="heldout-control",
                        timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
                        noise_control_path=noise_control_path,
                        noise_control_expected=noise_expected,
                    )
                    _check_verifier_budget(heldout_control, task_spec, budget_errors, "held-out control verifier")
                    heldout_measurement = heldout_result.get("measurement", {}) if isinstance(heldout_result.get("measurement"), dict) else {}
                    heldout_candidate_runs = list(heldout_measurement.get("candidate_runs", []))
                    heldout_baseline_runs = list(heldout_measurement.get("baseline_runs", []))
                    if not heldout_candidate_runs or len(heldout_candidate_runs) != len(heldout_baseline_runs):
                        raise ValueError("held-out verifier did not produce paired measurements")
                    heldout_case = {
                        "case_id": f"HELDOUT-{task_id}",
                        "paired_replay": True,
                        "same_fixture_id": f"HELDOUT-FIXTURE-{task_id}",
                        "independence_group": f"heldout-{task_id}",
                        "intervention_measurements": heldout_candidate_runs,
                        "baseline_measurements": heldout_baseline_runs,
                        "higher_is_better": bool(heldout_measurement.get("higher_is_better", False)),
                        "utility_scale": UTILITY_LOG_SCALE,
                        "scientific_ok": bool(heldout_scored.get("gates_passed", False)),
                        "quality_ok": bool(heldout_scored.get("gates_passed", False)) and bool(heldout_control_scored.get("gates_passed", False)),
                    }
                    heldout_interval = _case_effect_interval(heldout_case, delta=statistical_budget.validation)
                    if heldout_interval is None:
                        raise ValueError("held-out paired effect interval is unavailable")
                    heldout_effect, heldout_lcb, heldout_ucb = heldout_interval
                    regression_tolerance = float(
                        (task_spec.get("measurement") or {}).get("regression_tolerance", 0.0)
                    )
                    poison_case = execute_poison_probe(
                        task_spec, public_routing, proposals[0], realized_solution, solution_dir,
                        task_dir=tasks_root / task_id,
                        verifier_out=trial_dir / "poison-result.json",
                        timeout_s=float(task_spec.get("time_budget_s", budgets.wall_time_s)),
                        noise_control_path=noise_control_path,
                        noise_control_expected=noise_expected,
                    )
                    if poison_case.get("resource_blocked"):
                        budget_errors.append("poison verifier resource_blocked")
                    validation_evidence = {
                        "regression_tolerance": regression_tolerance,
                        # The promotion task is never reused as held-out
                        # evidence.  Disjoint family contexts are appended
                        # below and carry their own execution results.
                        "heldout_regression_cases": [{
                            **heldout_case,
                            "holdout_class": "replication",
                            "executed": True,
                            "execution_source": "verifier",
                            "effect_lcb": heldout_lcb,
                            "effect_ucb": heldout_ucb,
                        }],
                        "poison_probe_cases": [poison_case],
                    }
                    # The verifier seed holdout establishes replication.  The
                    # family view supplies two additional preregistered,
                    # disjoint contexts for transfer and boundary checks; they
                    # are evaluated by the same FamilyEnvironment oracle and
                    # never enter promotion evidence.
                    try:
                        from benchmark.families import family_views, FamilyEnvironment, EpisodeEnvironmentState
                        family_name = str(task_spec.get("family_id", task_spec.get("family", "compile")))
                        views = family_views(family_name, count=24, seed=int(item["outer_trial_index"]) + 17)
                        action_id = str(semantic_action_spec(
                            family_name,
                            proposals[0],
                            activation_certificate={
                                "task_id": task_id,
                                "proposal": proposals[0].get("intervention", {}),
                                "activation_metrics": {"causal": causal_result, "control": control_result},
                                "realized_digest": _workspace_digest(realized_solution),
                                "realization_digest": _workspace_digest(realized_solution),
                                "observed_signature": _workspace_digest(realized_solution),
                                "passed": True,
                                "expected_signature": "verifier-paired",
                                "verifier_artifacts": {"task_id": task_id},
                            },
                        )["action_id"])
                        env = FamilyEnvironment(family_name)
                        for holdout_class, pool_name in (("transfer", "representative_pool"), ("boundary", "sealed_boundary_pool")):
                            pool = views[pool_name]
                            instance = next((entry for entry in pool if entry.instance_id != task_id), pool[0])
                            held = env.evaluate(instance.parameters, (action_id,), EpisodeEnvironmentState())
                            base = env.evaluate(instance.parameters, (), EpisodeEnvironmentState())
                            validation_evidence["heldout_regression_cases"].append({
                                "case_id": f"HELDOUT-{holdout_class.upper()}-{instance.instance_id}",
                                "holdout_class": holdout_class,
                                "context": {"workload": dict(instance.parameters)},
                                "executed": True,
                                "execution_source": "family-environment",
                                "scientific_ok": all(held.scientific_gates.values()),
                                "utility": held.utility,
                                "effect": held.utility - base.utility,
                                "effect_lcb": held.utility - base.utility,
                                "effect_ucb": held.utility - base.utility,
                                "utility_policy_id": "family-outcome-v1",
                            })
                    except (KeyError, ValueError, TypeError):
                        pass
                except (OSError, ValueError, TypeError) as exc:
                    budget_errors.append(f"causal intervention realization failed: {exc}")
        if (
            str(item["condition"]) == "D"
            and causal_result is not None
            and control_result is not None
            and result.get("calibration_status") != "blocked"
        ):
            # The immutable verifier's paired outcome is fed back into the
            # canonical lifecycle.  Worker result JSON is never treated as an
            # EvidenceEvent source.
            measurement = causal_result.get("measurement", {}) if isinstance(causal_result.get("measurement"), dict) else {}
            control_measurement = control_result.get("measurement", {}) if isinstance(control_result.get("measurement"), dict) else {}
            on_runs, off_runs = list(measurement.get("candidate_runs", [])), list(control_measurement.get("candidate_runs", []))
            if on_runs and len(on_runs) == len(off_runs):
                effects = [utility_effect(float(on), float(off), higher_is_better=bool(measurement.get("higher_is_better", False)), log_scale=UTILITY_LOG_SCALE) for on, off in zip(on_runs, off_runs)]
                result["evidence_events"] = [
                    {
                        "schema_version": 2,
                        "event_id": f"formal-{task_id}-paired",
                        "context": build_public_context(public_routing),
                        "assignment": {"interventions": {str(task_id): 1}, "propensity": 0.5, "design_id": "formal-verifier-paired-v2"},
                        "evidence_stream": "representative",
                        "evidence_role": "promotion_representative",
                        "query_id": str(task_id),
                        "outcome_vector": {"utility": float(sum(effects) / len(effects)), "paired_effect": float(sum(effects) / len(effects)), "contrast": "on-minus-off"},
                        "scientific_gates": {"candidate": bool(causal_result.get("verdict") == "pass"), "control": bool(control_result.get("verdict") == "pass")},
                        "artifacts": {"causal_result": hashlib.sha256(json.dumps(causal_result, sort_keys=True, default=str).encode()).hexdigest(), "control_result": hashlib.sha256(json.dumps(control_result, sort_keys=True, default=str).encode()).hexdigest()},
                        "versions": {str(task_id): "1"},
                        "source_id": f"formal-verifier-{task_id}",
                        "independence_group": f"formal-task-{task_id}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trust_zone": "harness",
                        "attacker_controlled_fields": [],
                    }
                ]
        transition = _safe_post_task_update(
            budget_errors=budget_errors,
            condition=str(item["condition"]),
            store=store,
            task_id=task_id,
            result=result,
            scored=scored,
            core_repo=repo_root,
            out_dir=trial_dir,
            context_mode=str(item["context_mode"]),
            # Decisions are append-only and survive a campaign resume.  The
            # stream-local ledger is part of the condition store rather than
            # an in-memory side table.
            ledger=ledgers.setdefault(
                stream_id,
                EvolutionDecisionLedger(store / "evolution" / "decisions.jsonl"),
            ),
            allow_maintenance=not budget_errors and not (int(item.get("phase", 0)) >= 3 and str(item["condition"]) in {"C", "D"}),
            control_result=control_result,
            control_scored=control_scored,
            public_context=public_routing,
            family_id=str(task_spec.get("family_id", task_spec.get("family", ""))) or None,
            causal_result=causal_result,
            causal_scored=causal_scored,
            validation_evidence=validation_evidence,
            practical_epsilon=practical_effect_threshold(
                float((task_spec.get("measurement") or {}).get("min_improvement_percent", 0.0))
            ),
            seed=int(item["outer_trial_index"]),
            execution_validity="valid" if execution_valid else "resource_blocked",
            slot_id=str(item.get("slot_id", task_id)),
            visibility=str(item.get("visibility")) if item.get("visibility") is not None else None,
        )
        if transition.get("status") == "state_mutation_error":
            blocked_streams.add(stream_id)
        frozen_snapshot = None
        if int(item.get("phase", 0)) >= 3 and str(item.get("condition")) in {"C", "D"}:
            # Transfer probes use the pre-task canonical state and never feed
            # maintenance back into the snapshot.
            frozen_snapshot = {
                "store_digest": transition.get("pre_store_digest"),
                "condition": str(item["condition"]),
                "outer_trial_id": str(item["outer_trial_id"]),
                "maintenance_frozen": True,
            }
        if str(item["condition"]) == "D":
            evolution_budget = evolution_budget.add(
                replay_executions=len(transition.get("added_replay_case_ids", [])),
                wall_time_s=float(usage.get("wall_time_s", 0.0) or 0.0),
                tokens=int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
            )
            campaign["evolution_compute_budget"] = evolution_budget.as_dict()
            exceeded = evolution_budget.exceeds_limits()
            if exceeded:
                blocked_streams.add(stream_id)
                budget_errors.extend(f"evolution compute budget exceeded: {key}" for key in exceeded)
                transition["evolution_compute_budget"] = {"status": "blocked", "exceeded": exceeded, "state": evolution_budget.as_dict()}
            else:
                transition["evolution_compute_budget"] = {"status": "within_limits", "state": evolution_budget.as_dict()}
        attestation_ok, attestation_errors = conditions.verify_attestation(store)
        if not attestation_ok:
            budget_errors.extend(f"condition attestation failed: {error}" for error in attestation_errors)
            blocked_streams.add(stream_id)
        # Carry state is part of the condition trajectory.  Commit it only
        # after all protocol/budget checks and store attestation have passed;
        # invalid trials must not advance either state source.
        if item["context_mode"] == "carry" and not budget_errors and attestation_ok and worker_context_state.is_file():
            shutil.copy2(worker_context_state, state_path)
        record = {
            "experiment": manifest,
            "task_id": task_id,
            "slot_id": item.get("slot_id", task_id),
            "visibility": item.get("visibility"),
            "family": task_spec.get("family"),
            "family_id": task_spec.get("family_id", task_spec.get("family")),
            "anchor_instance_id": task_spec.get("anchor_instance_id", task_id),
            "family_instance_digest": task_spec.get("family_instance_digest"),
            "lineage_id": task_spec.get("lineage", {}).get("mutation_template_id", task_id),
            "kind": task_spec.get("kind"),
            "condition": item["condition"],
            "context_mode": item["context_mode"],
            "outer_trial_id": item["outer_trial_id"],
            "phase": item["phase"],
            "agent": agent,
            "agent_usage": usage,
            "budget_errors": budget_errors,
            "attestation_ok": attestation_ok,
            "execution_validity": "valid" if execution_valid else "resource_blocked",
            "task_outcome": str(scored.get("verdict", result.get("verdict", "error"))),
            "calibration_status": result.get("calibration_status", "not_evaluated"),
            "transfer_estimand": "frozen_transfer_probe" if frozen_snapshot else "online_stream",
            "frozen_transfer_snapshot": frozen_snapshot,
            # A protocol-valid candidate failure is still an observed,
            # score-zero efficacy cell.  Only infrastructure/protocol errors
            # are excluded from the efficacy matrix.
            "efficacy_eligible": bool(execution_valid and attestation_ok),
            "validity": "valid" if execution_valid and attestation_ok else "invalid",
            "transition": transition,
            "score": scored,
        }
        (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        records.append(record)
    campaign["status"] = "complete"
    campaign["results_claimed"] = bool(
        getattr(args, "claim_results", False)
        and _formal_claim_gate(campaign, records, repo_root / "benchmark" / "population_report.json")
    )
    required_cells = [
        (
            str(item["task_id"]),
            str(item["outer_trial_id"]),
            str(item["context_mode"]),
            str(item["condition"]),
        )
        for item in plan
    ]
    required_sealed_cells = [
        (str(item["task_id"]), str(item["outer_trial_id"]), str(item["context_mode"]), str(item["condition"]))
        for item in plan if str(item.get("visibility")) == "sealed"
    ]
    if getattr(args, "formal", False):
        try:
            claims_value_for_aggregate = miniyaml.load(str(claims_path)) if claims_path.suffix in {".yaml", ".yml"} else json.loads(claims_path.read_text(encoding="utf-8"))
        except Exception:
            claims_value_for_aggregate = {}
        campaign["aggregate"] = aggregate.aggregate_confirmatory(
            records, required_cells=required_sealed_cells, claims=claims_value_for_aggregate,
        )
    else:
        campaign["aggregate"] = aggregate.aggregate_trials(records, required_cells=required_cells)
    try:
        claims_value = miniyaml.load(str(claims_path)) if claims_path.suffix in {".yaml", ".yml"} else json.loads(claims_path.read_text(encoding="utf-8"))
    except Exception:
        claims_value = {}
    population_digest = hashlib.sha256(population_manifest.read_bytes()).hexdigest() if population_manifest.is_file() else "unknown"
    receipt = aggregate.unblinding_receipt(
        population_digest=population_digest,
        schedule=plan,
        claims=claims_value,
        records=records,
        claim_gate="pass" if campaign.get("results_claimed") else "withheld",
    )
    (out_dir / "unblinding_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign["unblinding_receipt"] = str(out_dir / "unblinding_receipt.json")
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return campaign


def _validate_formal_entry(
    repo_root: Path,
    population_manifest: Path,
    campaign_config: Path | None,
    claims_path: Path,
    analysis_plan_path: Path,
    sealed_tasks_root: Path | None = None,
) -> None:
    """Validate immutable formal inputs before scheduling any sealed cell."""
    missing = [str(path) for path in (population_manifest, claims_path, analysis_plan_path) if not path.is_file()]
    if missing:
        raise ValueError("formal entry artifacts missing: " + ", ".join(missing))
    if campaign_config is None or not Path(campaign_config).is_file():
        raise ValueError("formal campaign config is required")
    try:
        manifest = json.loads(population_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid population manifest: {exc}") from exc
    if manifest.get("sealed_total") != 35 or manifest.get("primary_population") != "sealed-35":
        raise ValueError("formal population must preregister sealed-35 as the primary population")
    if manifest.get("status") != "materialized_frozen":
        raise ValueError("formal population contents are not materialized_frozen")
    sealed_root = sealed_tasks_root or Path(str(manifest.get("sealed_root", "")))
    errors = validate_materialized_manifest(manifest, sealed_root)
    if errors:
        raise ValueError("invalid materialized formal population: " + "; ".join(errors))
    approval_path = repo_root / "benchmark" / "calibration" / "calibration_approval.json"
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("calibration approval is missing or invalid") from exc
    if approval.get("approved") is not True:
        raise ValueError("calibration approval is not PASS")
    pilot_path = repo_root / "benchmark" / "calibration" / "pilot_calibration.json"
    if not pilot_path.is_file() or approval.get("pilot_calibration_digest") in {None, "pending-review"}:
        raise ValueError("calibration approval is not bound to pilot_calibration.json")
    try:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pilot calibration artifact is invalid") from exc
    if pilot.get("artifact_digest") != approval.get("pilot_calibration_digest"):
        raise ValueError("calibration approval pilot digest mismatch")
    approval_errors = validate_calibration_approval(
        json.loads((repo_root / "benchmark" / "population_report.json").read_text(encoding="utf-8")),
        pilot,
        approval,
        repo_root=repo_root,
    )
    if approval_errors:
        raise ValueError("calibration approval validation failed: " + "; ".join(approval_errors))
    release_path = repo_root / "benchmark" / "formal_release_manifest.json"
    if not release_path.is_file():
        raise ValueError("formal_release_manifest.json is required")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("formal release manifest is invalid") from exc
    release_errors = validate_formal_release(
        release, repo_root=repo_root, population_manifest=population_manifest,
        approval_path=approval_path, claims_path=claims_path,
        protocol_path=analysis_plan_path, campaign_config=Path(campaign_config_path),
    )
    if release_errors:
        raise ValueError("formal release validation failed: " + "; ".join(release_errors))
    try:
        clean = not subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=False,
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        clean = False
    if not clean:
        raise ValueError("formal entry requires a clean git tree")


def _validate_campaign_config(config_path: Path, *, conditions: tuple[str, ...], context_modes: tuple[str, ...], outer_trials: int, schedule_seed: int, population_manifest: Path, repo_root: Path) -> dict[str, Any]:
    config = miniyaml.load(str(config_path))
    if str(config.get("mode")) != "formal":
        raise ValueError("campaign config mode must be formal")
    expected = {
        "conditions": list(conditions),
        "context_mode": context_modes[0] if len(context_modes) == 1 else list(context_modes),
        "outer_trials": int(outer_trials),
        "schedule_seed": int(schedule_seed),
    }
    for key, value in expected.items():
        if key in config and config.get(key) != value:
            raise ValueError(f"formal CLI/config mismatch for {key}: {config.get(key)!r} != {value!r}")
    configured_manifest = config.get("population_manifest")
    if configured_manifest is None:
        raise ValueError("formal campaign config must declare population_manifest")
    try:
        configured_path = Path(str(configured_manifest))
        if not configured_path.is_absolute():
            configured_path = (repo_root / configured_path).resolve()
        if configured_path != population_manifest.resolve():
            raise ValueError(f"formal CLI/config mismatch for population_manifest: {configured_manifest!r} != {population_manifest}")
    except OSError as exc:
        raise ValueError(f"invalid population_manifest path: {exc}") from exc
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--tasks-root", type=Path, default=root / "benchmark" / "tasks")
    parser.add_argument("--sealed-tasks-root", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=root / "benchmark" / "split" / "sequential.yaml")
    parser.add_argument("--skill-source", type=Path, default=root)
    parser.add_argument("--skill-view", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, default=None)
    parser.add_argument("--campaign-config", type=Path, default=None)
    parser.add_argument("--claims", type=Path, default=None)
    parser.add_argument("--analysis-plan", type=Path, default=None)
    parser.add_argument("--formal", action="store_true", help="enter sealed formal mode; enforce approval and frozen inputs")
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--context-modes", default="reset")
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--model-id", default="unidentified-agent")
    parser.add_argument("--agent-config", default="{}")
    parser.add_argument("--budgets", default=None, help='JSON, e.g. {"tokens":12000,"tool_calls":80,"wall_time_s":900}')
    parser.add_argument("--agent-command", default=None, help="shell template; receives SPE_* environment variables")
    parser.add_argument(
        "--executor-command",
        default=None,
        help="namespace/container executor template; receives {agent_command}, {worker_root}, {task_dir}, {solution_dir}, {retrieved_context}, {skill_view}, {executor_receipt}",
    )
    parser.add_argument("--executor-digest", default=None, help="allowlisted external executor/image digest")
    parser.add_argument(
        "--experiment-executor-command",
        default=None,
        help="external node/pair/three-way experiment executor; receives {request_json}, {result_json}, {work_root}",
    )
    parser.add_argument("--claim-results", action="store_true", help="claim only if the formal calibration gate is passed")
    parser.add_argument("--resume", action="store_true", help="resume an exact, digest-matched campaign from contiguous trial prefixes")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "schedule_size": result["schedule_size"], "results_claimed": result["results_claimed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
