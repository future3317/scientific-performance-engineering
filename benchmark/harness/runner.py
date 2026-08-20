#!/usr/bin/env python3
"""Sandbox materialization, subprocess isolation, and the paired measurement
driver (BENCHMARK_DESIGN.md section 6).

The verifier drives a task's ``benchmark.py`` (imported by path) through the
fixed S0-S6 order. This module owns:

- :func:`materialize_sandbox` — copy ``workspace/`` + ``public_tests/`` to a
  fresh temp dir, assert no VCS metadata, hash harness files.
- :func:`run_python_subprocess` — subprocess isolation with timeout.
- :func:`run_paired_measurement` — interleaved, seeded baseline/candidate
  measurement per section 6.1, with output-checksum recording and work-unit
  counter comparison.
- CUDA helpers: device selection, event+wall-clock bracketing, L2 thrash for
  kernel tasks.

Conventions expected of a task's ``benchmark.py`` (tolerant adapters below
accept simpler shapes):

- ``load_solution(path)`` -> solution object/module
- ``make_fixtures(seed)`` -> fixtures (dict recommended)
- ``run_correctness(solution, fixtures)`` -> bool or {"passed": bool, ...}
- ``run_scientific_gates(solution, fixtures)`` -> {name: bool or (bool, details)}
- ``run_performance(solution, fixtures)`` -> float, or dict with
  {"value": float, "work_units": {...}, "output_checksums": {...}, "timing": {...}}
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import random
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from . import anticheat


# ---------------------------------------------------------------------------
# Sandbox materialization (S0)
# ---------------------------------------------------------------------------


def materialize_sandbox(task_dir: str | Path, dest: str | Path | None = None) -> Path:
    """Copy ``workspace/`` + ``public_tests/`` of *task_dir* into a fresh sandbox.

    The agent-visible sandbox contains ONLY those two trees; ``benchmark.py``,
    ``hidden_verifier/``, ``oracle/``, ``task.yaml`` and ``metadata.json`` never
    enter it (section 4 rules). VCS metadata is asserted absent.
    """
    task_dir = Path(task_dir)
    workspace = task_dir / "workspace"
    if not workspace.is_dir():
        raise FileNotFoundError(f"task has no workspace/ directory: {task_dir}")
    anticheat.assert_no_vcs(task_dir)
    if dest is None:
        dest = tempfile.mkdtemp(prefix="spe_evo_sandbox_")
    sandbox = Path(dest)
    sandbox.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, sandbox / "workspace", dirs_exist_ok=True)
    public_tests = task_dir / "public_tests"
    if public_tests.is_dir():
        shutil.copytree(public_tests, sandbox / "public_tests", dirs_exist_ok=True)
    anticheat.assert_no_vcs(sandbox)
    return sandbox


def hash_harness_files(harness_dir: str | Path | None = None) -> dict[str, str]:
    """SHA-256 manifest of source harness files, excluding derived bytecode."""
    if harness_dir is None:
        harness_dir = Path(__file__).resolve().parent
    manifest = anticheat.hash_tree(harness_dir)
    return {
        path: digest
        for path, digest in manifest.items()
        if "__pycache__" not in Path(path).parts and not path.endswith(".pyc")
    }


# ---------------------------------------------------------------------------
# Module loading / seeds / devices
# ---------------------------------------------------------------------------


def import_module_by_path(path: str | Path, module_name: str | None = None):
    """Import a Python module from an explicit file path (benchmark.py, solutions)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    name = module_name or f"spe_evo_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Keep the repository root ahead of a task directory.  A task's
    # ``benchmark.py`` is itself named ``benchmark.py``; putting its directory
    # at ``sys.path[0]`` makes ``import benchmark.harness`` resolve that file
    # as a top-level module and shadows the real benchmark package.  The task
    # directory is still available for ordinary sibling imports, but only
    # after the package root.
    repo_root = Path(__file__).resolve().parents[2]
    task_packages_root = repo_root / "benchmark" / "tasks"
    # Task packages commonly use sibling names such as ``scientific_contract``.
    # Remove a prior task's sibling modules before loading the next package;
    # otherwise Python reuses the first task's module from ``sys.modules`` and
    # a multi-task campaign can execute the wrong verifier contract.
    for key, loaded in list(sys.modules.items()):
        loaded_path = getattr(loaded, "__file__", None)
        if not loaded_path:
            continue
        try:
            if Path(loaded_path).resolve().is_relative_to(task_packages_root.resolve()):
                del sys.modules[key]
        except (OSError, ValueError):
            continue
    added_paths: list[str] = []
    root_text = str(repo_root)
    task_text = str(path.parent)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        added_paths.append(root_text)
    if task_text not in sys.path:
        sys.path.append(task_text)
        added_paths.append(task_text)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for entry in reversed(added_paths):
            try:
                sys.path.remove(entry)
            except ValueError:
                pass
    return module


def set_global_seeds(seed: int) -> None:
    """Seed python ``random`` and torch (CPU + CUDA when present)."""
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def select_device(requires_cuda: bool) -> tuple[str, bool]:
    """Return (device, usable). CUDA-required tasks degrade to unusable on CPU hosts."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True
    except Exception:
        pass
    return "cpu", not requires_cuda


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------


def run_python_subprocess(
    snippet: str | None = None,
    module: str | None = None,
    args: tuple[str, ...] | list[str] = (),
    timeout: float = 600.0,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``sys.executable -c snippet`` or ``-m module`` with a hard timeout.

    Returns {exit_code, stdout, stderr, wall_time_s, timed_out}. Exactly one of
    *snippet* / *module* must be given.
    """
    if (snippet is None) == (module is None):
        raise ValueError("exactly one of snippet or module is required")
    command = [sys.executable]
    if snippet is not None:
        command += ["-c", snippet]
    else:
        command += ["-m", str(module)]
    command += list(args)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    start = time.perf_counter()
    timed_out = False
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = process.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout_tail, stderr_tail = process.communicate(timeout=2.0)
                stdout += stdout_tail or ""
                stderr += stderr_tail or ""
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout_tail, stderr_tail = process.communicate()
                stdout += stdout_tail or ""
                stderr += stderr_tail or ""
        exit_code = process.returncode if process.returncode is not None else -1
        stderr += f"\n[harness] subprocess timed out after {timeout}s"
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "wall_time_s": time.perf_counter() - start,
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# CUDA timing helpers
# ---------------------------------------------------------------------------

_L2_THRASH_SIZE_MB = 64


def l2_thrash(device: str = "cuda", size_mb: int = _L2_THRASH_SIZE_MB) -> None:
    """Evict L2 by writing a large buffer between trials (kernel-level tasks)."""
    if not device.startswith("cuda"):
        return
    import torch

    buffer = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    buffer.uniform_()
    torch.cuda.synchronize()


def timed_iteration(fn: Callable[[], Any], device: str = "cpu") -> dict[str, float]:
    """One bracketed iteration: CUDA events AND host wall clock, cross-checked.

    Returns {"event_ms", "wall_ms"}. On CPU only wall time is measured.
    """
    if device.startswith("cuda"):
        import torch

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return {"event_ms": start_event.elapsed_time(end_event), "wall_ms": wall_ms}
    wall_start = time.perf_counter()
    fn()
    return {"event_ms": None, "wall_ms": (time.perf_counter() - wall_start) * 1000.0}


# ---------------------------------------------------------------------------
# benchmark.py result adapters
# ---------------------------------------------------------------------------


def call_benchmark_fn(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a benchmark.py function with the subset of kwargs its signature accepts."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**accepted)


def normalize_correctness(result: Any) -> dict[str, Any]:
    if isinstance(result, bool):
        return {"passed": result, "details": {}}
    if isinstance(result, dict):
        return {"passed": bool(result.get("passed", False)), "details": result}
    raise TypeError(f"run_correctness returned unsupported type {type(result).__name__}")


def normalize_gates(result: Any) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    if not isinstance(result, dict):
        raise TypeError("run_scientific_gates must return a dict of gate results")
    for name, value in result.items():
        if isinstance(value, bool):
            gates[name] = {"passed": value, "details": {}}
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            gates[name] = {"passed": bool(value[0]), "details": value[1] if isinstance(value[1], dict) else {"raw": value[1]}}
        elif isinstance(value, dict):
            gates[name] = {"passed": bool(value.get("passed", False)), "details": value}
        else:
            raise TypeError(f"unsupported gate result for {name!r}: {type(value).__name__}")
    return gates


def normalize_performance(result: Any) -> dict[str, Any]:
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return {"value": float(result), "work_units": {}, "output_checksums": {}, "raw": result}
    if isinstance(result, dict):
        value = result.get("value", result.get("primary_metric"))
        return {
            "value": float(value) if isinstance(value, (int, float)) else None,
            "work_units": result.get("work_units", {}),
            "output_checksums": result.get("output_checksums", {}),
            "raw": result,
        }
    raise TypeError(f"run_performance returned unsupported type {type(result).__name__}")


# ---------------------------------------------------------------------------
# Fixture hashing / checksums / work units
# ---------------------------------------------------------------------------


def _hash_obj(obj: Any, hasher) -> None:
    """Recursively feed a fixture structure into a hasher (tensors by bytes)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in eval envs
        torch = None
    if torch is not None and isinstance(obj, torch.Tensor):
        hasher.update(obj.detach().cpu().contiguous().numpy().tobytes())
        hasher.update(str(tuple(obj.shape)).encode())
        hasher.update(str(obj.dtype).encode())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            hasher.update(str(key).encode())
            _hash_obj(obj[key], hasher)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _hash_obj(item, hasher)
    else:
        try:
            hasher.update(json.dumps(obj, sort_keys=True, default=str).encode())
        except TypeError:
            hasher.update(repr(obj).encode())


def fixture_hash(fixtures: Any) -> str:
    """SHA-256 of a fixture structure; recorded into results (section 13)."""
    hasher = hashlib.sha256()
    _hash_obj(fixtures, hasher)
    return hasher.hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_work_units(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Work-unit counters must match the baseline exactly (section 7)."""
    diffs: list[str] = []
    for key in sorted(set(baseline) | set(candidate)):
        if baseline.get(key) != candidate.get(key):
            diffs.append(f"work unit {key!r}: baseline={baseline.get(key)!r} candidate={candidate.get(key)!r}")
    return (not diffs, diffs)


# ---------------------------------------------------------------------------
# Interleaved paired measurement driver (section 6.1)
# ---------------------------------------------------------------------------


def run_paired_measurement(
    benchmark_module: Any,
    baseline_path: str | Path,
    candidate_path: str | Path,
    measurement_cfg: dict[str, Any],
    seed: int = 0,
    device: str = "cpu",
    l2_thrash_between: bool = False,
    reuse_fixture_per_repetition: bool = False,
) -> dict[str, Any]:
    """Run the seeded, interleaved paired measurement.

    ``measurement_cfg`` carries repetitions/warmup_iterations/measured_iterations
    from ``task.yaml: measurement``. Per repetition the order of
    baseline/candidate is shuffled with ``random.Random(seed)`` and recorded as
    ``run_order``; fixtures are re-drawn per repetition from a seeded generator.
    H2D tasks may request one generated/hashed fixture per repetition and provide
    ``clone_fixtures`` so each arm remains mutable-state isolated.
    Returns the raw measurement record for result.json.
    """
    repetitions = int(measurement_cfg.get("repetitions", 5))
    warmup = int(measurement_cfg.get("warmup_iterations", 5))
    iterations = int(measurement_cfg.get("measured_iterations", 30))
    rng = random.Random(seed)

    record: dict[str, Any] = {
        "run_order": [],
        "baseline_runs": [],
        "candidate_runs": [],
        "work_units": {},
        "output_checksums": {},
        "fixture_hashes": {},
        "timing": [],
        "fixture_build_time_s": 0.0,
        "fixture_hash_time_s": 0.0,
    }
    order_plan: list[tuple[str, int]] = []
    for rep in range(repetitions):
        pair = ["baseline", "candidate"]
        rng.shuffle(pair)
        for arm in pair:
            order_plan.append((arm, rep))

    fixtures_by_rep: dict[int, Any] = {}
    fixture_hash_by_rep: dict[int, str] = {}
    for arm, rep in order_plan:
        arm_started = time.perf_counter()
        record["run_order"].append(arm)
        fixture_seed = seed * 100003 + rep
        if reuse_fixture_per_repetition:
            if rep not in fixtures_by_rep:
                build_started = time.perf_counter()
                fixtures_by_rep[rep] = call_benchmark_fn(
                    benchmark_module.make_fixtures, seed=fixture_seed, device=device
                )
                record["fixture_build_time_s"] += time.perf_counter() - build_started
                hash_started = time.perf_counter()
                fixture_hash_by_rep[rep] = fixture_hash(fixtures_by_rep[rep])
                record["fixture_hash_time_s"] += time.perf_counter() - hash_started
            clone_fn = getattr(benchmark_module, "clone_fixtures", None)
            if not callable(clone_fn):
                raise ValueError("fixture reuse requires benchmark_module.clone_fixtures()")
            fixtures = call_benchmark_fn(clone_fn, fixtures=fixtures_by_rep[rep])
            record["fixture_hashes"][f"{arm}:{rep}"] = fixture_hash_by_rep[rep]
        else:
            build_started = time.perf_counter()
            fixtures = call_benchmark_fn(
                benchmark_module.make_fixtures, seed=fixture_seed, device=device
            )
            record["fixture_build_time_s"] += time.perf_counter() - build_started
            hash_started = time.perf_counter()
            record["fixture_hashes"][f"{arm}:{rep}"] = fixture_hash(fixtures)
            record["fixture_hash_time_s"] += time.perf_counter() - hash_started
        path = baseline_path if arm == "baseline" else candidate_path
        load_started = time.perf_counter()
        solution = call_benchmark_fn(benchmark_module.load_solution, path=str(path), device=device)
        load_wall = time.perf_counter() - load_started
        if l2_thrash_between:
            l2_thrash(device)
        performance_started = time.perf_counter()
        result = call_benchmark_fn(
            benchmark_module.run_performance,
            solution=solution,
            fixtures=fixtures,
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        performance_wall = time.perf_counter() - performance_started
        perf = normalize_performance(result)
        record[f"{arm}_runs"].append(perf["value"])
        record["work_units"][f"{arm}:{rep}"] = perf["work_units"]
        record["output_checksums"][f"{arm}:{rep}"] = perf["output_checksums"]
        raw = perf["raw"] if isinstance(perf["raw"], dict) else {}
        record["timing"].append({
            "arm": arm,
            "rep": rep,
            "repetition": rep,
            "arm_total_wall_s": time.perf_counter() - arm_started,
            "load_solution_wall_s": load_wall,
            "run_performance_wall_s": performance_wall,
            "timing": raw.get("timing", {}),
        })
    return record
