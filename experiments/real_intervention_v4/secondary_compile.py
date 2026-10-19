"""Independent, family-blocked runner for the preregistered compile/cache study.

The input roster is a timing-blind freeze. It must contain unique independent
families with an existing baseline-correctness reference and a family split,
plus explicit compile-canary families and shakedown contexts. See
``_validate_roster`` for the small JSON contract. The runner never imports or
changes the primary study configuration.

Run on the canonical server checkout, for example::

    python -m experiments.real_intervention_v4.secondary_compile \\
      --roster /path/to/compile-roster.json \\
      --results-root ~/RESULTS/evospe-rt-v4/EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019 \\
      --gpu-uuid GPU-... --arm-timeout-seconds 900 \\
      --cleanup-grace-seconds 20 --term-grace-seconds 5

Each worker is a fresh supervised process. Started attempts without a terminal
receipt remain UNKNOWN and are never replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock

from .cache_lifecycle import cache_artifact_digest, cache_usage, create_cache_scope
from .exposure_monitor import compare_arm_exposure
from .matched_sessions import wait_for_selected_gpu
from .outcomes import atomic_json, canonical_digest, classify_pair, recover_attempt
from .process_supervisor import execute_arm


_CONTRACT_PATH = Path(__file__).with_name("config") / "SECONDARY_COMPILE_AMORTIZATION.json"
_CACHE_ENV_NAMES = (
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "XDG_CACHE_HOME",
)
_SPLITS = {"development", "calibration", "final"}
_DEPLOYMENTS = {"COLD_NEW_GRAPH", "PERSISTENT_CACHE_AVAILABLE"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_contract(path: Path = _CONTRACT_PATH) -> dict[str, Any]:
    contract = _read_json(path)
    if (contract.get("study_id") != "EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019"
            or contract.get("intervention_id") != "COMPILE_AMORTIZATION_V1"
            or contract.get("comparison_class") != "systems"
            or contract.get("study_status") != "SEPARATE_PREREGISTERED_SECONDARY_NOT_RUN_BY_PRIMARY_RUNNER"
            or contract.get("S0") != "eager FP32 eval forward with original layout"
            or contract.get("S1") != "torch.compile backend=inductor mode=default dynamic=null; no channels_last, AMP or TF32"
            or contract.get("batches") != [1, 8]
            or contract.get("horizons") != [128, 1024, 4096]
            or contract.get("precision") != "float32"
            or contract.get("independent_family_unit") != "architecture lineage, never shapes, horizon or sessions"):
        raise ValueError("secondary compile contract differs from the preregistered identity")
    deployments = {row.get("id"): row for row in contract.get("deployment_contracts", [])}
    if set(deployments) != _DEPLOYMENTS:
        raise ValueError("secondary compile contract must define exactly the two frozen deployments")
    if deployments["COLD_NEW_GRAPH"].get("cache_state") != "EMPTY":
        raise ValueError("COLD_NEW_GRAPH requires an empty isolated cache")
    if deployments["COLD_NEW_GRAPH"].get("costs") != [
            "first_compile", "first_forward", "H_minus_1_forwards"]:
        raise ValueError("cold deployment costs differ from the preregistered contract")
    persistent = deployments["PERSISTENT_CACHE_AVAILABLE"]
    if (persistent.get("cache_state") != "PREFILLED"
            or persistent.get("costs") != ["cache_read", "first_forward", "H_minus_1_forwards"]
            or persistent.get("acquisition_cost_reported_separately") is not True
            or persistent.get("declared_reuse_counts") != [1, 10, 100]):
        raise ValueError("persistent-cache costs or reuse counts differ from the frozen contract")
    if contract.get("final_may_be_opened_only_after") != [
            "separate baseline-only roster and family split frozen",
            "model-scale compile and cache reload canaries pass",
            "engineering shakedown meets the preregistered reliability gate",
    ]:
        raise ValueError("final-open prerequisites differ from the preregistered contract")
    if contract.get("prohibition") != (
        "This secondary study cannot replace, modify, or be pooled with the CHANNELS_LAST_EAGER_LAYOUT_V1 primary study."
    ):
        raise ValueError("secondary study must remain separate from the channels-last primary")
    return contract


def _validate_roster(roster: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen, baseline-only family roster before timing is opened.

    Required top-level fields are ``baseline_only=true``, ``timing_opened=false``,
    ``family_split_frozen=true``, ``order_seed``, ``torchbench_root``,
    ``torchbench_commit``, ``dependency_overlay``, ``correctness``, ``families``,
    ``compile_canary_families``, ``shakedown_contexts``, and ``shakedown_gate``.
    Each family has ``family_id``, ``split``, ``model_spec`` (source/model/seed/
    shape), and ``qualification_by_batch`` with correctness status, input/weight
    digests, and a frozen reference path/digest for both contracted batches.
    """
    allowed_fields = {
        "schema_version", "study_id", "intervention_id", "baseline_only", "timing_opened",
        "family_split_frozen", "order_seed", "torchbench_root", "torchbench_commit",
        "dependency_overlay", "correctness", "families", "compile_canary_families",
        "shakedown_contexts", "shakedown_gate",
    }
    if set(roster) != allowed_fields:
        raise ValueError("baseline-only roster has unexpected fields; keep timing outcomes outside the freeze")
    if (roster.get("schema_version") != 1
            or roster.get("study_id") != contract["study_id"]
            or roster.get("intervention_id") != contract["intervention_id"]
            or roster.get("baseline_only") is not True
            or roster.get("timing_opened") is not False
            or roster.get("family_split_frozen") is not True):
        raise ValueError("roster must be a timing-blind, family-split freeze for this independent study")
    if not isinstance(roster.get("order_seed"), int) or isinstance(roster.get("order_seed"), bool):
        raise ValueError("roster must freeze an integer order_seed before measurement")
    if not isinstance(roster.get("torchbench_root"), str) or not roster["torchbench_root"]:
        raise ValueError("roster must freeze the TorchBench checkout path")
    if not isinstance(roster.get("torchbench_commit"), str) or len(roster["torchbench_commit"]) != 40:
        raise ValueError("roster must freeze the exact TorchBench commit")
    if not isinstance(roster.get("dependency_overlay"), str) or not roster["dependency_overlay"]:
        raise ValueError("roster must freeze the workload dependency overlay")
    correctness = roster.get("correctness")
    if (not isinstance(correctness, Mapping)
            or not isinstance(correctness.get("rtol"), (int, float))
            or not isinstance(correctness.get("atol"), (int, float))
            or not math.isfinite(float(correctness["rtol"]))
            or not math.isfinite(float(correctness["atol"]))
            or correctness["rtol"] < 0 or correctness["atol"] < 0):
        raise ValueError("roster must freeze non-negative FP32 correctness tolerances")

    families = roster.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("roster must contain at least one independent family")
    family_by_id: dict[str, dict[str, Any]] = {}
    required_batches = {str(item) for item in contract["batches"]}
    for raw in families:
        if not isinstance(raw, Mapping):
            raise ValueError("each family roster row must be an object")
        family = dict(raw)
        if set(family) != {"family_id", "split", "model_spec", "qualification_by_batch"}:
            raise ValueError("family rows may contain only frozen identity, split, model and baseline evidence")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not _SAFE_ID.fullmatch(family_id):
            raise ValueError("family_id must be a non-empty path-safe independent-family identifier")
        if family_id in family_by_id:
            raise ValueError(f"family appears more than once and would inflate independent n: {family_id}")
        if family.get("split") not in _SPLITS:
            raise ValueError(f"family {family_id} lacks its frozen development/calibration/final split")
        model_spec = family.get("model_spec")
        if (not isinstance(model_spec, Mapping)
                or set(model_spec) != {"source", "model", "seed", "shape"}
                or model_spec.get("source") not in {"torchvision", "torchbench"}
                or not isinstance(model_spec.get("model"), str)
                or not isinstance(model_spec.get("seed"), int) or isinstance(model_spec.get("seed"), bool)
                or not isinstance(model_spec.get("shape"), list)
                or not model_spec["shape"]
                or any(not isinstance(value, int) or value < 1 for value in model_spec["shape"])):
            raise ValueError(f"family {family_id} has an incomplete frozen source/model/input specification")
        qualifications = family.get("qualification_by_batch")
        if not isinstance(qualifications, Mapping) or set(qualifications) != required_batches:
            raise ValueError(f"family {family_id} must have baseline qualification for both contracted batches")
        for batch in required_batches:
            qualification = qualifications[batch]
            if (not isinstance(qualification, Mapping)
                    or set(qualification) != {
                        "correctness_status", "weights_sha256", "inputs_sha256",
                        "reference_path", "reference_sha256",
                    }
                    or qualification.get("correctness_status") != "PASS"
                    or not all(isinstance(qualification.get(key), str) and qualification[key]
                               for key in ("weights_sha256", "inputs_sha256", "reference_path", "reference_sha256"))):
                raise ValueError(f"family {family_id}, batch {batch} lacks a frozen correct baseline reference")
            for key in ("weights_sha256", "inputs_sha256", "reference_sha256"):
                value = qualification[key]
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                    raise ValueError(f"family {family_id}, batch {batch} has an invalid frozen {key}")
        family_by_id[family_id] = family

    if {family["split"] for family in family_by_id.values()} != _SPLITS:
        raise ValueError("the frozen roster must include development, calibration and final families")

    canary_ids = roster.get("compile_canary_families")
    if (not isinstance(canary_ids, list) or not canary_ids
            or len(set(canary_ids)) != len(canary_ids)
            or any(item not in family_by_id or family_by_id[item]["split"] != "development"
                   for item in canary_ids)):
        raise ValueError("compile/cache canaries must name unique development families before timing")
    gate = roster.get("shakedown_gate")
    shake = roster.get("shakedown_contexts")
    if (not isinstance(gate, Mapping) or not isinstance(gate.get("minimum_pairs"), int)
            or gate["minimum_pairs"] < 1
            or not isinstance(gate.get("minimum_matched_correct_fraction"), (float, int))
            or not 0 < gate["minimum_matched_correct_fraction"] <= 1
            or not isinstance(shake, list) or len(shake) < gate["minimum_pairs"]):
        raise ValueError("roster must freeze a concrete engineering-shakedown sample and reliability gate")
    shake_ids: set[tuple[str, int, int, str]] = set()
    for row in shake:
        if not isinstance(row, Mapping):
            raise ValueError("shakedown contexts must be objects")
        key = (str(row.get("family_id")), row.get("batch"), row.get("horizon"), str(row.get("deployment")))
        if (key[0] not in family_by_id or family_by_id[key[0]]["split"] != "development"
                or key[1] not in contract["batches"] or key[2] not in contract["horizons"]
                or key[3] not in _DEPLOYMENTS or key in shake_ids):
            raise ValueError(f"invalid or duplicated predeclared shakedown context: {key}")
        shake_ids.add(key)
    if {row[3] for row in shake_ids} != _DEPLOYMENTS:
        raise ValueError("engineering shakedown must cover both frozen cache deployments")
    return {**dict(roster), "families": [family_by_id[key] for key in sorted(family_by_id)],
            "family_by_id": family_by_id}


def _context_id(family_id: str, batch: int, horizon: int, deployment: str) -> str:
    return f"{family_id}-B{batch}-H{horizon}-{deployment}"


def _cache_state_for(arm: str, deployment: str) -> str:
    if arm not in {"S0", "S1"} or deployment not in _DEPLOYMENTS:
        raise ValueError("unknown arm or cache deployment")
    return "PREFILLED" if arm == "S1" and deployment == "PERSISTENT_CACHE_AVAILABLE" else "EMPTY"


def _pair_plan(roster: Mapping[str, Any], contract: Mapping[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    rng = random.Random(int(roster["order_seed"]))
    contexts = []
    if prefix == "formal":
        for family in roster["families"]:
            for batch in contract["batches"]:
                for horizon in contract["horizons"]:
                    for deployment in sorted(_DEPLOYMENTS):
                        contexts.append({"family_id": family["family_id"], "split": family["split"],
                                         "batch": batch, "horizon": horizon, "deployment": deployment})
    else:
        for row in roster["shakedown_contexts"]:
            family = roster["family_by_id"][row["family_id"]]
            contexts.append({"family_id": family["family_id"], "split": family["split"],
                             "batch": row["batch"], "horizon": row["horizon"],
                             "deployment": row["deployment"]})
    planned = []
    for index, context in enumerate(contexts):
        pair_id = f"{prefix}-{index:04d}-{_context_id(context['family_id'], context['batch'], context['horizon'], context['deployment'])}"
        if not _SAFE_ID.fullmatch(pair_id):
            raise ValueError(f"generated pair id is not path safe: {pair_id}")
        order = ["S0", "S1"] if rng.randrange(2) == 0 else ["S1", "S0"]
        planned.append({"pair_id": pair_id, **context, "order": order})
    return planned


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, timeout=10).stdout.strip()


def _server_identity(roster: Mapping[str, Any], results_root: Path, gpu_uuid: str) -> tuple[Path, str]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("secondary compile evidence is supported only on the configured server")
    repo = _repo_root()
    expected_repo = (Path.home() / "CODE" / "scientific-performance-engineering").resolve()
    expected_python = Path.home() / "miniconda3" / "envs" / "equivcompiler" / "bin" / "python"
    if repo != expected_repo or Path.cwd().resolve() != expected_repo:
        raise RuntimeError(f"run from the canonical server checkout: {expected_repo}")
    if Path(sys.executable) != expected_python:
        raise RuntimeError(f"use the project interpreter: {expected_python}")
    if _git("status", "--porcelain", "--untracked-files=all", cwd=repo):
        raise RuntimeError("refusing to produce evidence from a dirty canonical repository")
    commit = _git("rev-parse", "HEAD", cwd=repo)
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a physical GPU UUID must be frozen before the first arm")
    base = (Path.home() / "RESULTS" / "evospe-rt-v4").resolve()
    if results_root.parent != base or results_root.name != "EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019":
        raise ValueError(f"secondary results must use their independent root below {base}")
    torchbench = Path(str(roster["torchbench_root"])).expanduser().resolve()
    overlay = Path(str(roster["dependency_overlay"])).expanduser().resolve()
    if not torchbench.is_dir() or not overlay.is_dir():
        raise FileNotFoundError("frozen TorchBench checkout or dependency overlay is unavailable")
    if _git("rev-parse", "HEAD", cwd=torchbench) != roster["torchbench_commit"]:
        raise RuntimeError("TorchBench checkout differs from the baseline-only roster freeze")
    for family in roster["families"]:
        for batch, qualification in family["qualification_by_batch"].items():
            reference_path = Path(qualification["reference_path"]).expanduser().resolve(strict=True)
            if hashlib.sha256(reference_path.read_bytes()).hexdigest() != qualification["reference_sha256"]:
                raise ValueError(f"frozen baseline reference changed: {family['family_id']} batch {batch}")
    return repo, commit


def _freeze_manifest(contract: Mapping[str, Any], roster: Mapping[str, Any], *,
                     roster_bytes: bytes, source_commit: str, gpu_uuid: str,
                     resources: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "study_id": contract["study_id"], "intervention_id": contract["intervention_id"],
        "comparison_class": "systems",
        "primary_study_id": "EVOSPE_REAL_INTERVENTION_V4_CHANNELS_LAST_20261020_UUIDFIX",
        "contract": dict(contract), "contract_digest": canonical_digest(dict(contract)),
        "roster": {key: value for key, value in roster.items() if key != "family_by_id"},
        "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "source_commit": source_commit, "selected_gpu_uuid": gpu_uuid,
        "resources": dict(resources),
        "formal_plan": _pair_plan(roster, contract, prefix="formal"),
        "shakedown_plan": _pair_plan(roster, contract, prefix="shakedown"),
        "compile_canary_families": list(roster["compile_canary_families"]),
        "declared_reuse_counts": [1, 10, 100],
        "family_count": len(roster["families"]),
        "independent_family_ids": [row["family_id"] for row in roster["families"]],
    }
    body["manifest_digest"] = canonical_digest(body)
    return body


def _layout_signature(value: Any, torch: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            rows.append({"shape": list(item.shape), "stride": list(item.stride()),
                         "dtype": str(item.dtype), "device": str(item.device)})
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return rows


def _cpu_tree(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _all_float32(value: Any, torch: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or value.dtype == torch.float32
    if isinstance(value, Mapping):
        return all(_all_float32(child, torch) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_float32(child, torch) for child in value)
    return True


def _worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize exactly one eager or Inductor process; called only as a child."""
    import numpy as np
    import torch
    from experiments.real_intervention_v3.numerical_checks import (
        compare_outputs, configure_determinism, nested_digest,
    )
    from experiments.real_intervention_v3.source_adapters import load_model
    from .runtime_identity import capture_runtime

    configure_determinism(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    cpu_started = time.process_time()
    row: dict[str, Any] = {
        "status": "ERROR", "arm": spec["arm"], "deployment": spec["deployment"],
        "study_id": spec["study_id"], "intervention_id": spec["intervention_id"],
        "family_id": spec["family_id"], "split": spec["split"],
        "batch": spec["batch"], "horizon": spec["horizon"],
        "precision": "float32", "amp_enabled": False, "tf32_enabled": False,
        "layout_transform": "NONE; original model and input layouts retained",
    }
    try:
        if spec["arm"] not in {"S0", "S1"} or spec["intervention_id"] != "COMPILE_AMORTIZATION_V1":
            raise ValueError("worker arm or secondary intervention identity differs from the frozen contract")
        expected_cache_state = _cache_state_for(spec["arm"], spec["deployment"])
        if (spec["deployment"] not in _DEPLOYMENTS
                or spec["cache_state"] != expected_cache_state
                or int(spec["horizon"]) < 1):
            raise ValueError("deployment, cache state or horizon differs from the frozen compile contract")
        if spec["arm"] == "S1" and (spec.get("backend"), spec.get("compile_mode"), spec.get("dynamic")) != (
                "inductor", "default", None):
            raise ValueError("S1 must be torch.compile(backend='inductor', mode='default', dynamic=None)")
        model_spec = {**dict(spec["model_spec"]), "batch": int(spec["batch"]), "seed": seed}
        construction_started = time.perf_counter()
        model, inputs = load_model(model_spec)
        torch.cuda.synchronize()
        construction_seconds = time.perf_counter() - construction_started
        if not _all_float32(model.state_dict(), torch) or not _all_float32(inputs, torch):
            raise ValueError("the frozen compile study requires FP32 floating parameters and inputs")
        weights_digest = nested_digest(model.state_dict())
        inputs_digest = nested_digest(inputs)
        qualification = spec["qualification"]
        if weights_digest != qualification["weights_sha256"] or inputs_digest != qualification["inputs_sha256"]:
            raise ValueError("materialized weights or inputs differ from the frozen baseline-only qualification")
        runtime = capture_runtime(str(spec["gpu_uuid"]))
        reference_path = Path(spec["reference_path"])
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != qualification["reference_sha256"]:
            raise ValueError("baseline reference file differs from the frozen roster")
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        initial_layout = _layout_signature(inputs, torch)
        compile_wrapper_seconds = 0.0
        generated_kernel_count = 0
        executable = model
        if spec["arm"] == "S1":
            import torch._dynamo
            from torch._inductor import metrics

            torch._dynamo.config.suppress_errors = False
            metrics.reset()
            compile_started = time.perf_counter()
            executable = torch.compile(model, backend="inductor", mode="default", dynamic=None)
            compile_wrapper_seconds = time.perf_counter() - compile_started
        first_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            first_output = executable(*inputs)
            torch.cuda.synchronize()
            first_call_seconds = time.perf_counter() - first_started
            remaining_started = time.perf_counter()
            last_output = first_output
            for _ in range(int(spec["horizon"]) - 1):
                last_output = executable(*inputs)
            torch.cuda.synchronize()
            remaining_seconds = time.perf_counter() - remaining_started
        if spec["arm"] == "S1":
            generated_kernel_count = int(metrics.generated_kernel_count)
        cache_reuse_observed = (
            spec["arm"] == "S1"
            and spec["deployment"] == "PERSISTENT_CACHE_AVAILABLE"
            and spec["cache_state"] == "PREFILLED"
            and generated_kernel_count == 0
        )
        cold_compile_observed = (
            spec["arm"] == "S1"
            and spec["deployment"] == "COLD_NEW_GRAPH"
            and spec["cache_state"] == "EMPTY"
            and generated_kernel_count > 0
        )
        mechanism = {
            "active": cold_compile_observed or cache_reuse_observed,
            "torch_compile_called": spec["arm"] == "S1",
            "backend": "inductor" if spec["arm"] == "S1" else None,
            "mode": "default" if spec["arm"] == "S1" else None,
            "dynamic": None,
            "generated_kernel_count": generated_kernel_count,
            "cache_reuse_observed": cache_reuse_observed,
            "cold_compile_observed": cold_compile_observed,
            "compiled_callable_type": type(executable).__qualname__ if spec["arm"] == "S1" else None,
            "original_layout_retained": _layout_signature(inputs, torch) == initial_layout,
            "channels_last_requested": False,
            "amp_enabled": False, "tf32_enabled": False,
        }
        first_cpu, last_cpu = _cpu_tree(first_output), _cpu_tree(last_output)
        first_check = compare_outputs(reference, first_cpu, rtol=float(spec["correctness"]["rtol"]),
                                     atol=float(spec["correctness"]["atol"]))
        last_check = compare_outputs(reference, last_cpu, rtol=float(spec["correctness"]["rtol"]),
                                     atol=float(spec["correctness"]["atol"]))
        correct = first_check.passed and last_check.passed
        row.update({
            "status": "valid" if correct else "numeric_mismatch",
            "arm_execution": "COMPLETE", "scientific_correctness": "PASS" if correct else "FAIL",
            "correctness": {"passed": correct, "first_call": first_check.__dict__,
                            "last_call": last_check.__dict__},
            "mechanism_exposure": mechanism, "runtime_identity": runtime,
            "runtime_gpu_uuid": runtime["device_uuid"], "weights_sha256": weights_digest,
            "inputs_sha256": inputs_digest, "reference_sha256": qualification["reference_sha256"],
            "output_sha256": nested_digest(last_cpu),
            "input_layout_signature": initial_layout,
            "timing": {
                "model_input_construction_seconds": construction_seconds,
                "compile_wrapper_seconds": compile_wrapper_seconds,
                "generated_kernel_count": generated_kernel_count,
                "first_call_seconds": first_call_seconds,
                "remaining_H_minus_1_seconds": remaining_seconds,
                "measured_forward_seconds": first_call_seconds + remaining_seconds,
                "first_call_boundary": ("cold compile plus first forward" if
                    spec["arm"] == "S1" and spec["deployment"] == "COLD_NEW_GRAPH" else
                    "persistent cache lookup/read plus first forward" if
                    spec["arm"] == "S1" and spec["deployment"] == "PERSISTENT_CACHE_AVAILABLE" else
                    "eager first forward"),
            },
            "cache_scope_path": spec["cache_scope_path"],
            "cache_state": spec["cache_state"],
            "child_cpu_seconds": time.process_time() - cpu_started,
        })
    except Exception as exc:
        row.update({"status": "error", "arm_execution": "FAILED",
                    "scientific_correctness": "UNASSESSED", "error_type": type(exc).__name__,
                    "error": str(exc), "first_failure_stage": "SECONDARY_COMPILE_WORKER"})
    row["finished_monotonic"] = time.monotonic()
    row["worker_wall_seconds"] = time.perf_counter() - started
    return row


def _attempt_identity(*, manifest: Mapping[str, Any], pair: Mapping[str, Any], arm: str,
                      attempt_id: str) -> dict[str, Any]:
    context = {key: pair[key] for key in ("family_id", "split", "batch", "horizon", "deployment")}
    return {
        "study_id": manifest["study_id"], "intervention_id": manifest["intervention_id"],
        "pair_id": pair["pair_id"], "arm": arm, "purpose": pair["purpose"],
        "deployment": pair["deployment"], "family_id": pair["family_id"], "split": pair["split"],
        "source_commit": manifest["source_commit"], "config_digest": manifest["manifest_digest"],
        "context_digest": canonical_digest(context), "gpu_uuid": manifest["selected_gpu_uuid"],
        "attempt_id": attempt_id,
    }


def _run_arm(*, pair: Mapping[str, Any], arm: str, manifest: Mapping[str, Any],
             roster: Mapping[str, Any], root: Path, repo: Path,
             prefilled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attempt_id = f"{pair['pair_id']}-{arm}"
    if not _SAFE_ID.fullmatch(attempt_id):
        raise ValueError(f"attempt id is not path safe: {attempt_id}")
    attempt_dir = root / "raw" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    family = roster["family_by_id"][pair["family_id"]]
    qualification = family["qualification_by_batch"][str(pair["batch"])]
    cache_root = attempt_dir / "cache"
    cache_state = _cache_state_for(arm, str(pair["deployment"]))
    if (cache_state == "PREFILLED") != bool(prefilled):
        raise ValueError("persistent-cache S1 requires exactly one frozen prefilled artifact")
    spec = {
        "study_id": manifest["study_id"], "intervention_id": manifest["intervention_id"],
        "purpose": pair["purpose"], "family_id": pair["family_id"], "split": pair["split"],
        "pair_id": pair["pair_id"], "arm": arm, "deployment": pair["deployment"],
        "batch": int(pair["batch"]), "horizon": int(pair["horizon"]),
        "seed": int(family["model_spec"]["seed"]),
        "model_spec": {**dict(family["model_spec"]),
                       "torchbench_root": str(roster["torchbench_root"])},
        "qualification": dict(qualification), "correctness": dict(roster["correctness"]),
        "gpu_uuid": manifest["selected_gpu_uuid"], "cache_scope_path": str(cache_root.resolve()),
        "cache_state": cache_state, "backend": "inductor", "compile_mode": "default", "dynamic": None,
        "cache_artifact_digest": prefilled.get("artifact_digest") if prefilled else None,
        "cache_acquisition_cost_seconds": prefilled.get("acquisition_cost_seconds") if prefilled else None,
    }
    spec_path, output_path = attempt_dir / "spec.json", attempt_dir / "result.json"
    identity = _attempt_identity(manifest=manifest, pair=pair, arm=arm, attempt_id=attempt_id)
    full_identity = {**identity, "spec_digest": canonical_digest(spec)}
    prior = recover_attempt(attempt_dir, full_identity)
    if prior.get("arm_execution") != "NOT_STARTED":
        return prior
    if spec_path.is_file():
        if _read_json(spec_path) != spec:
            raise ValueError(f"attempt spec changed after its frozen identity: {attempt_id}")

    resource = manifest["resources"]
    if not wait_for_selected_gpu(
            root, str(manifest["selected_gpu_uuid"]),
            deadline=time.monotonic() + float(resource["gpu_wait_seconds"]),
            poll_seconds=float(resource["gpu_scan_interval_seconds"]),
            idle_memory_limit_mib=int(resource["idle_memory_limit_mib"])):
        return {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "SELECTED_GPU_NOT_IDLE", "identity": full_identity}

    materialization_started = time.monotonic()
    arm_deadline = materialization_started + float(resource["arm_timeout_seconds"])
    if prefilled:
        cache = create_cache_scope(
            attempt_dir, mode="PREFILLED", prefilled=prefilled["environment"],
            expected_artifact_digest=str(prefilled["artifact_digest"]),
            acquisition_cost_seconds=float(prefilled["acquisition_cost_seconds"]),
        )
    else:
        cache = create_cache_scope(attempt_dir, mode="EMPTY")
    materialization_seconds = time.monotonic() - materialization_started
    remaining_arm_seconds = arm_deadline - time.monotonic()
    if remaining_arm_seconds <= 0:
        return {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "CACHE_MATERIALIZATION_EXHAUSTED_ARM_BUDGET",
                "cache_materialization_seconds": materialization_seconds,
                "identity": full_identity}
    command = [sys.executable, "-m", "experiments.real_intervention_v4.secondary_compile",
               "--worker-spec", str(spec_path), "--worker-output", str(output_path)]
    overlay = str(Path(roster["dependency_overlay"]).expanduser().resolve())
    environment = {key: str(value) for key, value in cache["environment"].items()}
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (overlay, str(repo), os.environ.get("PYTHONPATH", ""))))
    result = execute_arm(
        results_root=root, attempt_id=attempt_id, command=command, spec=spec,
        identity=identity, gpu_uuid=str(manifest["selected_gpu_uuid"]),
        timeout_seconds=remaining_arm_seconds,
        cleanup_grace_seconds=float(resource["cleanup_grace_seconds"]),
        term_grace_seconds=float(resource["term_grace_seconds"]),
        heartbeat_seconds=float(resource["watchdog_heartbeat_seconds"]),
        sample_seconds=float(resource["gpu_sample_seconds"]), environment=environment,
    )
    recovered = recover_attempt(attempt_dir, full_identity)
    if recovered.get("receipt_verified"):
        recovered["cache_materialization_seconds"] = materialization_seconds
    return recovered if recovered.get("arm_execution") != "NOT_STARTED" else result


def _good_compiled_attempt(result: Mapping[str, Any], gpu_uuid: str) -> bool:
    worker = result.get("worker_result")
    return bool(
        result.get("arm_execution") == "COMPLETE" and result.get("receipt_verified") is True
        and isinstance(worker, Mapping) and worker.get("status") == "valid"
        and worker.get("scientific_correctness") == "PASS"
        and worker.get("runtime_gpu_uuid") == gpu_uuid
        and (worker.get("mechanism_exposure") or {}).get("active") is True
    )


def _acquire_cache(*, family_id: str, batch: int, manifest: Mapping[str, Any],
                   roster: Mapping[str, Any], root: Path, repo: Path) -> dict[str, Any]:
    attempt_id = f"cache-acquisition-{family_id}-B{batch}"
    pair = {"pair_id": attempt_id, "purpose": "CACHE_ACQUISITION", "family_id": family_id,
            "split": roster["family_by_id"][family_id]["split"], "batch": batch,
            "horizon": 1, "deployment": "COLD_NEW_GRAPH"}
    result = _run_arm(pair=pair, arm="S1", manifest=manifest, roster=roster,
                      root=root, repo=repo)
    if not _good_compiled_attempt(result, str(manifest["selected_gpu_uuid"])):
        execution = result.get("arm_execution")
        status = execution if execution in {"NOT_STARTED", "UNKNOWN"} else "FAILED"
        return {"status": status, "attempt": result}
    record_path = root / "cache_acquisitions" / f"{family_id}-B{batch}.json"
    if record_path.is_file():
        prior = _read_json(record_path)
        if (prior.get("study_id") != manifest["study_id"]
                or prior.get("family_id") != family_id or prior.get("batch") != batch
                or prior.get("source_receipt_sha256") != result.get("raw_receipt_sha256")):
            raise ValueError("cache acquisition record differs from its verified compile receipt")
        return {"status": "ACQUIRED", **prior}
    attempt_dir = root / "raw" / f"{attempt_id}-S1"
    scope = _read_json(attempt_dir / "CACHE_SCOPE.json")
    environment = {name: Path(scope["environment"][name]) for name in _CACHE_ENV_NAMES}
    artifact_digest, artifact_parts = cache_artifact_digest(environment)
    cache_sizes = {name: cache_usage(path) for name, path in environment.items()}
    if (any(item["status"] != "COMPLETE" for item in cache_sizes.values())
            or sum(int(item["bytes"]) for item in cache_sizes.values()) <= 0):
        return {"status": "FAILED", "attempt": result,
                "cache_status": "CACHE_ARTIFACT_EMPTY_OR_INCOMPLETE", "cache_sizes": cache_sizes}
    worker = result["worker_result"]
    timing = worker["timing"]
    acquisition_cost = float(timing["first_call_seconds"])
    parent_wall = result.get("parent_wall_seconds")
    record = {
        "study_id": manifest["study_id"], "intervention_id": manifest["intervention_id"],
        "family_id": family_id, "batch": batch, "attempt_id": f"{attempt_id}-S1",
        "artifact_digest": artifact_digest, "artifact_parts": artifact_parts,
        "cache_sizes": cache_sizes,
        "environment": {name: str(path) for name, path in environment.items()},
        "acquisition_cost_seconds": acquisition_cost,
        "acquisition_parent_wall_seconds": parent_wall,
        "acquisition_cache_state": scope["cache_state"],
        "source_receipt_sha256": result.get("raw_receipt_sha256"),
    }
    atomic_json(record_path, record, exclusive=True)
    return {"status": "ACQUIRED", **record}


def _pair_observation(*, planned: Mapping[str, Any], manifest: Mapping[str, Any],
                      roster: Mapping[str, Any], root: Path, repo: Path,
                      acquisition: Mapping[str, Any] | None) -> dict[str, Any]:
    arm_results: dict[str, dict[str, Any]] = {}
    acquisition_status = acquisition.get("status") if acquisition else None
    if planned["deployment"] == "PERSISTENT_CACHE_AVAILABLE" and acquisition_status in {
            "UNKNOWN", "NOT_STARTED"}:
        arm_results = {
            "S0": {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"},
            "S1": {"arm_execution": acquisition_status,
                   "scientific_correctness": "UNKNOWN" if acquisition_status == "UNKNOWN" else "UNASSESSED",
                   "first_failure_stage": "CACHE_ACQUISITION"},
        }
    else:
        for arm in planned["order"]:
            if arm == "S1" and planned["deployment"] == "PERSISTENT_CACHE_AVAILABLE":
                if not acquisition or acquisition_status != "ACQUIRED":
                    arm_results[arm] = {"arm_execution": "FAILED", "scientific_correctness": "UNASSESSED",
                                        "first_failure_stage": "CACHE_ACQUISITION",
                                        "root_cause": "PERSISTENT_CACHE_ARTIFACT_UNAVAILABLE"}
                    continue
                prefilled = acquisition
            else:
                prefilled = None
            arm_results[arm] = _run_arm(pair=planned, arm=arm, manifest=manifest,
                                        roster=roster, root=root, repo=repo, prefilled=prefilled)
            if arm_results[arm].get("arm_execution") in {"UNKNOWN", "NOT_STARTED"}:
                break
    for arm in ("S0", "S1"):
        arm_results.setdefault(arm, {"arm_execution": "NOT_STARTED",
                                     "scientific_correctness": "UNASSESSED"})
    exposure = compare_arm_exposure(arm_results["S0"], arm_results["S1"])
    worker = arm_results["S1"].get("worker_result")
    mechanism_active = ((worker.get("mechanism_exposure") or {}).get("active")
                        if isinstance(worker, Mapping) else None)
    classification = classify_pair(
        arm_results["S0"], arm_results["S1"], intervention_applicable=True,
        exposure_match=exposure.get("matched"),
        expected_gpu_uuid=str(manifest["selected_gpu_uuid"]),
        mechanism_active=mechanism_active,
    ).to_dict()
    observation = {
        "schema_version": 1, "study_id": manifest["study_id"],
        "intervention_id": manifest["intervention_id"], "manifest_digest": manifest["manifest_digest"],
        **dict(planned), "selected_gpu_uuid": manifest["selected_gpu_uuid"],
        "S0": arm_results["S0"], "S1": arm_results["S1"],
        "exposure": exposure, "classification": classification,
        "terminal_pair": all(value.get("arm_execution") in {"COMPLETE", "FAILED", "TIMEOUT"}
                              for value in arm_results.values()),
    }
    if acquisition:
        observation["cache_acquisition"] = {
            key: acquisition.get(key) for key in (
                "artifact_digest", "acquisition_cost_seconds", "acquisition_parent_wall_seconds",
                "source_receipt_sha256", "status",
            )
        }
    return observation


def _write_pair(observation: Mapping[str, Any], root: Path) -> None:
    atomic_json(root / "pairs" / f"{observation['pair_id']}.json", dict(observation))


def _canaries(manifest: Mapping[str, Any], roster: Mapping[str, Any], root: Path,
              repo: Path) -> dict[str, Any]:
    results = []
    batch = max(manifest["contract"]["batches"])
    for family_id in manifest["compile_canary_families"]:
        acquisition = _acquire_cache(family_id=family_id, batch=batch, manifest=manifest,
                                     roster=roster, root=root, repo=repo)
        pair_id = f"cache-reload-canary-{family_id}-B{batch}"
        planned = {"pair_id": pair_id, "purpose": "CACHE_RELOAD_CANARY",
                   "family_id": family_id, "split": "development", "batch": batch,
                   "horizon": 1, "deployment": "PERSISTENT_CACHE_AVAILABLE", "order": ["S1"]}
        if acquisition.get("status") == "ACQUIRED":
            reload_result = _run_arm(pair=planned, arm="S1", manifest=manifest, roster=roster,
                                     root=root, repo=repo, prefilled=acquisition)
        else:
            status = acquisition.get("status")
            execution = status if status in {"UNKNOWN", "NOT_STARTED"} else "FAILED"
            reload_result = {"arm_execution": execution,
                             "scientific_correctness": "UNKNOWN" if execution == "UNKNOWN" else "UNASSESSED",
                             "first_failure_stage": "CACHE_ACQUISITION"}
        passed = acquisition.get("status") == "ACQUIRED" and _good_compiled_attempt(
            reload_result, str(manifest["selected_gpu_uuid"]))
        row = {"family_id": family_id, "batch": batch,
               "cold_compile": acquisition, "persistent_cache_reload": reload_result,
               "passed": passed}
        atomic_json(root / "canaries" / f"{family_id}-B{batch}.json", row)
        results.append(row)
        if not passed:
            break

    def canary_execution(row: Mapping[str, Any], field: str) -> str:
        value = row[field]
        if field == "cold_compile":
            status = value.get("status")
            if status != "ACQUIRED":
                return str(status)
            return "COMPLETE"
        return str(value.get("arm_execution", "UNKNOWN"))

    states = [(canary_execution(row, "cold_compile"),
               canary_execution(row, "persistent_cache_reload")) for row in results]
    return {"study_id": manifest["study_id"], "manifest_digest": manifest["manifest_digest"],
            "planned_canaries": len(manifest["compile_canary_families"]),
            "attempted_canaries": len(results),
            "unattempted_canaries": len(manifest["compile_canary_families"]) - len(results),
            "unknown_canaries": sum("UNKNOWN" in pair for pair in states),
            "not_started_canaries": sum("NOT_STARTED" in pair for pair in states),
            "canaries": results,
            "passed": len(results) == len(manifest["compile_canary_families"])
            and bool(results) and all(row["passed"] for row in results)}


def _shakedown(manifest: Mapping[str, Any], roster: Mapping[str, Any], root: Path,
               repo: Path) -> dict[str, Any]:
    rows = []
    for planned in manifest["shakedown_plan"]:
        acquisition = None
        if planned["deployment"] == "PERSISTENT_CACHE_AVAILABLE":
            acquisition = _acquire_cache(family_id=planned["family_id"], batch=planned["batch"],
                                         manifest=manifest, roster=roster, root=root, repo=repo)
        row = _pair_observation(planned=planned, manifest=manifest, roster=roster,
                                root=root, repo=repo, acquisition=acquisition)
        _write_pair(row, root)
        rows.append(row)
        if any(result.get("arm_execution") in {"UNKNOWN", "NOT_STARTED"}
               for result in (row["S0"], row["S1"])):
            break
    valid = [row for row in rows if row["classification"].get("paired_comparability") == "MATCHED"
             and row["classification"].get("scientific_correctness") == "PASS"
             and row["classification"].get("policy_evidence") == "ELIGIBLE"]
    gate = roster["shakedown_gate"]
    fraction = len(valid) / len(manifest["shakedown_plan"]) if manifest["shakedown_plan"] else 0.0
    unknown_pairs = sum(any(result.get("arm_execution") == "UNKNOWN"
                            for result in (row["S0"], row["S1"])) for row in rows)
    unstarted_pairs = (len(manifest["shakedown_plan"]) - len(rows)
                       + sum(any(result.get("arm_execution") == "NOT_STARTED"
                                 for result in (row["S0"], row["S1"])) for row in rows))
    passes = (len(rows) == len(manifest["shakedown_plan"])
              and len(rows) >= int(gate["minimum_pairs"])
              and fraction >= float(gate["minimum_matched_correct_fraction"]))
    return {
        "study_id": manifest["study_id"], "manifest_digest": manifest["manifest_digest"],
        "planned_pairs": len(manifest["shakedown_plan"]), "observed_pairs": len(rows),
        "valid_pairs": len(valid), "unstarted_pairs": unstarted_pairs,
        "unknown_pairs": unknown_pairs,
        "independent_families_touched": len({row["family_id"] for row in rows}),
        "matched_correct_fraction": fraction,
        "frozen_gate": dict(gate), "passes": passes,
        "pairs": [row["pair_id"] for row in rows],
    }


def _pair_effect(row: Mapping[str, Any], reuse_count: int | None = None) -> dict[str, Any] | None:
    classification = row.get("classification") or {}
    if (classification.get("paired_comparability") != "MATCHED"
            or classification.get("scientific_correctness") != "PASS"
            or classification.get("policy_evidence") != "ELIGIBLE"):
        return None
    s0, s1 = row.get("S0", {}).get("worker_result"), row.get("S1", {}).get("worker_result")
    if not isinstance(s0, Mapping) or not isinstance(s1, Mapping):
        return None
    t0 = float(s0["timing"]["measured_forward_seconds"])
    t1 = float(s1["timing"]["measured_forward_seconds"])
    acquisition = 0.0
    if row["deployment"] == "PERSISTENT_CACHE_AVAILABLE":
        acquisition = float((row.get("cache_acquisition") or {})["acquisition_cost_seconds"])
        if reuse_count is None or reuse_count < 1:
            raise ValueError("persistent-cache effect requires one declared reuse count")
        t1 += acquisition / reuse_count
    if t0 <= 0 or t1 <= 0:
        return None
    return {"s0_seconds_per_deployment": t0, "s1_seconds_per_deployment": t1,
            "seconds_saved": t0 - t1, "log_benefit": math.log(t0 / t1),
            "reuse_count": reuse_count, "acquisition_cost_amortized_seconds": acquisition / reuse_count
                if reuse_count else None,
            "amortization_is_projection_from_one_measured_cache_reuse":
                row["deployment"] == "PERSISTENT_CACHE_AVAILABLE"}


def _summarize(pairs: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "independent_unit": "family_id; contexts and horizons are averaged within family",
        "persistent_first_call_boundary": "cache lookup/read and first forward are measured together because Inductor compilation is lazy",
        "reuse_counts_are_cost_projections": [1, 10, 100],
        "splits": {},
    }
    for split in sorted(_SPLITS):
        split_rows = [row for row in pairs if row["split"] == split]
        split_report: dict[str, Any] = {
            "independent_families": len({row["family_id"] for row in split_rows}),
            "planned_pairs": len(split_rows), "valid_pairs": sum(
                row["classification"].get("policy_evidence") == "ELIGIBLE" for row in split_rows),
            "deployments": {},
        }
        for deployment in sorted(_DEPLOYMENTS):
            deployment_rows = [row for row in split_rows if row["deployment"] == deployment]
            persistent = next(row for row in contract["deployment_contracts"]
                              if row["id"] == "PERSISTENT_CACHE_AVAILABLE")
            reuse_counts = [None] if deployment == "COLD_NEW_GRAPH" else list(
                persistent["declared_reuse_counts"]
            )
            deployment_report = {}
            for reuse_count in reuse_counts:
                by_family: dict[str, list[float]] = {}
                saved_by_family: dict[str, list[float]] = {}
                for row in deployment_rows:
                    effect = _pair_effect(row, reuse_count)
                    if effect is None:
                        continue
                    by_family.setdefault(row["family_id"], []).append(effect["log_benefit"])
                    saved_by_family.setdefault(row["family_id"], []).append(effect["seconds_saved"])
                family_log = {key: sum(value) / len(value) for key, value in sorted(by_family.items())}
                family_saved = {key: sum(value) / len(value) for key, value in sorted(saved_by_family.items())}
                key = "per_cold_deployment" if reuse_count is None else f"reuse_{reuse_count}"
                deployment_report[key] = {
                    "independent_families_with_valid_pairs": len(family_log),
                    "family_mean_log_benefit": (sum(family_log.values()) / len(family_log) if family_log else None),
                    "family_mean_seconds_saved": (sum(family_saved.values()) / len(family_saved) if family_saved else None),
                    "per_family_mean_log_benefit": family_log,
                }
            split_report["deployments"][deployment] = deployment_report
        report["splits"][split] = split_report
    return report


def _run_study(*, roster_path: Path, results_root: Path, gpu_uuid: str,
        arm_timeout_seconds: float, cleanup_grace_seconds: float,
        term_grace_seconds: float, gpu_wait_seconds: float = 1800.0,
        idle_memory_limit_mib: int = 512, gpu_scan_interval_seconds: float = 15.0,
        contract_path: Path = _CONTRACT_PATH) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    roster_bytes = roster_path.read_bytes()
    roster = _validate_roster(json.loads(roster_bytes.decode("utf-8")), contract)
    if arm_timeout_seconds <= 0 or cleanup_grace_seconds < 0 or term_grace_seconds < 0:
        raise ValueError("arm deadline must be positive and cleanup/termination grace non-negative")
    if term_grace_seconds > cleanup_grace_seconds or gpu_wait_seconds <= 0 or gpu_scan_interval_seconds <= 0:
        raise ValueError("termination grace must fit cleanup grace and GPU wait/scan periods must be positive")
    if idle_memory_limit_mib < 0:
        raise ValueError("idle memory threshold cannot be negative")
    root = results_root.expanduser().resolve()
    repo, source_commit = _server_identity(roster, root, gpu_uuid)
    resources = {
        "arm_timeout_seconds": float(arm_timeout_seconds),
        "cleanup_grace_seconds": float(cleanup_grace_seconds),
        "term_grace_seconds": float(term_grace_seconds),
        "gpu_wait_seconds": float(gpu_wait_seconds),
        "idle_memory_limit_mib": int(idle_memory_limit_mib),
        "gpu_scan_interval_seconds": float(gpu_scan_interval_seconds),
        "watchdog_heartbeat_seconds": 30.0,
        "gpu_sample_seconds": 2.0,
        "concurrency": 1, "gpu_workers": 1, "cpu_threads": 1,
        "worker_processes": 1,
    }
    manifest = _freeze_manifest(contract, roster, roster_bytes=roster_bytes,
                                source_commit=source_commit, gpu_uuid=gpu_uuid,
                                resources=resources)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("resume identity differs from the previously frozen secondary study")
    else:
        existing = [path for path in root.iterdir()
                    if path.name not in {"RUN_MANIFEST.json", ".study.lock"}]
        if existing:
            raise ValueError("refusing to adopt an unmanifested result root")
        atomic_json(manifest_path, manifest, exclusive=True)
    prior_completion = root / "RUN_COMPLETE.json"
    if prior_completion.is_file():
        completion = _read_json(prior_completion)
        if completion.get("manifest_digest") != manifest["manifest_digest"]:
            raise ValueError("terminal run record differs from the frozen manifest")
        return completion

    canaries = _canaries(manifest, roster, root, repo)
    atomic_json(root / "CANARY_REPORT.json", canaries)
    if not canaries["passed"]:
        if canaries.get("unknown_canaries", 0):
            return {"status": "CANARY_UNKNOWN", "manifest_digest": manifest["manifest_digest"],
                    "exit_code": 1, "final_opened": False}
        if canaries.get("not_started_canaries", 0):
            return {"status": "WAITING_FOR_CANARY_RESOURCES", "manifest_digest": manifest["manifest_digest"],
                    "exit_code": 1, "final_opened": False}
        completion = {"status": "COMPILE_OR_CACHE_CANARY_FAILED", "manifest_digest": manifest["manifest_digest"],
                      "final_opened": False, "exit_code": 3}
        atomic_json(root / "RUN_COMPLETE.json", completion)
        atomic_json(root / "RUN_EXIT.json", {"status": completion["status"], "exit_code": 3})
        return completion

    shake = _shakedown(manifest, roster, root, repo)
    atomic_json(root / "SHAKEDOWN_REPORT.json", shake)
    if not shake["passes"]:
        if shake["unknown_pairs"]:
            return {"status": "SHAKEDOWN_UNKNOWN", "manifest_digest": manifest["manifest_digest"],
                    "final_opened": False, "exit_code": 1}
        if shake["unstarted_pairs"]:
            return {"status": "WAITING_FOR_SHAKEDOWN_RESOURCES",
                    "manifest_digest": manifest["manifest_digest"], "final_opened": False,
                    "exit_code": 1}
        completion = {"status": "ENGINEERING_SHAKEDOWN_FAILED", "manifest_digest": manifest["manifest_digest"],
                      "final_opened": False, "shakedown": shake, "exit_code": 3}
        atomic_json(root / "RUN_COMPLETE.json", completion)
        atomic_json(root / "RUN_EXIT.json", {"status": completion["status"], "exit_code": 3})
        return completion

    final_opened = {
        "study_id": manifest["study_id"], "manifest_digest": manifest["manifest_digest"],
        "baseline_only_roster_frozen": True, "family_split_frozen": True,
        "compile_and_cache_reload_canaries_passed": True,
        "engineering_shakedown_passed": True,
    }
    final_open_path = root / "FINAL_OPENED.json"
    if final_open_path.exists():
        if _read_json(final_open_path) != final_opened:
            raise ValueError("final-open record differs from the frozen preflight gates")
    else:
        atomic_json(final_open_path, final_opened, exclusive=True)
    observations = []
    for planned in manifest["formal_plan"]:
        acquisition = None
        if planned["deployment"] == "PERSISTENT_CACHE_AVAILABLE":
            acquisition = _acquire_cache(family_id=planned["family_id"], batch=planned["batch"],
                                         manifest=manifest, roster=roster, root=root, repo=repo)
        observation = _pair_observation(planned=planned, manifest=manifest, roster=roster,
                                        root=root, repo=repo, acquisition=acquisition)
        _write_pair(observation, root)
        observations.append(observation)
        if not observation["terminal_pair"]:
            break

    summary = _summarize(observations, contract)
    atomic_json(root / "FAMILY_SUMMARY.json", summary)
    unknown = sum(any(result.get("arm_execution") == "UNKNOWN"
                      for result in (row["S0"], row["S1"])) for row in observations)
    unstarted = (len(manifest["formal_plan"]) - len(observations)
                 + sum(any(result.get("arm_execution") == "NOT_STARTED"
                           for result in (row["S0"], row["S1"])) for row in observations))
    unclosed = len(manifest["formal_plan"]) - sum(row["terminal_pair"] for row in observations)
    complete = unclosed == 0 and len(observations) == len(manifest["formal_plan"])
    result = {
        "status": "COMPLETE" if complete else "INCOMPLETE_TERMINAL_UNKNOWN_OR_UNSTARTED",
        "study_id": manifest["study_id"], "intervention_id": manifest["intervention_id"],
        "manifest_digest": manifest["manifest_digest"], "selected_gpu_uuid": gpu_uuid,
        "planned_pairs": len(manifest["formal_plan"]), "observed_pairs": len(observations),
        "unknown_pairs": unknown, "unstarted_pairs": unstarted,
        "unclosed_pairs": unclosed, "independent_families": len(roster["families"]),
        "final_opened": True, "family_summary": str(root / "FAMILY_SUMMARY.json"),
        "exit_code": 0 if complete else 1,
    }
    if complete:
        atomic_json(root / "RUN_COMPLETE.json", result)
    atomic_json(root / "RUN_EXIT.json", {"status": result["status"], "exit_code": result["exit_code"]})
    return result


def run(*, roster_path: Path, results_root: Path, gpu_uuid: str,
        arm_timeout_seconds: float, cleanup_grace_seconds: float,
        term_grace_seconds: float, gpu_wait_seconds: float = 1800.0,
        idle_memory_limit_mib: int = 512, gpu_scan_interval_seconds: float = 15.0,
        contract_path: Path = _CONTRACT_PATH) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("secondary compile evidence is supported only on the configured server")
    expected_repo = (Path.home() / "CODE" / "scientific-performance-engineering").resolve()
    expected_python = Path.home() / "miniconda3" / "envs" / "equivcompiler" / "bin" / "python"
    if _repo_root() != expected_repo or Path.cwd().resolve() != expected_repo:
        raise RuntimeError(f"run from the canonical server checkout: {expected_repo}")
    if Path(sys.executable) != expected_python:
        raise RuntimeError(f"use the project interpreter: {expected_python}")
    root = results_root.expanduser().resolve()
    allowed_root = (Path.home() / "RESULTS" / "evospe-rt-v4").resolve()
    expected_name = "EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019"
    if root.parent != allowed_root or root.name != expected_name:
        raise ValueError(f"secondary results must use {allowed_root / expected_name}")
    root.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(root / ".study.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            flock(lock_fd, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another secondary-study runner owns this independent result root") from exc
        return _run_study(
            roster_path=roster_path, results_root=root, gpu_uuid=gpu_uuid,
            arm_timeout_seconds=arm_timeout_seconds,
            cleanup_grace_seconds=cleanup_grace_seconds,
            term_grace_seconds=term_grace_seconds,
            gpu_wait_seconds=gpu_wait_seconds,
            idle_memory_limit_mib=idle_memory_limit_mib,
            gpu_scan_interval_seconds=gpu_scan_interval_seconds,
            contract_path=contract_path,
        )
    finally:
        flock(lock_fd, LOCK_UN)
        os.close(lock_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=_CONTRACT_PATH)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--arm-timeout-seconds", type=float)
    parser.add_argument("--cleanup-grace-seconds", type=float)
    parser.add_argument("--term-grace-seconds", type=float)
    parser.add_argument("--gpu-wait-seconds", type=float, default=1800.0)
    parser.add_argument("--idle-memory-limit-mib", type=int, default=512)
    parser.add_argument("--gpu-scan-interval-seconds", type=float, default=15.0)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec or args.worker_output:
        if not args.worker_spec or not args.worker_output:
            parser.error("worker requires both --worker-spec and --worker-output")
        atomic_json(args.worker_output, _worker(_read_json(args.worker_spec)))
        return
    required = (args.roster, args.results_root, args.gpu_uuid, args.arm_timeout_seconds,
                args.cleanup_grace_seconds, args.term_grace_seconds)
    if any(value is None for value in required):
        parser.error("study execution requires roster, independent results root, UUID and fixed arm deadlines")
    result = run(
        roster_path=args.roster, results_root=args.results_root, gpu_uuid=args.gpu_uuid,
        arm_timeout_seconds=args.arm_timeout_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        term_grace_seconds=args.term_grace_seconds,
        gpu_wait_seconds=args.gpu_wait_seconds,
        idle_memory_limit_mib=args.idle_memory_limit_mib,
        gpu_scan_interval_seconds=args.gpu_scan_interval_seconds,
        contract_path=args.contract,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(int(result.get("exit_code", 0)))


if __name__ == "__main__":
    main()
