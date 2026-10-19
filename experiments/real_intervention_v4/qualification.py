"""Performance-blind baseline and channels-last correctness qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import traceback
from typing import Any, Mapping


def _map_tensors(value: Any, transform) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return transform(value) if value.ndim == 4 else value
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    return value


def _shapes(value: Any) -> dict[str, list[int]]:
    import torch

    result: dict[str, list[int]] = {}

    def visit(item: Any, name: str) -> None:
        if isinstance(item, torch.Tensor):
            result[name] = list(item.shape)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{name}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]")

    visit(value, "input")
    return result


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def _profile_cuda_forward(model: Any, inputs: Any, torch: Any) -> dict[str, Any]:
    """Capture one qualification-only forward, outside every effect timing window."""
    started = time.perf_counter()
    evidence: dict[str, Any] = {
        "capture": "qualification_only",
        "included_in_performance_timing": False,
        "profiled_forwards": 0,
        "cuda_kernels_executed": False,
        "cuda_kernel_event_count": 0,
        "cuda_kernel_event_names": [],
    }
    try:
        profiler = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ])
        with torch.inference_mode(), profiler:
            model(*inputs)
            torch.cuda.synchronize()
            evidence["profiled_forwards"] = 1
        events = profiler.events()
        names = []
        cuda_events = []
        for event in events:
            device_type = getattr(event, "device_type", None)
            device_name = getattr(device_type, "name", str(device_type).rsplit(".", 1)[-1])
            event_name = event.name
            if device_name == "CUDA" and not any(
                marker in event_name.lower() for marker in ("memcpy", "memset")
            ):
                cuda_events.append(event)
                if event_name not in names:
                    names.append(event_name)
        evidence.update({
            "cuda_kernel_event_count": len(cuda_events),
            "cuda_kernel_event_names": names[:32],
        })
        evidence["cuda_kernels_executed"] = evidence["cuda_kernel_event_count"] > 0
        evidence["capture_status"] = "CAPTURED" if evidence["cuda_kernels_executed"] else "NO_CUDA_KERNEL_EVENTS"
    except Exception as exc:
        evidence.update({
            "capture_status": "PROFILER_FAILURE",
            "capture_error": f"{type(exc).__name__}: {exc}",
        })
    evidence["profiling_overhead_seconds"] = time.perf_counter() - started
    return evidence


def has_cuda_kernel_events(evidence: Any) -> bool:
    event_count = evidence.get("cuda_kernel_event_count") if isinstance(evidence, Mapping) else None
    return (
        isinstance(evidence, Mapping)
        and evidence.get("cuda_kernels_executed") is True
        and isinstance(event_count, int) and not isinstance(event_count, bool)
        and event_count > 0
        and bool(evidence.get("cuda_kernel_event_names"))
        and evidence.get("included_in_performance_timing") is False
    )


def has_pairable_kernel_exposure(exposure: Any) -> bool:
    return (
        isinstance(exposure, Mapping)
        and exposure.get("pairability_status") == "PAIRABLE"
        and exposure.get("active") is True
        and has_cuda_kernel_events(exposure.get("s0_cuda_kernel_path"))
        and has_cuda_kernel_events(exposure.get("s1_cuda_kernel_path"))
    )


def _layout_probe(model: Any, inputs: Any, torch: Any) -> dict[str, Any]:
    input_strides = []
    for value in _flatten(inputs):
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            input_strides.append({
                "shape": list(value.shape), "stride": list(value.stride()),
                "channels_last": value.is_contiguous(memory_format=torch.channels_last),
            })
    conv2d = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    calls: list[dict[str, Any]] = []
    handles = []

    def observe(module: Any, args: tuple[Any, ...]) -> None:
        tensor = args[0] if args else None
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            calls.append({
                "input_shape": list(tensor.shape), "input_stride": list(tensor.stride()),
                "input_channels_last": tensor.is_contiguous(memory_format=torch.channels_last),
                "weight_stride": list(module.weight.stride()),
                "weight_channels_last": module.weight.is_contiguous(memory_format=torch.channels_last),
            })

    for module in conv2d:
        handles.append(module.register_forward_pre_hook(observe))
    try:
        kernel_path = _profile_cuda_forward(model, inputs, torch)
    finally:
        for handle in handles:
            handle.remove()
    active = bool(calls) and all(item["input_channels_last"] and item["weight_channels_last"]
                                 for item in calls)
    return {
        "input_4d_tensors": input_strides,
        "conv2d_module_count": len(conv2d),
        "executed_conv2d_calls": calls,
        "active": active,
        "cuda_kernel_path": kernel_path,
    }


def _flatten(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten(item)


def _context(spec: Mapping[str, Any]) -> dict[str, Any]:
    from .source_registry import candidate_source_identity, family_lineage_id

    candidate = dict(spec["candidate"])
    source_identity = spec.get("source_identity") or candidate.get("source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = candidate_source_identity(
            candidate, source_revisions=spec.get("source_revisions"),
        )
    expected = candidate_source_identity(
        candidate, source_revisions=spec.get("source_revisions"),
    )
    identity_fields = ["candidate_id", "source", "model", "family_lineage_id", "registration"]
    if "source_revisions" in spec:
        identity_fields.append("source_revisions")
    identity_mismatches = [
        name for name in identity_fields if source_identity.get(name) != expected.get(name)
    ]
    return {
        "candidate_id": str(candidate.get("candidate_id", f"{candidate['source']}:{candidate['model']}")),
        "source": candidate.get("source"), "model": candidate.get("model"),
        "family_id": family_lineage_id(candidate),
        "family_lineage_id": family_lineage_id(candidate),
        "batch": int(spec["batch"]), "source_identity": dict(source_identity),
        "source_identity_valid": not identity_mismatches,
        "source_identity_mismatches": identity_mismatches,
        "candidate": candidate,
    }


def classify_qualification_failure(phase: str, stage: str) -> dict[str, str]:
    """Keep baseline failure, treatment failure, and platform identity distinct."""
    if stage == "RUNTIME_IDENTITY":
        return {
            "status": "ERROR", "eligibility": "RUNTIME_PREFLIGHT_FAILURE",
            "arm_execution": "FAILED", "scientific_correctness": "UNASSESSED",
        }
    if stage in {"S0_REFERENCE_READ", "S0_REFERENCE_WRITE"}:
        return {
            "status": "ERROR", "eligibility": "QUALIFICATION_REFERENCE_INVALID",
            "arm_execution": "FAILED", "scientific_correctness": "UNASSESSED",
        }
    if phase.startswith("S1"):
        return {
            "status": "ERROR", "eligibility": "CHANNELS_LAST_EXECUTION_FAILURE",
            "arm_execution": "FAILED", "scientific_correctness": "UNKNOWN",
        }
    return {
        "status": "BASELINE_INVALID", "eligibility": "BASELINE_INVALID",
        "arm_execution": "FAILED", "scientific_correctness": "UNASSESSED",
    }


def cross_process_qualification_identity_differences(
    reference: Mapping[str, Any], *, source_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any], expected_gpu_uuid: str,
    weights_sha256: str, inputs_sha256: str,
) -> dict[str, Any]:
    from .runtime_identity import (
        cross_process_runtime_differences, selected_gpu_identity_differences,
    )

    return {
        "runtime": cross_process_runtime_differences(
            dict(reference.get("runtime_identity") or {}), dict(runtime_identity),
        ),
        "selected_gpu": selected_gpu_identity_differences(
            dict(runtime_identity), expected_gpu_uuid,
        ),
        "source_identity_match": reference.get("source_identity") == dict(source_identity),
        "weights_match": reference.get("weights_sha256") == weights_sha256,
        "inputs_match": reference.get("inputs_sha256") == inputs_sha256,
    }


def _failure(spec: Mapping[str, Any], phase: str, stage: str,
             exc: BaseException) -> dict[str, Any]:
    result = {
        **_context(spec), "qualification_stage": phase,
        **classify_qualification_failure(phase, stage),
        "first_failure_stage": stage,
        "reason": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "candidate_denominator": True,
        "performance_timing_recorded": False,
    }
    return result


def _source_identity_failure(context: Mapping[str, Any], phase: str) -> dict[str, Any]:
    return {
        **dict(context), "qualification_stage": phase,
        "status": "ERROR", "eligibility": "QUALIFICATION_IDENTITY_MISMATCH",
        "arm_execution": "FAILED", "scientific_correctness": "UNASSESSED",
        "first_failure_stage": "SOURCE_IDENTITY",
        "identity_mismatches": context.get("source_identity_mismatches", []),
        "candidate_denominator": True, "performance_timing_recorded": False,
    }


def _paths(spec: Mapping[str, Any]) -> tuple[Path, Path]:
    candidate_id = str(spec["candidate"].get("candidate_id"))
    key = "".join(char if char.isalnum() or char in "-_." else "-" for char in candidate_id)
    batch = int(spec["batch"])
    if spec.get("reference_path"):
        reference_path = Path(str(spec["reference_path"]))
    else:
        reference_path = Path(str(spec["reference_root"])) / f"{key}-B{batch}.pt"
    metadata_path = Path(str(spec.get("metadata_path") or reference_path.with_suffix(".s0.json")))
    return reference_path, metadata_path


def _model_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(spec["candidate"]), "seed": int(spec["seed"]),
        "batch": int(spec["batch"]), "shape": list(spec.get("shape", (3, 224, 224))),
        "horizon": 1, "torchbench_root": str(spec["torchbench_root"]),
        "correctness": dict(spec["correctness"]), "precision": "float32",
        "mode": "eval", "gpu_uuid": str(spec["gpu_uuid"]),
    }


def _seed_process(seed: int, torch: Any, numpy: Any) -> None:
    from experiments.real_intervention_v3.numerical_checks import configure_determinism

    configure_determinism(torch)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_reference(torch: Any, reference_path: Path, output: Any) -> str:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=reference_path.parent, prefix="s0-aa-", suffix=".pt", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(_cpu_tree(output), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, reference_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(reference_path.read_bytes()).hexdigest()


def _read_reference(torch: Any, reference_path: Path, metadata: Mapping[str, Any]) -> Any:
    actual_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if actual_digest != metadata.get("reference_file_sha256"):
        raise ValueError("S0 reference file digest differs from its metadata")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    from experiments.real_intervention_v3.numerical_checks import nested_digest

    if nested_digest(reference) != metadata.get("output_sha256"):
        raise ValueError("S0 reference output differs from its recorded output identity")
    return reference


def _s0_within_process_aa(model: Any, inputs: Any, *, rtol: float,
                          atol: float) -> tuple[Any, dict[str, Any]]:
    """Compare two untimed eager forwards in the same qualification process."""
    import torch

    from experiments.real_intervention_v3.numerical_checks import compare_outputs, nested_digest

    with torch.inference_mode():
        first = model(*inputs)
        torch.cuda.synchronize()
        first_cpu = _cpu_tree(first)
        del first
        replicate = model(*inputs)
        torch.cuda.synchronize()
        replicate_cpu = _cpu_tree(replicate)
        del replicate
    comparison = compare_outputs(first_cpu, replicate_cpu, rtol=rtol, atol=atol)
    return first_cpu, {
        "passed": bool(comparison.passed),
        "comparison": comparison.__dict__,
        "first_output_sha256": nested_digest(first_cpu),
        "replicate_output_sha256": nested_digest(replicate_cpu),
        "performance_timing_recorded": False,
    }


def qualify_s0_process(spec: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    """Run S0 within-process A/A, then cross-process reference/replicate checks."""
    import numpy as np
    import torch

    from experiments.real_intervention_v3.numerical_checks import compare_outputs, nested_digest
    from .model_adapter import load_model
    from .outcomes import atomic_json
    from .runtime_identity import capture_runtime

    if phase not in {"reference", "replicate"}:
        raise ValueError("S0 phase must be reference or replicate")
    phase_name = "S0_REFERENCE" if phase == "reference" else "S0_AA_REPLICATE"
    context = _context(spec)
    if not context["source_identity_valid"]:
        return _source_identity_failure(context, "S0_REFERENCE" if phase == "reference" else "S0_AA_REPLICATE")
    reference_path, metadata_path = _paths(spec)
    stage = "RUNTIME_IDENTITY"
    try:
        _seed_process(int(spec["seed"]), torch, np)
        runtime = capture_runtime(str(spec["gpu_uuid"]))
        stage = "S0_MODEL_LOAD"
        model, inputs = load_model(_model_spec(spec))
        within_process_aa = None
        if phase == "reference":
            stage = "S0_WITHIN_PROCESS_AA"
            output_cpu, within_process_aa = _s0_within_process_aa(
                model, inputs, rtol=float(spec["correctness"]["rtol"]),
                atol=float(spec["correctness"]["atol"]),
            )
        else:
            stage = "S0_FORWARD"
            with torch.inference_mode():
                output = model(*inputs)
                torch.cuda.synchronize()
            output_cpu = _cpu_tree(output)
        kernel_path = (
            _profile_cuda_forward(model, inputs, torch)
            if phase == "replicate" else None
        )
        weights_digest = nested_digest(model.state_dict())
        inputs_digest = nested_digest(inputs)
        output_digest = nested_digest(output_cpu)
        result: dict[str, Any] = {
            **context, "qualification_stage": phase_name, "arm": "S0",
            "arm_execution": "COMPLETE", "scientific_correctness": "UNASSESSED",
            "runtime_identity": runtime, "runtime_gpu_uuid": runtime.get("device_uuid"),
            "source_identity": context["source_identity"],
            "weights_sha256": weights_digest, "inputs_sha256": inputs_digest,
            "output_sha256": output_digest, "candidate_denominator": True,
            "performance_timing_recorded": False,
        }
        if within_process_aa is not None:
            result["s0_within_process_aa"] = within_process_aa
        if kernel_path is not None:
            result["cuda_kernel_path"] = kernel_path
        if phase == "reference":
            stage = "S0_REFERENCE_WRITE"
            if reference_path.exists() or metadata_path.exists():
                raise FileExistsError("S0 reference already exists; qualification must resume from its recorded stage")
            reference_file_digest = _write_reference(torch, reference_path, output_cpu)
            metadata = {
                **context, "source_identity": context["source_identity"],
                "weights_sha256": weights_digest, "inputs_sha256": inputs_digest,
                "output_sha256": output_digest,
                "reference_file_sha256": reference_file_digest,
                "runtime_identity": runtime,
                "s0_within_process_aa": within_process_aa,
            }
            atomic_json(metadata_path, metadata)
            if within_process_aa["passed"]:
                result.update({
                    "status": "S0_REFERENCE_COMPLETE", "eligibility": "NOT_YET_QUALIFIED",
                    "reference_file_sha256": reference_file_digest,
                })
            else:
                result.update({
                    "status": "BASELINE_INVALID", "eligibility": "BASELINE_INVALID",
                    "scientific_correctness": "FAIL",
                    "first_failure_stage": "S0_WITHIN_PROCESS_AA",
                    "root_cause": "S0_WITHIN_PROCESS_OUTPUT_MISMATCH",
                    "reference_file_sha256": reference_file_digest,
                })
            return result

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage = "S0_REFERENCE_READ"
        reference = _read_reference(torch, reference_path, metadata)
        stage = "S0_CROSS_PROCESS_COMPARISON"
        comparison = compare_outputs(
            reference, output_cpu, rtol=float(spec["correctness"]["rtol"]),
            atol=float(spec["correctness"]["atol"]),
        )
        identity_differences = cross_process_qualification_identity_differences(
            metadata, source_identity=context["source_identity"], runtime_identity=runtime,
            expected_gpu_uuid=str(spec["gpu_uuid"]),
            weights_sha256=weights_digest, inputs_sha256=inputs_digest,
        )
        runtime_differences = identity_differences["runtime"]
        selected_gpu_differences = identity_differences["selected_gpu"]
        source_match = identity_differences["source_identity_match"]
        weights_match = identity_differences["weights_match"]
        inputs_match = identity_differences["inputs_match"]
        identity_match = not runtime_differences and not selected_gpu_differences and source_match and weights_match and inputs_match
        passed = bool(comparison.passed and identity_match)
        aa = {
            "passed": passed, "comparison": comparison.__dict__,
            "runtime_identity_differences": runtime_differences,
            "selected_gpu_identity_differences": selected_gpu_differences,
            "source_identity_match": source_match,
            "weights_match": weights_match, "inputs_match": inputs_match,
            "replicate_runtime_identity": runtime,
            "cuda_kernel_path": kernel_path,
        }
        metadata["s0_cross_process_aa"] = aa
        atomic_json(metadata_path, metadata)
        if runtime_differences or selected_gpu_differences:
            eligibility, status = "RUNTIME_PREFLIGHT_FAILURE", "ERROR"
            correctness = "UNASSESSED"
            root_cause = "S0_CROSS_PROCESS_HARDWARE_OR_RUNTIME_IDENTITY_MISMATCH"
        elif not source_match:
            eligibility, status = "QUALIFICATION_IDENTITY_MISMATCH", "ERROR"
            correctness = "UNASSESSED"
            root_cause = "S0_CROSS_PROCESS_SOURCE_IDENTITY_MISMATCH"
        elif not (source_match and weights_match and inputs_match) or not comparison.passed:
            eligibility, status = "BASELINE_INVALID", "BASELINE_INVALID"
            correctness = "FAIL" if not comparison.passed else "UNASSESSED"
            root_cause = "S0_CROSS_PROCESS_SOURCE_MODEL_OR_OUTPUT_MISMATCH"
        else:
            eligibility, status, correctness, root_cause = "S0_AA_PASS", "VALID", "PASS", None
        result.update({
            "status": status, "eligibility": eligibility,
            "scientific_correctness": correctness, "identity_match": identity_match,
            "cross_process_correctness": comparison.__dict__,
            "identity_differences": {
                "runtime": runtime_differences,
                "selected_gpu": selected_gpu_differences,
                "source_identity_match": source_match,
                "weights_match": weights_match, "inputs_match": inputs_match,
            },
            "reference_file_sha256": metadata["reference_file_sha256"],
            "root_cause": root_cause,
        })
        return result
    except Exception as exc:
        result = _failure(spec, phase_name, stage, exc)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if phase == "replicate":
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["s0_cross_process_aa"] = {
                    "passed": False, "failure_stage": stage,
                    "reason": result["reason"],
                }
                atomic_json(metadata_path, metadata)
            except (OSError, json.JSONDecodeError):
                pass
        return result


def qualify_s1_process(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run channels-last correctness in its own process against a passed S0 A/A."""
    import numpy as np
    import torch

    from experiments.real_intervention_v3.numerical_checks import (
        compare_outputs, nested_digest,
    )
    from .model_adapter import load_model
    from .runtime_identity import capture_runtime

    context = _context(spec)
    if not context["source_identity_valid"]:
        return _source_identity_failure(context, "S1_CORRECTNESS")
    reference_path, metadata_path = _paths(spec)
    stage = "S0_REFERENCE_READ"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (metadata.get("s0_within_process_aa") or {}).get("passed"):
            return {
                **context, "qualification_stage": "S1_CORRECTNESS",
                "status": "NOT_RUN", "eligibility": "BASELINE_WITHIN_PROCESS_AA_NOT_PASSED",
                "arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "candidate_denominator": True, "performance_timing_recorded": False,
                "reason": "S1 correctness requires a passed within-process S0 A/A",
            }
        if not (metadata.get("s0_cross_process_aa") or {}).get("passed"):
            return {
                **context, "qualification_stage": "S1_CORRECTNESS",
                "status": "NOT_RUN", "eligibility": "BASELINE_AA_NOT_PASSED",
                "arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "candidate_denominator": True, "performance_timing_recorded": False,
                "reason": "S1 correctness requires a passed cross-process S0 A/A",
            }
        stage = "RUNTIME_IDENTITY"
        _seed_process(int(spec["seed"]), torch, np)
        runtime = capture_runtime(str(spec["gpu_uuid"]))
        stage = "S1_MODEL_LOAD"
        model, inputs = load_model(_model_spec(spec))
        weights_digest = nested_digest(model.state_dict())
        inputs_digest = nested_digest(inputs)
        identity_differences = cross_process_qualification_identity_differences(
            metadata, source_identity=context["source_identity"], runtime_identity=runtime,
            expected_gpu_uuid=str(spec["gpu_uuid"]),
            weights_sha256=weights_digest, inputs_sha256=inputs_digest,
        )
        runtime_differences = identity_differences["runtime"]
        if runtime_differences or identity_differences["selected_gpu"] or not all(
            identity_differences[name] for name in
            ("source_identity_match", "weights_match", "inputs_match")
        ):
            return {
                **context, "qualification_stage": "S1_CORRECTNESS",
                "status": "ERROR", "eligibility": "QUALIFICATION_IDENTITY_MISMATCH",
                "arm_execution": "FAILED", "scientific_correctness": "UNKNOWN",
                "first_failure_stage": "S1_IDENTITY_CHECK",
                "identity_differences": identity_differences,
                "runtime_identity": runtime, "weights_sha256": weights_digest,
                "inputs_sha256": inputs_digest, "candidate_denominator": True,
                "performance_timing_recorded": False,
            }
        stage = "S0_REFERENCE_READ"
        reference = _read_reference(torch, reference_path, metadata)
        conv2d_count = sum(isinstance(module, torch.nn.Conv2d) for module in model.modules())
        shapes = _shapes(inputs)
        has_4d_input = any(len(shape) == 4 for shape in shapes.values())
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        summary = {
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "input_shapes": shapes,
            "input_numel": sum(math.prod(shape) for shape in shapes.values()),
            "parameter_count": parameter_count,
            "conv2d_module_count": conv2d_count,
            "weights_sha256": weights_digest, "inputs_sha256": inputs_digest,
            "reference_path": str(reference_path),
        }
        if not conv2d_count or not has_4d_input:
            return {
                **context, "qualification_stage": "S1_CORRECTNESS",
                "status": "NOT_APPLICABLE", "eligibility": "NOT_CHANNELS_LAST_APPLICABLE",
                "arm_execution": "COMPLETE", "scientific_correctness": "UNASSESSED",
                "reason": "requires a Conv2d module and a 4D input tensor",
                "conv2d_count": conv2d_count, "actual_input_shapes": shapes,
                "model_summary": summary, "runtime_identity": runtime,
                "candidate_denominator": True, "performance_timing_recorded": False,
            }
        stage = "S1_MEMORY_FORMAT_CONVERSION"
        model.to(memory_format=torch.channels_last)
        candidate_inputs = _map_tensors(
            inputs, lambda tensor: tensor.contiguous(memory_format=torch.channels_last),
        )
        torch.manual_seed(int(spec["seed"]))
        stage = "S1_FORWARD"
        with torch.inference_mode():
            output = model(*candidate_inputs)
            torch.cuda.synchronize()
        stage = "S1_NUMERICAL_COMPARISON"
        comparison = compare_outputs(
            reference, _cpu_tree(output), rtol=float(spec["correctness"]["rtol"]),
            atol=float(spec["correctness"]["atol"]),
        )
        stage = "S1_MECHANISM_PROBE"
        exposure = _layout_probe(model, candidate_inputs, torch)
        s0_kernel_path = (metadata.get("s0_cross_process_aa") or {}).get("cuda_kernel_path")
        s1_kernel_path = exposure["cuda_kernel_path"]
        pairable = (
            exposure["active"]
            and has_cuda_kernel_events(s0_kernel_path)
            and has_cuda_kernel_events(s1_kernel_path)
        )
        exposure.update({
            "s0_cuda_kernel_path": s0_kernel_path,
            "s1_cuda_kernel_path": s1_kernel_path,
            "pairability_status": "PAIRABLE" if pairable else "NO_MECHANISTIC_EXPOSURE",
        })
        candidate = context["candidate"]
        if not comparison.passed:
            eligibility, reason = "CHANNELS_LAST_CORRECTNESS_FAILURE", comparison.reason
            status, correctness = "NUMERIC_MISMATCH", "FAIL"
        elif not pairable:
            eligibility, reason = (
                "NO_MECHANISTIC_EXPOSURE",
                "pairability requires active channels-last Conv2d layout and named CUDA kernel events in both S0 and S1 qualification captures",
            )
            status, correctness = "VALID", "PASS"
        elif candidate.get("source") == "torchbench" and candidate.get("eval_deterministic") is not True:
            eligibility, reason = "STOCHASTIC_OR_UNVERIFIED_BASELINE", "source metadata does not establish deterministic eval behavior"
            status, correctness = "VALID", "PASS"
        else:
            eligibility, reason = "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY", None
            status, correctness = "VALID", "PASS"
        return {
            **context, "qualification_stage": "S1_CORRECTNESS", "status": status,
            "eligibility": eligibility, "reason": reason,
            "pairability_status": exposure["pairability_status"],
            "arm_execution": "COMPLETE", "scientific_correctness": correctness,
            "first_failure_stage": (
                "S1_NUMERICAL_COMPARISON" if correctness == "FAIL"
                else "S1_MECHANISM_PROBE" if not pairable else None
            ),
            "channels_last_correctness": comparison.__dict__,
            "weights_sha256": weights_digest, "inputs_sha256": inputs_digest,
            "reference_sha256": nested_digest(reference),
            "reference_file_sha256": metadata["reference_file_sha256"],
            "conv2d_count": conv2d_count, "actual_input_shapes": shapes,
            "parameter_count": parameter_count, "model_summary": summary,
            "runtime_identity": runtime, "output_shapes": _shapes(output),
            "kernel_exposure": exposure,
            "input_kind": candidate.get("input_kind"),
            "candidate_denominator": True, "performance_timing_recorded": False,
        }
    except Exception as exc:
        result = _failure(spec, "S1_CORRECTNESS", stage, exc)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return result


def aggregate_qualification(candidate: Mapping[str, Any], batch: int, *,
                            s0_reference: Mapping[str, Any] | None,
                            s0_replicate: Mapping[str, Any] | None,
                            s1_correctness: Mapping[str, Any] | None) -> dict[str, Any]:
    """Produce one terminal denominator row without conflating S0 and S1 failures."""
    from .source_registry import family_lineage_id

    base = {
        "candidate_id": str(candidate.get("candidate_id")),
        "source": candidate.get("source"), "model": candidate.get("model"),
        "family_id": family_lineage_id(dict(candidate)),
        "family_lineage_id": family_lineage_id(dict(candidate)),
        "batch": int(batch), "candidate_denominator": True,
        "performance_timing_recorded": False,
    }
    if s0_reference is None or s0_reference.get("arm_execution") in {"UNKNOWN", "NOT_STARTED"}:
        reference_execution = (
            s0_reference.get("arm_execution") if s0_reference is not None else "UNKNOWN"
        )
        return {
            **base, "status": "INCOMPLETE", "eligibility": "QUALIFICATION_INCOMPLETE",
            "arm_execution": reference_execution, "scientific_correctness": "UNKNOWN",
            "qualification_complete": False,
            "failure_phase": "S0_REFERENCE",
            "reason": "S0 reference has no terminal receipt; downstream phases were not started",
        }
    if s0_reference.get("arm_execution") != "COMPLETE":
        selected = dict(s0_reference)
        skipped = {"S0_AA_REPLICATE": "NOT_RUN_S0_REFERENCE_FAILED",
                   "S1_CORRECTNESS": "NOT_RUN_S0_REFERENCE_FAILED"}
        complete = True
    elif not isinstance(s0_reference.get("s0_within_process_aa"), Mapping):
        selected = {
            "status": "INCOMPLETE", "eligibility": "QUALIFICATION_INCOMPLETE",
            "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
            "first_failure_stage": "S0_WITHIN_PROCESS_AA_MISSING",
            "reason": "S0 reference lacks a within-process A/A record",
        }
        skipped = {"S0_AA_REPLICATE": "NOT_RUN_WITHIN_PROCESS_AA_MISSING",
                   "S1_CORRECTNESS": "NOT_RUN_WITHIN_PROCESS_AA_MISSING"}
        complete = False
    elif not s0_reference["s0_within_process_aa"].get("passed"):
        selected = dict(s0_reference)
        skipped = {"S0_AA_REPLICATE": "NOT_RUN_WITHIN_PROCESS_AA_FAILED",
                   "S1_CORRECTNESS": "NOT_RUN_WITHIN_PROCESS_AA_FAILED"}
        complete = True
    elif s0_replicate is None or s0_replicate.get("arm_execution") in {"UNKNOWN", "NOT_STARTED"}:
        replicate_execution = (
            s0_replicate.get("arm_execution") if s0_replicate is not None else "UNKNOWN"
        )
        selected = {
            "status": "INCOMPLETE", "eligibility": "QUALIFICATION_INCOMPLETE",
            "arm_execution": replicate_execution, "scientific_correctness": "UNKNOWN",
            "reason": "S0 cross-process replicate has no terminal result",
        }
        skipped = {"S0_AA_REPLICATE": replicate_execution, "S1_CORRECTNESS": "NOT_RUN_S0_AA_UNKNOWN"}
        complete = False
    elif s0_replicate.get("eligibility") != "S0_AA_PASS":
        selected = dict(s0_replicate)
        skipped = {"S1_CORRECTNESS": "NOT_RUN_S0_AA_FAILED"}
        complete = True
    elif s1_correctness is None or s1_correctness.get("arm_execution") in {"UNKNOWN", "NOT_STARTED"}:
        correctness_execution = (
            s1_correctness.get("arm_execution") if s1_correctness is not None else "UNKNOWN"
        )
        selected = {
            "status": "INCOMPLETE", "eligibility": "QUALIFICATION_INCOMPLETE",
            "arm_execution": correctness_execution, "scientific_correctness": "UNKNOWN",
            "reason": "S1 correctness stage has no terminal result after a passed S0 A/A",
        }
        skipped = {"S1_CORRECTNESS": correctness_execution}
        complete = False
    else:
        selected = dict(s1_correctness)
        skipped = {}
        complete = True
        if selected.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY":
            exposure = selected.get("kernel_exposure") or {}
            if not has_pairable_kernel_exposure(exposure):
                selected.update({
                    "eligibility": "NO_MECHANISTIC_EXPOSURE",
                    "pairability_status": "NO_MECHANISTIC_EXPOSURE",
                    "reason": "PAIRABLE requires layout exposure and named CUDA kernel events for both S0 and S1",
                })
    result = {**base, **selected}
    result["candidate_denominator"] = True
    result["qualification_complete"] = complete
    result["phase_summary"] = {
        "S0_REFERENCE": s0_reference.get("status"),
        "S0_WITHIN_PROCESS_AA": (
            "PASS" if (s0_reference.get("s0_within_process_aa") or {}).get("passed") is True
            else "FAIL" if (s0_reference.get("s0_within_process_aa") or {}).get("passed") is False
            else "MISSING"
        ),
        "S0_AA_REPLICATE": s0_replicate.get("status") if s0_replicate else skipped.get("S0_AA_REPLICATE"),
        "S1_CORRECTNESS": s1_correctness.get("status") if s1_correctness else skipped.get("S1_CORRECTNESS"),
    }
    result["s0_within_process_aa"] = s0_reference.get("s0_within_process_aa")
    result["s0_cross_process_aa"] = (s0_replicate or {}).get("cross_process_correctness")
    if s0_reference.get("arm_execution") != "COMPLETE":
        result["failure_phase"] = "S0_REFERENCE"
    elif result["phase_summary"]["S0_WITHIN_PROCESS_AA"] != "PASS":
        result["failure_phase"] = "S0_WITHIN_PROCESS_AA"
    elif s0_replicate is None or s0_replicate.get("eligibility") != "S0_AA_PASS":
        result["failure_phase"] = "S0_AA_REPLICATE"
    elif result.get("eligibility") != "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY":
        result["failure_phase"] = "S1_CORRECTNESS"
    else:
        result["failure_phase"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=(
        "candidate", "s0-reference", "s0-replicate", "s1-correctness",
    ), default="candidate")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("qualification spec must be a JSON object")
    if args.phase == "candidate":
        result = {
            **_context(spec), "qualification_stage": "ORCHESTRATION_REQUIRED",
            "status": "NOT_RUN", "eligibility": "CROSS_PROCESS_QUALIFICATION_REQUIRED",
            "arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
            "candidate_denominator": True, "qualification_complete": False,
            "performance_timing_recorded": False,
            "reason": "run_all must supervise S0 reference, S0 replicate, and S1 in separate processes",
        }
    elif args.phase == "s0-reference":
        result = qualify_s0_process(spec, phase="reference")
    elif args.phase == "s0-replicate":
        result = qualify_s0_process(spec, phase="replicate")
    else:
        result = qualify_s1_process(spec)
    result["finished_monotonic"] = time.monotonic()
    from .outcomes import atomic_json
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
