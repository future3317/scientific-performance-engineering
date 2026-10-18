from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import load_model
from .contracts import digest, validate_manifest


# Fixed before outcomes. Each entry is one architecture/model lineage; variants
# of the same named family are not counted as separate lineages in this cohort.
MODEL_CANDIDATES = (
    "alexnet", "densenet121", "mobilenet_v2", "mnasnet1_0", "resnet18",
    "shufflenet_v2_x1_0", "squeezenet1_1", "vgg16", "timm_efficientnet",
    "timm_nfnet", "timm_regnet", "timm_resnest", "timm_vision_transformer",
    "basic_gnn_edgecnn", "basic_gnn_gcn", "basic_gnn_gin", "basic_gnn_sage",
    "BERT_pytorch", "hf_Albert", "hf_Bart", "hf_Bert", "hf_GPT2",
    "pytorch_unet", "pyhpc_turbulent_kinetic_energy",
)


def _lineage(model: str) -> str:
    return "bert" if model in {"BERT_pytorch", "hf_Bert"} else model


def _ordered_models(eligible: list[str], lineages: list[str]) -> list[str]:
    lineage_order = {lineage: index for index, lineage in enumerate(lineages)}
    return sorted(eligible, key=lambda model: (lineage_order[_lineage(model)], model))


def _source_digest(root: Path, model: str) -> str:
    folder = root / "torchbenchmark" / "models" / model
    hasher = hashlib.sha256()
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        hasher.update(path.relative_to(folder).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _torchvision_digest(model: str, network: Any) -> str:
    import inspect
    import torchvision
    source = Path(inspect.getsourcefile(network.__class__) or models.__file__)
    return digest({"model": model, "torchvision_version": str(torchvision.__version__),
                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})


def freeze_study(*, torchbench_root: str | Path, output_root: str | Path,
                 project_root: str | Path, device: str, gpu_uuid: str,
                 seed: int = 20261018) -> dict[str, Any]:
    tb_root, project = Path(torchbench_root).resolve(), Path(project_root).resolve()
    run_root = Path(output_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    cache_root = run_root / "model-cache"
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "hub")
    sys.path.insert(0, str(tb_root))
    import torch
    source_commit = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                   check=True, capture_output=True, text=True).stdout.strip()
    benchmark_commit = subprocess.run(["git", "-C", str(tb_root), "rev-parse", "HEAD"],
                                       check=True, capture_output=True, text=True).stdout.strip()
    intervention_code_digest = digest({
        name: hashlib.sha256((project / "experiments" / "real_intervention_v2" / name).read_bytes()).hexdigest()
        for name in ("worker.py", "adapters.py")
    })
    benchmark_source_bytes = sum(path.stat().st_size for path in tb_root.rglob("*")
                                 if path.is_file() and ".git" not in path.parts)
    if benchmark_source_bytes > 10737418240:
        raise RuntimeError("pinned TorchBench source exceeds the declared external-download ceiling")
    if benchmark_commit != "61227cc119913f01e78e8e115a68208c4ca0541b":
        raise ValueError(f"unexpected TorchBench commit: {benchmark_commit}")
    api_matrix = []
    materialized: dict[tuple[str, int], dict[str, Any]] = {}
    download_cap = 10737418240

    def cache_bytes() -> int:
        return sum(path.stat().st_size for path in cache_root.rglob("*") if path.is_file()) if cache_root.exists() else 0

    for model in MODEL_CANDIDATES:
        model_rows = []
        for batch_size in (1, 2):
            row = {"model": model, "batch_size": batch_size, "adapter": "torchbench"}
            try:
                network, inputs = load_model(row, "cpu")
                shapes, dtypes = [], []
                def visit(value):
                    if hasattr(value, "shape") and hasattr(value, "dtype"):
                        shapes.append(list(value.shape))
                        dtypes.append(str(value.dtype).replace("torch.", ""))
                    elif isinstance(value, dict):
                        for key in sorted(value): visit(value[key])
                    elif isinstance(value, (list, tuple)):
                        for item in value: visit(item)
                visit(inputs)
                downloaded_bytes = cache_bytes()
                if benchmark_source_bytes + downloaded_bytes > download_cap:
                    raise RuntimeError(f"isolated model cache exceeded {download_cap} bytes")
                record = {"status": "api_eligible", "input_shapes": shapes,
                          "input_dtypes": dtypes,
                          "model_parameter_count": sum(p.numel() for p in network.parameters()),
                          "model_source_digest": _source_digest(tb_root, model),
                          "isolated_cache_bytes_after_model": downloaded_bytes}
                materialized[(model, batch_size)] = record
                model_rows.append(record)
            except Exception as exc:
                if benchmark_source_bytes + cache_bytes() > download_cap:
                    raise RuntimeError(f"isolated model cache exceeded {download_cap} bytes") from exc
                if isinstance(exc, RuntimeError) and "isolated model cache exceeded" in str(exc):
                    raise
                record = {"status": "api_unavailable", "error_type": type(exc).__name__,
                          "error": str(exc)[:500]}
                model_rows.append(record)
            api_matrix.append({"model": model, "batch_size": batch_size, **record})
        if not all(row["status"] == "api_eligible" for row in model_rows):
            materialized.pop((model, 1), None)
            materialized.pop((model, 2), None)

    eligible = [model for model in MODEL_CANDIDATES
                if (model, 1) in materialized and (model, 2) in materialized]
    fallback = len(eligible) < 2
    if fallback:
        materialized.clear()
        api_matrix.append({"status": "torchbench_api_blocked", "eligible_candidates": eligible,
                           "fallback_reason": "fewer_than_two_candidate_models_materialized"})
        fallback_models = ("resnet18", "mobilenet_v3_small", "squeezenet1_0")
        for model in fallback_models:
            for batch_size in (1, 2):
                row = {"model": model, "batch_size": batch_size, "adapter": "torchvision",
                       "input_shape": [batch_size, 3, 224, 224], "seed": seed}
                try:
                    network, inputs = load_model(row, "cpu")
                    record = {"status": "api_eligible", "input_shapes": [list(inputs[0].shape)],
                              "input_dtypes": [str(inputs[0].dtype).replace("torch.", "")],
                              "model_parameter_count": sum(p.numel() for p in network.parameters()),
                              "model_source_digest": _torchvision_digest(model, network),
                              "isolated_cache_bytes_after_model": cache_bytes(),
                              "input_source_type": "materialized_random_tensor"}
                    materialized[(model, batch_size)] = record
                    api_matrix.append({"model": model, "batch_size": batch_size,
                                       "adapter": "torchvision", **record})
                except Exception as exc:
                    api_matrix.append({"model": model, "batch_size": batch_size,
                                       "adapter": "torchvision", "status": "api_unavailable",
                                       "error_type": type(exc).__name__, "error": str(exc)[:500]})
        eligible = [model for model in fallback_models
                    if (model, 1) in materialized and (model, 2) in materialized]
    lineages = sorted({_lineage(model) for model in eligible},
                      key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    n_dev = int(len(lineages) * 0.5)
    n_cal = int(len(lineages) * 0.25)
    splits = {lineage: ("development" if index < n_dev else
                        "gate_calibration" if index < n_dev + n_cal else "final_untouched")
              for index, lineage in enumerate(lineages)}
    contexts = []
    ordered = _ordered_models(eligible, lineages)
    for model in ordered:
        source_digest = materialized[(model, 1)]["model_source_digest"]
        for batch_size in (1, 2):
            metadata = materialized[(model, batch_size)]
            for horizon in (1, 16, 128):
                seed_row = {"model": model, "batch_size": batch_size, "horizon": horizon, "seed": seed}
                context = {
                    "lineage_id": _lineage(model),
                    "split": splits[_lineage(model)],
                    "benchmark_source": "TORCHVISION_REAL_MODEL_MATERIALIZED_INPUT" if fallback else "TORCHBENCH",
                    "benchmark_commit": benchmark_commit,
                    "adapter": "torchvision" if fallback else "torchbench",
                    "model": model,
                    "model_id": model,
                    "task_mode": "eval",
                    "batch_size": batch_size,
                    "planned_horizon": horizon,
                    "input_shapes": metadata["input_shapes"],
                    "dtype": metadata["input_dtypes"][0] if metadata["input_dtypes"] else "unknown",
                    "shape_regime": "static",
                    "model_parameter_count": metadata["model_parameter_count"],
                    "device_type": device,
                    "gpu_uuid": gpu_uuid if device == "cuda" else None,
                    "torch_version": str(torch.__version__),
                    "compiler_backend": "inductor",
                    "model_source_digest": source_digest,
                    "input_manifest_digest": digest(seed_row),
                    "declared_cold_cache": True,
                    "seed": seed,
                    "sessions": [
                        {"session_id": "session-1", "order": "eager_first"},
                        {"session_id": "session-2", "order": "compile_first"},
                    ],
                }
                if fallback:
                    context["input_shape"] = metadata["input_shapes"][0]
                    context["input_source_type"] = "materialized_random_tensor"
                context["context_id"] = digest({key: value for key, value in context.items() if key != "context_id"})
                contexts.append(context)

    manifest = {
        "schema_version": 1,
        "run_id": "evospe-rt-v2-real-compile-support-20261018-r1",
        "source_commit": source_commit,
        "torchbench_commit": benchmark_commit,
        "benchmark_source": "TORCHVISION_REAL_MODEL_MATERIALIZED_INPUT" if fallback else "TORCHBENCH",
        "fallback_reason": "torchbench_api_blocked_fewer_than_two_models" if fallback else None,
        "benchmark_source_bytes": benchmark_source_bytes,
        "intervention_id": "EVOSPE_RT_COMPILE_DEFAULT_V1",
        "intervention_code_digest": intervention_code_digest,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "intervention": {"baseline": "eager", "candidate": "torch.compile(model, backend='inductor', mode='default', dynamic=None)"},
        "device": device,
        "seed": seed,
        "provider_calls": 0,
        "sampling_unit": "model_lineage",
        "planned_context_count": len(contexts),
        "planned_pair_sessions": sum(len(row["sessions"]) for row in contexts),
        "aa_control_context_ids": [row["context_id"] for row in contexts
                                    if row["split"] == "development" and row["batch_size"] == 1
                                    and row["planned_horizon"] == 1],
        "resource_caps": {"cpu_seconds": 172800, "gpu_seconds": 172800,
                          "wall_seconds": 172800, "storage_bytes": 53687091200,
                          "download_bytes": 10737418240},
        "rtol": 1e-4,
        "atol": 1e-5,
        "minimum_horizon": 10**18,
        "break_even_horizon": 10**18,
        "acre_rules": [],
        "acre_states": [],
        "acre_regimes": {},
        "contexts": contexts,
    }
    identity_payload = {
        "run_id": manifest["run_id"], "source_commit": source_commit,
        "torchbench_commit": benchmark_commit, "device": device,
        "gpu_uuid": gpu_uuid, "intervention_id": manifest["intervention_id"],
        "intervention_code_digest": intervention_code_digest,
        "contexts_digest": digest(contexts), "aa_control_context_ids": manifest["aa_control_context_ids"],
        "resource_caps": manifest["resource_caps"],
    }
    manifest["run_identity"] = digest(identity_payload)
    validate_manifest(manifest)
    (run_root / "MODEL_API_MATRIX.json").write_text(json.dumps(api_matrix, indent=2, sort_keys=True) + "\n")
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (run_root / "APPROVAL_BINDING.json").write_text(json.dumps({
        "approved": True,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_source": "user explicitly authorized direct execution in chat",
        "run_identity": manifest["run_identity"],
        "manifest_digest": digest(manifest),
        "permission_scope": ["pinned TorchBench source", "model admission and S0/S1 correctness",
                             "paired development/calibration/final", "offline policy replay",
                             "optional ZINC only after main study closes"],
        "resource_caps": manifest["resource_caps"],
        "provider_calls": 0,
    }, indent=2, sort_keys=True) + "\n")
    return {"api_matrix_rows": len(api_matrix), "eligible_model_lineages": len(lineages),
            "contexts": len(contexts), "paired_sessions": manifest["planned_pair_sessions"],
            "splits": {name: sum(1 for lineage in lineages if splits[lineage] == name)
                       for name in ("development", "gate_calibration", "final_untouched")},
            "run_root": str(run_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchbench-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-uuid", default="")
    args = parser.parse_args()
    print(json.dumps(freeze_study(torchbench_root=args.torchbench_root,
                                  output_root=args.output_root,
                                  project_root=args.project_root,
                                  device=args.device, gpu_uuid=args.gpu_uuid), indent=2))


if __name__ == "__main__":
    main()
