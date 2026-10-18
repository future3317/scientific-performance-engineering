from __future__ import annotations

import argparse
import hashlib
import json
import random
from statistics import median
import time
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    import torch
    hasher = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                hasher.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            hasher.update(str(type(item).__name__).encode())
            for child in item: visit(child)
        else:
            hasher.update(repr(item).encode())

    visit(value)
    return hasher.hexdigest()


def _model_digest(model: Any) -> str:
    return _digest({key: value for key, value in model.state_dict().items()})


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .adapters import load_model
    worker_started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = config["device"]
    setup_started = time.perf_counter()
    try:
        model, inputs = load_model(config["context"], device)
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "phase": "model_setup", "error_type": type(exc).__name__,
                "setup_s": time.perf_counter() - setup_started}
    except Exception as exc:
        return {"status": "execution_error", "phase": "model_setup", "error_type": type(exc).__name__,
                "error": str(exc), "setup_s": time.perf_counter() - setup_started}
    setup_s = time.perf_counter() - setup_started
    input_digest = _digest(inputs)
    model_digest = _model_digest(model)
    if config["arm"] == "compile":
        compile_wrapper_started = time.perf_counter()
        model = torch.compile(model, backend="inductor", mode="default", dynamic=None)
        compile_wrapper_s = time.perf_counter() - compile_wrapper_started
    else:
        compile_wrapper_s = 0.0
    start = time.perf_counter()
    output = None
    try:
        with torch.inference_mode():
            for _ in range(int(config["horizon"])):
                output = model(*inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "error_type": type(exc).__name__, "setup_s": setup_s,
                "execution_s": time.perf_counter() - start, "input_digest": input_digest}
    except Exception as exc:
        return {"status": "compile_error" if config["arm"] == "compile" else "execution_error",
                "error_type": type(exc).__name__, "error": str(exc), "setup_s": setup_s,
                "execution_s": time.perf_counter() - start, "input_digest": input_digest}
    execution_s = time.perf_counter() - start
    process_startup_s = max(0.0, worker_started - float(config["parent_launch_monotonic"]))
    output_digest = _digest(output)
    # Temporary server-local output is used only for cross-process correctness
    # comparison and is removed once the pair receipt is complete.
    torch.save(output, config["scratch_output"])
    warm = []
    for _ in range(int(config["warmups"])):
        with torch.inference_mode(): model(*inputs)
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    for _ in range(int(config["repetitions"])):
        if device.startswith("cuda"): torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(): model(*inputs)
        if device.startswith("cuda"): torch.cuda.synchronize(device)
        warm.append(time.perf_counter() - started)
    return {"status": "valid", "process_startup_s": process_startup_s,
            "setup_s": setup_s, "compile_wrapper_s": compile_wrapper_s,
            "cold_horizon_execution_s": execution_s,
            "total_job_s": process_startup_s + setup_s + compile_wrapper_s + execution_s,
            "horizon": int(config["horizon"]),
            "input_digest": input_digest, "output_digest": output_digest,
            "model_digest": model_digest,
            "warm_s": warm, "warm_median_s": median(warm)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("result")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(config)
    Path(args.result).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
