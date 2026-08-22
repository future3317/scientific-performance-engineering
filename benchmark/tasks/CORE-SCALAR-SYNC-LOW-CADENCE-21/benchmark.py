from __future__ import annotations
import hashlib
import importlib.util
import statistics
import time
from pathlib import Path
from typing import Any
import torch

_TASK_DIR = Path(__file__).resolve().parent

def _load(path: str | Path, prefix: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_CHECKS = _load(_TASK_DIR / "hidden_verifier" / "checks.py", "candidate_checks")
_SCIENCE = _load(_TASK_DIR / "scientific_contract.py", "candidate_science")

def load_solution(path: str | Path, device: str | None = None) -> Any:
    path = Path(path)
    if path.is_dir():
        path = path / "solution.py"
    module = _load(path, "candidate_solution")
    required = REQUIRED_API
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError("API violations: missing " + ", ".join(missing))
    return module

def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

REQUIRED_API = ('build_model','train_step','run_training')


def make_fixtures(seed:int, device:str="cpu"):
    g=torch.Generator().manual_seed(seed); in_dim=16
    x=torch.randn(64,in_dim,generator=g); y=torch.randn(64,generator=g)
    probe=_load(_TASK_DIR/'workspace'/'solution.py','probe'); m=probe.TinyRegressor(in_dim)
    for p in m.parameters(): p.data.normal_(0,0.03,generator=g)
    return {'device':'cpu','requested_device':device,'in_dim':in_dim,'batch':(x,y),'logical_batch_size':64,'lr':0.01,'scalar_syncs_per_step':4,'metric_cadence':8,'init_state':m.state_dict()}


def run_correctness(solution, fixtures): return _CHECKS.check_training(solution, fixtures)


def run_scientific_gates(solution, fixtures): return {'metric_semantics_preserved': _SCIENCE.metric_semantics_preserved(solution, fixtures)}


def _item_call_count(module):
    import ast
    tree=ast.parse(Path(module.__file__).read_text())
    return sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='item' for n in ast.walk(tree))
def run_activation_evidence(solution, baseline_solution, fixtures):
    return {'candidate_metrics':{'aggregated_scalar_syncs':_item_call_count(solution)},'baseline_metrics':{'aggregated_scalar_syncs':_item_call_count(baseline_solution)}}


def run_performance(solution: Any, fixtures: dict[str, Any], warmup: int = 3, iterations: int = 20, device: str = "cpu") -> dict[str, Any]:
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(fixtures.get("lr", 0.01)))
    batch = fixtures["batch"]
    for _ in range(warmup):
        solution.train_step(model, batch, optimizer)
    _sync(str(fixtures.get("device", device)))
    times = []
    last = None
    for _ in range(iterations):
        start = time.perf_counter()
        last = solution.train_step(model, batch, optimizer)
        _sync(str(fixtures.get("device", device)))
        times.append((time.perf_counter() - start) * 1000.0)
    value = statistics.median(times)
    loss = last.get("loss") if isinstance(last, dict) else None
    checksum = None
    if isinstance(loss, torch.Tensor):
        checksum = hashlib.sha256(loss.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "value": value,
        "work_units": {"samples": int(fixtures.get("logical_batch_size", 1)) * iterations, "optimizer": iterations},
        "output_checksums": {"loss": checksum},
        "timing": {"metric": "step_ms_p50", "step_times_ms": times, "median_ms": value},
    }
