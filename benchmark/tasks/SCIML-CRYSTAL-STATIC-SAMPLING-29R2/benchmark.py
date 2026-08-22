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

REQUIRED_API=('build_sampler','sample_step','sample')
def make_fixtures(seed:int,device:str='cpu'):
    neighbor_count=8; sample_count=32; geometry_variation=0.2
    config={'neighbor_count':neighbor_count,'sample_count':sample_count,'geometry_variation':geometry_variation}
    # The declared axes construct the lattice, cutoff, and per-step motion.
    pts=[]
    for i in range(4):
        for j in range(8): pts.append([float(i),float(j),0.0])
    initial=torch.tensor(pts,dtype=torch.float32)
    initial=initial+geometry_variation*0.01*torch.sin(initial)
    cutoff=1.45+0.03*(neighbor_count-8)
    config['cutoff']=cutoff
    return {'device':'cpu','initial':initial,'cutoff':cutoff,'num_steps':12,'config':config,'neighbor_count':neighbor_count,'sample_count':sample_count,'geometry_variation':geometry_variation}
def run_correctness(solution,fixtures): return _CHECKS.check_sample(solution,fixtures)
def run_scientific_gates(solution,fixtures): return {'neighbor_consistency':_SCIENCE.neighbor_consistency(solution,fixtures)}
def _sample_with_count(module,fixtures):
    sampler=module.build_sampler(fixtures); count={'n':0}; original=module.sample_step
    def wrapped(*args,**kwargs): count['n']+=1; return original(*args,**kwargs)
    module.sample_step=wrapped
    try: out=module.sample(sampler,fixtures,fixtures['num_steps'])
    finally: module.sample_step=original
    return out,count['n'],sampler
def _trace(module,fixtures): return _sample_with_count(module,fixtures)[2].rebuild_count
def run_activation_evidence(solution,baseline_solution,fixtures): return {'candidate_metrics':{'graph_rebuild_count':_trace(solution,fixtures)},'baseline_metrics':{'graph_rebuild_count':_trace(baseline_solution,fixtures)}}
def run_performance(solution,fixtures,warmup=0,iterations=1,device='cpu'):
    times=[]; out=None; calls=[]
    for _ in range(5):
        st=time.perf_counter(); out,step_calls,s=_sample_with_count(solution,fixtures); times.append((time.perf_counter()-st)*1000); calls.append(s.rebuild_count)
    return {'value':statistics.median(times),'work_units':{'sample_steps':step_calls},'output_checksums':{'sample':hashlib.sha256(out.numpy().tobytes()).hexdigest()},'timing':{'metric':'step_ms_p50','run_times_ms':times,'graph_rebuild_count':calls}}
