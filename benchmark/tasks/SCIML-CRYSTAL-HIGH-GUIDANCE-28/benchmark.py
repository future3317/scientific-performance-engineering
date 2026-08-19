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
    g=torch.Generator().manual_seed(seed); n=32; target=torch.stack([torch.arange(n,dtype=torch.float32)*0.8,torch.zeros(n),torch.zeros(n)],1); initial=target+0.6*torch.randn(n,3,generator=g); noise=torch.randn(100,n,3,generator=g); return {'device':'cpu','target':target,'initial':initial,'noise':noise,'guidance_scale':4.0,'num_steps':100}
def run_correctness(solution,fixtures): return _CHECKS.check_sample(solution,fixtures)
def run_scientific_gates(solution,fixtures): return {'structure_validity':_SCIENCE.structure_validity(solution,fixtures)}
def _sample_with_count(module,fixtures):
    sampler=module.build_sampler(fixtures); count={'n':0}; original=module.sample_step
    def wrapped(*args,**kwargs): count['n']+=1; return original(*args,**kwargs)
    module.sample_step=wrapped
    try: out=module.sample(sampler,fixtures,fixtures['num_steps'])
    finally: module.sample_step=original
    return out,count['n'],sampler
def _calls(module,fixtures): return _sample_with_count(module,fixtures)[1]
def run_activation_evidence(solution,baseline_solution,fixtures): return {'candidate_metrics':{'crystal_sampler_calls':_calls(solution,fixtures)},'baseline_metrics':{'crystal_sampler_calls':_calls(baseline_solution,fixtures)}}
def run_performance(solution,fixtures,warmup=0,iterations=1,device='cpu'):
    st=time.perf_counter(); out,calls,s=_sample_with_count(solution,fixtures); elapsed=time.perf_counter()-st; reached=bool((out-fixtures['target']).square().mean().sqrt()<0.005); return {'value':elapsed,'work_units':{'sampler_steps':calls},'output_checksums':{'sample':hashlib.sha256(out.numpy().tobytes()).hexdigest()},'timing':{'metric':'time_to_quality_s','reached':reached,'wall_time_s':elapsed,'sampler_calls':calls}}
