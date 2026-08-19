from __future__ import annotations
import hashlib
import importlib.util
import statistics
import json
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

REQUIRED_API=('run_episode_task',)
def make_fixtures(seed:int,device:str='cpu'): return {'device':device,'condition':'C','budget':{'max_wall_time_s':120},'seed':seed}
def _run(solution,fixtures):
    skill_view={'condition':fixtures['condition']}; budget={**fixtures['budget'],'seed':int(fixtures.get('seed',0))}; return solution.run_episode_task(str(_TASK_DIR/'workspace'),skill_view,budget)
def run_correctness(solution,fixtures):
    try: r=_run(solution,fixtures); return {'passed':isinstance(r.get('episode_score'),(int,float)),'details':{'keys':sorted(r)}}
    except Exception as exc: return {'passed':False,'details':{'error':repr(exc)}}
def run_scientific_gates(solution,fixtures):
    r=_run(solution,fixtures); return {'state_transition_valid':_SCIENCE.state_transition_valid(r),'specialization_applied':_SCIENCE.specialization_applied(r.get('episode_metrics',{}))}
def run_activation_evidence(solution,baseline_solution,fixtures):
    c=_run(solution,fixtures); b=_run(baseline_solution,fixtures); return {'candidate_metrics':{'transition_applied':c.get('condition_used')=='D'},'baseline_metrics':{'transition_applied':b.get('condition_used')=='D'}}
def run_performance(solution,fixtures,warmup=0,iterations=1,device='cpu'):
    st=time.perf_counter(); r=_run(solution,fixtures); wall=time.perf_counter()-st; gates={'state_transition_valid':_SCIENCE.state_transition_valid(r),'specialization_applied':_SCIENCE.specialization_applied(r.get('episode_metrics',{}))}; return {'value':float(r.get('episode_score',0.0)),'work_units':{'episode_runs':1},'output_checksums':{'result':str(sorted(r))},'timing':{'wall_time_s':wall},'episode_result':r,'episode_gates':gates,'episode_gate_details':{}}
