from __future__ import annotations
import hashlib
import importlib.util
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
def make_fixtures(seed:int,device:str='cpu'): return {'device':device,'public_context':{'workload':{'runtime_version':'A','context_width':4,'drift_rate':0.3,'fixture_index':2}},'budget':{'max_wall_time_s':120},'seed':seed}
def _run(solution,fixtures):
    skill_view={'public_context':fixtures['public_context']}; budget={'max_wall_time_s':float(fixtures['budget']['max_wall_time_s'])}; return solution.run_episode_task(str(_TASK_DIR/'workspace'),skill_view,budget)
def run_correctness(solution,fixtures):
    try: r=_run(solution,fixtures); return {'passed':isinstance(r.get('action'),dict),'details':{'keys':sorted(r)}}
    except Exception as exc: return {'passed':False,'details':{'error':repr(exc)}}
def run_scientific_gates(solution,fixtures):
    r=_run(solution,fixtures); return {'declarative_action_valid': isinstance(r.get('action'),dict)}
def run_activation_evidence(solution,baseline_solution,fixtures):
    c=_run(solution,fixtures); b=_run(baseline_solution,fixtures); return {'candidate_action':c.get('action',{}),'baseline_action':b.get('action',{})}
def run_performance(solution,fixtures,warmup=0,iterations=1,device='cpu'):
    r=_run(solution,fixtures)
    if not isinstance(r.get('action'),dict):
        raise TypeError('episode solution must return an action mapping')
    return {'value':0.0,'action':dict(r['action'])}

def score_harness_episode(result):
    metrics=result.get('metrics',{})
    precision=metrics.get('rule_precision')
    negative=metrics.get('negative_transfer_rate')
    return (float(precision) if isinstance(precision,(int,float)) else 0.0) * (1.0-(float(negative) if isinstance(negative,(int,float)) else 0.0))

def gates_harness_episode(result):
    scored={'episode_score':score_harness_episode(result),'episode_metrics':result.get('metrics',{}),'condition_used':result.get('condition')}
    return {'state_transition_valid':_SCIENCE.state_transition_valid(scored),'specialization_applied':_SCIENCE.specialization_applied(scored.get('episode_metrics',{}))}
