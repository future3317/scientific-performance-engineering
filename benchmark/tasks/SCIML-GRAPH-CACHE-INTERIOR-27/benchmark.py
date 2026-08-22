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

REQUIRED_API=('build_model','energy_fn','forces_fn')
def make_fixtures(seed:int,device:str='cpu'):
    g=torch.Generator().manual_seed(seed); n=128; displacement=0.03; base=torch.stack([torch.arange(n,dtype=torch.float32),torch.zeros(n),torch.zeros(n)],dim=1); disp=torch.zeros_like(base); disp[:,1]=displacement*torch.sin(torch.arange(n,dtype=torch.float32)); pos=base+disp; cutoff=1.6; edge=((torch.cdist(base,base)<cutoff)&(torch.cdist(base,base)>0)).nonzero(as_tuple=False).T
    skin=0.4; dynamic_rate=0.2
    probe=_load(_TASK_DIR/'workspace'/'solution.py','probe'); m=probe.Model(); return {'device':device,'positions':pos,'base_positions':base,'edge_index':edge,'cutoff':cutoff,'graph_cache_config':{'geometry_displacement':displacement,'skin':skin,'graph_size':n,'dynamic_rate':dynamic_rate},'init_state':m.state_dict()}
def run_correctness(solution,fixtures): return _CHECKS.check_energy_force(solution,fixtures)
def run_scientific_gates(solution,fixtures): return {'energy_force_consistency':_SCIENCE.energy_force_consistency(solution,fixtures)}
def _trace(module,fixtures):
    from unittest.mock import patch
    m=module.build_model(fixtures); p=fixtures['positions'].to(next(m.parameters()).device)
    with patch.object(torch,'cdist',wraps=torch.cdist) as fn: module.energy_fn(m,p,edge_index=fixtures['edge_index'],cutoff=fixtures['cutoff'])
    return fn.call_count
def run_activation_evidence(solution,baseline_solution,fixtures):
    c=_trace(solution,fixtures); b=_trace(baseline_solution,fixtures); return {'candidate_metrics':{'cache_hit_without_rebuild':c==0},'baseline_metrics':{'cache_hit_without_rebuild':b==0}}
def run_performance(solution,fixtures,warmup=1,iterations=10,device='cpu'):
    m=solution.build_model(fixtures); p=fixtures['positions'].to(next(m.parameters()).device); graph={'edge_index':fixtures['edge_index'],'cutoff':fixtures['cutoff']}
    for _ in range(warmup): solution.forces_fn(m,p,**graph)
    times=[]; last=None
    for _ in range(iterations):
        st=time.perf_counter(); last=solution.forces_fn(m,p,**graph); times.append((time.perf_counter()-st)*1000)
    return {'value':statistics.median(times),'work_units':{'force_evaluations':iterations},'output_checksums':{'force':hashlib.sha256(last.detach().cpu().numpy().tobytes()).hexdigest()},'timing':{'metric':'step_ms_p50','step_times_ms':times}}
