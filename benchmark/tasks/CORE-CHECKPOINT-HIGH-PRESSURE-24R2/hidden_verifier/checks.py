from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict(); return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)

def check_gradient_step(solution,fixtures):
    model=solution.build_model(fixtures)
    if not _arch_ok(model, fixtures['init_state']): return {'passed':False,'output_checksum':None,'reason':'architecture mismatch'}
    r=solution.run_training(fixtures,1); return {'passed':bool(torch.isfinite(r['final_loss'])), 'output_checksum':float(r['final_loss'])}
