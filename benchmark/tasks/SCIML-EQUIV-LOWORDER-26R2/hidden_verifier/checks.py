from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict()
    return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)

def check_output(solution,fixtures):
    m=solution.build_model(fixtures); p=fixtures['eval_positions'].to(next(m.parameters()).device); y=m(p); return {'passed':bool(_arch_ok(m, fixtures['init_state']) and torch.isfinite(y).all() and y.shape==p.shape), 'output_checksum':float(y.detach().mean().cpu())}
