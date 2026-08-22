from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict()
    return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)

def check_vjp(solution, fixtures):
    model = solution.build_model(fixtures)
    x = fixtures['batch'][0].to(next(model.parameters()).device)
    got = solution.jacobian_features(model, x)
    xr = x.detach().requires_grad_(True)
    y = model(xr).mean(dim=0)
    refs = []
    for i in range(y.numel()):
        refs.append(torch.autograd.grad(y[i], xr, retain_graph=True, create_graph=True)[0])
    ref = torch.stack(refs)
    err = float((got-ref).abs().max().detach().cpu())
    return {'passed': bool(_arch_ok(model, fixtures['init_state']) and torch.isfinite(got).all() and got.shape == ref.shape and err < 2e-6), 'output_checksum': float(got.detach().abs().mean().cpu()), 'max_vjp_error': err}


