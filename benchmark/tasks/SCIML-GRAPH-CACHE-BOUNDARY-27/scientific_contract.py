from __future__ import annotations
import torch

def energy_force_consistency(solution,fixtures):
    m=solution.build_model(fixtures); dev=next(m.parameters()).device; p=fixtures['positions'].to(dev).detach().clone().requires_grad_(True); graph={'edge_index':fixtures['edge_index'],'cutoff':fixtures['cutoff']}; e=solution.energy_fn(m,p,**graph); ref=-torch.autograd.grad(e,p)[0]; got=solution.forces_fn(m,p.detach(),**graph); err=float((ref-got).abs().max().cpu()); return (err<2e-5, {'max_force_error':err})
