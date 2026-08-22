from __future__ import annotations
import torch

def _reference(model, positions, cutoff):
    p = positions.detach().clone().requires_grad_(True)
    d = torch.cdist(p, p)
    edge = ((d < cutoff) & (d > 0)).nonzero(as_tuple=False).T
    src, dst = edge
    r2 = (p[src] - p[dst]).square().sum(-1)
    energy = 0.5 * model.log_scale.exp() * torch.exp(-r2).sum()
    force = -torch.autograd.grad(energy, p)[0]
    return energy.detach(), force.detach()

def check_energy_force(solution, fixtures):
    model = solution.build_model(fixtures)
    dev = next(model.parameters()).device
    p = fixtures['positions'].to(dev)
    graph = {'edge_index': fixtures['edge_index'], 'cutoff': fixtures['cutoff']}
    got_e = solution.energy_fn(model, p, **graph).detach()
    got_f = solution.forces_fn(model, p, **graph).detach()
    ref_e, ref_f = _reference(model, p, fixtures['cutoff'])
    eerr = float((got_e-ref_e).abs().max().cpu())
    ferr = float((got_f-ref_f).abs().max().cpu())
    return {'passed': bool(torch.isfinite(got_e) and torch.isfinite(got_f).all() and eerr < 2e-5 and ferr < 2e-5), 'output_checksum': float(got_e.cpu()), 'energy_error': eerr, 'force_error': ferr}
