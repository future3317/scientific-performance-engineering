from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict()
    return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)

def check_pair(solution, fixtures):
    model = solution.build_model(fixtures)
    x1, x2, _, _ = fixtures['batch']
    dev = next(model.parameters()).device
    with torch.no_grad():
        got1, got2 = model.forward_pair(x1.to(dev), x2.to(dev))
        ref1 = model.h1(model.backbone(x1.to(dev))).squeeze(-1)
        ref2 = model.h2(model.backbone(x2.to(dev))).squeeze(-1)
    err = max(float((got1-ref1).abs().max().cpu()), float((got2-ref2).abs().max().cpu()))
    distinct = not torch.allclose(ref1, ref2, rtol=1e-5, atol=1e-6)
    return {'passed': _arch_ok(model, fixtures['init_state']) and err < 1e-6 and distinct, 'output_checksum': float((got1.mean()+got2.mean()).cpu()), 'max_reference_error': err}
