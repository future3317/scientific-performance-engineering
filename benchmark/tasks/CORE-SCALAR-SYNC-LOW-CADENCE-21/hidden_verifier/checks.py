from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict()
    return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)
import torch.nn.functional as F

def check_training(solution, fixtures):
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    x, y = fixtures['batch']
    dev = next(model.parameters()).device
    with torch.no_grad():
        pred = model(x.to(dev))
        loss = F.mse_loss(pred, y.to(dev))
        expected = [
            float(loss.detach().cpu()),
            float(pred.detach().cpu().mean()),
            float(pred.detach().cpu().abs().mean()),
            float(pred.detach().cpu().square().mean()),
        ]
    result = solution.train_step(model, fixtures['batch'], optimizer)
    got = [float(v) for v in result.get('metrics', [])]
    ok = len(got) == 4 and all(abs(a-b) < 1e-6 for a,b in zip(got, expected))
    return {'passed': _arch_ok(model, fixtures['init_state']) and ok and bool(torch.isfinite(result['loss'])), 'output_checksum': float(result['loss']), 'metric_error': max([abs(a-b) for a,b in zip(got, expected)] or [999.0])}
