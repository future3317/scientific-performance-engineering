from __future__ import annotations
import torch

def _arch_ok(model, init_state):
    state=model.state_dict()
    return set(state)==set(init_state) and all(tuple(state[k].shape)==tuple(init_state[k].shape) for k in init_state)

def check_batch(solution, fixtures):
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    seen = {}
    def capture_input(_module, args):
        seen.setdefault('x', args[0].detach().cpu().clone())

    handle = model.register_forward_pre_hook(capture_input)
    result = solution.train_step(model, fixtures['batch'], optimizer)
    handle.remove()
    x, _ = fixtures['batch']
    captured = seen.get('x')
    ok = captured is not None and captured.shape == x.shape and torch.allclose(captured, x, rtol=0, atol=0)
    return {'passed': bool(_arch_ok(model, fixtures['init_state']) and ok and torch.isfinite(result['loss'])), 'output_checksum': float(result['loss'].detach().cpu()), 'captured_batch': bool(captured is not None)}
