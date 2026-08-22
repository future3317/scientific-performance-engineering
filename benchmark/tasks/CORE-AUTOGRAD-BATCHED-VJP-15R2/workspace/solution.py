from __future__ import annotations
import torch
from torch import nn

TASK_VARIANT = "CORE-AUTOGRAD-BATCHED-VJP-15R2"

class Model(nn.Module):
    def __init__(self, input_dim, output_count, output_mask=None):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, output_count)); self.register_buffer("vjp_output_mask", output_mask.clone() if output_mask is not None else torch.ones(output_count, dtype=torch.bool))
    def forward(self, x): return self.net(x)

def build_model(fixtures):
    model = Model(fixtures["input_dim"], fixtures["output_count"], fixtures["vjp_output_mask"]); model.load_state_dict(fixtures["init_state"]); return model.to(fixtures["device"])

def jacobian_features(model, x):
    x = x.detach().requires_grad_(True); y = model(x); rows = []
    for index in torch.nonzero(model.vjp_output_mask, as_tuple=False).flatten().tolist(): rows.append(torch.autograd.grad(y[:, index].mean(), x, retain_graph=True, create_graph=True)[0])
    return torch.stack(rows)

def train_step(model, batch, optimizer):
    x = batch[0].to(next(model.parameters()).device); optimizer.zero_grad(); y = model(x); j = jacobian_features(model, x); loss = y.square().mean() + 0.1 * j.square().mean(); loss.backward(); optimizer.step()
    return {"loss": loss.detach(), "work_units": {"vjp": int(model.vjp_output_mask.sum()), "optimizer": 1}}

def run_training(fixtures, steps):
    model = build_model(fixtures); optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"]); result = None
    for _ in range(steps): result = train_step(model, fixtures["batch"], optimizer)
    return {"final_loss": result["loss"]}


