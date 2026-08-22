from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

TASK_VARIANT = "CORE-MEM-RETAINED-GRAPH-13R2"

class SegmentedResMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, segment_count, checkpointed_segments):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.segments = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(segment_count))
        self.output = nn.Linear(hidden_dim, 1)
        self.segment_count = segment_count
        self.checkpointed_segments = checkpointed_segments
        self.retained_history = []

    def forward(self, x):
        h = torch.relu(self.input(x))
        for index, segment in enumerate(self.segments):
            out = checkpoint(segment, h, use_reentrant=False) if index < self.checkpointed_segments else segment(h)
            h = h + out
            self.retained_history.append(h.square().mean())
        return self.output(h)

def build_model(fixtures):
    config = fixtures["model_config"]
    model = SegmentedResMLP(config["in_dim"], config["hidden_dim"], config["segment_count"], config["checkpointed_segments"])
    model.load_state_dict(fixtures["init_state"])
    return model.to(fixtures["device"])

def train_step(model, batch, optimizer):
    device = next(model.parameters()).device
    x, target = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), target); loss.backward(); optimizer.step()
    return {"loss": loss.detach(), "work_units": {"forward": 1, "backward": 1, "optimizer": 1}}

def run_training(fixtures, steps):
    model = build_model(fixtures); optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"]); losses = []
    for i in range(steps):
        size = fixtures["batch_sizes"][i % len(fixtures["batch_sizes"])]
        offset = (i * size) % (fixtures["inputs"].shape[0] - size)
        losses.append(train_step(model, (fixtures["inputs"][offset:offset + size], fixtures["targets"][offset:offset + size]), optimizer)["loss"])
    return {"losses": torch.stack(losses), "final_loss": losses[-1].item()}


