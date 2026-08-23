import torch


def finite_loss_gate(solution, fixtures):
    model = solution.build_model(fixtures); optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"]); losses = []
    for i, size in enumerate(fixtures["batch_sizes"][:3]):
        out = solution.train_step(model, (fixtures["inputs"][i * size:(i + 1) * size], fixtures["targets"][i * size:(i + 1) * size]), optimizer); losses.append(out["loss"])
    values = torch.stack(losses)
    return bool(torch.isfinite(values).all()), {"losses": values.tolist()}



