import torch


def gradient_equivalence(solution, fixtures):
    model = solution.build_model(fixtures); x = fixtures["batch"][0].to(fixtures["device"]); got = solution.jacobian_features(model, x)
    expected = []
    x_req = x.detach().requires_grad_(True); y = model(x_req)
    for index in torch.nonzero(model.vjp_output_mask, as_tuple=False).flatten().tolist():
        expected.append(torch.autograd.grad(y[:, index].mean(), x_req, retain_graph=True, create_graph=True)[0])
    reference = torch.stack(expected); error = float((got - reference).abs().max())
    return error < 2e-6 and torch.isfinite(got).all().item(), {"shape": list(got.shape), "max_vjp_error": error}



