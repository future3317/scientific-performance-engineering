import torch


def gradient_equivalence(solution, fixtures):
    model = solution.build_model(fixtures); x = fixtures["batch"][0].to(next(model.parameters()).device); jacobian = solution.jacobian_features(model, x)
    return (tuple(jacobian.shape[:2]) == (1, 8) and bool(torch.isfinite(jacobian).all())), {"shape": list(jacobian.shape)}



