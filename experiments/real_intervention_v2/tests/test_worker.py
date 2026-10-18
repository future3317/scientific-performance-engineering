import torch

from experiments.real_intervention_v2.worker import _digest


def test_digest_supports_scalar_tensors_nested_in_outputs():
    first = {"score": torch.tensor(3, dtype=torch.int64), "logits": (torch.tensor([1.0]),)}
    same = {"score": torch.tensor(3, dtype=torch.int64), "logits": (torch.tensor([1.0]),)}
    changed = {"score": torch.tensor(4, dtype=torch.int64), "logits": (torch.tensor([1.0]),)}

    assert _digest(first) == _digest(same)
    assert _digest(first) != _digest(changed)
