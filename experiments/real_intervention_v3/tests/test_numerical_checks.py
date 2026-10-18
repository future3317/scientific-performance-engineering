import torch
from types import SimpleNamespace

from experiments.real_intervention_v3.numerical_checks import compare_outputs, configure_determinism, nested_digest


def test_determinism_configuration_disables_tf32_and_cudnn_benchmark():
    backends = SimpleNamespace(
        cudnn=SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=True),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
    )
    torch_stub = SimpleNamespace(
        set_num_threads=lambda count: None,
        set_num_interop_threads=lambda count: None,
        use_deterministic_algorithms=lambda enabled: setattr(torch_stub, "algorithms", enabled),
        backends=backends,
    )
    configure_determinism(torch_stub)
    assert torch_stub.algorithms is True
    assert backends.cudnn.deterministic is True
    assert backends.cudnn.benchmark is False
    assert backends.cudnn.allow_tf32 is False
    assert backends.cuda.matmul.allow_tf32 is False


def test_nested_outputs_compare_metadata_finiteness_and_tolerance():
    expected = {"logits": (torch.tensor([1.0, 2.0]), [torch.tensor([3.0])])}
    close = {"logits": (torch.tensor([1.0, 2.000001]), [torch.tensor([3.0])])}
    assert compare_outputs(expected, close, rtol=1e-4, atol=1e-5).passed
    assert compare_outputs(expected, {"logits": torch.tensor([1.0, 2.0])}, rtol=1e-4, atol=1e-5).reason.startswith("output_type_mismatch")
    assert compare_outputs(expected, {"logits": (torch.tensor([1.0, float("nan")]), [torch.tensor([3.0])])},
                           rtol=1e-4, atol=1e-5).reason.startswith("non_finite_output")


def test_digest_tracks_logical_tensor_values_and_nested_structure():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert nested_digest(tensor) == nested_digest(tensor.t().contiguous().t())
    assert nested_digest((tensor,)) != nested_digest([tensor])
    assert nested_digest(tensor) != nested_digest(tensor + 1)
