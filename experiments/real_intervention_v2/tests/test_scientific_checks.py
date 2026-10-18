import pytest

torch = pytest.importorskip("torch")

from experiments.real_intervention_v2.scientific_checks import compare_outputs


def test_nested_tensor_outputs_compare_within_fixed_tolerance():
    result = compare_outputs({"logits": (torch.tensor([1.0]),)},
                             {"logits": (torch.tensor([1.00001]),)}, rtol=1e-4, atol=1e-6)
    assert result.passed
    assert result.tensor_count == 1


def test_structure_mismatch_and_nonfinite_outputs_fail():
    assert not compare_outputs([torch.ones(1)], (torch.ones(1),), rtol=0, atol=0).passed
    assert compare_outputs(torch.tensor([float("nan")]), torch.tensor([0.0]),
                           rtol=0, atol=0).reason == "non_finite_output:output"
