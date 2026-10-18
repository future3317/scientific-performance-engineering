import json

import torch

from experiments.real_intervention_v3 import model_admission, numerical_checks, source_adapters


def test_admission_runs_only_eager_reproducibility_and_persists_reference(tmp_path, monkeypatch):
    model = torch.nn.Linear(3, 2).eval()
    inputs = (torch.tensor([[1.0, 2.0, 3.0]]),)
    monkeypatch.setattr(source_adapters, "load_model", lambda spec: (model, inputs))
    monkeypatch.setattr(model_admission, "_seed_all", lambda torch_module, seed: torch_module.manual_seed(seed))
    monkeypatch.setattr(numerical_checks, "configure_determinism", lambda torch_module: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    spec = {"source": "torchvision", "model": "resnet18", "seed": 31,
            "reference_path": str(tmp_path / "baseline.pt"),
            "correctness": {"rtol": 1e-4, "atol": 1e-5}}
    output = tmp_path / "admission.json"

    result = model_admission.admit(spec, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == saved["status"] == "valid"
    assert result["arm_executed"] == "S0_ONLY"
    assert result["admission_status"] == "ADMITTED_BASELINE_ONLY"
    assert result["parameter_count"] == 8
    assert result["actual_input_shapes"] == {"input[0]": [1, 3]}
    assert (tmp_path / "baseline.pt").is_file()
    assert result["weights_sha256"] and result["inputs_sha256"] and result["reference_sha256"]
