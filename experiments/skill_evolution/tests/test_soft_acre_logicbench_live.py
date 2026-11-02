import pytest

from experiments.skill_evolution.soft_acre_logicbench_live import (
    deepseek_api_key,
    validate_final_policy_decisions,
)


def test_deepseek_key_takes_precedence_over_generic_openai_key():
    assert deepseek_api_key({
        "DEEPSEEK_API_KEY": "deepseek-key",
        "OPENAI_API_KEY": "other-provider-key",
    }) == "deepseek-key"


def test_generic_key_remains_a_fallback():
    assert deepseek_api_key({"DEEPSEEK_API_KEY": None, "OPENAI_API_KEY": "key"}) == "key"


def test_final_decision_validation_uses_the_frozen_id_set():
    expected = [f"final-{index}" for index in range(76)]
    policy = {"per_skill": [{"final_decisions": [{"instance_id": item} for item in expected]}]}
    validate_final_policy_decisions(policy, expected)

    policy["per_skill"][0]["final_decisions"].pop()
    with pytest.raises(ValueError, match="frozen final cohort"):
        validate_final_policy_decisions(policy, expected)
