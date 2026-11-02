from experiments.skill_evolution.soft_acre_logicbench_live import deepseek_api_key


def test_deepseek_key_takes_precedence_over_generic_openai_key():
    assert deepseek_api_key({
        "DEEPSEEK_API_KEY": "deepseek-key",
        "OPENAI_API_KEY": "other-provider-key",
    }) == "deepseek-key"


def test_generic_key_remains_a_fallback():
    assert deepseek_api_key({"DEEPSEEK_API_KEY": None, "OPENAI_API_KEY": "key"}) == "key"
