from experiments.external_real.alfworld_runner import extract_action, task_family_name


def test_extract_action_accepts_exact_action_after_label():
    assert extract_action("Action: go to cabinet 1", ["go to cabinet 1", "look"]) == "go to cabinet 1"


def test_extract_action_returns_none_for_non_admissible_output():
    assert extract_action("pick up the invisible object", ["look", "go to cabinet 1"]) is None


def test_task_family_name_uses_official_alfworld_ids():
    assert task_family_name(3) == "clean_place"
    assert task_family_name(4) == "heat_place"
    assert task_family_name(5) == "cool_place"
