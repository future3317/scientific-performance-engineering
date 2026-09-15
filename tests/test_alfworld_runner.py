from pathlib import Path
import sys
import types

from experiments.external_real.alfworld_runner import (
    SMOKE_TASK_TYPE_IDS,
    _alfworld_config,
    extract_action,
    task_family_name,
)


def test_extract_action_accepts_exact_action_after_label():
    assert extract_action("Action: go to cabinet 1", ["go to cabinet 1", "look"]) == "go to cabinet 1"


def test_extract_action_returns_none_for_non_admissible_output():
    assert extract_action("pick up the invisible object", ["look", "go to cabinet 1"]) is None


def test_task_family_name_uses_official_alfworld_ids():
    assert task_family_name(3) == "clean_place"
    assert task_family_name(4) == "heat_place"
    assert task_family_name(5) == "cool_place"


def test_alfworld_config_uses_official_numeric_task_ids():
    config = _alfworld_config(Path("/data/alfworld"), SMOKE_TASK_TYPE_IDS, 60)
    assert config["env"]["task_types"] == [3, 4, 5]


def test_deepseek_client_uses_bounded_single_request(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **request: captured.update(request)
                        or types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(content="", reasoning_content="Action: look")
                            )
                        ],
                        usage=None,
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from experiments.external_real.alfworld_runner import DEFAULT_MODEL, DeepSeekClient

    DeepSeekClient("key", "https://api.deepseek.com", "deepseek-flash")
    assert DEFAULT_MODEL == "deepseek-flash"
    assert captured["timeout"] == 120.0
    assert captured["max_retries"] == 0

    DeepSeekClient("key", "https://api.deepseek.com", "deepseek-flash").choose_action("obs", ["look"], None)
    assert captured["model"] == "deepseek-flash"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["max_tokens"] == 512


def test_saved_action_replays_through_online_path_and_env_step(monkeypatch, tmp_path):
    raw_receipt = "Action: go to fridge 1"
    admissible_commands = ["go to cabinet 1", "go to fridge 1", "look"]
    provider_reasoning = "The saved response selected the fridge as the next location."

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **request: types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(
                                    content=raw_receipt,
                                    reasoning_content=provider_reasoning,
                                )
                            )
                        ],
                        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from experiments.external_real.alfworld_runner import DeepSeekClient, extract_action, run_episode

    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "traj_data.json").write_text('{"task_type": "pick_clean_then_place_in_recep"}', encoding="utf-8")

    class ReplayEnvironment:
        def __init__(self):
            self.actions = []

        def reset(self):
            return ["initial observation"], {
                "admissible_commands": [admissible_commands],
                "extra.gamefile": [str(game_dir / "game.tw-pddl")],
            }

        def step(self, commands):
            self.actions.append(commands)
            return ["next observation"], [0.0], [True], {
                "won": [False],
                "admissible_commands": [["look"]],
            }

    actual_client = DeepSeekClient("key", "https://api.deepseek.com", "deepseek-flash")
    observed = {}

    class ReplayClient:
        def choose_action(self, observation, admissible, skill):
            completion = actual_client.choose_action(observation, admissible, skill)
            observed["completion"] = completion
            return completion

    environment = ReplayEnvironment()
    row = run_episode(environment, ReplayClient(), None, max_steps=1)
    completion = observed["completion"]
    raw_provider_response = completion["provider_raw"]
    response_field_used_by_online_runner = completion["response_field_used"]
    parser_input = completion["parser_input"]
    parsed_action = extract_action(parser_input, admissible_commands)
    action_argument_actually_passed_to_env = environment.actions[0][0]

    print("raw_provider_response:", raw_provider_response)
    print("response_field_used_by_online_runner:", response_field_used_by_online_runner)
    print("parsed_action:", parsed_action)
    print("repr(parsed_action):", repr(parsed_action))
    print("admissible_commands[0]:", admissible_commands[0])
    print("action_argument_actually_passed_to_env:", action_argument_actually_passed_to_env)

    assert raw_provider_response["content"] == raw_receipt
    assert response_field_used_by_online_runner == "message.content"
    assert parsed_action == "go to fridge 1"
    assert parsed_action in admissible_commands
    assert action_argument_actually_passed_to_env == parsed_action
    assert row["invalid_actions"] == 0
