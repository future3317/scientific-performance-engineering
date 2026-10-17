from experiments.external_real.alfworld_stateful_runner import (
    EpisodeState,
    ExecutionGuard,
    HistoryCompressor,
    run_episode,
)


def test_thinking_request_caches_stable_prefix_and_preserves_usage(monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
    from experiments.external_real.alfworld_runner import DeepSeekClient

    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Action: look", reasoning_content="Action: invalid"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=400, completion_tokens=60,
                prompt_cache_hit_tokens=256, prompt_cache_miss_tokens=144,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=50),
            ),
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    ))
    client = DeepSeekStatefulClient("key", "https://api.deepseek.com", "deepseek-flash")
    state = EpisodeState.start("Your task is to: put a clean apple in fridge.", 60)
    first = client.choose_action(state, ["look"], "Inspect first.", "progress one", None)
    state.remaining_steps = 59
    client.choose_action(state, ["look"], "Inspect first.", "progress two", "retry")
    prefix = requests[0]["messages"][1]["content"].split("progress one")[0]
    assert "Inspect first." in prefix
    assert requests[1]["messages"][1]["content"].startswith(prefix)
    assert requests[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert requests[0]["max_tokens"] == 8192
    assert first["raw_output"] == "Action: look"
    assert first["cache_hit_tokens"] == 256
    assert first["cache_miss_tokens"] == 144
    assert first["reasoning_tokens"] == 50
    assert first["finish_reason"] == "stop"
    legacy = DeepSeekClient("key", "https://api.deepseek.com", "deepseek-flash")
    row = legacy.choose_action("Observation", ["look"], "Inspect first.")
    assert row["parser_input"] == "Action: look"
    assert row["cache_hit_tokens"] == 256
    assert requests[-1]["max_tokens"] == 8192
    assert requests[-1]["extra_body"] == {"thinking": {"type": "enabled"}}

    # A truncated thinking response must not execute a command from private reasoning.
    def reasoning_only(**request):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", reasoning_content="Action: look"),
            finish_reason="length",
        )], usage=None)

    client.client.chat.completions.create = reasoning_only
    row = client.choose_action(state, ["look"], None, "progress", None)
    assert row["raw_output"] == ""
    assert row["finish_reason"] == "length"
    assert row["cache_hit_tokens"] is None


class ReplayClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def choose_action(self, state, admissible, skill, progress, replan_reason):
        self.calls.append((state, tuple(admissible), progress, replan_reason))
        action = self.actions.pop(0)
        return {"raw_output": f"Action: {action}", "input_tokens": None, "output_tokens": None}


class ReplayEnvironment:
    def __init__(self, observations, actions, *, won=False):
        self.observations = list(observations)
        self.actions = list(actions)
        self.won = won
        self.passed = []

    def reset(self):
        return [self.observations[0]], {
            "admissible_commands": [self.actions[0]],
            "extra.gamefile": ["must-not-be-read"],
        }

    def step(self, commands):
        self.passed.extend(commands)
        index = len(self.passed)
        done = index == len(self.actions)
        return [self.observations[index]], [0.0], [done], {
            "won": [self.won and done],
            "admissible_commands": [self.actions[min(index, len(self.actions) - 1)]],
        }


def test_environment_failure_preserves_provider_usage_and_unknown_outcome():
    class BrokenEnvironment(ReplayEnvironment):
        def step(self, commands):
            raise RuntimeError("environment unavailable")

    env = BrokenEnvironment(["Your task is to: put a clean apple in fridge."], [["look"]])
    row = run_episode(env, ReplayClient(["look"]), skill="Inspect visible state", max_steps=1)
    assert row["success"] is None
    assert row["env_steps"] == 0
    assert row["trajectory"][0]["selected_action"] == "look"
    assert "input_tokens" in row["trajectory"][0]
    assert row["skill_rendered"] and row["skill_read_or_used"] == "unknown"


def test_visible_state_tracks_goal_inventory_transform_and_progress_without_gamefile():
    state = EpisodeState.start(
        "You are in kitchen. Your task is to: put a clean apple in fridge.",
        remaining_steps=8,
    )
    assert state.target_object == "apple"
    assert state.target_receptacle == "fridge"
    assert state.required_transformation == "clean"
    assert "clean" not in state.completed_subgoals

    state.observe("You cannot take the apple from the countertop.", "take apple from countertop")
    assert state.inventory is None
    assert "acquire_object" not in state.completed_subgoals
    state.observe("You pick up the apple from the countertop.", "take apple from countertop")
    assert "acquire_object" in state.completed_subgoals
    state.observe("You cannot clean the apple with sinkbasin 1.", "clean apple with sinkbasin 1")
    assert "clean" not in state.completed_subgoals
    state.observe("You clean the apple with sinkbasin 1.", "clean apple with sinkbasin 1")
    assert "clean" in state.completed_subgoals
    state.observe("You put the apple in fridge 1.", "put apple in fridge 1")
    assert "place_object" in state.completed_subgoals
    assert "gamefile" not in repr(state).casefold()


def test_episode_runs_visible_subgoals_and_only_passes_admissible_actions():
    observations = [
        "You are in kitchen. Your task is to: put a clean apple in fridge.",
        "You arrive at the countertop. You see an apple.",
        "You cannot take the apple from the countertop.",
        "You pick up the apple from the countertop.",
        "You clean the apple with sinkbasin 1.",
        "You arrive at the fridge.",
        "You open fridge 1.",
        "You put the apple in fridge 1.",
        "Task complete.",
    ]
    actions = [
        ["go to countertop"],
        ["take apple from countertop"],
        ["take apple from countertop"],
        ["clean apple with sinkbasin 1"],
        ["go to fridge 1"],
        ["open fridge 1"],
        ["put apple in fridge 1"],
        ["look"],
    ]
    env = ReplayEnvironment(observations, actions, won=True)
    client = ReplayClient(["go to countertop", "take apple from countertop", "take apple from countertop", "clean apple with sinkbasin 1", "go to fridge 1", "open fridge 1", "put apple in fridge 1", "look"])

    result = run_episode(env, client, max_steps=8)

    assert result["success"] is True
    assert len(env.passed) == 8
    assert all(action in allowed for action, allowed in zip(env.passed, actions))
    assert result["state"]["required_transformation"] == "clean"
    assert "clean" in result["state"]["completed_subgoals"]
    assert "place_object" in result["state"]["completed_subgoals"]
    assert result["trajectory"][6]["subgoal"].startswith("verify visible task completion")
    assert "acquire_object" not in result["trajectory"][1]["completed_subgoals"]
    assert "acquire_object" in result["trajectory"][2]["completed_subgoals"]
    assert result["state"]["completed_subgoals"][-1] == "environment_verified_completion"
    assert "gamefile" not in result
    assert result["trajectory"][0]["prompt_chars_by_section"]["state_chars"] > 0


def test_invalid_model_action_triggers_replan_and_never_reaches_environment():
    env = ReplayEnvironment(["Start", "Finished"], [["look"]], won=True)
    client = ReplayClient(["delete everything", "look"])

    result = run_episode(env, client, max_steps=2)

    assert env.passed == ["look"]
    assert client.calls[1][3] == "action_not_admissible"
    assert any(event.get("kind") == "replan" for event in result["progress_events"])
    assert result["invalid_actions"] == 1


def test_guard_requests_replan_after_repeated_no_progress_and_history_is_bounded():
    state = EpisodeState.start("Same observation", remaining_steps=8)
    guard = ExecutionGuard(repeat_limit=2)
    assert guard.observe_result(state, "Same observation", "Same observation", "look") is None
    assert guard.observe_result(state, "Same observation", "Same observation", "look") == "repeated_state_action"

    history = [{"step": index, "action": f"action-{index}"} for index in range(20)]
    _, sizes = HistoryCompressor(recent_events=3).render(state, history)
    assert sizes["recent_history_chars"] < len(str(history))


def test_unknown_inventory_does_not_claim_acquisition():
    state = EpisodeState.start("Your task is to: pick up the mug.", remaining_steps=4)
    state.observe("You see a mug on the table.")
    assert state.inventory is None
    assert "acquire_object" not in state.completed_subgoals


def test_guard_detects_alternating_navigation_cycle():
    guard = ExecutionGuard(repeat_limit=3)
    state = EpisodeState.start("Your task is to: put a clean apple in fridge.", 60)
    for _ in range(2):
        assert guard.observe_result(state, "at fridge", "at cabinet", "go to cabinet 1") is None
        assert guard.observe_result(state, "at cabinet", "at fridge", "go to fridge 1") is None
    assert guard.observe_result(state, "at fridge", "at cabinet", "go to cabinet 1") == "repeated_state_action"


def test_failed_placement_does_not_advance_until_native_success_acknowledgement():
    state = EpisodeState.start("Your task is to: put a clean apple in fridge.", remaining_steps=4)
    state.observe("You cannot put the apple in fridge 1.", "put apple in fridge 1")
    assert "place_object" not in state.completed_subgoals
    state.observe("You put the apple in fridge 1.", "put apple in fridge 1")
    assert "place_object" in state.completed_subgoals
import pytest


@pytest.mark.parametrize("goal,obj,count,recep,family", [
    ("put two keychain in dresser", "keychain", 2, "dresser", "pick_two_obj_and_place"),
    ("find two keychain and put them in dresser", "keychain", 2, "dresser", "pick_two_obj_and_place"),
    ("put two mugs on shelf", "mug", 2, "shelf", "pick_two_obj_and_place"),
    ("put a keychain in dresser", "keychain", 1, "dresser", "pick_and_place_simple"),
    ("put some keychain on desk", "keychain", 1, "desk", "pick_and_place_simple"),
    ("put a clean mug in cabinet", "mug", 1, "cabinet", "pick_clean_then_place_in_recep"),
    ("put a hot apple in fridge", "apple", 1, "fridge", "pick_heat_then_place_in_recep"),
    ("put a cool cup in cabinet", "cup", 1, "cabinet", "pick_cool_then_place_in_recep"),
    ("look at a mug under the desklamp", "mug", 1, "desklamp", "look_at_obj_in_light"),
    ("examine the laptop with the desklamp", "laptop", 1, "desklamp", "look_at_obj_in_light"),
    # The six supported families each have two public goal_library templates.
    ("clean some mug and put it in cabinet", "mug", 1, "cabinet", "pick_clean_then_place_in_recep"),
    ("heat some mug and put it in cabinet", "mug", 1, "cabinet", "pick_heat_then_place_in_recep"),
    ("cool some mug and put it in cabinet", "mug", 1, "cabinet", "pick_cool_then_place_in_recep"),
    ("look at mug under the desklamp", "mug", 1, "desklamp", "look_at_obj_in_light"),
])
def test_public_goal_contract(goal, obj, count, recep, family):
    state = EpisodeState.start(f"Your task is to: {goal}.", 60)
    assert (state.target_object, state.target_count, state.target_receptacle,
            state.task_family, state.goal_parse_status) == (obj, count, recep, family, "PARSED")


@pytest.mark.parametrize("won", [False, True])
def test_native_examine_goal_reaches_online_action_path_without_provider(monkeypatch, won):
    import sys
    from types import SimpleNamespace
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
    from experiments.skill_evolution.attribution import LENGTH_CONTROL

    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            id="public-goal-regression", model="fake-model", system_fingerprint="fake-fingerprint",
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="Action: use desklamp 1", reasoning_content="Action: must-not-execute"),
                finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    ))
    env = ReplayEnvironment([
        "Inventory: laptop 1.\nYour task is to: examine the laptop with the desklamp.",
        "You turn on the desklamp 1.",
    ], [["use desklamp 1"]], won=won)
    client = DeepSeekStatefulClient("fake-key", "https://api.deepseek.com", "deepseek-flash")
    result = run_episode(env, client, skill=LENGTH_CONTROL, max_steps=1)

    assert len(requests) == 1
    prompt = requests[0]["messages"][1]["content"]
    assert '"target_object": "laptop"' in prompt and '"target_count": 1' in prompt
    assert '"task_family": "look_at_obj_in_light"' in prompt
    assert "use the requested light with the held goal object" in prompt
    assert f"Reusable skill: {LENGTH_CONTROL}\n" in prompt
    assert env.passed == ["use desklamp 1"]
    assert result["env_steps"] == result["provider_calls"] == 1
    assert result["trajectory"][0]["response_field_used"] == "message.content"
    assert result["trajectory"][0]["invalid_action"] is False
    assert result["environment_won"] is won and result["environment_done"] is True
    assert result["success"] is won


def test_unparsed_goal_keeps_unknown_count_and_original_public_text():
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient, SubgoalPlanner
    goal = "do something unspecified"
    state = EpisodeState.start(f"Your task is to: {goal}", 60)
    assert state.goal == goal
    assert state.target_count is None and state.target_object is None
    assert state.goal_parse_status == "UNPARSED"
    assert "unresolved" in SubgoalPlanner.next_subgoal(state, ["look"])
    assert not state.completed_subgoals
    class NoRequest:
        def complete_text(self, messages):
            pytest.fail("unparsed policy goal must stop before any provider request")
    with pytest.raises(ValueError, match="public_goal_unparsed"):
        DeepSeekStatefulClient.choose_action(NoRequest(), state, ["look"], None, "", None)
    class Policy(NoRequest):
        choose_action = DeepSeekStatefulClient.choose_action
    env = ReplayEnvironment([f"Your task is to: {goal}", "Must not execute"], [["look"]])
    result = run_episode(env, Policy(), max_steps=1)
    assert result["status"] == "GOAL_PARSE_UNCERTAIN" and result["success"] is None
    assert result["provider_calls"] == result["env_steps"] == 0
    assert env.passed == [] and result["trajectory"] == []


def test_two_object_progress_requires_distinct_acknowledged_instances():
    from experiments.external_real.alfworld_stateful_runner import SubgoalPlanner
    state = EpisodeState.start("Your task is to: put two keychain in dresser", 60)
    put1 = "put keychain 1 in/on dresser 1"
    state.observe("Nothing happens.", put1)
    assert not state.placed_target_objects
    state.observe("You put keychain 1 in/on dresser 1.", put1)
    state.observe("You put keychain 1 in/on dresser 1.", put1)
    assert state.placed_target_objects == ["keychain 1"]
    assert "place_object" not in state.completed_subgoals
    assert SubgoalPlanner.next_subgoal(state, ["go to drawer 1"]).startswith("locate the goal object")
    state.observe("You cannot put keychain 2 in/on dresser 1.", "put keychain 2 in/on dresser 1")
    assert len(state.placed_target_objects) == 1
    state.observe("You put keychain 2 in/on dresser 1.", "put keychain 2 in/on dresser 1")
    assert state.placed_target_objects == ["keychain 1", "keychain 2"]
    assert "place_object" in state.completed_subgoals
    state.observe("You pick up keychain 1 from dresser 1.", "take keychain 1 from dresser 1")
    assert state.placed_target_objects == ["keychain 2"]
    assert "place_object" not in state.completed_subgoals


def test_real_prompt_path_carries_corrected_target_and_count_without_provider():
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
    from experiments.skill_evolution.attribution import LENGTH_CONTROL
    class Capture:
        def complete_text(self, messages):
            return messages
    state = EpisodeState.start("Your task is to: put two keychain in dresser", 60)
    progress, _ = HistoryCompressor().render(state, [])
    messages = DeepSeekStatefulClient.choose_action(Capture(), state, ["look"], LENGTH_CONTROL, progress, None)
    prompt = messages[1]["content"]
    assert '"target_object": "keychain"' in prompt and '"target_count": 2' in prompt
    assert "Current subgoal: locate the goal object" in prompt
    assert f"Reusable skill: {LENGTH_CONTROL}\n" in prompt


def test_two_object_state_never_overrides_official_failure():
    env = ReplayEnvironment([
        "Your task is to: put two keychain in dresser",
        "You put keychain 1 in/on dresser 1.",
        "You put keychain 2 in/on dresser 1.",
    ], [["put keychain 1 in/on dresser 1"], ["put keychain 2 in/on dresser 1"]], won=False)
    result = run_episode(env, ReplayClient(["put keychain 1 in/on dresser 1", "put keychain 2 in/on dresser 1"]), max_steps=2)
    assert "place_object" in result["state"]["completed_subgoals"]
    assert result["success"] is False
    assert result["environment_won"] is False and result["environment_done"] is True



@pytest.mark.parametrize("transform", ["clean", "heat", "cool"])
def test_native_transformation_acknowledgements_require_positive_result(transform):
    state = EpisodeState.start(f"Your task is to: put a {transform} apple in fridge.", remaining_steps=4)
    state.observe(f"You cannot {transform} the apple.", f"{transform} apple with appliance 1")
    assert transform not in state.completed_subgoals
    state.observe(f"You {transform} the apple with appliance 1.", f"{transform} apple with appliance 1")
    assert transform in state.completed_subgoals


def test_episode_state_caps_accumulated_visible_state_sets():
    state = EpisodeState.start("Start", remaining_steps=100)
    for index in range(40):
        state.observe(f"You arrive at room {index}. You see object {index}.")
    assert len(state.visited_locations) <= 12
    assert len(state.seen_objects_and_receptacles) <= 32
    assert len(state.progress_markers) <= 12
