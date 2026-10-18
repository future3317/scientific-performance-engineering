from experiments.skill_evolution.native_verifier import (
    official_outcome,
    score_official_episode,
    teacher_diagnostic_episode,
)


def test_official_success_requires_terminal_done_and_won():
    assert official_outcome(True, True) == {"success": True, "status": "VERIFIED_SUCCESS"}
    assert official_outcome(False, False) == {"success": None, "status": "IN_PROGRESS"}
    assert official_outcome(False, True) == {"success": None, "status": "VERIFIER_INCONSISTENT"}
    assert official_outcome(True, False) == {"success": False, "status": "VERIFIED_FAILURE"}
    assert official_outcome(True, None) == {"success": None, "status": "INCONCLUSIVE"}


def test_negative_verifier_contracts_are_not_success_even_with_reward():
    legal_incomplete = score_official_episode(
        done=False, won=False, reward=1, steps=1, max_steps=5,
    )
    assert legal_incomplete["success"] is None
    assert legal_incomplete["status"] == "IN_PROGRESS"

    illegal = score_official_episode(
        done=True, won=True, reward=1, steps=1, max_steps=5, action_admissible=False,
    )
    assert illegal["success"] is None
    assert illegal["status"] == "API_CONTRACT_FAILURE"

    exhausted = score_official_episode(
        done=False, won=False, reward=0, steps=5, max_steps=5,
    )
    assert exhausted["success"] is False
    assert exhausted["status"] == "VERIFIED_FAILURE"

    early_end_with_reward = score_official_episode(
        done=True, won=False, reward=1, steps=2, max_steps=5,
    )
    assert early_end_with_reward["success"] is False
    assert early_end_with_reward["status"] == "VERIFIED_FAILURE"


def test_provider_or_missing_native_values_are_not_recast_as_task_failures():
    missing = score_official_episode(done=None, won=False, steps=5, max_steps=5)
    assert missing["success"] is None
    assert missing["status"] == "INCONCLUSIVE"

    class bool_:
        def __init__(self, value):
            self.value = value

        def __bool__(self):
            return self.value

    assert official_outcome(bool_(True), bool_(True))["success"] is True


class _TeacherEnvironment:
    def __init__(self, *, action="look", done=True, won=True, reward=1):
        self.action = action
        self.done = done
        self.won = won
        self.reward = reward
        self.step_calls = 0
        self.gamefile = "/data/json_2.1.1/train/pick_clean_then_place_in_recep-Mug-Cabinet-1/trial_1/game.tw-pddl"

    def reset(self):
        return ["observation"], {
            "extra.gamefile": [self.gamefile],
            "admissible_commands": [["look"]],
            "extra.expert_plan": [[self.action]],
        }

    def step(self, actions):
        self.step_calls += 1
        assert actions == [self.action]
        return ["next"], [self.reward], [self.done], {
            "won": [self.won], "admissible_commands": [["look"]],
            "extra.expert_plan": [["look"]],
        }


def test_teacher_positive_is_tagged_and_never_counted_as_agent_success():
    environment = _TeacherEnvironment()
    result = teacher_diagnostic_episode(environment, split="train", max_steps=5)
    assert result["status"] == "VERIFIED_SUCCESS"
    assert result["teacher_success"] is True
    assert result["agent_success"] is False
    assert result["evidence_kind"] == "teacher_or_scripted_diagnostic"
    assert result["provider_calls"] == 0


def test_teacher_illegal_expert_action_is_rejected_before_environment_step():
    environment = _TeacherEnvironment(action="take mug from cabinet")
    result = teacher_diagnostic_episode(environment, split="train", max_steps=5)
    assert result["status"] == "API_CONTRACT_FAILURE"
    assert result["teacher_success"] is None
    assert result["agent_success"] is False
    assert environment.step_calls == 0


def test_teacher_diagnostic_rejects_non_train_split():
    environment = _TeacherEnvironment()
    try:
        teacher_diagnostic_episode(environment, split="untouched_eval", max_steps=5)
    except ValueError as error:
        assert "restricted to ALFWorld train" in str(error)
    else:
        raise AssertionError("teacher diagnostic accepted a non-train split")
