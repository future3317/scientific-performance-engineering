"""Visible-state ALFWorld policy with bounded history and explicit progress."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, ProviderUsageBudget, manifest_digest
from experiments.external_real.alfworld_runner import (
    _alfworld_config,
    _scalar,
    _select_game_files,
    extract_action,
    task_family_name,
)


POLICY_REVISION = "alfworld-stateful-visible-thinking"
_TRANSFORMATIONS = ("clean", "heat", "cool")
_STATE_HISTORY_LIMIT = 12
_SEEN_ENTITY_LIMIT = 32
_DEEPSEEK_MAX_TOKENS = 8192


def _prompt_token_bound(messages: Sequence[Mapping[str, str]]) -> int:
    """Bound this exact prompt by UTF-8 content bytes plus chat framing.

    This is accounting, not a tokenizer: it assumes content tokenization uses
    no more tokens than UTF-8 bytes and reserves 16 tokens per message plus 3
    for the provider's chat framing/assistant prefix.
    """
    return sum(len(message["content"].encode("utf-8")) + 16 for message in messages) + 3


def _goal_from_observation(observation: str) -> str:
    match = re.search(r"(?:your task is to|task is to)\s*:?\s*(.+?)(?:\n|$)", observation, re.I)
    return (match.group(1) if match else observation).strip()


def _goal_entities(goal: str) -> tuple[str | None, str | None]:
    task = re.sub(r"^\s*put\s+(?:a|an|the|some)\s+", "", goal, flags=re.I)
    task = re.sub(r"^(?:clean(?:ed)?|heat(?:ed)?|cool(?:ed)?|hot|cold)\s+", "", task, flags=re.I)
    object_match = re.search(r"(?:pick up|take|clean|heat|cool)\s+(?:(?:a|an|the|some)\s+)?([\w -]+?)(?:\s+(?:and|then|in|on|with|using)\b|[,.]|$)", goal, re.I)
    if not object_match and task != goal:
        object_match = re.search(r"^([\w -]+?)(?:\s+(?:in|inside|on|onto)\b|[,.]|$)", task, re.I)
    destination_match = re.search(r"\b(?:in|inside|on|onto)\s+(?:the\s+)?([\w -]+?)(?:\s+\d+)?(?:[,.]|$)", goal, re.I)
    return (
        object_match.group(1).strip().casefold() if object_match else None,
        destination_match.group(1).strip().casefold() if destination_match else None,
    )


def _visible_inventory(observation: str) -> tuple[str, ...] | None:
    match = re.search(r"(?:inventory|you are carrying|you have)\s*:?\s*([^\n.]+)", observation, re.I)
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value.casefold() in {"nothing", "nothing."}:
        return ()
    return tuple(item.strip().casefold() for item in re.split(r",| and ", value) if item.strip())


def _transformation_from_goal(goal: str) -> str | None:
    lowered = goal.casefold()
    return next((action for action in _TRANSFORMATIONS if re.search(
        rf"\b{action}(?:ed|ing)?\b" + (r"|\bhot\b" if action == "heat" else r"|\bcold\b" if action == "cool" else ""),
        lowered)), None)


def _mentions(entity: str, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(entity)}\b", text.casefold()))


@dataclass
class EpisodeState:
    """Episode memory derived only from visible observations and actions."""

    goal: str
    current_room_or_location: str | None = None
    visited_locations: list[str] = field(default_factory=list)
    seen_objects_and_receptacles: list[str] = field(default_factory=list)
    inventory: tuple[str, ...] | None = None
    target_object: str | None = None
    target_receptacle: str | None = None
    required_transformation: str | None = None
    completed_subgoals: list[str] = field(default_factory=list)
    open_containers: list[str] = field(default_factory=list)
    last_actions: list[str] = field(default_factory=list)
    progress_markers: list[str] = field(default_factory=list)
    stuck_counter: int = 0
    remaining_steps: int = 0
    last_observation: str = ""
    fact_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    observation_step: int = -1

    @classmethod
    def start(cls, observation: str, remaining_steps: int) -> "EpisodeState":
        goal = _goal_from_observation(observation)
        target_object, target_receptacle = _goal_entities(goal)
        state = cls(
            goal=goal,
            target_object=target_object,
            target_receptacle=target_receptacle,
            required_transformation=_transformation_from_goal(goal),
            remaining_steps=remaining_steps,
        )
        state.observe(observation)
        return state

    def observe(self, observation: str, action: str | None = None) -> None:
        self.observation_step += 1
        previous = self.last_observation
        self.last_observation = observation
        location = re.search(r"you are in (?:the )?([^.!\n]+)", observation, re.I)
        if not location:
            location = re.search(r"you arrive at (?:the )?([^.!\n]+)", observation, re.I)
        if location:
            value = location.group(1).strip().casefold()
            self.current_room_or_location = value
            self.fact_sources["location"] = {"status": "known", "step": self.observation_step}
            if value not in self.visited_locations:
                self.visited_locations.append(value)
                self.visited_locations = self.visited_locations[-_STATE_HISTORY_LIMIT:]
                self.progress_markers.append(f"location_seen:{value}")
        inventory = _visible_inventory(observation)
        if inventory is not None:
            contradicted = self.inventory is not None and inventory != self.inventory
            self.fact_sources["inventory"] = {"status": "known", "step": self.observation_step,
                                              "previous_contradicted": contradicted}
            if self.inventory != inventory:
                self.progress_markers.append("inventory_observed")
            if self.target_object and any(_mentions(self.target_object, item) for item in inventory):
                if "acquire_object" not in self.completed_subgoals:
                    self.completed_subgoals.append("acquire_object")
                    self.progress_markers.append("subgoal_observed:acquire_object")
            self.inventory = inventory
        names = re.findall(r"\b(?:[a-z][a-z -]*\s+\d+|[a-z]+(?:[ -][a-z]+)*)\b", observation.casefold())
        for name in names[:32]:
            normalized = name.strip()
            if normalized and normalized not in self.seen_objects_and_receptacles:
                self.seen_objects_and_receptacles.append(normalized)
        self.seen_objects_and_receptacles = self.seen_objects_and_receptacles[-_SEEN_ENTITY_LIMIT:]
        for container in re.findall(r"(?:open|opened)\s+(?:the\s+)?([\w -]+?)(?:\s+\d+)?(?:\b|[,.])", observation, re.I):
            normalized = container.strip().casefold()
            if normalized and normalized not in self.open_containers:
                self.open_containers.append(normalized)
        self.open_containers = self.open_containers[-_STATE_HISTORY_LIMIT:]

        visible = observation.casefold()
        if self.goal:
            visible = visible.replace(self.goal.casefold(), "")
        if self.target_object and _mentions(self.target_object, visible) and "locate_object" not in self.completed_subgoals:
            self.completed_subgoals.append("locate_object")
            self.progress_markers.append("subgoal_observed:locate_object")
        transform = self.required_transformation
        descriptor = {"clean": r"clean(?:ed)?", "heat": r"(?:heat(?:ed)?|hot)",
                      "cool": r"(?:cool(?:ed)?|cold)"}.get(transform, "")
        transform_observed = (transform and self.target_object and _mentions(self.target_object, visible)
                              and re.search(rf"\byou\s+{transform}(?:ed)?\b|\b(?:is|now)\s+(?:(?:a|an)\s+)?{descriptor}\b", visible))
        if transform_observed and transform not in self.completed_subgoals:
            self.completed_subgoals.append(transform)
            self.progress_markers.append(f"subgoal_observed:{transform}")
            self.fact_sources[str(transform)] = {"status": "known", "step": self.observation_step}
        if action:
            self.last_actions.append(action)
            self.last_actions = self.last_actions[-8:]
            self._record_action_acknowledgement(observation, action)
        self.progress_markers = self.progress_markers[-_STATE_HISTORY_LIMIT:]
        if observation != previous:
            self.stuck_counter = 0

    def _record_action_acknowledgement(self, observation: str, action: str) -> None:
        """Apply task-state changes only when ALFWorld visibly acknowledges success."""
        text = observation.casefold().replace(self.goal.casefold(), "")
        action = action.casefold().strip()
        pickup = re.match(r"(?:pick up|take)\s+(?:the\s+)?([\w -]+?)(?:\s+from\b|$)", action)
        if pickup and self.target_object and _mentions(self.target_object, pickup.group(1)):
            item = pickup.group(1).strip()
            if re.search(rf"\byou\s+(?:pick up|take)\s+(?:the\s+)?{re.escape(item)}\b", text):
                inventory = list(self.inventory or ())
                if item not in inventory:
                    inventory.append(item)
                self.inventory = tuple(inventory)
                self.fact_sources["inventory"] = {"status": "known", "step": self.observation_step}
                if "acquire_object" not in self.completed_subgoals:
                    self.completed_subgoals.append("acquire_object")
                    self.progress_markers.append("subgoal_acknowledged:acquire_object")

        transformation = next((value for value in _TRANSFORMATIONS if action.startswith(value + " ")), None)
        if transformation and self.target_object and _mentions(self.target_object, action) and re.search(
            rf"\byou\s+{transformation}(?:ed)?\b|\b(?:is|now)\s+{transformation}(?:ed)?\b", text
        ):
            if transformation not in self.completed_subgoals:
                self.completed_subgoals.append(transformation)
                self.progress_markers.append(f"subgoal_acknowledged:{transformation}")
                self.fact_sources[transformation] = {"status": "known", "step": self.observation_step}

        placement = re.match(r"(?:put|place|move)\s+(?:the\s+)?([\w -]+?)\s+(?:in/on|in|inside|on|onto|to)\s+(.+)$", action)
        if (
            placement and self.target_object and self.target_receptacle
            and _mentions(self.target_object, placement.group(1))
            and _mentions(self.target_receptacle, placement.group(2))
        ):
            item, destination = placement.group(1).strip(), placement.group(2).strip()
            acknowledged = re.search(
                rf"\byou\s+(?:put|place|move)\s+(?:the\s+)?{re.escape(item)}\s+(?:in/on|in|inside|on|onto|to)\s+(?:the\s+)?{re.escape(destination)}\b",
                text,
            )
            if acknowledged:
                if self.inventory is not None:
                    self.inventory = tuple(value for value in self.inventory if not _mentions(item, value))
                    self.fact_sources["inventory"] = {"status": "known", "step": self.observation_step}
                if "place_object" not in self.completed_subgoals:
                    self.completed_subgoals.append("place_object")
                    self.progress_markers.append("subgoal_acknowledged:place_object")


class SubgoalPlanner:
    """Choose a textual subgoal from state facts, without environment oracle data."""

    @staticmethod
    def next_subgoal(state: EpisodeState, admissible_actions: Sequence[str]) -> str:
        if "place_object" in state.completed_subgoals:
            return "verify visible task completion; otherwise replan from the latest observation"
        visible_observation = state.last_observation.casefold().replace(state.goal.casefold(), "")
        held = state.target_object is not None and state.inventory is not None and any(
            _mentions(state.target_object, item) for item in state.inventory
        )
        if state.target_object and not held:
            object_visible = _mentions(state.target_object, visible_observation) or any(
                _mentions(state.target_object, action) for action in admissible_actions
            )
            matching_pickup = any(
                action.casefold().startswith(("pick up ", "take ")) and _mentions(state.target_object, action)
                for action in admissible_actions
            )
            if matching_pickup or object_visible:
                return "acquire the visible goal object and confirm the inventory change"
            return "locate the goal object using visible observations and admissible navigation actions"
        if state.required_transformation and state.required_transformation not in state.completed_subgoals:
            return f"perform and visibly confirm the required {state.required_transformation} transformation"
        if state.target_receptacle and not _mentions(state.target_receptacle, visible_observation):
            return "locate the goal receptacle using admissible navigation actions"
        if state.target_receptacle and "place_object" not in state.completed_subgoals:
            return "place the goal object in the requested receptacle and inspect the result"
        return "verify visible task completion; otherwise replan from the latest observation"


class HistoryCompressor:
    """Render compact state and recent events; report characters, not fake tokens."""

    def __init__(self, recent_events: int = 6) -> None:
        self.recent_events = recent_events

    def render(self, state: EpisodeState, history: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, int]]:
        state_text = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
        history_text = json.dumps(list(history[-self.recent_events :]), ensure_ascii=False, sort_keys=True)
        sections = {"state_chars": state_text, "recent_history_chars": history_text}
        unknowns = [name for name in ("inventory", "location", state.required_transformation)
                    if name and name not in state.fact_sources]
        return "\n".join((f"Visible state: {state_text}", f"Unknown facts: {', '.join(unknowns) or 'none'}",
                          f"Recent events: {history_text}")), {
            key: len(value) for key, value in sections.items()
        }


class ExecutionGuard:
    def __init__(self, repeat_limit: int = 3) -> None:
        if repeat_limit < 1:
            raise ValueError("repeat_limit must be positive")
        self.repeat_limit = repeat_limit
        self._signature_counts: dict[tuple[Any, ...], int] = {}

    def select(self, raw_output: str, admissible_actions: Sequence[str]) -> tuple[str | None, str | None]:
        action = extract_action(raw_output, admissible_actions)
        return (action, None) if action is not None else (None, "action_not_admissible")

    def observe_result(self, state: EpisodeState, previous_observation: str, observation: str, action: str) -> str | None:
        signature = (state.current_room_or_location, state.inventory,
                     tuple(state.completed_subgoals), observation, action.casefold())
        self._signature_counts[signature] = self._signature_counts.get(signature, 0) + 1
        count = self._signature_counts[signature]
        # Returning to the same observed state/action is also a loop, even
        # when each navigation step changes the observation (A -> B -> A).
        state.stuck_counter = count if observation == previous_observation or count > 1 else 0
        return "repeated_state_action" if state.stuck_counter >= self.repeat_limit else None


class DeepSeekStatefulClient:
    """Small provider adapter; construction is explicit and never done by fixtures."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        budget: ProviderUsageBudget | None = None,
        max_completion_tokens: int = _DEEPSEEK_MAX_TOKENS,
        persist_budget: Any = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.budget = budget
        self.max_completion_tokens = max_completion_tokens
        self.persist_budget = persist_budget
        self.response_identity: tuple[Any, Any] | None = None

    def choose_action(
        self,
        state: EpisodeState,
        admissible_actions: Sequence[str],
        skill: str | None,
        progress: str,
        replan_reason: str | None,
    ) -> dict[str, Any]:
        skill_text = skill or "No reusable skill is provided."
        prompt = (
            "Choose exactly one command from the admissible list. Never invent commands or infer hidden state.\n"
            f"Reusable skill: {skill_text}\n"
            f"Goal: {state.goal}\n{progress}\n"
            f"Current subgoal: {SubgoalPlanner.next_subgoal(state, admissible_actions)}\n"
            f"Replan note: {replan_reason or 'none'}\n"
            f"Current observation: {state.last_observation}\n"
            "Admissible commands:\n" + "\n".join(f"- {item}" for item in admissible_actions)
            + "\nReply only as Action: <exact command>."
        )
        messages = [
            {"role": "system", "content": "Follow the action format exactly."},
            {"role": "user", "content": prompt},
        ]
        return self.complete_text(messages)

    def complete_text(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Shared metered final-content path for actions and Skill proposals.

        This text-only protocol deliberately does not send native API tools.
        Private reasoning is neither executable nor persisted in exported traces.
        """
        input_bound = _prompt_token_bound(messages)
        usd_before = self.budget.usd_used if isinstance(self.budget, ApprovedProviderBudget) else None
        if input_bound > getattr(self, "max_input_per_request", input_bound):
            raise ProviderBudgetExceeded("per_request_input_cap")
        if self.budget is not None:
            self.budget.reserve(input_bound, self.max_completion_tokens)
            if self.persist_budget is not None:
                self.persist_budget(self.budget.as_dict())
        started = time.perf_counter()
        try:
            request_timeout = ({"timeout": max(0.001, min(120.0, self.budget.remaining_wall_seconds))}
                               if isinstance(self.budget, ApprovedProviderBudget) else {})
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                extra_body={"thinking": {"type": "enabled"}},
                messages=messages,
                **request_timeout,
            )
        except Exception:
            if self.budget is not None:
                self.budget.mark_unknown("provider_exception_usage_unknown")
                if self.persist_budget is not None:
                    self.persist_budget(self.budget.as_dict())
            raise
        latency = time.perf_counter() - started
        try:
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            if self.budget is not None:
                if (isinstance(input_tokens, int) and isinstance(output_tokens, int)
                        and input_tokens >= 0 and output_tokens >= 0):
                    if isinstance(self.budget, ApprovedProviderBudget):
                        self.budget.record_usage(input_tokens, output_tokens,
                            cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", None),
                            cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None))
                    else:
                        self.budget.record_usage(input_tokens, output_tokens)
                else:
                    self.budget.mark_unknown()
        except Exception:
            if self.budget is not None:
                self.budget.mark_unknown()
            raise
        finally:
            if self.budget is not None and self.persist_budget is not None:
                self.persist_budget(self.budget.as_dict())
        message = response.choices[0].message
        identity = (getattr(response, "model", None), getattr(response, "system_fingerprint", None))
        if self.response_identity is not None and identity != self.response_identity:
            if self.budget is not None:
                self.budget.stop("provider_model_identity_changed")
                if self.persist_budget is not None:
                    self.persist_budget(self.budget.as_dict())
            raise RuntimeError("provider model identity changed within a frozen run")
        self.response_identity = identity
        content = message.content or ""
        parser_input = content
        return {
            "raw_output": parser_input,
            "prompt_sha256": manifest_digest({"messages": messages}),
            "provider_raw": {"content": content},
            "model_response_identity": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "response_field_used": "message.content",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
            "cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "latency_seconds": latency,
            "estimated_provider_usd": (str(self.budget.usd_used - usd_before)
                                       if usd_before is not None else None),
        }


def run_episode(
    environment: Any,
    client: Any,
    skill: str | None = None,
    max_steps: int = 60,
    *,
    repeat_limit: int = 3,
    skill_version: str | None = None,
    receipt_metadata: Mapping[str, Any] | None = None,
    expected_gamefiles: Sequence[str] | None = None,
) -> dict[str, Any]:
    from experiments.skill_evolution.native_verifier import official_outcome

    started_at = time.perf_counter()
    observations, infos = environment.reset()
    actual_gamefile = _scalar(infos, "extra.gamefile") if expected_gamefiles is not None else None
    if expected_gamefiles is not None and actual_gamefile not in expected_gamefiles:
        raise ValueError("environment reset outside the frozen task manifest")
    observation = str(observations[0])
    admissible = list(_scalar(infos, "admissible_commands") or [])
    state = EpisodeState.start(observation, max_steps)
    planner, guard, compressor = SubgoalPlanner(), ExecutionGuard(repeat_limit), HistoryCompressor()
    history: list[dict[str, Any]] = []
    progress_events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    replan_reason: str | None = None
    termination_reason = "step_limit"
    won: bool | None = None
    done: bool | None = None

    for step_index in range(max_steps):
        state.remaining_steps = max_steps - step_index
        progress, chars = compressor.render(state, history)
        try:
            completion = client.choose_action(state, admissible, skill, progress, replan_reason)
        except ProviderBudgetExceeded as exc:
            termination_reason = "provider_budget_exhausted"
            progress_events.append({
                "step": step_index,
                "kind": "provider_budget_exhausted",
                "reason": str(exc),
            })
            break
        except Exception as exc:  # external client errors are terminal evidence, not hidden retries
            termination_reason = "policy_error"
            trajectory.append({"step": step_index, "observation": observation,
                               "selected_action": None, "invalid_action": False,
                               "failure_stage": "provider", "exception_type": type(exc).__name__,
                               "input_tokens": None, "output_tokens": None})
            progress_events.append({"step": step_index, "kind": "policy_error", "error": type(exc).__name__})
            break
        action, invalid_reason = guard.select(str(completion.get("raw_output", "")), admissible)
        if action is None:
            trajectory.append({
                "step": step_index,
                "observation": observation,
                "admissible_actions": admissible,
                "selected_action": None,
                "invalid_action": True,
                "fallback_reason": invalid_reason,
                "prompt_chars_by_section": chars,
                **completion,
            })
            progress_events.append({"step": step_index, "kind": "replan", "reason": invalid_reason})
            replan_reason = invalid_reason
            continue

        try:
            next_observations, rewards, dones, next_infos = environment.step([action])
        except Exception as exc:
            termination_reason = "environment_error"
            trajectory.append({"step": step_index, "observation": observation,
                               "admissible_actions": admissible, "selected_action": action,
                               "invalid_action": False, "failure_stage": "environment",
                               "exception_type": type(exc).__name__, **completion})
            progress_events.append({"step": step_index, "kind": "environment_error", "error": type(exc).__name__})
            break
        next_observation = str(next_observations[0])
        previous_observation = observation
        state.observe(next_observation, action)
        cycle_reason = guard.observe_result(state, previous_observation, next_observation, action)
        done = bool(dones[0])
        native_won = _scalar(next_infos, "won")
        won = bool(native_won) if native_won is not None else None
        event = {
            "step": step_index,
            "action": action,
            "observation_changed": next_observation != observation,
            "subgoal": planner.next_subgoal(state, admissible),
            "completed_subgoals": list(state.completed_subgoals),
            "stuck_counter": state.stuck_counter,
        }
        history.append(event)
        progress_events.append({"step": step_index, "kind": "action_result", **event})
        trajectory.append({
            "step": step_index,
            "observation": observation,
            "admissible_actions": admissible,
            "selected_action": action,
            "invalid_action": False,
            "reward": rewards[0],
            "done": done,
            "won": won,
            "subgoal": event["subgoal"],
            "completed_subgoals": list(state.completed_subgoals),
            "stuck_counter": state.stuck_counter,
            "prompt_chars_by_section": chars,
            **completion,
        })
        native_status = official_outcome(done, won)["status"]
        if native_status in {"INCONCLUSIVE", "VERIFIER_INCONSISTENT"}:
            termination_reason = "verifier_inconclusive"
            break
        if done:
            termination_reason = "success" if won else "environment_done"
            if won:
                state.completed_subgoals.append("environment_verified_completion")
            break
        replan_reason = cycle_reason
        if cycle_reason:
            progress_events.append({"step": step_index, "kind": "replan", "reason": cycle_reason})
            if state.stuck_counter >= 2 * repeat_limit:
                termination_reason = "agent_policy_failure"
                break
        observation = next_observation
        admissible = list(_scalar(next_infos, "admissible_commands") or [])

    success = None if termination_reason in {
        "policy_error", "environment_error", "provider_budget_exhausted", "verifier_inconclusive"
    } else (won if won is not None else None)
    if not any("reward" in row for row in trajectory) and any(row["invalid_action"] for row in trajectory):
        termination_reason = "api_contract_failure"
        success = None
    status = {"policy_error": "PROVIDER_FAILURE", "environment_error": "ENVIRONMENT_FAILURE",
              "provider_budget_exhausted": "BUDGET_EXHAUSTED", "verifier_inconclusive": "INCONCLUSIVE",
              "api_contract_failure": "API_CONTRACT_FAILURE",
              "agent_policy_failure": "AGENT_POLICY_FAILURE"}.get(
                  termination_reason, "VERIFIED_SUCCESS" if success is True else
                  "VERIFIED_FAILURE" if success is False else "INCONCLUSIVE")
    return {
        **dict(receipt_metadata or {}),
        "schema_version": 2,
        "policy_revision": POLICY_REVISION,
        "split": (receipt_metadata or {}).get("split", "train"),
        "success": success,
        "status": status,
        "environment_won": won,
        "environment_done": done,
        "model_response_identity": (client.response_identity[0] if getattr(client, "response_identity", None)
                                    else (receipt_metadata or {}).get("model_response_identity")),
        "system_fingerprint": (client.response_identity[1] if getattr(client, "response_identity", None) else None),
        "wall_seconds": time.perf_counter() - started_at,
        "provider_calls": len(trajectory),
        "input_tokens": (sum(row["input_tokens"] for row in trajectory)
                         if trajectory and all(type(row.get("input_tokens")) is int for row in trajectory) else None),
        "completion_tokens": (sum(row["output_tokens"] for row in trajectory)
                              if trajectory and all(type(row.get("output_tokens")) is int for row in trajectory) else None),
        **({"gamefile": actual_gamefile} if expected_gamefiles is not None else {}),
        "steps": len(trajectory),
        "env_steps": sum("reward" in row for row in trajectory),
        "skill_version": skill_version,
        "skill_proposed": None,
        "skill_retrieved": None,
        "skill_rendered": skill is not None,
        "skill_injected": skill is not None,
        "skill_read_or_used": "unknown",
        "activation_source": "prompt_rendering" if skill else None,
        "subgoal_followed": None,
        "invalid_actions": sum(int(row["invalid_action"]) for row in trajectory),
        "termination_reason": termination_reason,
        "provider_budget": client.budget.as_dict() if getattr(client, "budget", None) is not None else None,
        "state": asdict(state),
        "progress_events": progress_events,
        "trajectory": trajectory,
    }


def run_smoke(
    data_root: Path,
    output_dir: Path,
    client: Any,
    episodes: int = 3,
    max_steps: int = 60,
    seed: int = 0,
    task_types: Sequence[int] = (3, 4, 5),
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    budget = getattr(client, "budget", None)
    (output_dir / "config.json").write_text(json.dumps({
        "benchmark": "ALFWorld", "split": "train", "policy_revision": POLICY_REVISION,
        "episodes": episodes, "max_steps": max_steps, "seed": seed,
        "task_types": list(task_types), "model": getattr(client, "model", None),
        "temperature": getattr(client, "temperature", None), "thinking": "enabled",
        "max_completion_tokens": getattr(client, "max_completion_tokens", _DEEPSEEK_MAX_TOKENS),
        "provider_budget": budget.as_dict() if budget is not None else None,
        "readiness_plan": dict(plan) if plan is not None else None,
        "run_manifest": getattr(client, "run_manifest", None),
        "approval_scope": budget.approval if isinstance(budget, ApprovedProviderBudget) else None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = _alfworld_config(data_root, task_types, max_steps)
    alfred = AlfredTWEnv(config, train_eval="train")
    if plan is not None:
        from experiments.skill_evolution.native_probe import probe_game_files
        selected = probe_game_files(alfred.game_files, dict(plan))
        if len(selected) != episodes:
            raise ValueError("episode count differs from the frozen readiness plan")
    else:
        selected = _select_game_files(alfred.game_files, episodes, seed)
    alfred.game_files = selected
    alfred.num_games = len(selected)
    environment = alfred.init_env(batch_size=1)
    environment.seed(seed)
    rows = []
    try:
        for _ in range(episodes):
            if isinstance(budget, ApprovedProviderBudget):
                try:
                    budget.ensure_capacity(max_steps, max_steps * client.max_input_per_request,
                                           max_steps * client.max_completion_tokens)
                except ProviderBudgetExceeded:
                    break
            metadata = getattr(client, "receipt_metadata", None)
            episode_kwargs = ({"receipt_metadata": metadata, "expected_gamefiles": selected}
                              if metadata is not None else {})
            row = run_episode(environment, client, max_steps=max_steps, **episode_kwargs)
            if metadata is not None:
                from experiments.skill_evolution.readiness import _train_identity
                template, task_id = _train_identity(row["gamefile"])
                if any(previous.get("task_id") == task_id for previous in rows):
                    raise ValueError("duplicate native reset in the frozen readiness batch")
                planned_task = next(item for item in plan["episodes"] if item["gamefile"] == row["gamefile"])
                row.update(task_id=task_id, template_id=template, cluster_id=planned_task["cluster_id"],
                           task_family=task_family_name(planned_task["task_type"]),
                           trial_id=f"{metadata['run_id']}:{len(rows)}", seed=seed)
                row["receipt_sha256"] = manifest_digest(row)
            rows.append(row)
            (output_dir / "results.json").write_text(json.dumps({
                "benchmark": "ALFWorld", "split": "train", "status": "IN_PROGRESS",
                "episodes": rows, "provider_budget": budget.as_dict() if budget is not None else None,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if rows[-1]["termination_reason"] == "provider_budget_exhausted" or (budget is not None and budget.stopped):
                break
    finally:
        environment.close()
    result = {
        "schema_version": 2,
        "benchmark": "ALFWorld",
        "split": "train",
        "policy_revision": POLICY_REVISION,
        "episodes": rows,
        "scheduled_episodes": episodes,
        "nonlaunched_episodes": episodes - len(rows),
        "provider_budget": budget.as_dict() if budget is not None else None,
        "summary": {
            "episode_count": len(rows),
            "success_rate": (sum(row["success"] is True for row in rows) / len(rows)
                             if rows and all(row["success"] is not None for row in rows) else None),
            "known_successes": sum(row["success"] is True for row in rows),
            "missing_outcomes": sum(row["success"] is None for row in rows),
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "termination_reasons": {
                reason: sum(row["termination_reason"] == reason for row in rows)
                for reason in sorted({row["termination_reason"] for row in rows})
            },
        },
    }
    if getattr(client, "receipt_metadata", None) is not None:
        from experiments.skill_evolution.real_study import viability_gate
        result["viability_gate"] = viability_gate(rows)
        failures_report = ["# Train-side Agent readiness", "", str(result["viability_gate"]), "",
                           "| Family | Planned | Started | Wins | Unknown | Termination |",
                           "|---|---:|---:|---:|---:|---|"]
        for family in sorted({item["task_family"] for item in plan["episodes"]}):
            tasks = [item for item in plan["episodes"] if item["task_family"] == family]
            family_rows = [row for row in rows if row["task_id"] in {item["task_id"] for item in tasks}]
            failures_report.append(f"| {family} | {len(tasks)} | {len(family_rows)} | "
                                   f"{sum(row['success'] is True for row in family_rows)} | "
                                   f"{sum(row['success'] is None for row in family_rows)} | "
                                   f"{', '.join(row['termination_reason'] for row in family_rows)} |")
        (output_dir / "PER_FAMILY_AGENT_FAILURES.md").write_text("\n".join(failures_report) + "\n", encoding="utf-8")
        (output_dir / "REPORT.md").write_text(
            "# Real train readiness\n\n" + str(result["viability_gate"]) +
            "\n\nSee PER_FAMILY_AGENT_FAILURES.md and full trajectories in results.json.\n", encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    from experiments.skill_evolution.real_study import main as study_main
    study_main()


if __name__ == "__main__":
    main()
