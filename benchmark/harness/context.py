"""Explicit task-context lifecycle for reset/carry evaluations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class TaskContext:
    def __init__(self, mode: str = "reset") -> None:
        if mode not in {"reset", "carry"}:
            raise ValueError("context mode must be reset or carry")
        self.mode = mode
        self._state: dict[str, Any] = {}

    def begin_task(self, task_id: str) -> dict[str, Any]:
        if self.mode == "reset":
            self._state = {}
        self._state["task_id"] = task_id
        return deepcopy(self._state)

    def record(self, key: str, value: Any) -> None:
        self._state[key] = deepcopy(value)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)
