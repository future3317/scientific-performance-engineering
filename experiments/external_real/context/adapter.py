from __future__ import annotations

from typing import Callable, Literal, Mapping, Sequence

from ..models import ExternalTask
from ..protocol import assert_information_flow


class ContextAdapter:
    """Extract policy-visible context without exposing evaluator outcomes."""

    def __init__(
        self,
        *,
        mode: Literal["native_structured", "raw_task_to_typed"],
        public_fields: Sequence[str],
        parser: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        parser_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.mode = mode
        self.public_fields = tuple(public_fields)
        self._parser = parser
        self.parser_metadata = dict(parser_metadata or {})
        if mode == "raw_task_to_typed":
            required = {"model", "version", "prompt", "schema", "revision"}
            if parser is None or not required.issubset(self.parser_metadata):
                raise ValueError("raw_task_to_typed requires frozen parser metadata and parser")

    def extract(self, task: ExternalTask) -> dict[str, object]:
        raw: Mapping[str, object]
        if self.mode == "native_structured":
            raw = task.native_payload
        else:
            assert self._parser is not None
            raw = self._parser(task.native_payload)
        if not isinstance(raw, Mapping):
            raise ValueError("context parser must return an object")
        assert_information_flow(raw)
        return {field: raw[field] for field in self.public_fields if field in raw}
