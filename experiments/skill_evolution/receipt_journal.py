"""Durable, single-user journal for provider intents and raw content receipts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import fcntl
from typing import Any, Mapping

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DurableProviderJournal:
    """Append-only intent log plus atomic per-call content receipts.

    The journal is deliberately opt-in and intended for the Linux server spool.
    An intent without a durable response is permanently uncertain for that call ID.
    """

    def __init__(self, server_spool_directory: str | os.PathLike[str], *, run_identity: str | None = None) -> None:
        if os.name == "nt":
            raise RuntimeError("provider receipt spool must be configured on the server")
        root = Path(server_spool_directory)
        if not root.is_absolute() or root.resolve() == Path(root.anchor):
            raise ValueError("receipt spool must be an absolute non-root server path")
        self.root = root.resolve()
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / "active.lock").open("a")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError("another invocation owns the provider spool") from None
        self.events_path = self.root / "journal.jsonl"
        self._restored_budget_objects: set[Any] = set()
        events = self.events()
        self._usage_ledger_calls = {event["call_id"] for event in events
                                    if event.get("event") == "usage_ledger"}
        if not any(event.get("event") == "campaign_started" for event in events):
            self._append({"event": "campaign_started", "started_at_utc": _utc_now(), "run_identity": run_identity})
        elif next(e for e in events if e.get("event") == "campaign_started").get("run_identity") != run_identity:
            self.close()
            raise RuntimeError("spool belongs to a different frozen manifest")
        self.response_identity: tuple[Any, Any] | None = None
        self.response_ids: set[str] = set()

    def close(self) -> None:
        self._lock_file.close()

    def _append(self, event: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(event), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False) + "\n"
        existed = self.events_path.exists()
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if not existed:
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        data = self.events_path.read_bytes()
        if data and not data.endswith(b"\n"):
            # A killed/failed append may leave only a tail fragment. The atomic
            # response file, not the response/usage index event, is authoritative.
            boundary = data.rfind(b"\n") + 1
            with (self.root / "journal.incomplete-tail").open("ab") as saved:
                saved.write(data[boundary:]); saved.flush(); os.fsync(saved.fileno())
            with self.events_path.open("r+b") as handle:
                handle.truncate(boundary); handle.flush(); os.fsync(handle.fileno())
            data = data[:boundary]
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    def _receipt_path(self, call_id: str) -> Path:
        return self.receipts / f"{_digest(call_id)}.json"

    def _existing(self, call_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        intent = next((event for event in self.events()
                       if event.get("event") == "intent" and event.get("call_id") == call_id), None)
        receipt_path = self._receipt_path(call_id)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
        return intent, receipt

    def reserve_intent(self, *, call_id: str, episode_id: str, arm: str,
                       skill_version: str | None, turn: int, prompt_identity: str,
                       configuration_identity: str, input_token_bound: int,
                       completion_token_bound: int) -> dict[str, Any] | None:
        intent, receipt = self._existing(call_id)
        identity = {"call_id": call_id, "episode_id": episode_id, "arm": arm,
                    "skill_version": skill_version, "turn": turn,
                    "prompt_identity": prompt_identity,
                    "configuration_identity": configuration_identity}
        if intent is not None:
            if any(intent.get(key) != value for key, value in identity.items()):
                raise RuntimeError("durable call identity changed; refusing provider call")
            if receipt is None:
                raise RuntimeError("provider call has an unresolved durable intent; refusing duplicate call")
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise RuntimeError("saved provider receipt identity mismatch")
            return receipt
        if receipt is not None:
            raise RuntimeError("provider receipt exists without its intent")
        for previous in self.events():
            if (previous.get("event") == "intent"
                    and not self._receipt_path(previous["call_id"]).exists()):
                raise RuntimeError("unresolved provider intent blocks further calls")
        self._append({"event": "intent", **identity,
                      "input_token_bound": input_token_bound,
                      "completion_token_bound": completion_token_bound,
                      "created_at_utc": _utc_now()})
        return None

    def existing_response(self, *, call_id: str, episode_id: str, arm: str,
                          skill_version: str | None, turn: int,
                          prompt_identity: str, configuration_identity: str) -> dict[str, Any] | None:
        intent, receipt = self._existing(call_id)
        if intent is None:
            if receipt is not None:
                raise RuntimeError("provider receipt exists without its intent")
            return None
        identity = {"call_id": call_id, "episode_id": episode_id, "arm": arm,
                    "skill_version": skill_version, "turn": turn,
                    "prompt_identity": prompt_identity,
                    "configuration_identity": configuration_identity}
        if any(intent.get(key) != value for key, value in identity.items()):
            raise RuntimeError("durable call identity changed; refusing provider call")
        if receipt is None:
            raise RuntimeError("provider call has an unresolved durable intent; refusing duplicate call")
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError("saved provider receipt identity mismatch")
        return receipt

    def save_response(self, receipt: Mapping[str, Any]) -> None:
        call_id = str(receipt["call_id"])
        path = self._receipt_path(call_id)
        if path.exists():
            raise RuntimeError("raw provider receipt already exists")
        payload = json.dumps(dict(receipt), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=self.receipts)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(self.receipts, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        self._append({"event": "response", "call_id": call_id,
                      "receipt_file": path.name, "saved_at_utc": _utc_now()})

    def restore_budget(self, budget: Any) -> int:
        """Restore saved response usage once into a fresh existing budget object."""
        if budget in self._restored_budget_objects:
            return 0
        if budget.calls_reserved or budget.tokens_used or getattr(budget, "input_used", 0) or getattr(budget, "output_used", 0):
            raise ValueError("restore_budget requires a fresh budget instance")
        events = self.events()
        intents: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.get("event") == "intent":
                intents[event["call_id"]] = event
        receipts: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for call_id, intent in intents.items():
            path = self._receipt_path(call_id)
            if not path.exists():
                raise RuntimeError(f"unresolved provider intent blocks resume: {call_id}")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if any(receipt.get(key) != intent.get(key) for key in (
                "call_id", "episode_id", "arm", "skill_version", "turn", "prompt_identity", "configuration_identity"
            )):
                raise RuntimeError("provider receipt call identity mismatch")
            response_id = receipt.get("response_id")
            identity = receipt.get("model"), receipt.get("system_fingerprint")
            if response_id is not None and response_id in self.response_ids:
                raise RuntimeError("duplicate provider response identity in durable receipts")
            if self.response_identity is not None and identity != self.response_identity:
                raise RuntimeError("provider identity drift in durable receipts")
            self.response_identity = identity
            if response_id is not None:
                self.response_ids.add(response_id)
            receipts.append((intent, receipt))

        campaign_start = next(event["started_at_utc"] for event in events
                              if event.get("event") == "campaign_started")
        started_utc = datetime.fromisoformat(campaign_start)
        if started_utc.tzinfo is None:
            raise RuntimeError("campaign start timestamp has no timezone")
        budget.started = time.monotonic() - max(0.0, (datetime.now(timezone.utc) - started_utc).total_seconds())
        restored = 0
        for intent, receipt in receipts:
            inputs, outputs = receipt.get("input_tokens"), receipt.get("completion_tokens")
            if type(inputs) is int and type(outputs) is int:
                budget.reserve(int(intent["input_token_bound"]), int(intent["completion_token_bound"]))
                budget.record_usage(inputs, outputs,
                                    cache_hit_tokens=receipt.get("cache_hit_tokens"),
                                    cache_miss_tokens=receipt.get("cache_miss_tokens"))
            else:
                budget.reserve(int(intent["input_token_bound"]), int(intent["completion_token_bound"]))
                budget.mark_unknown("provider_usage_unknown_recovered_from_receipt")
                self._record_usage_event(receipt)
                break
            self._record_usage_event(receipt)
            restored += 1
        self._restored_budget_objects.add(budget)
        return restored

    def mark_budget_restored(self, budget: Any, receipt: Mapping[str, Any]) -> None:
        """Mark usage already accounted by the live request path."""
        self._record_usage_event(receipt)
        self._restored_budget_objects.add(budget)

    def _record_usage_event(self, receipt: Mapping[str, Any]) -> None:
        call_id = receipt["call_id"]
        if call_id in self._usage_ledger_calls:
            return
        self._append({"event": "usage_ledger", "call_id": call_id,
                      "input_tokens": receipt.get("input_tokens"),
                      "completion_tokens": receipt.get("completion_tokens"),
                      "cache_hit_tokens": receipt.get("cache_hit_tokens"),
                      "cache_miss_tokens": receipt.get("cache_miss_tokens"),
                      "recorded_at_utc": _utc_now()})
        self._usage_ledger_calls.add(call_id)

    def budget_was_restored(self, budget: Any) -> bool:
        return budget in self._restored_budget_objects

