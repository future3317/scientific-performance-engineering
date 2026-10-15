"""Resource accounting that reserves already scheduled execution work.

This ledger is deliberately independent from the statistical alpha budget.
It answers one operational question: may an optional evolution query consume
the portion of compute reserved for future scheduled cells?
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import time
from typing import Any, Mapping


def manifest_digest(value: Mapping[str, Any]) -> str:
    """Identity of a frozen run/split, excluding its optional self reference."""
    payload = {k: v for k, v in value.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ProviderBudgetExceeded(RuntimeError):
    """A provider request cannot be admitted under the frozen usage budget."""


class ProviderUsageBudget:
    """Call and total-token caps with one in-flight request reservation.

    ``max_total_tokens`` counts provider-reported input plus completion tokens.
    An unknown usage result freezes the budget because the remaining allowance
    can no longer be established.
    """

    def __init__(self, max_calls: int, max_total_tokens: int) -> None:
        if max_calls < 0 or max_total_tokens < 0:
            raise ValueError("provider usage caps must be non-negative")
        self.max_calls = int(max_calls)
        self.max_total_tokens = int(max_total_tokens)
        self.calls_reserved = 0
        self.tokens_used = 0
        self.tokens_unknown = False
        self.bound_violation = False
        self.stop_reason: str | None = None
        self._reservation: dict[str, int | str] | None = None
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def reserve(self, input_token_bound: int, max_completion_tokens: int) -> None:
        """Reserve a complete request; never lower its requested output cap."""
        if input_token_bound < 0 or max_completion_tokens < 0:
            raise ValueError("provider token reservations must be non-negative")
        if self._stopped:
            raise ProviderBudgetExceeded("provider_usage_unknown_or_over_budget")
        if self._reservation is not None:
            raise RuntimeError("a provider request reservation is already in flight")
        if self.calls_reserved >= self.max_calls:
            raise ProviderBudgetExceeded("provider_call_cap")
        required = int(input_token_bound) + int(max_completion_tokens)
        if self.tokens_used + required > self.max_total_tokens:
            raise ProviderBudgetExceeded("provider_token_cap")
        self.calls_reserved += 1
        self._reservation = {
            "input_token_bound": int(input_token_bound),
            "max_completion_tokens": int(max_completion_tokens),
            "total_token_reserve": required,
            "status": "in_flight",
        }

    def record_usage(self, input_tokens: int, completion_tokens: int) -> None:
        """Commit actual provider usage and release the request reservation."""
        if self._reservation is None or self._reservation["status"] != "in_flight":
            raise RuntimeError("no provider request reservation is in flight")
        if input_tokens < 0 or completion_tokens < 0:
            self.mark_unknown()
            return
        self.tokens_used += int(input_tokens) + int(completion_tokens)
        input_bound = int(self._reservation["input_token_bound"])
        self._reservation = None
        if input_tokens > input_bound:
            self.bound_violation = True
            self._stopped = True
            self.stop_reason = "input_token_bound_violation"
        elif self.tokens_used > self.max_total_tokens:
            self._stopped = True
            self.stop_reason = "provider_token_cap_exceeded_by_actual_usage"

    def mark_unknown(self, reason: str = "provider_usage_unknown") -> None:
        """Keep the reservation charged and stop all later provider calls."""
        if self._reservation is not None:
            self._reservation["status"] = "usage_unknown"
        self.tokens_unknown = True
        self._stopped = True
        self.stop_reason = reason

    def stop(self, reason: str) -> None:
        """Stop a run whose identity/contract changed without inventing charges."""
        self._stopped = True
        self.stop_reason = reason

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable accounting snapshot for receipts."""
        return {
            "max_calls": self.max_calls,
            "max_total_tokens": self.max_total_tokens,
            "calls_reserved": self.calls_reserved,
            "tokens_used_known": self.tokens_used,
            "tokens_unknown": self.tokens_unknown,
            "reservations": None if self._reservation is None else dict(self._reservation),
            "bound_violation": self.bound_violation,
            "stopped": self._stopped,
            "stop_reason": self.stop_reason,
        }


class ApprovedProviderBudget(ProviderUsageBudget):
    """One run's explicitly approved call/token/USD/time envelope.

    Approval is supplied by the human operator, never inferred from an API key.
    Prices are an attributed snapshot, not a claim about the provider's bill.
    Cache discounts are charged only when both counters reconcile with input.
    """

    def __init__(self, manifest: Mapping[str, Any], approval: Mapping[str, Any],
                 pricing: Mapping[str, Any], *, controller_commit: str,
                 stage: str, model: str, base_url: str) -> None:
        if approval.get("provider_enabled") is not True:
            raise ProviderBudgetExceeded("APPROVAL_REQUIRED")
        for key, expected in (("controller_commit", controller_commit), ("model", model),
                              ("study_id", manifest.get("study_id")),
                              ("run_id", manifest.get("run_id")),
                              ("approved_manifest_sha256", manifest_digest(manifest)),
                              ("approved_split_manifest_sha256", manifest.get("split_manifest_sha256"))):
            if not expected or approval.get(key) != expected:
                raise ProviderBudgetExceeded(f"approval_scope_mismatch:{key}")
        if stage not in approval.get("permitted_stages", []) or stage != manifest.get("stage"):
            raise ProviderBudgetExceeded("approval_stage_mismatch")
        if base_url.rstrip("/") not in approval.get("api_base_allowlist", []):
            raise ProviderBudgetExceeded("approval_api_base_mismatch")
        for key in ("approved_by", "approval_reference", "approved_at_utc", "expiration_utc"):
            if not approval.get(key):
                raise ProviderBudgetExceeded(f"approval_missing:{key}")
        self.expiration = datetime.fromisoformat(str(approval["expiration_utc"]).replace("Z", "+00:00"))
        if self.expiration.tzinfo is None or self.expiration <= datetime.now(timezone.utc):
            raise ProviderBudgetExceeded("approval_expired")
        for key in ("max_provider_calls", "max_input_tokens", "max_completion_tokens",
                    "per_episode_max_steps", "per_request_max_completion_tokens", "max_wall_seconds"):
            if type(approval.get(key)) is not int or approval[key] <= 0:
                raise ProviderBudgetExceeded(f"approval_invalid_cap:{key}")
        if approval.get("max_failed_retries") != 0:
            raise ProviderBudgetExceeded("only_zero_retries_supported")
        if pricing.get("currency") != "USD" or pricing.get("model") != model:
            raise ValueError("pricing snapshot must match model and USD")
        for key in ("pricing_source_url", "checked_at_utc", "price_period", "estimate_quality"):
            if not pricing.get(key):
                raise ValueError(f"pricing snapshot missing {key}")
        self.rates = {key: Decimal(str(pricing[key])) for key in
                      ("input_hit_per_million", "input_miss_per_million", "output_per_million")}
        if any(not rate.is_finite() or rate < 0 for rate in self.rates.values()):
            raise ValueError("pricing rates must be finite and nonnegative")
        self.max_usd = Decimal(str(approval.get("max_total_usd", 0)))
        if not self.max_usd.is_finite() or self.max_usd <= 0:
            raise ProviderBudgetExceeded("approval_invalid_cap:max_total_usd")
        self.approval, self.pricing = dict(approval), dict(pricing)
        self.started = time.monotonic()
        self.input_used = self.output_used = 0
        self.usd_used = Decimal(0)
        self.pending_usd = Decimal(0)
        super().__init__(approval["max_provider_calls"],
                         approval["max_input_tokens"] + approval["max_completion_tokens"])

    def _cost(self, inputs: int, outputs: int, hits: int = 0) -> Decimal:
        return ((inputs - hits) * self.rates["input_miss_per_million"]
                + hits * self.rates["input_hit_per_million"]
                + outputs * self.rates["output_per_million"]) / Decimal(1_000_000)

    @property
    def remaining_wall_seconds(self) -> float:
        return min(self.approval["max_wall_seconds"] - (time.monotonic() - self.started),
                   (self.expiration - datetime.now(timezone.utc)).total_seconds())

    def ensure_capacity(self, calls: int, input_tokens: int, completion_tokens: int) -> None:
        """Admit a whole episode/pair before starting its first arm."""
        if self.stopped or datetime.now(timezone.utc) >= self.expiration:
            raise ProviderBudgetExceeded("approval_expired_or_budget_stopped")
        if time.monotonic() - self.started >= self.approval["max_wall_seconds"]:
            raise ProviderBudgetExceeded("provider_wall_cap")
        if self.calls_reserved + calls > self.max_calls:
            raise ProviderBudgetExceeded("provider_call_cap")
        if self.input_used + input_tokens > self.approval["max_input_tokens"]:
            raise ProviderBudgetExceeded("provider_input_cap")
        if self.output_used + completion_tokens > self.approval["max_completion_tokens"]:
            raise ProviderBudgetExceeded("provider_completion_cap")
        worst_input_rate = max(self.rates["input_hit_per_million"], self.rates["input_miss_per_million"])
        cost = (input_tokens * worst_input_rate + completion_tokens * self.rates["output_per_million"]) / Decimal(1_000_000)
        if self.usd_used + cost > self.max_usd:
            raise ProviderBudgetExceeded("provider_usd_cap")

    def reserve(self, input_token_bound: int, max_completion_tokens: int) -> None:
        if max_completion_tokens > self.approval["per_request_max_completion_tokens"]:
            raise ProviderBudgetExceeded("per_request_completion_cap")
        self.ensure_capacity(1, input_token_bound, max_completion_tokens)
        super().reserve(input_token_bound, max_completion_tokens)
        self.pending_usd = max(self._cost(input_token_bound, max_completion_tokens),
                               self._cost(input_token_bound, max_completion_tokens, input_token_bound))

    def record_usage(self, input_tokens: int, completion_tokens: int, *,
                     cache_hit_tokens: int | None = None, cache_miss_tokens: int | None = None) -> None:
        reservation = self._reservation
        if reservation is None:
            raise RuntimeError("no provider request reservation is in flight")
        if type(input_tokens) is not int or type(completion_tokens) is not int or min(input_tokens, completion_tokens) < 0:
            self.mark_unknown()
            return
        reconciled = (type(cache_hit_tokens) is int and type(cache_miss_tokens) is int
                      and min(cache_hit_tokens, cache_miss_tokens) >= 0
                      and cache_hit_tokens + cache_miss_tokens == input_tokens)
        charge = (self._cost(input_tokens, completion_tokens, cache_hit_tokens) if reconciled
                  else max(self._cost(input_tokens, completion_tokens),
                           self._cost(input_tokens, completion_tokens, input_tokens)))
        self.input_used += input_tokens
        self.output_used += completion_tokens
        self.usd_used += charge
        self.pending_usd = Decimal(0)
        output_bound = int(reservation["max_completion_tokens"])
        super().record_usage(input_tokens, completion_tokens)
        if completion_tokens > output_bound or self.usd_used > self.max_usd:
            self._stopped = True
            self.stop_reason = "provider_actual_usage_exceeds_reservation"

    def as_dict(self) -> dict[str, object]:
        return {**super().as_dict(), "input_tokens_known": self.input_used,
                "completion_tokens_known": self.output_used,
                "estimated_usd_known": str(self.usd_used),
                "reserved_usd": str(self.pending_usd), "max_total_usd": str(self.max_usd),
                "billing_status": "BILLING_UNKNOWN" if self.tokens_unknown else "ESTIMATE_NOT_BILL",
                "approval_reference": self.approval["approval_reference"], "pricing": self.pricing}


@dataclass(frozen=True)
class EvolutionAdmission:
    accepted: bool
    reason: str
    query_id: str
    cost: float
    evolution_available_before: float
    evolution_available_after: float


@dataclass(frozen=True)
class ExecutionReserveBudget:
    """Immutable, replayable ledger for optional evolution compute."""

    total_budget: float
    scheduled_minimums: tuple[float, ...]
    evolution_spent: float = 0.0
    task_index: int = 0
    evolution_frozen: bool = False

    def __post_init__(self) -> None:
        if self.total_budget < 0.0:
            raise ValueError("total_budget must be non-negative")
        if any(float(value) < 0.0 for value in self.scheduled_minimums):
            raise ValueError("scheduled execution minima must be non-negative")
        if not 0 <= self.task_index <= len(self.scheduled_minimums):
            raise ValueError("task_index must be within the scheduled task range")
        if self.evolution_spent < 0.0:
            raise ValueError("evolution_spent must be non-negative")
        if self.evolution_spent + self.execution_reserve > self.total_budget + 1e-12:
            raise ValueError("scheduled execution reserve exceeds total budget")

    @property
    def execution_reserve(self) -> float:
        return float(sum(self.scheduled_minimums[self.task_index:]))

    @property
    def evolution_available(self) -> float:
        return max(0.0, float(self.total_budget - self.evolution_spent - self.execution_reserve))

    @property
    def scheduled_execution_allowed(self) -> bool:
        """A frozen evolution ledger never blocks the scheduled task path."""
        return self.execution_reserve <= self.total_budget - self.evolution_spent + 1e-12

    def admit_evolution(self, cost: float, *, query_id: str) -> tuple["ExecutionReserveBudget", EvolutionAdmission]:
        if cost < 0.0 or not query_id:
            raise ValueError("evolution cost must be non-negative and query_id must be non-empty")
        before = self.evolution_available
        if self.evolution_frozen:
            return self, EvolutionAdmission(False, "evolution_frozen", query_id, float(cost), before, before)
        if float(cost) > before + 1e-12:
            frozen = replace(self, evolution_frozen=True)
            return frozen, EvolutionAdmission(False, "execution_reserve", query_id, float(cost), before, before)
        updated = replace(self, evolution_spent=self.evolution_spent + float(cost))
        if updated.evolution_available <= 1e-12:
            updated = replace(updated, evolution_frozen=True)
        return updated, EvolutionAdmission(True, "admitted", query_id, float(cost), before, updated.evolution_available)

    def advance_task(self, minimum_cost: float | None = None) -> "ExecutionReserveBudget":
        """Consume the current scheduled minimum and release its reservation."""
        if self.task_index >= len(self.scheduled_minimums):
            raise IndexError("no scheduled task remains")
        expected = self.scheduled_minimums[self.task_index]
        if minimum_cost is not None and abs(float(minimum_cost) - expected) > 1e-12:
            raise ValueError("minimum_cost does not match the frozen schedule")
        return replace(self, task_index=self.task_index + 1)

    def freeze_evolution(self) -> "ExecutionReserveBudget":
        return replace(self, evolution_frozen=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_budget": self.total_budget,
            "scheduled_minimums": list(self.scheduled_minimums),
            "evolution_spent": self.evolution_spent,
            "task_index": self.task_index,
            "execution_reserve": self.execution_reserve,
            "evolution_available": self.evolution_available,
            "evolution_frozen": self.evolution_frozen,
            "scheduled_execution_allowed": self.scheduled_execution_allowed,
        }


@dataclass(frozen=True)
class PairReservation:
    """Separate task/evolution caps with an all-or-none paired-task reserve.

    The task and evolution caps use their respective caller-defined units.
    ``measured_pair_costs`` records actual task usage when known; ``None``
    remains explicitly unknown.
    """

    task_execution_budget: float
    evolution_search_budget: float
    pair_costs: tuple[tuple[str, float], ...]
    reserved: bool
    reason: str
    evolution_spent: float = 0.0
    completed_pairs: tuple[str, ...] = ()
    measured_pair_costs: tuple[tuple[str, float | None], ...] = ()

    def __post_init__(self) -> None:
        if self.task_execution_budget < 0.0 or self.evolution_search_budget < 0.0:
            raise ValueError("task and evolution budgets must be non-negative")
        ids = [pair_id for pair_id, _ in self.pair_costs]
        id_set = set(ids)
        if len(ids) != len(id_set) or any(not pair_id for pair_id in ids):
            raise ValueError("pair ids must be unique and non-empty")
        if any(cost < 0.0 for _, cost in self.pair_costs):
            raise ValueError("pair cost reservations must be non-negative")
        if self.reserved and sum(cost for _, cost in self.pair_costs) > self.task_execution_budget + 1e-12:
            raise ValueError("reserved pairs exceed task_execution_budget")
        if self.evolution_spent < 0.0 or self.evolution_spent > self.evolution_search_budget + 1e-12:
            raise ValueError("evolution spending exceeds its independent budget")
        if any(pair_id not in id_set for pair_id in self.completed_pairs):
            raise ValueError("completed pair is not in the frozen reservation")
        measured_ids = [pair_id for pair_id, _ in self.measured_pair_costs]
        if len(measured_ids) != len(set(measured_ids)) or any(pair_id not in id_set for pair_id in measured_ids):
            raise ValueError("measured pair costs must reference unique reserved pairs")
        if any(cost is not None and cost < 0.0 for _, cost in self.measured_pair_costs):
            raise ValueError("measured pair costs must be non-negative or unknown")

    @classmethod
    def reserve_all_pairs(
        cls,
        pair_costs: tuple[tuple[str, float], ...],
        *,
        task_execution_budget: float,
        evolution_search_budget: float,
    ) -> "PairReservation":
        """Reserve every matched pair, or reserve none of them."""
        if task_execution_budget < 0.0 or evolution_search_budget < 0.0:
            raise ValueError("task and evolution budgets must be non-negative")
        required = sum(cost for _, cost in pair_costs)
        fits = required <= task_execution_budget + 1e-12
        return cls(
            task_execution_budget=float(task_execution_budget),
            evolution_search_budget=float(evolution_search_budget),
            pair_costs=tuple((pair_id, float(cost)) for pair_id, cost in pair_costs),
            reserved=fits,
            reason="reserved" if fits else "task_execution_budget_insufficient",
        )

    @property
    def evolution_available(self) -> float:
        return max(0.0, self.evolution_search_budget - self.evolution_spent)

    @property
    def launch_allowed(self) -> bool:
        return self.reserved

    def record_evolution(self, cost: float) -> "PairReservation":
        if cost < 0.0:
            raise ValueError("evolution cost must be non-negative")
        if cost > self.evolution_available + 1e-12:
            raise ValueError("evolution_search_budget_insufficient")
        return replace(self, evolution_spent=self.evolution_spent + float(cost))

    def complete_pair(self, pair_id: str, measured_cost: float | None) -> "PairReservation":
        if not self.reserved:
            raise ValueError("cannot complete a pair before all pairs are reserved")
        pair_ids = {item for item, _ in self.pair_costs}
        if pair_id not in pair_ids:
            raise ValueError("pair_id is not in the frozen reservation")
        if pair_id in self.completed_pairs:
            raise ValueError("pair is already complete")
        if measured_cost is not None and measured_cost < 0.0:
            raise ValueError("measured_cost must be non-negative or unknown")
        return replace(
            self,
            completed_pairs=self.completed_pairs + (pair_id,),
            measured_pair_costs=self.measured_pair_costs + ((pair_id, measured_cost),),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "task_execution_budget": self.task_execution_budget,
            "evolution_search_budget": self.evolution_search_budget,
            "pair_costs": dict(self.pair_costs),
            "reserved": self.reserved,
            "reason": self.reason,
            "evolution_spent": self.evolution_spent,
            "evolution_available": self.evolution_available,
            "completed_pairs": list(self.completed_pairs),
            "measured_pair_costs": dict(self.measured_pair_costs),
        }


__all__ = [
    "EvolutionAdmission",
    "ExecutionReserveBudget",
    "PairReservation",
    "ProviderBudgetExceeded",
    "ProviderUsageBudget",
    "ApprovedProviderBudget",
    "manifest_digest",
]
