"""Typed timing observations and rules for comparing like measurements."""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Callable, Literal


TimingPhase = Literal["PREFILL", "CACHE_RELOAD", "FIRST", "STEADY", "JOB_TOTAL"]
TimingClock = Literal["HOST_MONOTONIC", "CUDA_EVENT"]
TimingUnit = Literal["s", "ms"]
CacheState = Literal["EMPTY", "PREFILLED"]
Synchronization = Literal["BEFORE_AFTER", "CUDA_EVENT", "NONE"]


@dataclass(frozen=True)
class TimingObservation:
    """One raw sample; ``calls`` is per statement execution.

    A sample represents ``calls * number_per_run`` invocations. ``repeats``
    counts independent raw samples and does not multiply either duration.
    """

    phase: TimingPhase
    clock: TimingClock
    unit: TimingUnit
    value: float
    calls: int
    repeats: int
    number_per_run: int
    cache_state: CacheState
    synchronization: Synchronization
    includes_compile: bool
    workload_scope: str

    def __post_init__(self) -> None:
        if self.phase not in ("PREFILL", "CACHE_RELOAD", "FIRST", "STEADY", "JOB_TOTAL"):
            raise ValueError(f"unsupported timing phase: {self.phase}")
        if self.clock not in ("HOST_MONOTONIC", "CUDA_EVENT"):
            raise ValueError(f"unsupported timing clock: {self.clock}")
        if self.unit not in ("s", "ms"):
            raise ValueError(f"unsupported timing unit: {self.unit}")
        if self.cache_state not in ("EMPTY", "PREFILLED"):
            raise ValueError(f"unsupported cache state: {self.cache_state}")
        if self.synchronization not in ("BEFORE_AFTER", "CUDA_EVENT", "NONE"):
            raise ValueError(f"unsupported synchronization: {self.synchronization}")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("timing value must be finite and non-negative")
        if self.calls < 1 or self.repeats < 1 or self.number_per_run < 1:
            raise ValueError("calls, repeats, and number_per_run must be positive")
        if not self.workload_scope:
            raise ValueError("workload_scope must identify the measured workload")

    @property
    def aggregate_seconds(self) -> float:
        """Elapsed time represented by one raw sample/batch, in seconds."""
        return self.value if self.unit == "s" else self.value / 1000

    @property
    def per_call_seconds(self) -> float:
        """Seconds per invocation: aggregate / (calls * number_per_run)."""
        return self.aggregate_seconds / (self.calls * self.number_per_run)


@dataclass(frozen=True)
class TimingComparison:
    status: Literal["COMPARABLE", "CROSSCHECK_NOT_COMPARABLE"]
    reasons: tuple[str, ...]


def compare_timing(left: TimingObservation, right: TimingObservation) -> TimingComparison:
    """Check that a per-call cross-check compares the same measurement contract."""
    checks = (
        (left.workload_scope == right.workload_scope, "same_workload_scope"),
        (left.phase == right.phase, "same_measurement_phase"),
        (left.cache_state == right.cache_state, "same_cache_state"),
        (left.unit == right.unit, "same_units"),
        (left.clock == right.clock, "same_clock"),
        (left.synchronization == right.synchronization, "same_synchronization"),
        (left.includes_compile == right.includes_compile, "same_compile_scope"),
    )
    reasons = tuple(name for passed, name in checks if not passed)
    status = "COMPARABLE" if not reasons else "CROSSCHECK_NOT_COMPARABLE"
    return TimingComparison(status, reasons)


@dataclass(frozen=True)
class BenchmarkMicrofixtureRecord:
    clock: TimingClock
    raw_times_seconds: tuple[float, ...]
    median_seconds_per_call: float
    number_per_run: int
    repeats: int
    host_aggregate_seconds: float
    host_per_call_seconds: float
    cuda_event_per_call_seconds: float | None


def run_microfixture(
    workload: Callable[[], object],
    *,
    clock: TimingClock,
    number_per_runs: tuple[int, ...] = (1, 10, 100, 400),
    repeats: int = 5,
) -> tuple[BenchmarkMicrofixtureRecord, ...]:
    """Record benchmark batches and an independently synchronized host sample.

    Pass a CPU workload with ``HOST_MONOTONIC`` or a CUDA workload with
    ``CUDA_EVENT``. This helper performs measurements only when called.
    """
    import torch
    import torch.utils.benchmark as benchmark

    if clock not in ("HOST_MONOTONIC", "CUDA_EVENT"):
        raise ValueError(f"unsupported timing clock: {clock}")
    timer = benchmark.Timer(
        stmt="workload()", globals={"workload": workload}, num_threads=1
    )
    records = []
    for number_per_run in number_per_runs:
        measurements = [timer.timeit(number_per_run) for _ in range(repeats)]
        raw_times = tuple(raw for item in measurements for raw in item.raw_times)

        if clock == "CUDA_EVENT":
            torch.cuda.synchronize()
        host_start = time.perf_counter()
        for _ in range(number_per_run):
            workload()
        if clock == "CUDA_EVENT":
            torch.cuda.synchronize()
        host_aggregate = time.perf_counter() - host_start

        event_per_call = None
        if clock == "CUDA_EVENT":
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(number_per_run)]
            for start_event, end_event in events:
                start_event.record()
                workload()
                end_event.record()
            torch.cuda.synchronize()
            event_per_call = sum(start.elapsed_time(end) / 1000 for start, end in events) / number_per_run

        records.append(BenchmarkMicrofixtureRecord(
            clock=clock,
            raw_times_seconds=raw_times,
            median_seconds_per_call=statistics.median(item.median for item in measurements),
            number_per_run=number_per_run,
            repeats=repeats,
            host_aggregate_seconds=host_aggregate,
            host_per_call_seconds=host_aggregate / number_per_run,
            cuda_event_per_call_seconds=event_per_call,
        ))
    return tuple(records)
