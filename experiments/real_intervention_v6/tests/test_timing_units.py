from experiments.real_intervention_v6.benchmark_timing_contract import (
    TimingObservation,
    compare_timing,
)


def observation(*, phase="STEADY", value=0.002, number_per_run=1, unit="s"):
    return TimingObservation(
        phase=phase,
        clock="HOST_MONOTONIC",
        unit=unit,
        value=value,
        calls=1,
        repeats=5,
        number_per_run=number_per_run,
        cache_state="PREFILLED",
        synchronization="BEFORE_AFTER",
        includes_compile=False,
        workload_scope="efficientformer-b1-h128",
    )


def test_benchmark_number_per_run_normalizes_aggregate_time():
    per_call = 0.002

    observations = [
        observation(value=per_call * number, number_per_run=number)
        for number in (1, 10, 100, 400)
    ]

    assert [item.aggregate_seconds for item in observations] == [
        per_call * number for number in (1, 10, 100, 400)
    ]
    assert [item.per_call_seconds for item in observations] == [
        per_call,
        per_call,
        per_call,
        per_call,
    ]


def test_prefill_job_and_steady_per_call_are_not_comparable():
    prefill = observation(phase="PREFILL", value=1.4, number_per_run=1)
    steady = observation(phase="STEADY", value=0.0035, number_per_run=1)

    comparison = compare_timing(prefill, steady)

    assert comparison.status == "CROSSCHECK_NOT_COMPARABLE"
    assert comparison.reasons == ("same_measurement_phase",)
