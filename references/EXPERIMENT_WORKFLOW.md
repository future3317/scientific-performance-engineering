# Performance experiment workflow

This workflow governs performance candidates without changing the ACRE or
benchmark semantics.

## Candidate isolation

Each candidate is developed in its own branch/worktree and contains one
declared intervention. A failed candidate worktree is discarded. A successful
candidate is integrated only after its evidence is reviewed; the integration
commit is not itself benchmark evidence.

## Test-first intervention

Before changing the candidate, add the smallest reproducer for the claimed
hotspot and run it against the reference implementation. Record the reference
result, input shape/dtype/layout, forward/backward case, and expected failure or
baseline behavior. The candidate may not change the test to make it pass.

## Fresh-context review

The implementer does not self-certify a candidate when a fresh-context review
is available. The reviewer receives only the contract, diff, tests, raw
benchmark record, and profiler evidence. It checks benchmark validity, hidden
synchronization, work mismatch, silent fallback, and numerical regression.

## GPU comparability

Record GPU power limit, average power, SM and memory clocks, P-state,
temperature, throttle reason, MIG mode, persistence mode, and competing GPU
processes for matched baseline/candidate windows. Material clock, power, or
thermal differences make the comparison inconclusive.

## Profilers and sanitizers

Profiler traces diagnose causes; they do not decide speedup. Speedup is decided
only by matched unprofiled benchmark windows. For new CUDA/C++/Triton kernels,
run `scripts/run_cuda_sanitizer.py` with `memcheck`; add `racecheck`,
`initcheck`, or `synccheck` when the kernel uses shared/async memory or
explicit synchronization. A missing sanitizer is a blocked gate, not a pass.

## Clean production run

After evidence is accepted, rerun the benchmark without profiler/debug
instrumentation. Preserve correctness and rollback evidence, remove temporary
debug hooks, and keep the accepted change as one small, reviewable commit.
