# CORE-KERNEL-FUSION-09R2 — launch-fragmented pointwise chain

## Goal

`solution.py` implements a pointwise tensor chain over a moderate fp32 tensor
(`[1024, 4096]`, ~16 MB):

```
z   = a1 * x + b1                  # affine scale/shift
h   = SiLU(z) = z * sigmoid(z)
y   = h + residual                 # residual add
yc  = clamp(y, clamp_min, clamp_max)
out = a2 * yc + b2                 # second affine
```

Make `forward` **measurably faster** while producing the *same* numerical result
(verified against an fp64 reference on fresh random inputs you do not see).

## API (fixed — `kernel_module_v1`)

- `init(fixtures) -> context` — build any state you need from the fixtures
  (scalar parameters, device). Called once per measurement run.
- `forward(context, x, residual) -> torch.Tensor` — the hot path under
  measurement. Same shapes/dtypes every call; inputs change between verdicts.

## Constraints

- stdlib + torch only. No triton, no PyG, no network, no file I/O outside the
  workspace.
- The numerical semantics must not change: the same chain, same clamp bounds,
  applied to the actual inputs of every call (results cached across calls are
  detected by adversarial probes and score 0).
- Do not touch timing, randomness, or torch internals; do not spawn threads or
  processes.
- `forward` is timed with host wall clock (plus CUDA events on GPU hosts) over
  repeated calls; a warmup budget is provided, so one-time setup inside `init`
  (e.g. compilation) is allowed and encouraged if it pays for itself.

## Time budget

600 s total for your session; each verification run of the task finishes in
well under a minute.

## Required diagnosis

Before you finish, state the **bottleneck mechanism** you identified (one of:
`launch_fragmentation`, `scalar_sync`, `h2d_blocking`, `repeated_compute`,
`compile_break`, `memory_pressure`) and justify it with the evidence you
collected. The mechanism name is scored.

## Smoke test

```
python public_tests/smoke_test.py
```

runs your workspace `solution.py` on a tiny fixture and checks the output
shape/finiteness and agreement with a plain eager recomputation.


