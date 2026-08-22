# CORE-H2D-PIPELINE-03R2 — blocking host-to-device data pipeline

## Goal

`solution.py` implements a tiny training loop over an in-memory synthetic
dataset. The measured DataLoader uses the declared batch size, worker count,
prefetch factor, and pinned-memory setting. The baseline remains correct but
wastes time moving batches from host to device. Make the hot `train_step` faster while keeping
the trained model numerically equivalent to the baseline (verified against an
fp64 reference on fresh inputs you do not see).

## API (fixed — `train_loop_v1`)

- `build_model(fixtures) -> torch.nn.Module`
- `train_step(model, batch, optimizer) -> dict` with at least `"loss"` (tensor)
  and `"work_units"` (dict of counters).
- `run_training(fixtures, steps: int) -> dict` with final metrics.

`train_step` is the hot loop body that the harness times.

## Constraints

- stdlib + torch only. No PyG, no network, no file I/O outside the workspace.
- The same forward/backward/optimizer work units must happen on every step
  (the harness compares counters).
- Do not touch timing, randomness, or torch internals; do not spawn threads or
  processes explicitly.
- The device is CUDA for this task; CI hosts without CUDA degrade to
  `inconclusive`, not failure.

## Time budget

300 s total for your session; a single verification run finishes in under a
minute.

## Required diagnosis

Before you finish, state the **bottleneck mechanism** you identified (one of
the choices in `task.yaml`) and justify it with evidence. The mechanism name is
scored.

## Smoke test

```
python public_tests/smoke_test.py
```

runs your workspace `solution.py` on a tiny fixture and checks shapes,
finiteness, and API conformance.
