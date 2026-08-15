# CORE-COMPILE-DYNAMIC-11 — torch.compile graph break + dynamic-shape recompilation

## Goal

`solution.py` implements a tiny training loop wrapped in `torch.compile`. The
baseline contains a Python control-flow graph break (an `.item()`-dependent
branch) and the measurement loop cycles through 8 different batch sizes. Each
new shape triggers a dynamo recompilation, and the `.item()` call synchronizes
the host. Make the hot `train_step` measurably faster while keeping the trained
model numerically equivalent to the baseline (verified against an fp64 reference
on fresh inputs you do not see).

## API (fixed — `train_loop_v1`)

- `build_model(fixtures) -> torch.nn.Module`
- `train_step(model, batch, optimizer) -> dict` with at least `"loss"` (tensor)
  and `"work_units"` (dict of counters).
- `run_training(fixtures, steps: int) -> dict` with final metrics.

`train_step` is the hot loop body that the harness times.

## Constraints

- stdlib + torch only. No PyG, no network, no file I/O outside the workspace.
- The same forward/backward/optimizer work units must happen on every step.
- Do not touch timing, randomness, or torch internals; do not spawn threads or
  processes explicitly.
- `torch.compile` warmup is allowed inside `build_model`/`run_training`; the
  task is sized so that the warmup stays within the time budget on the target
  host.

## Time budget

300 s total for your session; a single verification run finishes in well under
a minute.

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
