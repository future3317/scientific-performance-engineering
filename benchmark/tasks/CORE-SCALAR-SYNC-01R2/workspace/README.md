# CORE-SCALAR-SYNC-01R2: Per-step scalar synchronization

## Goal
Speed up the tiny MLP training loop while preserving its mathematical output.

## API
Implement the `train_loop_v1` contract in `solution.py`:

- `build_model(fixtures) -> torch.nn.Module`
- `train_step(model, batch, optimizer) -> dict` with at least `'loss'` (tensor) and `'work_units'` (dict).
- `run_training(fixtures, steps) -> dict` with final metrics.

You may only edit files inside this workspace.

## Constraints
- Plain torch only (no PyG, no network).
- Deterministic for a fixed seed.
- The same forward/backward/optimizer work must be performed (work-unit counters checked).
- Final returned metrics must stay within correctness tolerance of the fp64 reference.

## Task
The baseline pays a per-step host-read tax: it calls `.item()` on the loss **and** on every per-parameter-group gradient norm every step, then recomputes running statistics as Python `float`s. Name the bottleneck mechanism in your diagnosis.

## Time budget
Target total runtime for one `run-task` evaluation is well under the 600 s task budget; the hot loop is intentionally small.
