# CORE-REPEATED-BACKBONE-02: Repeated backbone computation

## Goal
Speed up the multi-head regression loop while preserving its mathematical output, including under a regime where the "fixed" input actually changes every step.

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
- Final returned metrics must stay within correctness tolerance of the fp64 reference in BOTH the `fixed` and `changing` regimes.

## Task
The baseline recomputes the embedding projection of a fixed input batch for every head and recomputes the shared backbone output separately for every head each step. Name the bottleneck mechanism in your diagnosis. Be careful: a naive unconditional cache of the fixed-input projection will pass the `fixed` regime but fail when the inputs change per step.

## Time budget
Target total runtime for one `run-task` evaluation is well under the 600 s task budget.
