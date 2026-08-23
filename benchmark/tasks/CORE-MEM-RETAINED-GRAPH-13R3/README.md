# CORE-MEM-RETAINED-GRAPH-13R3 — long-horizon retained autograd history

## Goal

The baseline runs a segmented residual training step with 12 segments and
retains each segment output together with its autograd history across steps;
its redundant `retain_graph=True` makes that lifetime explicit.
The oracle clears that cross-step history and stores detached diagnostics only.
The forward, backward, checkpoint, and optimizer work remain the same.

The benchmark is intended to expose memory pressure and retained-graph
overhead over a 240-iteration measured horizon. It is not a correctness change
to the model or objective.

## API (fixed — `train_loop_v1`)

- `build_model(fixtures) -> torch.nn.Module`
- `train_step(model, batch, optimizer) -> dict` with `loss` and `work_units`
- `run_training(fixtures, steps: int) -> dict`

## Contract

- `memory_pressure=0.82`, `segment_count=12`, `recompute_ratio=0.5`.
- The first six segments use the same non-reentrant checkpoint path in both
  arms.
- The baseline retains live autograd history; the oracle retains detached
  activation diagnostics and clears them at the next forward.
- Repetitions, improvement threshold, and noise criterion are unchanged.

## Required diagnosis

Report `memory_pressure` and support it with the contrastive activation trace:
the baseline must expose live retained graph edges while the oracle must not.

## Smoke test

```
python smoke_test.py
```
