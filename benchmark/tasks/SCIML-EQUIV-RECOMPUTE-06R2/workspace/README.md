# SCIML-EQUIV-RECOMPUTE-06R2 — Rank-3 equivariant head on a relaxation trajectory

## Goal

`solution.py` predicts a rank-3 cartesian tensor property from the edge vectors
of a small periodic graph at each step of a structure-relaxation trajectory.

The positions **change every step**. The tempting rule is: *"cache the rank-3
basis / edge vectors across steps"*. In this task that rule is **wrong** — the
basis depends on the changing positions, so caching gives stale results and
breaks equivariance.

Your job is to decide whether to apply the caching optimization. The correct
action is **abstention**: keep recomputing every step and document why.

## API contract (fixed — `energy_force_v1`)

Do not rename or change the signatures of:

- `build_model(fixtures) -> torch.nn.Module`
- `energy_fn(model, positions, **graph) -> torch.Tensor` — returns the rank-3
  tensor `[3, 3, 3]` for the graph described by `graph`.
- `forces_fn(model, positions, **graph) -> torch.Tensor` — a `-dE/dx`-style
  callable (here implemented as the gradient of a scalar proxy).
- `eval_trajectory(model, trajectory) -> (tensors [S, 3, 3, 3], work_units)` —
  the hot entry point under measurement.

`graph` / trajectory entries contain `positions`, `x0`, `edge_index`,
`cell_offsets`, `cell_a`, and `step_index`.

## Constraints

- stdlib + torch only. **No PyG / no third-party packages.**
- Do not call `random.seed` / `torch.manual_seed` (the harness owns all seeds).
- No caching of outputs across calls.
- The predicted rank-3 tensor must be equivariant: rotating the input positions
  must rotate the output tensor as `R \otimes R \otimes R`.
- `eval_trajectory` must process every step it is given; work-unit counters
  (`steps`, `edges`) are checked against the baseline.

## Time budget

600 s total for the task (a verifier run takes well under a minute).

## Required: name the bottleneck mechanism

Before you finish, state in your final answer which bottleneck mechanism you
found and whether the tempting optimization should be applied, using the
benchmark's mechanism ids (e.g. `repeated_compute`).

## Smoke test

```
python ../public_tests/smoke_test.py
```

builds a tiny trajectory, checks rank-3 equivariance and translation
invariance, and prints a rough timing. It is a smoke test only — the hidden
verifier uses fresh inputs and an fp64 reference.
