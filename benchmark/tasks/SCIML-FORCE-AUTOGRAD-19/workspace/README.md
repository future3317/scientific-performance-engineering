# SCIML-FORCE-AUTOGRAD-19 — Ragged batch of crystal energy/force evaluations

## Goal

`solution.py` evaluates energies and atomic forces (`F = -dE/dx`, via autograd)
for a batch of small periodic crystals using an invariant message-passing model.
It is **correct but too slow**. Make the evaluation measurably faster while
keeping the scientific outputs unchanged.

## API contract (fixed — `energy_force_v1`)

Do not rename or change the signatures of:

- `build_model(fixtures) -> torch.nn.Module` — builds the model from
  `fixtures["config"]` and loads `fixtures["init_state"]`.
- `energy_fn(model, positions, **graph) -> energy tensor` — energy of ONE graph;
  must support autograd w.r.t. `positions`. `graph` carries `x0`, `edge_index`,
  `cell_offsets`, `cell_a`.
- `forces_fn(model, positions, **graph) -> forces tensor` — must equal `-dE/dx`.
- `eval_batch(model, graphs) -> (energies [G], forces [N_total, 3], work_units)`
  — the hot entry point under measurement. `graphs` is a ragged list of per-graph
  dicts (`positions` [N_i, 3], `x0` [N_i, F], `edge_index` [2, E_i],
  `cell_offsets` [E_i, 3] integer PBC offsets, `cell_a` scalar cubic lattice).

You may add new functions/methods and restructure internals freely.

## Constraints

- stdlib + torch only. **No PyG / no third-party packages.**
- Do not call `random.seed` / `torch.manual_seed` (the harness owns all seeds).
- No caching of outputs across calls: the verifier re-runs you on fresh seeded
  inputs and probes you with mutated inputs.
- Forces must remain exactly `-dE/dx` of your energy (autograd); energies must
  remain invariant to a global translation of all positions.
- `eval_batch` must process every graph/atom/edge it is given; the returned
  `work_units` counters (`graphs`, `atoms`, `edges`) are checked against the
  baseline.

## Time budget

600 s total for the task (a verifier run takes well under a minute).

## Required: name the bottleneck mechanism

Before you finish, state in your final answer which bottleneck mechanism(s) you
found and fixed, using the benchmark's mechanism ids (e.g. `ragged_loops`,
`autograd_overhead`, `repeated_compute`, `graph_rebuild`, `scalar_sync`,
`launch_fragmentation`).

## Smoke test

```
python ../public_tests/smoke_test.py
```

builds two tiny graphs, checks that forces match `-dE/dx`, and prints a rough
per-iteration timing. It is a smoke test only — the hidden verifier uses fresh
inputs, an fp64 reference, and additional scientific gates.
