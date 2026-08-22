# SCIML-GRAPH-REBUILD-08R2 — Langevin sampler with changing positions

## Goal

`solution.py` runs a tiny Langevin-style sampler over atom positions in a cubic
box.  A simple pair potential is evaluated on a radius neighbor graph.

This task is a **do-not-apply** counterexample.  The baseline already behaves
correctly: it **rebuilds the neighbor graph every step** because the positions
change every step.  The tempting optimization — cache the neighbor list across
steps to avoid rebuild overhead — is **scientifically wrong** here and will be
rejected by the verifier.

Your correct action is to abstain from applying the cache-neighbor-list rule.
You may add a short comment in `solution.py` explaining why caching is unsafe
when positions change every step.  Do not otherwise change the sampler math.

## API contract (fixed — `sampler_v1`)

Do not rename or change the signatures of:

- `build_sampler(fixtures) -> sampler context`
- `sample_step(sampler, state, step_index: int) -> new state`
- `sample(sampler, fixtures, num_steps: int) -> positions tensor`

`state` is a dict with key `"positions"` of shape `[B, N, 3]`.  The returned
`sample` tensor has the same shape.

The solution also exposes `energy_forces(sampler, positions)` for the scientific
gate; do not remove it.

## Constraints

- stdlib + torch only. No PyG, no third-party packages, no network, no file I/O
  outside the workspace.
- Do not call `random.seed` / `torch.manual_seed` (the harness owns all seeds).
- Do not cache outputs across calls.
- Keep the radius-graph builder and the force update equations unchanged.
- Do not reduce `num_steps`, `num_atoms`, or the step size.

## Time budget

600 s total for your session; a single verifier run finishes in well under a
minute.

## Required: name the bottleneck mechanism

Before you finish, state the mechanism id and whether the rule should be
applied.  The correct answer for this task is `graph_rebuild` + **do not apply**
(caching the neighbor list is unsafe because positions change every step).

## Smoke test

```bash
python ../public_tests/smoke_test.py
```

runs the sampler on a tiny fixture and checks finiteness, shape, and a loose
reference agreement.
