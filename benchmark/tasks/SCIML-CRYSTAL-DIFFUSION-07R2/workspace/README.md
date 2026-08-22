# SCIML-CRYSTAL-DIFFUSION-07R2 — Tiny DDPM crystal sampler

## Goal

`solution.py` implements a correct but inefficient DDPM-style sampler over
fractional coordinates and orthorhombic lattice lengths for tiny synthetic
crystals. Make the sampler measurably faster while preserving the exact
scientific outputs:

- generated crystal structures must remain structurally valid (no interatomic
  distance below 0.5 Å after Cartesian conversion),
- the distribution of generated samples must match the seeded reference
  distribution.

The harness measures **time-to-quality**: wall-clock seconds to reach a validity
rate ≥ 0.9 on a fixed, seeded evaluation batch.

## API contract (fixed — `sampler_v1`)

Do not rename or change the signatures of:

- `build_sampler(fixtures) -> sampler context`
- `sample_step(sampler, state, step_index: int) -> new state`
- `sample(sampler, fixtures, num_steps: int) -> samples tensor`

`state` is a dict with keys `"frac"` ([B, N, 3]) and `"lengths"` ([B, 3]).
The returned `sample` tensor has shape [B, N*3 + 3] and concatenates fractional
coordinates and lattice lengths.

## Constraints

- stdlib + torch only. No PyG, no third-party packages, no network, no file I/O
  outside the workspace.
- Do not call `random.seed` / `torch.manual_seed` (the harness owns all seeds).
- Do not cache outputs across calls; the verifier uses fresh inputs and
  zero/mutated probes.
- The numerical sampling equations must stay the same; only the implementation
  efficiency should improve.
- `eval_frac` / `eval_lengths` in the fixtures are fixed; do not reduce batch
  size or step count.

## Time budget

600 s total for your session; a single verifier run finishes in well under a
minute.

## Required: name the bottleneck mechanism(s)

Before you finish, state which bottleneck mechanism(s) you fixed using the
benchmark ids, e.g. `scalar_sync`, `launch_fragmentation`, `ragged_loops`,
`repeated_compute`, `h2d_blocking`.

## Smoke test

```bash
python ../public_tests/smoke_test.py
```

runs your solution on a tiny fixture and checks finiteness, shape, and a loose
reference agreement.
