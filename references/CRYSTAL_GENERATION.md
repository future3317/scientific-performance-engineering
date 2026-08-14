# Crystal generation workloads

Use this reference for crystal diffusion, flow matching, score-based sampling, and generation campaigns. Read `PERFORMANCE_PLAYBOOK.md` for shared measurement and runtime rules. Read `EQUIVARIANT_OPERATOR_DESIGN.md` only when an explicitly authorized denoiser representation/operator change is in scope.

## 1. Separate training from sampling

Standard diffusion/flow training usually samples one time per training example; sampling repeatedly evaluates the denoiser. Diagnose training throughput and sampling throughput as separate workloads. Do not claim that a sampling NFE reduction accelerates training, or that a faster training step reduces sampling cost without measuring the denoiser path.

Measure sampling as

\[
T_{\rm sample}\approx\sum_{k=1}^{N_{\rm FE}}(T_{\rm graph}^{(k)}+T_{\rm net}^{(k)}+T_{\rm guidance}^{(k)})+T_{\rm solver}+T_{\rm IO}.
\]

Record actual model evaluations per sample, not configured solver steps. Euler commonly uses one FE per step; Heun/RK, predictor-corrector, and classifier-free guidance can use multiple evaluations. Count conditional and unconditional guidance calls separately.

## 2. Optimize in generation order

1. Reduce NFE only through an authorized sampler/model experiment.
2. Reduce the generated state or exclude known variables.
3. Remove unnecessary predictor/corrector, guidance, trajectory, and output branches.
4. Sweep generation batch size, cache invariant state, and remove I/O from the hot path.
5. Profile graph rebuild, periodic geometry, spherical harmonics, denoiser, guidance, solver, and I/O independently.
6. Optimize the fixed-NFE denoiser with the operator/runtime modules.

Kernel acceleration reduces per-FE cost; NFE or state reduction changes the sampling problem. Do not combine them in an initial comparison. Start with the smallest sweep that can eliminate a choice; run a larger or held-out confirmation only for candidates that survive the initial comparison.

## 3. Treat state reduction as an authorized model/task experiment

- If composition is given for CSP, do not denoise atom types. If atom count is known or independently sampled, do not carry padded \(N_{\max}\) state through the trajectory.
- For symmetry-constrained generation, compare asymmetric-unit/Wyckoff or space-group-conditioned generation against generate-then-filter. The symmetry map, composition/count contract, and target feasibility must be explicit; do not assume a structure is forbidden from an unverified label convention.
- Treat fractional coordinates as periodic/torus variables. A Euclidean coordinate path that breaks the \(0\leftrightarrow1\) boundary is a model change, not a numerical shortcut.
- Flow path, base distribution, branch-specific anti-annealing, solver, distillation, consistency, and one/few-step generation are separately trained or separately validated experiments. Compare NFE-quality Pareto points, not nominal solver order or step count.

## 4. Keep dynamic geometry dynamic

When fractional coordinates \(F_t\) or lattice \(L_t\) change, distances, edge vectors, spherical harmonics, and possibly neighbor topology change too. Do not cache them across NFE. Cache only values invariant across the trajectory: composition/condition embeddings, masks, batch indices, species metadata, and schedule-only tables such as \(\sigma_t\), \(\alpha_t\), or loss weights.

Measure neighbor rebuild/PBC, spherical harmonics, denoiser, guidance, and solver separately. For an independent-sample generation job, distribute sample shards across GPUs; trajectories are serial in time but independent across samples. Within one GPU, bucket compatible atom counts/shapes only when it preserves the requested output distribution.

### Routing and candidate multiplicity

When a route/class can have multiple realizations, verify that catalogue multiplicity does not inflate the route's marginal probability. Prefer an explicit factorization such as `P(route | condition) × P(realization | route, condition)`, or prove that the existing global softmax is mathematically equivalent. Test normalization and marginal probabilities with equal-score synthetic candidates before tuning priors.

## 5. Guidance, correctors, and I/O are first-class costs

Sweep predictor-corrector count, guidance scale, and solver settings on a fixed quality panel. Compare target hit, validity, novelty/diversity, stability, and actual FE/GPU-hour; do not assume more corrector steps or larger guidance is better. Batch conditional/unconditional passes only when memory permits, and still count both computations.

Treat trajectory recording as an output-contract question, not a default. Preserve it when the user, repository, or downstream consumer requires trajectories; otherwise disable it because it can add GPU-to-CPU transfers, host serialization, memory growth, and disk I/O. Include relaxation and downstream screening in a campaign result:

\[
T_{\rm campaign}=N_{\rm gen}T_{\rm sample}+N_{\rm relax}T_{\rm relax}+N_{\rm DFT}T_{\rm DFT}.
\]

## 6. Generation acceptance

On an identical checkpoint, software/hardware environment, conditioning panel, seeds, and requested output count, report actual NFE/sample, time/FE, crystals/s, p50/p95 latency, allocated/reserved memory, validity, novelty/diversity, structural stability, and relaxation/downstream cost when it recurs in the intended campaign.

Use the repository’s gates. If no campaign gate exists, propose `stable structures / GPU-hour` or `target-hit / GPU-hour` in addition to raw sampling rate. Treat a relaxation/DFT filter as an operations-policy experiment, not a generator improvement: do not invent a score or retention rate. Compare only an existing downstream decision rule, and audit rejected candidates on a held-out fully evaluated panel before claiming campaign savings. Accept the candidate only when the declared quality and constraint gates pass; a faster generator that increases later relaxation or DFT cost can be rejected.
