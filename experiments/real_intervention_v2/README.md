# Real fixed-intervention study (v2)

This channel studies the fixed inference intervention `torch.compile` with
`backend="inductor"`, `mode="default"`, and `dynamic=None` against eager
execution. It is not evidence that EvoSPE invented or generated that compiler
optimization.

The current source is an **unapproved implementation draft**. The observed
server has PyTorch 2.6.0+cu126 and torchvision 0.21.0+cu126, but does not have
TorchBench installed. No TorchBench dependency, model data, or performance
measurement has been run for this channel. The paired executor uses one
subprocess and one compiler-cache directory per arm; however, hard aggregate
storage enforcement and a completed external model admission run still need
verification before the runner is eligible for approval.

## Current preflight

- Server source base: `90d4b542ad2ec4c0511436771d1ebbbfa5d85023`; execution
  interpreter: `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Runtime census: PyTorch 2.6.0+cu126, CUDA build 12.6, six RTX 4090 GPUs,
  96 Xeon Platinum 8160 logical CPUs, and about 245 GiB free on the RESULTS
  filesystem. TorchBench is absent; torchvision 0.21.0+cu126 is installed.
- Focused server tests: 15 passed. Import checks passed. These tests do not
  invoke `torch.compile`, load benchmark models, or establish performance.
- Provider/API calls, external model downloads, model executions, and
  performance measurements for this channel: zero.
- The proposed caps cover 24 lineages × 6 contexts × 2 sessions (576 arm
  processes) at up to 2,880 GPU-minutes, 172,800 CPU wall-seconds, 50 GiB
  experiment storage, and 10 GiB downloads. This is a ceiling, not a measured
  completion-time estimate; the server has no known USD/GPU-hour rate.

Before any external download, model execution, correctness qualification, or
timing study, obtain one overall resource approval bound to the pinned
benchmark, frozen model-lineage cohort, code revision, and caps in
`config/resource_approval_v1.json`. The proposed caps are not authorization.

Final policy replay must use only execution-time context. ACRE routing is
delegated to the repository's canonical `AcreEngine.route`; this package does
not implement another router. The post-hoc pairwise oracle is scoring-only.
# Real fixed-intervention study

The only measured comparison in this package is a fixed eager versus
`torch.compile(..., backend="inductor", mode="default", dynamic=None)`
inference intervention. TorchBench source, model lineage split, two sessions per
context, arm order, cold compiler cache, numerical tolerance, and resource
ceiling are bound in the server-generated `RUN_MANIFEST.json` before the first
S1 outcome. Development and calibration receipts are completed before
`POLICY_FROZEN.json` is published; final decisions are then frozen before final
arm receipts are created.

All runtime commands below execute on `dbcloud` with the project
`equivcompiler` interpreter. The pinned TorchBench checkout and model cache live
outside Git under `/home/workspace/lrh/DATA/evospe-rt-v2`; results live under
`/home/workspace/lrh/RESULTS/evospe-rt-v2/20261018-one-shot`.

```bash
PY=/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
TB=/home/workspace/lrh/DATA/evospe-rt-v2/torchbench
PROJECT=/home/workspace/lrh/CODE/worktrees/scientific-performance-engineering-real-intervention-v2
OUT=/home/workspace/lrh/RESULTS/evospe-rt-v2/20261018-one-shot
CUDA_VISIBLE_DEVICES=GPU-ff4b8a6c-b1ad-8ae1-d2e8-741a64fd24a1 \
  PYTHONPATH="$PROJECT:$TB" $PY -m experiments.real_intervention_v2.prepare_study \
  --torchbench-root "$TB" --project-root "$PROJECT" --output-root "$OUT" \
  --device cuda --gpu-uuid GPU-ff4b8a6c-b1ad-8ae1-d2e8-741a64fd24a1
$PY -m experiments.real_intervention_v2.run_one_window \
  --run-root "$OUT" --project-root "$PROJECT" --torchbench-root "$TB"
$PY -m experiments.real_intervention_v2.finalize --run-root "$OUT"
```

The first command performs model API/input-shape admission and writes a frozen
manifest plus the user authorization binding. It does not execute model
forward passes or read candidate outcomes. If fewer than 18 independent
lineages are API-eligible, the run remains measured but is labeled
underpowered. No model is substituted based on S1 speed. The policy report
includes the canonical ACRE route; absent a governed support certificate, that
route abstains rather than inventing a rule. The pairwise oracle is scoring-only.
