# EvoSPE v4 real channels-last study

This is a new, pre-registered study identity. Its primary comparison is eager
FP32 inference in the original layout (S0) versus the same model, weights,
inputs, precision, and eager execution with channels-last preparation included
in the cold cost (S1). `torch.compile`, AMP, and TF32 are excluded from the
primary intervention. Compile/cache amortization is a separate study identity
and cannot replace a primary result.

The v3 result roots listed in `config/PRIMARY_LAYOUT_STUDY.json` are read-only
inputs to `audit_v3.py`. Their measurements, final split, and unresolved arms
are not modified or replayed. The full v4 model roster is assigned to
development/calibration/final by architecture lineage before qualification;
unavailable or ineligible families stay in their assigned denominator.

## Run

All project execution is server-only, from the canonical repository and the
`equivcompiler` environment, as required by the repository `AGENTS.md`:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python \
  -m experiments.real_intervention_v4.run_all \
  --config experiments/real_intervention_v4/config/PRIMARY_LAYOUT_STUDY.json \
  --results-root /home/workspace/lrh/RESULTS/evospe-rt-v4/EVOSPE_REAL_INTERVENTION_V4_CHANNELS_LAST_20261020_SEEDFIX_REPAIR \
  --resume-safe
```

The runner validates the read-only v3 audit, executes its CPU fault-injection
suite, selects and pins one physical GPU UUID for the study, completes
performance-blind model qualification, freezes the roster and manifest, runs
the engineering shakedown, then executes the frozen paired splits. An arm
whose durable lifecycle started without a verified terminal receipt remains
unknown and is never silently replayed. Each new arm has a unique cache scope;
its hard deadline includes cleanup grace.

The UUID-prefix and seed-propagation shakedown failures remain immutable
under their original result roots. After fixing seed propagation, the next
complete run uses the new `SEEDFIX_REPAIR` study identity above; it does not
resume or reuse either failed root.

The primary final gate is independent of ACRE promotion: it requires the
pre-registered family minima, closed development/calibration evidence, a
passing shakedown, and a policy freeze. ACRE can be scored or deployed only
with its canonical, scope-bound support certificate. If it is not fitted,
the final still evaluates the primary effect and frozen simple comparators,
while ACRE results remain `ACRE_NOT_FITTED`—not learned abstention or efficacy.

## Current contract and outputs

`config/PRIMARY_LAYOUT_STUDY.json` is the study contract and
`config/SPLIT_FREEZE.json` fixes family assignment. `CONTRACT.md` describes
measurement, validity, and stopping semantics. A completed run writes
`REPORT.md`, `METRICS.json`, `MODEL_REGISTRY.csv`,
`QUALIFICATION_MATRIX.csv`, `FROZEN_MANIFEST.json`, `PAIRS.jsonl`,
`POLICY_FREEZE.json`, `POLICY_RESULTS.csv`, `FAILURE_TAXONOMY.csv`,
`MISSINGNESS_SENSITIVITY.md`, `PAPER_CLAIM_MATRIX.md`, and a unique-member
`EVIDENCE_BUNDLE.tar.gz` with a SHA256 member manifest. Raw attempt receipts
and lifecycle events are bundled; caches, model weights, and secrets are not.

The final report separates primary efficacy from ACRE certification, and
observed measurements from mechanism hypotheses. A compile/cache secondary
study is not launched unless its own frozen contract and available resource
window permit it.
