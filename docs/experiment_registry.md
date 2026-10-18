# EVOSPE experiment registry

This registry is the repository-side index for reproducibility and claim
boundaries. Raw results stay under `/home/workspace/lrh/RESULTS`; this file
does not copy or rewrite them.

## Current status

```text
HOSTED_EFFICACY_STATUS = UNSUPPORTED
HOSTED_FEASIBILITY_STATUS = INSUFFICIENT
SEALED_C_D_STATUS = NOT_RUN
RESULTS_CLAIMED = FALSE
```

Freeze manifest: `docs/EVOSPE_EVIDENCE_FREEZE_20260905.json`
Manifest SHA256: `b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`

## Registered evidence

| Evidence | Root or artifact | Revision identity | SHA256 / status | Paper use |
|---|---|---|---|---|
| Provider-free method evidence | `/home/workspace/lrh/RESULTS/evospe-method-evidence-026e733-20260902` | method `026e733` ancestry | indexed in paper evidence map | applicability, relation, observability |
| Provider-free lifecycle evidence | `/home/workspace/lrh/RESULTS/evospe-method-efficacy-d69a626-20260902/component_ablations.json` | `d69a626` | raw-source digest in paper freeze | lifecycle mechanism only |
| Resource ablation | `/home/workspace/lrh/RESULTS/evospe-budget-aware-ablation-59fe1a5-20260902-tight-reserve/report.json` | `59fe1a5bd4af06d368eecb1ffd6e50df161fb384` | `db9331024a39bd79c5bb403d44668649eaa1d599c96f48393532176f5a7ff503` | resource governance |
| Reserve stress replay | `/home/workspace/lrh/RESULTS/evospe-budget-aware-reserve-stress-7995dac-20260902/report.json` | `7995dac98d4107e8048075b8d805e300a574cc64` | `93b7027b6b9d221f40da732437c72c0619439d3bc691960c993aeeac5570dbb8` | scheduled completion |
| Historical formal boundary | `/home/workspace/lrh/RESULTS/formal-partial-identification-dce4d19-20260902` | benchmark `dce4d19`; controller `db0d25d` | immutable historical | treatment-coupled observation only |
| Hosted feasibility v1/v2 | `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-*` | benchmark `a8ede49`/`6ef43cb`; controller `08f41bf`/`3e422775` | feasibility only | failure progression |
| Hosted feasibility v3 | `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-calibration-v3-f104f2d-d72219a-gpu5-20260904-executor-recovery2` | benchmark `f104f2d59701ffeeef622099f4fca33c9fc4c561`; controller `d72219a490a4c2e4c0bb6ea1fe5bb40be20468c2` | protocol `1de6c4fd08e1a0c4324d3d4b14c974690688c18ff637f6a416c9c85b5cbf0346`; recovery manifest `e3150492080b6f03c497dd1bc44865b4bf095f130e261bda460d931ed45f0c89` | feasibility only; no C/D |

## Interpretation rules

- The historical 228-cell campaign is immutable and is never pooled with
  provider-free or feasibility evidence.
- Executor-valid is not measurement-valid; the v3 18-cell screen is not a
  treatment experiment.
- No provider rerun, budget increase, sealed comparison, or unblinding is
  authorized by this registry.
- Any future evidence must record its exact benchmark/controller revisions,
  protocol, seed policy, environment, command receipt, and digest before it is
  eligible for a paper claim.

## Post-freeze fixed-intervention study

`EVOSPE_RT_V3_20261018` is a separate exploratory identity, not a revision of
the September freeze or its historical hosted feasibility v3.

- Current evidence root: `/home/workspace/lrh/RESULTS/evospe-rt-v3-20261019-gpu-resume`.
- Frozen manifest digest: `890b32e9befbf303b231f10c0c60090e318905993541c0437088d7d4f3a8abbd`.
- Protocol source: `463f2833d7580d9cf068665bd92793cba6ec421b`;
  inherited measurement: `9d5a9ebe568ad3824882937a9ae078f61910954d`;
  resumed measurement: `49ee7d8293beaef8a4a67944eaaa236cba1c69e5`.
- Zero-GPU analysis: `1ad8ad2aedd947c04997f3200be3a60249341cb1`;
  study documentation: `8e67b9340c72b4c605ee26320338c862a832a7af`.
- Runtime: equivcompiler Python; torch 2.6.0+cu126, torchvision 0.21.0+cu126,
  Triton 3.2.0, RTX 4090. Original GPU UUID
  `GPU-ff4b8a6c-b1ad-8ae1-d2e8-741a64fd24a1` contributed 16 valid pairs;
  resumed GPU UUID `GPU-f190a619-3c7d-72a9-4f20-40d7b7d35c20` contributed 3.
- Frozen design: 18 families, 10 development / 5 calibration / 3 final;
  H=1/16/128, seed 31, two AB/BA sessions, 108 planned pairs.
- Outcome: 90 pair records, 19 valid, 11 numeric failures, 6 cache
  infrastructure errors, 1 baseline failure, 53 unresolved. Two interrupted
  candidate attempts overlap the unresolved count and were not retried.
- Valid subset: six independent families; all 19 cold effects negative;
  family-equal mean log benefit -3.599646, 95% family-bootstrap CI
  [-3.98265, -3.23516] (seed 101, 10,000 draws).
- Release: `UNDERPOWERED_NOT_FINAL`, valid calibration families 3/5;
  18 final pairs unstarted. Both ACRE variants: `ACRE_NOT_FITTED`.
- Claim boundary: real model computation with materialized/miniature inputs,
  not dataset accuracy, scientific time-to-quality, universal compiler harm,
  or selective-gate efficacy. Cold timing includes compilation and H forwards,
  not process startup/model-input setup or the complete job lifecycle.
- Full net deployment utility and complete resource costs remain unknown;
  observed parent windows are not GPU busy seconds. The historical 5619-second
  baseline parent window exceeded the declared 900-second cap.

The evidence bundle, member hashes, raw receipts, final exits and tests remain
in that RESULTS root; the local delivery is under
`E:/DATA/evospe-rt-v3-20261019-gpu-resume`. See the
[study README](../experiments/real_intervention_v3/README.md) for the execution
contract. Neither integration into `main` nor future code maintenance
reauthorizes measurements or changes the immutable run identities.
