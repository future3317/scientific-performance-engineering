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
