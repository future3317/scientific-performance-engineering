# EVOSPE claim matrix

This file is the manuscript claim boundary. A result is usable only within the
scope recorded here; hosted feasibility and efficacy are intentionally separate.

## C1 — Typed applicability certificates

- Paper location: RQ1 / applicability identification.
- Exact wording: “ACRE forms conservative, typed applicability certificates from positive evidence and certified counterexamples.”
- Status: supported.
- Evidence: 1,100 held-out family-surface evaluations across 11 families; ACRE unsafe-generalization rate 0; defined-family F1 = 1.00 on six families.
- Benchmark revision: provider-free method evidence revision recorded in `results/provider_free_method_evidence.json`; raw applicability source `/home/workspace/lrh/RESULTS/evospe-method-efficacy-1f68c3c-20260902/applicability_methods.json`.
- Controller revision: not applicable; provider calls = 0.
- Allowed scope: tested family-generated provider-free surfaces and their registered oracles.
- Forbidden extrapolation: no claim of hosted-agent efficacy or universal applicability.

## C2 — Safety–coverage trade-off

- Paper location: RQ1 / applicability comparison.
- Exact wording: “Counterexample-guided typing reaches zero unsafe generalization on these surfaces with lower coverage than nearest-experience retrieval.”
- Status: supported, bounded to the tested surfaces.
- Evidence: ACRE recall .857, abstention .484; nearest-experience recall .949, unsafe rate .0418; paired unsafe-rate interval includes zero.
- Experiment: provider-free applicability comparison with family-cluster bootstrap.
- Allowed scope: policy-level safety/coverage characterization, not a leaderboard claim.
- Forbidden extrapolation: do not claim statistically superior end-to-end utility.

## C3 — Resource-aware evidence governance

- Paper location: RQ2 / resource-aware evidence governance.
- Exact wording: “Resource-aware ACRE reduces optional evidence work while preserving scheduled execution under the tested provider-free budget pressures.”
- Status: supported for the tested provider-free regimes.
- Evidence: queries 6 -> 3, evidence cost 9.0 -> 4.0, relation calls 16 -> 4, and 2,902/2,902 scheduled cells completed with missing = 0 in the stress replay.
- Benchmark revision: method revision `59fe1a5bd4af06d368eecb1ffd6e50df161fb384`.
- Result artifact: `/home/workspace/lrh/RESULTS/evospe-budget-aware-ablation-59fe1a5-20260902-tight-reserve/report.json`.
- Allowed scope: provider-free resource reservation, certificate-critical acquisition, lazy relation admission, and reuse semantics.
- Forbidden extrapolation: this does not establish hosted-agent efficacy, D > C, or lower total hosted cost.

## C4 — Lifecycle governance

- Paper location: RQ3 / lifecycle evidence.
- Exact wording: “Governed lifecycle handling avoids measured negative transfer in the checked poison, compiler-drift, and equivariant-specialization fixtures while retaining conservative routing.”
- Status: supported as mechanism and episode evidence only.
- Evidence: three provider-free episodes; governed negative-transfer rate is 0 in each checked fixture; poison-handling safety metric is 1.0 for governed handling.
- Result artifact: provider-free component/lifecycle source `/home/workspace/lrh/RESULTS/evospe-method-efficacy-d69a626-20260902/component_ablations.json`.
- Allowed scope: checked-in deterministic fixtures and their registered lifecycle contracts.
- Forbidden extrapolation: no broad claim that governed evolution improves hosted utility.

## C5 — Treatment-coupled observability and partial identification

- Paper location: RQ4 / historical hosted boundary and controlled simulation.
- Exact wording: “When treatment-owned compute controls whether outcomes are observed, complete-case contrasts can be biased and support bounds make the unidentified component explicit.”
- Status: supported and bounded.
- Evidence: historical 228-cell campaign had 20/57 downstream D cells unlaunched after registered evolution-budget exhaustion; controlled simulation at .35 censoring produced complete-case bias approximately -0.11053 under treatment-coupled observation and support-bound coverage 1.0.
- Historical benchmark revision: `dce4d194354b998d94ac9eb5b02a3d3d804828c9`.
- Historical controller revision: `db0d25d17dfd905ce53788f3aa9c170b7c10b3c7`.
- Allowed scope: the registered historical campaign and the controlled finite-population simulation.
- Forbidden extrapolation: do not treat the 20 cells as zero outcomes or infer D/C efficacy from complete cases.

## C6 — Hosted end-to-end efficacy

- Paper location: hosted feasibility boundary in RQ4/Appendix E.
- Exact wording: “The fixed hosted configuration did not establish feasibility sufficient for a treatment comparison; no hosted efficacy claim is made.”
- Status: unsupported and explicitly forbidden.
- Evidence: v1 6 cells: execution-valid 6, correctness 0, measurement 0, positive 0; v2 12 cells: execution-valid 12, correctness 4, measurement 4, positive 0; v3 recovery2 18 cells: executor-valid 18, correctness 0, measurement 0, positive 0, with five nested noise-control resource-blocked records.
- Benchmark revision: `f104f2d59701ffeeef622099f4fca33c9fc4c561`.
- Controller revision: `d72219a490a4c2e4c0bb6ea1fe5bb40be20468c2`.
- Allowed scope: feasibility-boundary and failure-forensics reporting only.
- Forbidden extrapolation: no D-C comparison, no sealed campaign, no end-to-end utility improvement claim.

## Frozen status lines

```text
HOSTED_EFFICACY_STATUS = UNSUPPORTED
HOSTED_FEASIBILITY_STATUS = INSUFFICIENT
SEALED_C_D_STATUS = NOT_RUN
RESULTS_CLAIMED = FALSE
```
