# Budget-aware ACRE completion report

## Contents

- [Scope and evidence boundary](#scope-and-evidence-boundary)
- [Implemented changes](#implemented-changes)
- [Verification](#verification)
- [Provider-free resource evidence](#provider-free-resource-evidence)
- [Condition-blind calibration closure](#condition-blind-calibration-closure)
- [Historical hosted funnel](#historical-hosted-funnel)
- [Supported and unsupported claims](#supported-and-unsupported-claims)
- [Paper closure](#paper-closure)
- [Remaining limitations](#remaining-limitations)
- [Commit and provenance summary](#commit-and-provenance-summary)

Date: 2026-09-05

## Scope and evidence boundary

This revision addresses resource governance in EVOSPE/ACRE without changing the
deployability certificate, scientific gates, causal estimands, or historical
formal evidence. The historical `dce4d19` hosted campaign remains immutable and
was not rerun. No provider call, budget increase, sealed campaign, unblinding,
or efficacy rescue was performed.

The implementation source is the server repository
`/home/workspace/lrh/CODE/scientific-performance-engineering`, on `main` at
`c5eb494d2ac9b17712c446a0b7b8eecc37d1d897`. The method/controller evidence
freeze remains tagged at `097a36a0c49812ba20d626310040f7e84f2192b3`. This report is kept separate
from the server runtime evidence. The local Windows checkout contains
only user-local untracked materials in addition to this tracked source; those
materials were not staged.

## Implemented changes

- Added an immutable, replayable `ExecutionReserveBudget` ledger for optional
  evidence acquisition. It computes the remaining scheduled-execution reserve,
  rejects optional evolution work that would consume it, preserves accounting,
  and enters `evolution_frozen` without disabling scheduled execution.
- Added certificate-critical acquisition using existing observable confidence
  quantities and cost-normalized priority. The legacy decision-aware path remains
  available as the comparison condition.
- Added lazy relation admission before the factorial engine, reserve-aware
  underidentification, and early stopping for conclusive confidence sequences.
- Added dependency-aware certificate reuse/invalidation with provenance receipts;
  unchanged dependencies reuse evidence and changed dependencies invalidate only
  affected clauses.
- Added public-only solution preflight and near-budget finalization policy. The
  preflight is symmetric and does not expose hidden oracle, verifier,
  anti-cheat, or performance-target information. Hosted agent reasoning is
  explicitly disabled in the attestation.
- Changed the formal campaign's evolution-exhaustion path from stream blocking to
  ordinary evolution freeze. State mutation, attestation, executor, hardware,
  and protocol errors remain infrastructure-invalid and still block their stream.
- Updated one existing noise-control test fixture to satisfy the now-enforced
  public API contract. No production verifier behavior was weakened.
- Made thread-topology configuration idempotent for repeated same-process
  calibration setup, while preserving PyTorch's error on a genuinely different
  interop-thread request.

## Verification

All commands below used the project environment on `dbcloud`:

```text
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
```

- Targeted budget/ACRE/controller/closure tests: **39 passed, 1 warning** in
  8.71 s, including the seeded reserve stress regression.
- Fresh affected-path selection after calibration: **94 passed, 1 warning** in
  143.79 s.
- Regression test after the public-contract fixture correction: 1 passed,
  1 existing torch-geometric deprecation warning.
- Earlier full project suite at commit `3bb2c2b`: **382 passed, 8 warnings**,
  545.81 s.

A fresh full-suite attempt on 2026-09-03 reached 381 passed and one failure:
`benchmark/tests/test_compile_calibration.py::test_dynamic_compile_activation_is_harness_owned_and_contrastive`
failed during TorchInductor CUDA-context initialization with an out-of-memory
error. At the time, the fixed GPU5 had an unrelated foreign process using
13,894 MiB and 98% utilization. No process was terminated and no runtime or
scientific configuration was changed. The earlier 382/382 run at this same
code revision and the directly affected 94-test selection remain recorded
separately; that attempt is classified as shared-resource
infrastructure-blocked, not as a code assertion failure.

After the GPU5 resource was released, the affected selection passed with
**38 passed, 1 warning** in 34.88 s, and the final full suite at commit
`0504635` passed with **383 passed, 8 warnings** in 610.87 s.

On 2026-09-05, the final `main` full-suite verification at
`f28e5e317a40b52e2c4a41012b25c53a89875caf` recorded **419 passed and 1
failed** in 1,659.11 s. The failure was the process-group cleanup assertion
in `test_forced_owned_worker_cleanup_is_reported_and_valid`; the same test
passed alone and the complete `test_calibration_runner.py` selection passed
with **36 passed, 1 warning**. The retained failure log identified a child-ready
race: the parent exited after a fixed `0.2s` before the worker was guaranteed
to have joined the process group. The regression now uses a child-ready
handshake with no fixed sleep. The final `main` at
`c5eb494d2ac9b17712c446a0b7b8eecc37d1d897` recorded **420 passed, 8 warnings**.

The warnings are dependency/deprecation or fixture warnings; no test was skipped
to hide a failure.

## Provider-free resource evidence

Canonical evidence:

```text
Path: /home/workspace/lrh/RESULTS/evospe-budget-aware-ablation-59fe1a5-20260902-tight-reserve/report.json
SHA256: db9331024a39bd79c5bb403d44668649eaa1d599c96f48393532176f5a7ff503
Method revision: 59fe1a5bd4af06d368eecb1ffd6e50df161fb384
Provider calls: 0
Seed: 7
```

Observed fixture results:

| Variant | Queries | Evidence cost | Factorial calls |
|---|---:|---:|---:|
| Legacy ACRE | 6 | 9.0 | 16 (eager reference) |
| Execution reserve | 6 | 9.0 | 16 |
| Certificate-critical | 3 | 4.0 | 16 |
| Lazy relation | — | — | 4 |
| Full budget-aware ACRE | 3 | 4.0 | 4 |

Thus the checked-in fixture reduced acquisition queries by 50% and fixture
evidence cost by 55.6% relative to legacy, while lazy relation admission saved
12 of 16 factorial invocations (75%). The tight-reserve case rejected the
optional query, froze evolution, completed 2/2 scheduled cells, and produced
zero missing scheduled cells. Deterministic replay produced the same digest.

The split contained 8 calibration instances and 8 final-surface instances with
seed, promotion, and evidence disjointness; labels were not exposed to policy,
and the split was excluded from treatment-effect estimation.

### Seeded reserve stress evidence

The single tight-reserve fixture was extended to 256 deterministic schedules,
covering schedule lengths 1--32, four execution-cost levels, six optional-query
costs, and tight/moderate/slack optional-budget levels. Across all levels, 2,902
scheduled cells completed and zero were missing. Tight pressure admitted 0 of
1,317 optional-query opportunities and abstained on all 1,317; moderate pressure
admitted 250 at cost 183.25; slack pressure admitted 628 at cost 1,184.5.
Thus the stress evidence demonstrates the intended operational direction—budget
pressure reduces optional evolution and increases abstention without cancelling
scheduled execution—but does not estimate utility or hosted efficacy.

Canonical stress artifact:

```text
Path: /home/workspace/lrh/RESULTS/evospe-budget-aware-reserve-stress-7995dac-20260902/report.json
SHA256: 93b7027b6b9d221f40da732437c72c0619439d3bc691960c993aeeac5570dbb8
Code revision: 7995dac98d4107e8048075b8d805e300a574cc64
Seed: 17; schedules: 256; provider calls: 0
```

The formula/documentation consistency check found that the implementation
already reserves future execution from the total envelope; the manuscript now
names that quantity `B_total^remain` and applies any independent evolution cap
as a separate minimum bound.

These are provider-free mechanism and resource-governance results. They are not
hosted-model efficacy results and do not establish that EVOSPE improves task
utility.

## Condition-blind calibration closure

The independent condition-blind calibration is now complete. The derived index
is:

```text
Path: /home/workspace/lrh/RESULTS/evospe-condition-blind-calibration-8tasks-20260903-complete/calibration_funnel.json
SHA256: ca8cd8b5eeee3db1e25bccc832fe416dd78c713e37601018080c2ff859c4dab2
Benchmark/method revision: 3bb2c2b20f399c021f6e48879f489401c74aad50
Calibration protocol digest: fdf4d06f33d8cdfcf8b350b8ead745c7888d277e13bd7dd9eaf9c68d649d5d20
Population manifest digest: ae40e9c217ade74ad6d1409561797ddad3b46b753875f4674d76ebff4a08e2ad
Provider calls: 0
```

The index references eight task-specific raw roots and preserves their raw,
noise-control, and envelope artifacts in place. The 24-cell funnel is:

| Quantity | Count |
|---|---:|
| Expected / submitted | 24 / 24 |
| Execution-valid | 24 |
| Correctness-pass | 24 |
| Eligible | 24 |
| Public-contract applicable / pass | 15 / 15 |
| Atomic-performance measurement reached | 15 |
| Atomic verified-positive / inconclusive | 9 / 6 |
| Evolution-episode evidence cells | 9 |
| Protocol-invalid / unresolved invalid | 0 / 0 |

The completed set uses one revision, one calibration protocol digest, one
population manifest digest, `shared_observed` resource mode, GPU UUID
`GPU-a40e8c73-adfc-eb76-9aa1-63b6569cd9f7`, and `CUDA_VISIBLE_DEVICES=5`.
The earlier interrupted repeated-backbone root and the failed resume attempt
remain separate historical evidence; neither was edited or used as a source
for the three completed repeated-backbone cells. The failed resume attempt was
an output-root/CLI precondition error, not a calibration outcome.

This closes the condition-blind feasibility check: the selected tasks can reach
their public contract and correctness/measurement paths under the fixed runtime.
It does not establish hosted efficacy, compare treatments, or authorize a
provider campaign by itself.

## Historical hosted funnel

The prior hosted campaign remains a motivating observation-boundary case:

- 228 scheduled cells;
- 208 execution-valid observations;
- 20 downstream D cells not launched after the registered global evolution
  budget was exhausted;
- all observed scores were zero through distinct failure, correctness, and
  unverified-measurement paths;
- no provider rerun was used to repair the matrix.

Its D--B and D--C contrasts remain partially identified under
treatment-coupled observability. The historical record is not pooled with the
new provider-free revision and is not presented as end-to-end efficacy.

## Supported and unsupported claims

Supported by the current evidence:

- typed applicability and certified counterexample governance can produce zero
  unsafe generalizations on the defined provider-free family surfaces, with the
  measured conservative-coverage trade-off;
- lifecycle poison/drift cases exercise abstention, specialization,
  revalidation, and retirement behavior;
- certificate-critical acquisition and lazy relation admission reduce measured
  provider-free evidence work;
- optional evolution can be frozen while scheduled execution continues in the
  tested reserve path;
- treatment-coupled observation can bias complete-case analysis, while support
  bounds retain coverage in the controlled simulation.

Not supported:

- a hosted end-to-end superiority or net-utility claim;
- a claim that the historical zero-score campaign demonstrates treatment
  efficacy;
- a claim that the small provider-free fixture predicts hosted token savings;
- a claim that resource governance alone makes D causally superior to C.

## Paper closure

The manuscript repository is `E:\PAPER\EvoSPE`, branch
`codex/paper-evidence-restructure-20260905`. It now
uses the deployable-knowledge narrative, gives treatment-coupled observability
and execution reservation mathematical definitions, includes the provider-free
resource ablation, removes implementation-only identifiers from the manuscript,
and removes unverified proceedings-author placeholders from the bibliography.

The final local build is:

```text
E:\PAPER\EvoSPE\main.pdf
```

LaTeX build completed successfully with BibTeX. The final PDF has 15 pages.
The final log has no undefined references/citations, no overfull boxes, and no
duplicate hyperref destinations after setting `hypertexnames=false`. Two
non-fatal underfull vertical-box diagnostics remain around intentionally
page-filling figure material; they do not cause clipping or overlap. Pages 14
and 15 were freshly rendered and checked for clipping, overlap, blank-page
errors, and table/figure placement; the full-page render set is retained under
the paper repository's QA directory.

## Remaining limitations

- The reserve ledger is fully exercised by provider-free policy and execution
  fixtures. The formal runner's production guarantee is currently implemented
  through the explicit non-blocking `evolution_frozen` path; a future hosted
  revision would need a calibrated per-cell minimum-cost schedule before using
  a numeric total-compute reserve as a confirmatory contract.
- The condition-blind calibration is complete for the eight selected tasks, but
  it is a feasibility sample rather than a held-out end-to-end treatment study;
  broader task-population calibration remains outside this revision.
- The hosted funnel still has no positive verified-speedup events, so no new
  hosted efficacy claim is available.
- Prompt-cache hit/miss counters were not persisted in historical receipts and
  were not collected by modifying the experiment.
- Several recent neighboring-work citations remain arXiv or documentation
  records; bibliographic metadata should receive a final author/title check
  before submission.

## Commit and provenance summary

The budget-aware implementation is integrated into `main` through the staged
commits ending at:

```text
05046354da9b9a27ba66e39b5a0741e454178345
```

The latest tracked commit records the thread-topology idempotence regression
fix and its verification. The
resource evidence remains bound to
`59fe1a5bd4af06d368eecb1ffd6e50df161fb384` and the historical formal evidence
remains bound to its original benchmark/controller revisions.
