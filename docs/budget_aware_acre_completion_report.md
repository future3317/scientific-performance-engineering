# Budget-aware ACRE completion report

Date: 2026-09-02

## Scope and evidence boundary

This revision addresses resource governance in EVOSPE/ACRE without changing the
deployability certificate, scientific gates, causal estimands, or historical
formal evidence. The historical `dce4d19` hosted campaign remains immutable and
was not rerun. No provider call, budget increase, sealed campaign, unblinding,
or efficacy rescue was performed.

The implementation source is the server repository
`/home/workspace/lrh/CODE/scientific-performance-engineering`, on `main` at
`7995dac98d4107e8048075b8d805e300a574cc64`; this report is a documentation-only
follow-up commit. The local Windows checkout contains
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

## Verification

All commands below used the project environment on `dbcloud`:

```text
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
```

- Targeted budget/ACRE/controller tests: 23 passed, 13 passed, 1 stress test
  passed, and 1 passed in the earlier staged checks.
- Regression test after the public-contract fixture correction: 1 passed,
  1 existing torch-geometric deprecation warning.
- Full project suite at commit `84dd160`: **381 passed, 8 warnings**, 636.37 s.

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

The manuscript repository is `E:\PAPER\EvoSPE`, branch `paper-update`. It now
uses the deployable-knowledge narrative, gives treatment-coupled observability
and execution reservation mathematical definitions, includes the provider-free
resource ablation, removes implementation-only identifiers from the manuscript,
and removes unverified proceedings-author placeholders from the bibliography.

The final local build is:

```text
E:\PAPER\EvoSPE\main.pdf
```

LaTeX build completed successfully with BibTeX. The final PDF has 15 pages.
The final log has no undefined references/citations and no overfull or underfull
box warnings. All 15 rendered pages were visually checked for clipping,
overlap, and figure/table layout problems.

## Remaining limitations

- The reserve ledger is fully exercised by provider-free policy and execution
  fixtures. The formal runner's production guarantee is currently implemented
  through the explicit non-blocking `evolution_frozen` path; a future hosted
  revision would need a calibrated per-cell minimum-cost schedule before using
  a numeric total-compute reserve as a confirmatory contract.
- The hosted funnel still has no positive verified-speedup events, so no new
  hosted efficacy claim is available.
- Prompt-cache hit/miss counters were not persisted in historical receipts and
  were not collected by modifying the experiment.
- Several recent neighboring-work citations remain arXiv or documentation
  records; bibliographic metadata should receive a final author/title check
  before submission.

## Commit and provenance summary

The budget-aware implementation was integrated into `main` through the staged
commits ending at:

```text
7995dac98d4107e8048075b8d805e300a574cc64
```

The latest commit only aligns the existing noise-control test with the public
API preflight. The resource evidence remains bound to
`59fe1a5bd4af06d368eecb1ffd6e50df161fb384` and the historical formal evidence
remains bound to its original benchmark/controller revisions.
