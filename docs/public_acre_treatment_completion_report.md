# Public ACRE treatment completion report

Historical branch report imported during repository integration. Its numbers
are retained as originally reported, not revalidated or promoted into the
September frozen claim matrix by this merge. Current execution authority is
the repository AGENTS.md and experiment registry; no new run is authorized.

Revision: `public-acre-treatment-20260915-v1`  
Implementation commit used for the recorded runs: `4037c01`  
Provider: DeepSeek, one pinned configuration

## Completed evidence

### Fixed-skill public audit

The audit covers 266 held-out instances from ScienceWorld (100), Mind2Web
(100), ToolBench (50), and MCP-Bench (16), using the SkillGen repository at
`3c4537bb12ac287ceb1b5d410b491206089fdcb7`. Versioned v1 execution made 894
provider calls. The v2 re-projection reused the completed held-out paired
outcomes and made 96 new train-side evidence calls.

Global reuse and the public semantic router covered all 266 instances. The
strict ACRE policy covered none because this generic fixed intervention did not
produce a train-side positive-and-no-harm support group. This result is an
external deployment-boundary diagnostic, not SkillGen-generation efficacy and
not an ACRE superiority result.

### Sequential ScienceWorld treatment

The A/C/D treatment used 24 held-out ScienceWorld tasks and 144 provider calls.
All three arms were observation-complete. C had mean gain `0.0908645`, one
negative-transfer task, and 57,743 total tokens. D had mean gain `0.1884443`,
zero negative-transfer tasks, 23 immutable skill versions, and 110,979 total
tokens. The paired D-C mean was `0.0975798` with bootstrap 95% interval
`[0.0533120, 0.1434263]` (13 positive, 3 negative, 8 ties).

This is a limited one-benchmark treatment-calibration signal. It supports no
cross-benchmark, multi-model, or sealed confirmatory efficacy claim.

### Aborted public run

A larger verified-skill ScienceWorld audit was started but stopped after a
provider stall. It has a partial decision ledger and no final report, so it is
not used as paper evidence or in any aggregate.

## Verification

- Focused implementation checks: `17 passed`.
- Full server suite: `471 passed, 8 warnings`, exit code 0.
- LaTeX build: exit code 0; final manuscript build is 17 pages.
- Historical formal/provider-free evidence was not overwritten.
- The public results use new revisioned artifact roots and retain their raw
  provenance, model, benchmark, task-order, and provider-call metadata.

## Supported paper claims

- Context-indexed, intervention-grounded support certification on the controlled
  SPE-EvoBench substrate.
- External public evidence that global/semantic reuse can deploy a fixed
  intervention without intervention-specific support; the strict policy can
  abstain when that evidence is absent.
- A bounded D-C calibration signal on the recorded ScienceWorld stream.

## Claims not supported

- Cross-benchmark or multi-model ACRE efficacy.
- Prospective SkillGen-generation improvement.
- A sealed confirmatory C/D treatment estimate.
- Universal safety or universal applicability of the learned support rule.
