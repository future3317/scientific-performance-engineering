# Skill-evolution implementation and zero-provider study

Date: 2026-10-15. Source action plan was read in full. The user's subsequent
direction is to establish provider-free behavior before any new API experiment.
Three Luna work packages covered agent/skills, ACRE semantics, and budget/study;
the main agent integrated changes and ran all validation on `dbcloud`.

## Revisions and artifacts

- Baseline: `49b267e45222d49639636bba324dd9e80a3d1255`.
- Development branch: `codex/skill-evolution-study`, pushed to origin.
- Historical audit code: `3386655`; native environment probe: `26aa9cf`.
- Initial implementation: `5793365`; corrections: `8fa4e1f`.
- Initial successful regression/qualification: `d7b104d`.
- Finite-lattice scope and lossless paired-evidence correction: `2a9355d`;
  test-contract clarification: `3ed1055`.
- Server results root: `/home/workspace/lrh/RESULTS/skill-evolution-study-20261015`.
- Audit baseline and missing-file findings: [CODEX_AUDIT_BASELINE.md](CODEX_AUDIT_BASELINE.md).
- Study assumptions, frozen grids and statistical units: [PROTOCOL.md](../../experiments/skill_evolution/PROTOCOL.md).

Every test/experiment used the canonical server checkout and
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
Resource contract for final validation: CPU only, CUDA hidden, with all seven
declared variables set to 1: OMP, MKL, OpenBLAS, NumExpr, Inductor compile,
SPE Torch intra-op and SPE Torch inter-op threads;
no model training, GPU campaign, or provider calls. Local Windows use was
restricted to editing and static inspection. Pre-existing local dirty files
were not staged or discarded; the separate dirty server worktree was untouched.

## Completed implementation

| Stage | Implemented and exercised | Evidence limit |
|---|---|---|
| P0 | Read-only historical parser, cell/group CSV, closed failure funnel, fresh-output protection; visible-state ALFWorld runner, admissible-action guard, bounded history, progress/replan trace | Native environment probe is reset/step integration, not agent viability |
| P1 | Train-only success/failure contrast mining; typed procedural steps; immutable skill contract; bounded edits; independent paired verifier; low-power `UNDECIDED`; candidate persistence/rejection/rollback through RuleSpec, StateStore, MutationJournal | Deterministic extraction/patch API and synthetic replay; no LLM proposal efficacy or canonical promotion claim |
| P2 | Scope/version-bound global/selective/uncertain/rejected decisions; overall and stratum bounds; operational/diagnostic distinction; fresh vocabulary; bounded search; explicit inductive vs transductive vocabulary; compact reporting; selective deployment restricted to certified feature projections on the declared lattice | Learned experiment is synthetic; real-task power remains untested |
| P3 | All-or-none pair reservations, separate evolution cap; raw missing costs stay unknown; lexical/BM25 body ranking ends in AcreEngine.route | Dense/cross-encoder retrieval and learned VOI acquisition remain future empirical work |
| P4 | Frozen synthetic train/evaluation protocol, paired outcomes, cluster uncertainty, explicit denominators, prospective confidence allocation, search stress; concrete train-only Phase-I proposal | Real Phase I and all later real phases remain NOT_RUN |

The existing stateless runner and one-shot synthesis file were not edited.
The optional new regime filter runs before the existing canonical/stable-state
router; it cannot promote a candidate or bypass governed routing.

## Observed results

Historical diagnosis reproduces **228 scheduled / 208 execution-valid / 20
nonlaunched**. Of the 208, **145** terminated before a verifier record:
78 input-token-limit, 53 turn-limit, 14 controller failure. The 63 verifier
records divide into 48 correctness failures (41 API, 7 anti-cheat) and 15
measurement-only records. No verified positive was inferred from zero scores.
All 499 source files recorded by the audit were compared after execution: unchanged.
The campaign SHA256 remains
`b67df06db177b7042e14931e293d3e5ea6db0cc84228876e465d8923770536f8`.

The official ALFWorld train probe completed **3 resets and 3 batched admissible
steps**, with observations/actions/rewards saved under `native-probe/`. Data
was located from the actual archived receipt, at
`/home/workspace/jy/ig_memory_active_diag_20260811/data/alfworld`; the expected
`/home/workspace/lrh/DATA/alfworld` directory was empty. No datasets were moved.

The learned synthetic study used train seeds 101/211, evaluation seeds 307/401,
three regimes, 60 groups per stratum per seed, and 1,024 repetitions per group.
It reports 720 independent evaluation groups; repeated observations are not
counted as independent clusters. Confidence allocation was serialized before
evaluation: .025 measurement, .0125 regime classification, .0125 CEGIS.
The repetition count follows the pre-execution worst-case radius target .15
(computed radius .1463), not a search for favorable evaluation outcomes.

| Synthetic regime | New classification | New coverage | Incremental success vs base | Harm/deployed |
|---|---|---:|---:|---:|
| Broad benefit | GLOBAL_VERIFIED | 1.00 | .85009 | 0 |
| Selective transfer | SELECTIVE_IDENTIFIED | .50 | .42524 | 0 |
| Null effect | UNCERTAIN | 0 | 0 | undefined |

The broad positive control exposes the intended conservative old-selective
abstention; the selective positive control learns the observable stratum and
retains the repairs while avoiding the constructed harmful stratum. These are
strongly separated synthetic positive controls with known simulation structure,
not evidence that real skills admit an equally separable boundary.

Search stress at 8/16 values completed exactly; 32/64 values reached the fixed
1,000-candidate cap and returned `search_exhausted`, never identified. Measured
Python traced peaks were about 38/139/313/333 KB; serialized summaries stayed
about 1.9–2.3 KB initially, and 2.7–4.2 KB after storing the decision-lattice
projection needed for scope checks. These are traced Python allocations, not process or GPU memory.
Initial qualification wall time: 5.11 s on the server.

The synthetic evolution replay produced three expected states:
64 independent positive pairs -> CANDIDATE_ACCEPTED;
64 regression pairs -> REJECTED and rollback;
one positive pair -> UNDECIDED. Journal/store consistency was checked per case.
Acceptance here retains candidate status, pending independent calibration and
governed promotion. No canonical skill was installed.

## Validation record

Initial tests: 6 failed, 31 passed before interruption; the log is retained as
`tests-initial.txt`. The interruption followed a real cubic membership path in
pair-reservation revalidation; ID membership now uses sets. Subsequent run:
96 passed, 2 failed. The remaining planner terminal-state and exact-enumeration
API regressions were corrected. The affected-path suite at `d7b104d`:
**98 passed in 21.44 s, exit 0**, logged in `tests-final.txt`.

The command covered all six new test modules, legacy ALFWorld parser tests,
ACRE v0.4/lifecycle, budget-aware ACRE, and formal-authority closure tests.
Full-suite status and final finite-lattice follow-up are recorded below after
their actual server execution; no pending run is described as passed.

The initial full-suite attempt was interrupted with 329 passed / 4 failed.
One CLI failure was caused by the operator supplying only OMP/MKL rather than
the required complete topology. Two supported compile/lifecycle tests passed
on untouched baseline `49b267e` (2 passed, 19.87 s), establishing that the new
default 100k/5 s search envelope was a regression. The corrected default derives
the finite combination count from the declared grammar, with a 30 s wall cap.
Explicit study limits remain 1,000 candidates / 2 s; incomplete search abstains.

The expanded focused run at `2a9355d` produced 110 passed / 1 failed; the sole
failure was a ten-second fingerprint subprocess timeout. The exact command
completed successfully in 9.94 s outside that deadline. This test verifies
thread-environment propagation, not import performance; only its test startup
allowance was changed to 30 s, retaining the exact topology assertions. No
production timing, certificate, or scientific task budget changed. Original
failure logs remain in the results root.

The standalone boundary script originally assumed version-space cardinality
must decrease when a new observed numeric threshold is introduced. That
assumption encoded the stale-vocabulary behavior being fixed. Its replacement
asserts incremental/fresh-fit equality and that every retained predicate admits
the anchor and rejects every counterexample; it passed at `3ed1055` (exit 0).

The next full run at `3ed1055` was interrupted after **333 passed, 6 warnings**
(exit 2), while investigating diagnostic CEGIS cost; it is not a completed suite.
The culprit was retaining syntactically distinct hypotheses in no-lattice
diagnostics, then repeatedly fitting that large list. At `f63ec84`, these calls
retain evidence-equivalence representatives and remain `underidentified` because
no deployment lattice was declared. Explicit-lattice identification still uses
that complete declared comparison domain. The new agent also detects alternating
navigation cycles rather than only identical consecutive observations.

Latest focused verification at `f63ec84`: **50 new tests passed in 15.93 s,
exit 0**. The full suite at this same revision is **IN_PROGRESS**, not passed:
server PID 689281, log `full-suite-bounded.txt`. The 120 s faulthandler output
is a diagnostic stack dump, not a test timeout verdict. The scheduled thread
follow-up `evospe` checks this actual job and will record the terminal result.

The dry-run manifest writer consumes the executable study's own preregistration
as its input, avoiding a second split/budget config. Its final command check
remains pending until the running suite finishes and server source can be
synchronized without changing revisions mid-run.

`qualification-scoped.json` and `evolution-scoped/` record the follow-up at
`2a9355d`, with the same frozen synthetic parameters and all three expected
regime/candidate states. Earlier files remain available. Paired evidence now
preserves raw off/on utility and the untruncated paired delta, records the actual
agent/benchmark revisions, and explicitly normalizes the Rule OS utility field
by a factor of two to respect its [-1,1] range.

## Provider configuration update

The user enabled thinking for subsequent `deepseek-flash` action calls and
requested a task-grounded reassessment of the completion ceiling. The current
8,192-token setting (reasoning plus answer) is an uncalibrated engineering cap,
not an approved formal-study operating point.
Both legacy and stateful action clients use this setting; prior receipts are
unchanged. The stateful prompt now places the fixed skill before changing
progress/state fields, preserving an identical prefix for provider-side prompt
caching. Responses persist cache-hit/miss tokens, reasoning tokens and finish
reason. Missing provider counters remain unknown, not zero. Only final content
is executable; private reasoning is never parsed as an action. No local answer
cache is used, and paired arms still make independent requests. Cache savings
must be measured from provider usage, not assumed from prefix ordering.

The Phase I proposal reflects the new thinking/output setting. Its total
call/token ceilings remain unchanged; this configuration change does not launch
that experiment or authorize an unbounded campaign.

Verification at `09819a6`: 23 action-runner tests passed on the server. Two
real requests through `DeepSeekStatefulClient` both returned an admissible
`look` action in final content, with thinking present and finish reason `stop`.
Each used 161 input tokens; completion usage was 60 and 49 tokens, including
56 and 45 reasoning tokens. Both reported 0 cache-hit tokens and 161 cache-miss
tokens. Thus connectivity/thinking/action extraction are verified, but this
short immediate-repeat probe does not demonstrate a cache benefit. No episode
or provider experiment was started. API keys were never printed.

These trivial `look` requests cannot calibrate complex decisions. At the
proposed 720-call maximum, an 8,192-token completion cap permits 5,898,240 output
tokens alone, exceeding the proposed 2M total-token ceiling before input usage.
The two ceilings can coexist only as independently enforced limits, not as a
guarantee that every scheduled pair can finish. Representative train-only
planning/recovery decisions must inform cap selection; paired completion must
be budgeted before a campaign starts. Cache hits reduce billing where the
provider supports it, not context length or reasoning/output token usage.

## Remaining real-study work

The supplied plan is a multi-stage research program. It has not been completed
as real-task research. Phase I proposes 6 matched train-only legacy/stateful
pairs, 60 steps, DeepSeek `deepseek-flash`, thinking enabled, no retries, at
most 720 calls and 2M total tokens. This is a planning ceiling, not a spending
approval. The user asked to prioritize zero-API work, so provider calls remain 0.

Before real execution: finalize the clean train slice excluding old diagnostic
episodes; operationally enforce actual call/token/money caps; add the provider
proposal/independent outer-validation path and confirm activation telemetry on
real rollouts. Preserve separation of agent improvement, skill-content changes,
and routing. Dense retrieval, learned acquisition, joint-system ablations and
cross-environment/model efficacy are not implemented empirical results.

No long training/provider job is running. A 30-minute thread follow-up was
created only after confirming the long full-suite validation process was live;
it may inspect validation results and finish documentation, but cannot start
provider work. It will be paused when this validation is complete. Historical
paper results and efficacy statuses are unchanged.
