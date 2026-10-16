# EvoSPE / ACRE Phase3 execution report

Source: `EvoSPE_Codex_Phase3_Skill_Efficacy_and_ACRE_Taskbook_20261016.md`,
read completely. This is zero-provider preparation, not a completed efficacy
experiment. T1–T4 have no user spending authorization.

## A. Repository, identities and protection

Entry actual/report HEAD: `7cdbeee7760575d248c5f1ac920d483e639b2b7b`, branch
`codex/skill-evolution-study`, server clean and explicit branch pull up to date.
Windows pre-existing tracked/untracked user changes are preserved and excluded.
All runtime checks use dbcloud canonical repository and
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`. CPU only, CUDA
hidden, OMP/MKL/OpenBLAS/NumExpr/Inductor/SPE intra/inter-op threads all one.
Server AGENTS.md specifies that environment; local additional historical-freeze
constraints are retained. No historical hosted/formal/sealed/C-D is launched.

New root: `/home/workspace/lrh/RESULTS/skill-evolution-phase3-20261016`.
Phase2 B source remains `713bd65027873e7d121642c2d2e1d9d3936bd1a2`.
Entry audit verifies each of its three canonical receipt digests, task/cluster/
parent-split membership, official won/done, 56 journal requests, known closing
usage, one length turn and actual exit0. See `entry_audit/audit.json`.
All 499 historical source files match; the historical freeze manifest remains
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`.
Per-file comparisons are retained rather than replacing prior audit records.

B-time game content hashes and response IDs were not persisted. The entry
audit records current bytes of the three already exposed games separately;
it does not retroactively claim B-time game identity or response-ID dedup.
The Phase2 report's stale approval-required sentence, implementation-log header
and evidence-scope status are corrected. Original B result/config/journal and
the older test identities are unchanged.

## B. Data split and final reserve

Parent: dev462 / inner461 / outer460, 1,383 semantic clusters, digest
`adbf45195039868fafd1d9a568b5097a681263567f3f2b868ca4ee897745337c`.
The parent is never rewritten. `split_extension_v3.json` adds children under
parent_split=train_outer only, with unknown/exposed groups quarantined.
Metadata includes stable semantic/template/possible-overlap keys; neither the
sampler nor reservation opens games, outcomes, descriptions or expert plans.

Predeclared allocation: equal thirds within each official family; largest
remainder with rotating family tie priority. This is a practical exploratory
reservation, not a powered success-superiority or low-harm certification.
Actual child counts and digests are recorded in the server preparation summary.
Final reserved task IDs are metadata-only; no final environment is reset.

T1: study_id=`phase3_baseline_diagnostic`, seed31, exactly two previously
unexposed dev clusters per official family. Rank SHA256 of UTF8 compact JSON
array `[study_id,family,cluster_id,seed]`. Complete reserve order and inclusion
probabilities are retained, but no automatic replacement is authorized.
No outcomes/expert lengths/difficulty guesses affect selection. Existing
88-template history, old probes, B and teacher exposure are excluded as groups.
Only the twelve selected T1 game files are then read as opaque bytes to bind
execution identities. Reserved final files remain unopened.

## C. Provider authorization and execution

| Stage | Authorization | Execution | Remaining dependency |
|---|---|---|---|
| T1 | NOT_APPROVED | NOT_RUN | Human approval of one exact T1 option |
| T2 | NOT_APPROVED | NOT_RUN | Compliant train_dev sources, source-mode/family choice, exact dependent manifest |
| T3 | NOT_APPROVED | NOT_RUN | Frozen inner-selected candidate and independent outer manifest |
| T4 | NOT_APPROVED | NOT_RUN / NOT_QUALIFIED | Real Skill utility/heterogeneity, independent calibration and final power/approval |

New Phase3 provider requests and spend: **0**. No credential or provider ping.
T1 planned/started/completed/not-started: **12/0/0/12**. Observed success and
cost distributions for these twelve tasks are not measured, not fabricated
zeros. T2–T4 dependent task/candidate identities remain null and unlaunchable.

T1 uses thinking-enabled deepseek-flash, Chat Completions, final content only,
no tools, no retries, 60 decisions per task, 32,768 input reservation and 8,192
completion cap per request. `reasoning_effort` was NOT_EXPLICITLY_SENT in B;
temperature0 was sent but thinking documentation says it has no effect.
The prompt is unchanged. Added response-ID telemetry/dedup and selected-only
native loading define BASE_AGENT_V2_TELEMETRY; no pooling with B.

Two approval options share exactly the same tasks/order/token budget:

| Option | Episode USD cap | Stage USD cap | Worst full token envelope |
|---|---:|---:|---:|
| bounded (recommended) | 1.25 | 15 | 14.155776 |
| wide | 4 | 48 | 14.155776 |

720 calls is a ceiling (12×60), not a target or independent sample size.
Stage allowances: 23,592,960 input and 5,898,240 completion tokens. At retained
official peak rates, one full episode envelope is USD1.179648, so both options
fit the same frozen horizon. The wide dollar cap grants no extra tokens/calls.
Smaller caps can cause informative censoring and require a new manifest;
failure-early savings cannot count as efficacy. Expected from B's three-task
mean is USD0.371944944 for12, a weak cross-family extrapolation, not a guarantee.
HIGH uses twelve times B's observed maximum episode cost, not a reliable
population quantile. Both are in `budget_summary.json`. Prices must be checked
again before live approval; the retained snapshot is attributed, not a bill.
All budgets exclude Codex/Luna costs, which are not available as a task bill.

T2 proposal: one initial + one feedback patch, six inner tasks, newly executed
OFF/ON for both candidate versions (24 episodes, 1,442 call ceiling, USD30 cap
proposal). This deliberately does not assume reusable OFF controls.
T3 proposal: eight new outer pairs (16 episodes, 960-call ceiling, USD20 cap
proposal). These figures are planning envelopes, not permission or finalized
candidate-dependent manifests. T4 remains unqualified without a fabricated cap.

## D. Real no-Skill diagnostic

T1 NOT_RUN. The only autonomous evidence remains historical B: clean21,
cool12, heat22 environment steps; all won/done true; heat had one recovered
length/empty-content decision. It is not a failed episode. B has three distinct
success clusters but no failure contrast. Those sources are train_inner and
therefore historical diagnostics, not Phase3 train_dev induction sources.
New gate fields separate VERIFIED_TRAIN_VIABLE, SUCCESS_ONLY_READY and
CONTRASTIVE_READY from Skill efficacy and provider permission.

## E. Real model Skill evolution

Real initial proposal/patch/candidate/inner outcomes: **NOT_RUN**.
Source mode SUCCESS_ONLY_INDUCTION permits no failure examples and requires
legal train_dev real successes; it is never labelled contrastive repair.
CONTRASTIVE_REPAIR requires real independently resolved success/failure.
Initial/patch consume final content only; invalid, truncated, hidden-answer,
wrong-parent, wrong-split or outer-derived proposals are rejected and journaled.
The code is a preparation contract, not evidence that a real Skill was generated.
The exact integration/negative tests and server results are recorded below.

## F. Independent outer and ACRE evidence

Real outer, gate calibration and held-out policy comparisons: **NOT_RUN**.
Frozen candidates require a separate T3 approval; T2 cannot consume outer.
Outer decisions retain favorable-pilot/undecided/rejected, never automatic
GLOBAL_VERIFIED or canonical promotion. ACRE still uses AcreEngine.route.
Static frozen decisions over real OFF/ON may be evaluated as
POLICY_VALUE_ON_REAL_PAIRED_ROLLOUTS, not online execution per gate; dynamic
routers cannot reuse these two arms. Future held-out remains unopened.

Statistics distinguish raw00/01/10/11, planned/started/complete/missing, support
retention, conditional harm (undefined at zero deployment), both-success
efficiency distributions and all-in cost. Cheaper failed episodes cannot
create an efficiency gain. The fixed token-saving screen is zero observed
regressions plus >=10% median both-success total-token saving; USD is secondary.
Zero-harm exact bounds require declared fixed, independent iid unselected units;
otherwise UNCERTIFIED. Break-even is undefined without positive measured saving.

## G. Server tests and code delivery

Entry full suite at7cdbeee: **579 passed, 8 warnings, 701.98s, exit0**.
UTC start03:48:02.302299/end03:59:46.952645 on2026-10-16. This verifies entry
HEAD, not subsequent Phase3 implementation commits.
Raw command/environment/stdout/stderr/exit are under
`test_receipt/entry-7cdbeee/`. Later implementation validation must cite its own
commit and scope; overlapping test counts are not summed. Windows runs no tests.

Source mapping:

| Package | Existing production files |
|---|---|
| E | split_manifest.py; corrected Phase2 report/log/evidence scope |
| F | real_study.py, native_verifier.py, resource_budget.py; runner telemetry only |
| G | core/skills/proposal.py, contract.py, evolution.py; real_learning.py |
| H/I | evaluation.py with canonical route, shared-control/efficiency contracts |

Three coherent Luna packages handled E, G and H/I; main agent handled evidence
audit, F budgeting/execution entry, integration and server acceptance. No new
provider framework, parallel registry, worktree or PR was created. Only owned
files are staged; historical artifacts and user modifications remain untouched.

## H. Scientific boundary and next decision

Evidence category: historical real AGENT readiness only. New REAL_INNER,
REAL_OUTER and HELDOUT effects remain NOT_RUN. No manuscript/figure or historical
numerical claim is changed. Expected delivery is READY_FOR_APPROVAL for T1,
not Phase3 scientific completion. After separate T1 approval, execute only its
exact frozen stage, retain all outcomes, choose source mode by observed facts,
and return for T2 approval; no automatic expansion, task hunting or self-approval.
