# EvoSPE Phase2 execution report

## Contents

- [1. Repository / commit / evidence identity](#1-repository--commit--evidence-identity)
- [2. Implementation by work package](#2-implementation-by-work-package)
- [3. Real Agent viability](#3-real-agent-viability)
- [4. Model-driven Skill evolution](#4-model-driven-skill-evolution)
- [5. Fixed-Skill gate and held-out](#5-fixed-skill-gate-and-held-out)
- [6. Total cost and budgets](#6-total-cost-and-budgets)
- [7. Test integrity and scientific claim scope](#7-test-integrity-and-scientific-claim-scope)
- [8. Artifacts and no-overwrite proof](#8-artifacts-and-no-overwrite-proof)
- [9. Approved real B execution and decision](#9-approved-real-b-execution-and-decision)

Source: `EvoSPE_Codex_Phase2_Real_Efficacy_Taskbook_20261016.md`, especially
sections 0 and 12. This is a new study, not a revision of historical results.

## 1. Repository / commit / evidence identity

Baseline: `1796fa36360cf0fbe0a175caec3b4868d4b55e2f`, branch
`codex/skill-evolution-study`, matching local/server/origin on entry.
The canonical server checkout was clean; pre-existing local modified and
untracked files were preserved and excluded from these changes.

Runtime: `/home/workspace/lrh/CODE/scientific-performance-engineering`,
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`, via `dbcloud`.
CPU only, CUDA hidden; OMP/MKL/OpenBLAS/NumExpr/Inductor/SPE intra-op/inter-op
thread counts all one. The approved B run used the server credential without
printing it; no GPU job was started.

New evidence root:
`/home/workspace/lrh/RESULTS/skill-evolution-phase2-20261016`.
The historical formal root and October 15 evidence are read-only inputs.
Runtime implementation checkpoint: `d03cf90`. A newer documentation commit
does not relabel any test as having run on that commit.

## 2. Implementation by work package

| Package | Delivered implementation | Execution status |
|---|---|---|
| A | `split_manifest.py`, extended `readiness.py`, `native_verifier.py`; semantic-cluster separation, exposure ledger, official scorer | Server-tested; actual partition and native controls produced |
| B | Stateful runner, `real_study.py`, `ApprovedProviderBudget`; visible state/native grammar and scope/expiry/call/token/USD/wall guards | Completed approved readiness: 3/3 official wins; VIABLE_BUT_NOT_QUALIFIED |
| C | `core/skills/proposal.py`, contract/evolution extensions, `real_learning.py`; JSON proposal/patch, matched inner/outer, rejected proposals and outer rollback | Interfaces and regression tested; real optimization NOT_RUN |
| D | `core/skills/pack.py`, `evaluation.py`; standard pack/readback, canonical routing decisions, paired analysis and CSV/JSON/report output | Offline contracts tested; real content/gating benefit NOT_RUN |

Three coherent Luna packages covered A, C/pack and D. Main-agent work covered
experimental judgment, budget/approval, Agent integration, cross-package review
and central server acceptance. No new user-owned task, PR, registry, benchmark
or provider framework was created.

Actual split: **1,383 semantic groups**, **462 train_dev / 461 train_inner /
460 train_outer**. The historical **88 template directories** and **3 old
native-probe episodes** are recorded/excluded. Unknown historical or selected
task identities: **0**. No final outcome/expert plan was opened. Final held-out
IDs are not selected yet, rather than invented.

Grouping uses train-only typed target relations, floor plan and layout, not
task-description wording or seed identity. These fields do not enter policy
prompts. The first actual pass exposed a loader bug: nine demonstrations had
`traj_data.json` but no compiled game. Metadata recovery must not require a
compiled game. The corrected CLI resolves them; the blocked artifact is kept.

## 3. Real Agent viability

Completed under the user's separate B approval: three real autonomous train
episodes at `713bd65`, all official successes, 56 provider requests, exit 0.
Section 9 contains the exact receipts and cost. The taskbook alone did not
authorize those calls. C/D remain NOT_RUN. Native teacher diagnostics below
are separate environment/scorer controls, not autonomous Agent evidence.

Three already-exposed train tasks completed with the official handcoded teacher:
**9 / 22 / 12 steps**, all `won=True`, provider calls **0**. See
`teacher-47482ef-{0,1,2}/results.json`. The SSH shell wrapper returned 1 after
the three native commands because of a trailing Windows CR; this is not
presented as an exit-0 run. The official successful outcome receipts remain
environment controls. Later shell commands use explicit exit/status capture.

Saved native observations exposed actual Agent bugs before paid work:
`heat some mug` became target `some mug`, and native `move ... to ...` plus
`to the ...` feedback were unrecognized. These are fixed against public grammar
regressions. Inventory/transformation facts require visible acknowledgment of
the goal object. Private reasoning/expert-plan fields never become actions.

Executed B: three independent train_inner source tasks, clean/heat/cool, no
Skill, unchanged 60-action horizon, at most 180 requests. All three completed. Source
tasks can never serve as their own validation pairs. This is engineering
viability, not an efficacy sample-size design.

Stage B is stateful/no-Skill first; H1 legacy-current vs stateful remains
untested. Zero real wins stops paid Skill optimization. One win remains
insufficient to start C. Two different template-group wins plus independent
failure evidence permits requesting C approval, not automatic execution.

## 4. Model-driven Skill evolution

`REAL_OPTIMIZATION_NOT_RUN`. No real proposal, refinement or paired inner/outer
rollout has been executed in Phase2. Code/mocks cannot supply these outcomes.
The model-driven path must preserve source task/template/trace identities,
frozen candidates, independent outer tasks and rejected versions.

The initial/patch APIs use the same metered final-content client. The real CLI
currently executes one bounded patch against a supplied parent, then fresh
inner pairs and, only after inner acceptance, outer pairs. It does not collect
a usable same-family contrast automatically when B lacks one. That next source
plan needs actual B evidence and separate approval, not fabricated data.

Both raw arm trajectories are persisted. Source digests, native task paths,
semantic groups and fixed Agent/configuration identities are checked. Outer
rejection is journaled and retires the candidate; outer undecided stays
undecided. Both stages must accept before review/export; no automatic canonical
promotion occurs. B-only approval cannot authorize a Skill proposal.

## 5. Fixed-Skill gate and held-out

`NOT_RUN`. No new SkillBank has real-effect qualification. No final held-out
outcome has been opened. Content improvement and routing improvement are
different contrasts; all fixed-content gates must share the same matched
base/Skill outcomes. Zero coverage means zero incremental success and
undefined conditional harm, not zero conditional risk.

The evaluator separates content-on/off from gate/global effects, preserves raw
four-way counts/missingness, and reports planned bounds and exploratory
semantic-cluster uncertainty where declared sample conditions hold. Shared
controls are neither new independent samples nor repeated costs. An abstainer
equals base and an always-reuser equals global even when that shared outcome
is missing: those contrast zeros are identities, not imputed task outcomes.
Actual unknown outcomes stay unknown. ACRE still uses `AcreEngine.route()` and
its original canonical/governance gates.

The pack provides standard name/description frontmatter, typed sidecars and
readback; no scripts are exported. Real pack loading, behavioral Skill use and
held-out benefit have not been demonstrated.

## 6. Total cost and budgets

Phase2 approved B provider requests: **56**, all with known usage. Input:
**69,393**; completion: **60,203**, including **59,720 reasoning tokens**.
Cache hit/miss: **256 / 69,137** input tokens. Estimated cost at the retained
peak USD snapshot: **0.092986236**, not a provider bill. This excludes
Codex/subagent costs. There was no additional provider call during closeout.

The user approved this exact B manifest with a hard USD 4 cap, one hour,
180 calls and zero retries; `approval/user_approved_readiness.json` records it.
The old two connectivity calls do not extend authorization. The completion
cap is not calibrated by those calls. Cache savings are not assumed.
The disabled approval example is not permission to spend; prices must be
attributed to a checked official snapshot before live execution.

An actual public pricing snapshot was captured server-side at
`2026-10-15T19:05:06+00:00`: official peak USD/million rates are cache hit
**0.006**, cache miss **0.3**, completion including reasoning **1.2**.
`pricing-source.html`, `pricing-table.json`, `pricing_snapshot.json` retain
source/extraction. Estimates are not provider bills or promises of future rates.

Proposed B envelope: 32,768 input allowance and 8,192 completion cap per request,
180 calls: **5,898,240 input + 1,474,560 completion tokens**, worst estimate
**USD 3.538944** with no cache discount. Suggested ceiling **USD 4**, B only,
zero retries, one fixed fresh live directory. The 8,192 cap is an uncalibrated
pilot rejection bound, not a scientific default. Input reservation uses declared
UTF-8/chat-framing assumptions, not an exact provider tokenizer. Actual usage
violations/unknown billing stop further requests. C/D or higher caps need new
approval; the disabled example is not that approval.

## 7. Test integrity and scientific claim scope

The entry full-suite run is pinned to `1796fa3`; raw command, environment,
stdout/stderr, start/end and exit code are under `baseline-tests/`.
Result: **497 passed, 8 warnings, 681.60 seconds, exit 0**. The separately
saved exit code is authoritative. This verifies the entry baseline, not the
new Phase2 changes. The previous 473-test result remains specific to
`f63ec84`. No synthetic qualification campaign is rerun.

| Revision | Actual server result | Log directory |
|---|---|---|
| `bcae5cd` | 122 passed / 3 failed, exit 1 | `targeted-bcae5cd` |
| `47482ef` | 125 passed, exit 0 | `targeted-47482ef` |
| `47482ef` | 563 passed / 2 failed, 8 warnings, 735.70 s, exit 1 | `full-suite-47482ef` |
| `54ccec5` | 166 passed, 1 warning, 34.12 s, exit 0 | `targeted-54ccec5` |
| `54ccec5` | **570 passed, 8 warnings, 675.93 s, exit 0** | `full-suite-54ccec5` |
| `2f9d548` | 171 passed / 3 failed, 1 warning, 34.49 s, exit 1 | `targeted-2f9d548` |
| `d03cf90` | **175 passed, 1 warning, 35.34 s, exit 0** | `targeted-d03cf90` |

Batches overlap and are not added. The full suite applies to `54ccec5`, not
the subsequent native-grammar/outer-journal/contrast-identity changes. Those
received the final 175-test affected/adjacent-path batch under taskbook 7.1.
No current-HEAD full-suite claim is made.

The two full-order failures were worker fixtures assuming signal-handler
readiness after 0.2 s and inheriting the capture pipe. Explicit ready-message
synchronization/separate child pipe replace the timing assumption. Production
cleanup, isolation, evidence persistence, task budgets and scientific gates
were unchanged; original assertions remain. Both tests and the full suite pass
at `54ccec5`. The later native-acknowledgment failures exposed the destination
article bug and were fixed, not waived.

The three B receipts demonstrate train-side autonomous Agent task success,
not Agent superiority, Skill benefit or ACRE efficacy.
The manuscript and its numerical evidence are unchanged.

## 8. Artifacts and no-overwrite proof

`frozen-source-check.json`: **499 historical files, zero mismatches**.
Historical campaign identity remains
`b67df06db177b7042e14931e293d3e5ea6db0cc84228876e465d8923770536f8`.
The historical cross-repository manifest still matches
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`.

Actual split: `data-54ccec5/split_manifest.json`, identity
`adbf45195039868fafd1d9a568b5097a681263567f3f2b868ca4ee897745337c`;
its `SPLIT_OVERLAP_REPORT.md` records exposure/grouping. Earlier failed and
blocked outputs remain development records, not replacement evidence.

The active proposal is generated under `approval/` after final documentation
sync, binding the exact current revision. `prepared-54ccec5/` is an older,
non-active preparation. Local mirror:
`E:/DATA/EvoSPE/skill-evolution-phase2-20261016/`.

## 9. Approved real B execution and decision

Runtime commit: `713bd65027873e7d121642c2d2e1d9d3936bd1a2`.
Manifest: `639d108719117f35916225965f0455cc705f411e94e2def03891693703d9048c`.
Output: `live-readiness-713bd65/`; disabled proposal remains unchanged.
PID **1952418**, UTC **2026-10-16 02:45:52.524746** to
**02:53:20.058652**, elapsed **447.53 s**, recorded **exit 0**.
Scheduled/started/completed/nonlaunched: **3 / 3 / 3 / 0**.
Model request/response: `deepseek-flash`; thinking enabled; no Skill injected.

| Family | Official won/done | Decision turns | Environment steps | Invalid turns | Termination |
|---|---|---:|---:|---:|---|
| Clean & Place | true / true | 21 | 21 | 0 | success |
| Cool & Place | true / true | 12 | 12 | 0 | success |
| Heat & Place | true / true | 23 | 22 | 1 | success |

All receipts are `REAL_AGENT / VERIFIED_SUCCESS`, distinct frozen semantic
clusters. Journal request count and trajectory count both equal **56** in
this run; generally trajectory length alone is not proof of billed requests.
All usage is known; no pending reservation or budget-bound violation remains.
Finish reasons: **55 stop / 1 length**. Heat turn 15 reached the 8,192-token
completion cap with empty final content, yielding `selected_action=null` and
`action_not_admissible`; it was not passed to the environment. The next
decision recovered without an API retry, and the episode eventually won.
Maximum completion was **8,192**, maximum input **1,562**. The truncation is
an observed cap limitation, not a failure episode or evidence for increasing
the cap without approval. Cache hits were only 256 tokens; substantial cache
savings were not observed and are not claimed.

Gate: **VIABLE_BUT_NOT_QUALIFIED**, **3 independent win clusters / 0 independent
failure clusters**, action **STOP_PAID_SKILL_OPTIMIZATION**. Real task success
is established for these three train tasks, but the prescribed success/failure
contrast for C is absent. No source collection, Skill proposal, C/D or held-out
evaluation is authorized by this result. Further train-only contrast collection
requires a new bounded plan and approval. No parameter or prompt was modified
after observing outcomes; all original trajectories and budget journal remain.

The follow-up monitor is paused after closeout. Automatic Skill benefit,
independent content efficacy and real ACRE efficacy remain **NOT_RUN /
UNSUPPORTED**. The manuscript and historical numerical evidence are unchanged.
