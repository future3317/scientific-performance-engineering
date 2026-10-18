# Hosted A/reset Feasibility Failure Forensics

## Contents

- [Follow-up: schedule-fixed v2 calibration pilot](#follow-up-schedule-fixed-v2-calibration-pilot-2026-09-04)
- [Scope and decision](#scope-and-decision)
- [Frozen identities and evidence roots](#frozen-identities-and-evidence-roots)
- [Six-cell execution funnel](#six-cell-execution-funnel)
- [CORE: correctness failures](#core-correctness-failures)
- [SCIML: token-limit failures](#sciml-token-limit-failures)
- [Current status and bounded next action](#current-status-and-bounded-next-action)

Initial report date: 2026-09-03; v2 follow-up: 2026-09-04

## Follow-up: schedule-fixed v2 calibration pilot (2026-09-04)

The schedule-fixed replacement pilot is now complete. This section is the
current status; the six-cell analysis below remains historical evidence from
the preceding pilot.

### Frozen identities and scope

- Root:
  `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-calibration-v2-6ef43cb-3e422775-gpu5-20260903-schedulefix`
- Protocol: `EVOSPE_HOSTED_FEASIBILITY_CALIBRATION_V2`
- Protocol SHA256: `d7c6461d837424f14491d739d43f4dbd2c405e66aec01d919330c67ca0dbfcf5`
- Benchmark revision: `6ef43cbfe2a6c78095975b23472ba8546dbb537d`
- Controller revision: `3e422775e7f10136ea216c56f82571ddd7542f71`
- Benchmark branch: `codex/hosted-pilot-public-contract-eligibility`
- Controller branch: `codex/host-controller-no-thinking`
- Eligibility manifest SHA256: `69f46c24d7fe5dff1275c34f81cface1b004768db9f164f3b008cf6526233fb2`
- Model: `deepseek-v4-flash`, thinking disabled, temperature 0
- Condition: A-only, reset, 4 tasks × 3 outer trials = 12 cells
- `results_claimed=false`; no B/C/D or sealed campaign was run

### v2 funnel

| stage | result |
|---|---:|
| scheduled | 12/12 |
| worker/executor execution-valid | 12/12 |
| executor-attested and sandbox-valid | 12/12 |
| protocol-invalid recorded | 0 recorded |
| correctness-pass | 4/12 |
| measurement reached | 4/12 |
| verified-positive | 0/12 |

The four correctness and measurement reaches are concentrated as follows:

- `CORE-AUTOGRAD-VJP-SMALL-25R2`: 3/3 correctness-pass and 3/3 reached
  measurement; none verified positive.
- `SCIML-EQUIV-LOWORDER-26R2`: 1/3 correctness-pass and 1/3 reached
  measurement; the verified speedup was false.
- `CORE-DATALOADER-FANOUT-16R2`: 0/3 correctness-pass.
- `CORE-KERNEL-FUSION-09R2`: 0/3 correctness-pass.

The recorded task outcomes are 4 `inconclusive`, 4 `error`, and 4 `fail`.
Here, `trial.json:measurement_eligible` is an eligibility-manifest field; it
is not the observation of whether S5 measurement actually ran. Measurement
reach above is derived from a persisted result containing paired baseline and
candidate runs.

### Exact failure decomposition

- Four persisted verifier results stopped at S2 correctness: two dataloader
  shape errors (`16x256` multiplied by `16x64` or `64x2048`) and two kernel
  errors from `forward()` receiving an unexpected keyword argument `x`.
  Their public preflight passed, but the candidate implementation was not
  correct for the task interface. These are agent candidate/API-implementation
  failures, not scientific-gate failures; S3 was not reached.
- `SCIML-EQUIV-LOWORDER-26R2`, outer-001, reached public preflight but failed
  because `build_model`, `train_step`, and `run_training` were missing. This is
  a candidate public-contract failure. A read-only recheck with the server
  environment confirms that the worker's baseline `solution/solution.py`
  passes, while the submitted candidate at
  `submitted_solution/solution.py` fails with exactly those three missing
  callables; the recorded preflight result therefore reflects the submitted
  candidate and is not an aggregation or coverage bug.
- Three cells hit the registered controller input-token limit before a
  verifier result was emitted: SCIML outer-000 (66,420 input tokens),
  dataloader outer-002 (74,665), and kernel outer-002 (74,525). Their agent
  outcomes are `input_token_limit`; no correctness or measurement result is
  inferable from those cells.

No scientific-correctness failure was observed in this pilot: the four
correctness failures stopped before the scientific gates, and the three
token-limited cells never reached the verifier.

### Token and telemetry evidence

Across all 12 controller receipts: 430,356 input tokens, 106,247 output
tokens, 130 tool calls, and 826.979 seconds of reported wall time. The three
token-limited cells each have a `host_controller_partial.json` with complete
turn-level usage and tool-call trace:

| cell | turns | last cumulative input | last cumulative output | tool calls |
|---|---:|---:|---:|---:|
| SCIML outer-000 | 14 | 66,420 | 14,913 | 14 |
| dataloader outer-002 | 13 | 74,665 | 16,384 | 13 |
| kernel outer-002 | 14 | 74,525 | 15,041 | 14 |

This closes the prior trace-loss problem for the observed turns, but it does
not satisfy every field listed by the v2 protocol: the partial artifacts do
not contain `finalize_reserve_triggered`, trigger token counts, remaining
reserve, candidate presence, public-preflight status, or submit status.
Therefore partial telemetry is present but not complete under the
pre-registered finalization-field contract.

### Decision

The pre-registered task-level stability gates are not met: no two distinct
tasks have at least 2/3 correctness passes, no two distinct tasks have at
least 2/3 measurement reaches, and there are no verified-positive cells.
The correct state remains:

`STOP_HOSTED_FEASIBILITY_INSUFFICIENT`

This pilot is feasibility evidence only. It does not unlock C/D, sealed
evaluation, or any treatment-effect claim. No rerun, budget increase,
certificate change, hidden-verifier change, or scientific-gate relaxation was
performed in this forensic pass.

## Scope and decision

This report is a read-only forensic analysis of the six condition-A/reset
feasibility cells. No DeepSeek call, rerun, budget change, task expansion,
certificate change, C/D run, formal run, or sealed run was performed.

The registered decision remains:

`HOSTED_FEASIBILITY_INSUFFICIENT`

The six cells are execution-valid and executor-attested, but none reached
correctness or measurement. They therefore provide feasibility evidence only;
they are not treatment-effect observations.

## Frozen identities and evidence roots

- Benchmark revision: `a8ede491b232e45c4b63302402fa2bf45d5ae6b5`
- Controller revision: `08f41bf079cbe5567b88c4d4301c920a5bdce20c`
- Controller branch: `codex/host-controller-no-thinking`
- Population digest: `a61b150f030cb21b3e9969132468b98835c2e198157a3c4060a4300dd6d4635a`
- Eligibility digest: `a06723740cec9eaaeaa27e0c19cb0094f6d3a40e35ad41e579373f0e153eab66`
- Summary manifest: `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-v1-a8ede49-08f41bf-gpu5-20260903/hosted-pilot-summary-v1/hosted_pilot_funnel_manifest.json`
- Summary manifest SHA256: `0ddfe5021390b3ac74463dd5abdc993d6fbb0dcf32da75dc95fa936bee226112`

Raw roots:

- CORE outer-000: `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-v1-a8ede49-08f41bf-gpu5-20260903/public-contract-canary-rebound-v7`
- SCIML outer-000: `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-v1-a8ede49-08f41bf-gpu5-20260903/public-contract-canary-rebound-v8`
- Remaining outer-001/002: `/home/workspace/lrh/RESULTS/evospe-hosted-feasibility-v1-a8ede49-08f41bf-gpu5-20260903/public-contract-pilot-remaining-v1`

## Six-cell execution funnel

| task | outer | controller outcome | input/output tokens | tools | executor | correctness | measurement |
|---|---:|---|---:|---:|---|---:|---:|
| CORE-REPEATED-BACKBONE-02R2 | 000 | completed; S1 hard fail | 12,505 / 5,628 | 8 | valid | 0 | 0 |
| CORE-REPEATED-BACKBONE-02R2 | 001 | completed; S1 hard fail | 12,045 / 5,399 | 8 | 0 | 0 |
| CORE-REPEATED-BACKBONE-02R2 | 002 | completed; S2 error | 14,793 / 6,783 | 8 | 0 | 0 |
| SCIML-GNN-RAGGED-05R2 | 000 | input_token_limit | 70,458 / 15,237 | 16 | valid | not reached | 0 |
| SCIML-GNN-RAGGED-05R2 | 001 | input_token_limit | 67,811 / 11,883 | 15 | valid | not reached | 0 |
| SCIML-GNN-RAGGED-05R2 | 002 | input_token_limit | 68,414 / 12,847 | 15 | valid | not reached | 0 |

Across raw receipts: 246,026 input tokens, 57,777 output tokens, 6 provider
calls, 6/6 executor attestations, 6/6 sandbox canary passes, 0 protocol-
invalid cells, 0/6 correctness passes, 0/6 measurement reaches, and
`results_claimed=false`.

The summary field `protocol_token_limit_outcomes=1` is not a faithful count of
the raw controller outcomes: all three SCIML receipts report
`host_controller.outcome=input_token_limit`. The raw receipts are the source
of truth for this forensic count.

## CORE: correctness failures

### Observed pipeline

The task's public README documents a three-key batch:
`{"step", "fixed", "targets"}`. The harness's S2 candidate path passes that
dictionary to `train_step`, then later invokes the model's
`project_fixed` and `forward_head` methods. The public callable list in
`public_task.json` names only `build_model`, `train_step`, and `run_training`,
so the model-method dependency is not explicit in that callable list.

The verifier's S1 scan rejects candidate calls to `torch.manual_seed` because
the harness owns the seeds. This is the primary observed failure for outer-000
and outer-001.

### Outer-000 — candidate implementation/contract violation with semantic mismatch

Evidence:

- `result.json` records S1 `seed_tampering` at `solution.py:line 34` and
  `solution.py:line 133`, with `anticheat.hard_fail=true`.
- The candidate calls `torch.manual_seed` in both `build_model` and
  `run_training`.
- Its `train_step` destructures the batch as `x, y = batch`, which is
  incompatible with the documented three-key batch.
- Its model applies a different architecture: it does not implement the
  reference fixed-context `project_fixed`/`forward_head` interface, and its
  `run_training` creates fresh random targets instead of using the supplied
  task targets.

Classification: primary `API/implementation contract violation`; the code also
has a separate scheme-level semantic mismatch because it solves a different
workload. This was rejected before S2, so it is not a scientific-gate failure.

### Outer-001 — candidate implementation/contract violation with semantic mismatch

Evidence:

- `result.json` records S1 `seed_tampering` at `solution.py:line 31` and
  `solution.py:line 94`, with `anticheat.hard_fail=true`.
- The candidate again calls `torch.manual_seed` in `build_model` and
  `run_training`.
- Its `train_step` again uses `x, y = batch` for the documented three-key
  dictionary.
- Its model does not expose the reference `project_fixed`/`forward_head`
  methods and does not implement the documented fixed-context computation.

Classification: primary `API/implementation contract violation`, with the same
scheme-level semantic mismatch. S2 and the scientific gates were not reached.

### Outer-002 — direct S2 implementation error

Evidence:

- `result.json` records the exact error:
  `S2 correctness raised: ValueError('too many values to unpack (expected 2)')`.
- The candidate's `train_step` contains `x, y = batch`, while the public task
  passes a three-key dictionary.
- The candidate also lacks the model methods that the S2 path later invokes:
  `project_fixed` and `forward_head`.
- `anticheat.findings` is empty, so this is distinct from the S1 failures.

Classification: `API/implementation error` at S2. The first concrete error is
the invalid batch destructuring; the missing model methods are a subsequent
contract failure that would be exposed after that error. No S3 scientific
gate ran, so there is no evidence for a scientific-correctness classification.

### CORE conclusion

The requested three-way split does not produce one failure of each type. The
raw evidence supports candidate implementation/contract failures and semantic
scheme mismatch, but **no scientific correctness failure was observed**:
scientific gates are empty because all three cells stopped before S3. Calling
these failures “scientific correctness errors” would overstate what was tested.

## SCIML: token-limit failures

### What is confirmed

All three cells have an attested executor receipt, `network_mode=none`, all
sandbox canary checks passing, and `worker_result.outcome=input_token_limit`.
The controller usage is:

| outer | input | output | total | tool calls | wall time (s) |
|---:|---:|---:|---:|---:|---:|
| 000 | 70,458 | 15,237 | 85,695 | 16 | 114.81 |
| 001 | 67,811 | 11,883 | 79,694 | 15 | 87.84 |
| 002 | 68,414 | 12,847 | 81,261 | 15 | 101.98 |

The controller declares an input limit of 65,536 and an output limit of
16,384. `UsageState.add_provider_usage` raises `BudgetOutcome("input_token_limit")`
once accumulated input crosses the input limit. The experiment-level total
budget is 81,920, but outer-001 and outer-002 hit the controller's input-side
limit even though their input+output totals are below 81,920.

Each cell did write a candidate `worker/solution/solution.py`, but none has a
`submitted_solution` or task `result.json`; correctness and measurement were
never invoked. The three final candidate files differ from the baseline,
which proves that the agents made progress before the budget outcome.

### Why the exact last turns cannot be reconstructed

The controller sends the full accumulated `messages` list on every provider
request. Each assistant response and each tool result is appended to that
list. Therefore input-token usage includes repeated transmission of growing
conversation context, not only the initial task bundle. The static task
artifacts are small (public task 1,355 bytes, README 2,496 bytes, retrieved
context 106 bytes), so the observed 67.8k–70.5k input totals are consistent
with dynamic multi-turn context growth.

However, the raw artifacts do not contain the last assistant messages or
per-turn tool-result payloads. More importantly, `host_agent.py` keeps
`tool_trace` local to `_run_conversation`; when `BudgetOutcome` is raised, the
assignment in `run()` never receives the partial tuple, so the outer trace
remains empty. The three receipts consequently report `tool_trace=[]` despite
15/16 tool calls.

Thus the evidence supports:

- confirmed: input-side budget exhaustion after substantial multi-turn work;
- confirmed: dynamic context accumulation was part of the cost path;
- confirmed: no completed finalization/submission was recorded;
- not identifiable from persisted evidence: whether the last turns were
  invalid exploration, repeated debugging, verbose tool output, or a missing
  near-budget finalize decision.

It would be incorrect to attribute the three cases specifically to repeated
debugging or invalid exploration without the missing trace. This is a
controller observability limitation, not a reason to rerun the pilot.

## Current status and bounded next action

The correct scientific/operational status is still:

`STOP_HOSTED_FEASIBILITY_INSUFFICIENT`

No C/D comparison or sealed campaign is justified. `results_claimed=false`
must remain unchanged.

Potential future engineering work, only after separate authorization, is
limited to:

1. persist partial turn/tool metadata when a budget exception interrupts the
   conversation;
2. make the CORE model-method dependency explicit in the public contract or
   make S2 use only the declared callable interface; and
3. add a condition-blind, model-semantic-preserving near-budget finalization
   policy if a later pilot is authorized.

None of these changes was made in this forensic pass, and no new provider
experiment should start from this report alone.
