# SkillGen ScienceWorld fixed-skill audit protocol

This document freezes the external audit inputs before any held-out result is
used for a routing decision.

## Revisions

- SkillGen repository: `yccm/SkillGen`, commit `3c4537bb12ac287ceb1b5d410b491206089fdcb7`.
- ScienceWorld repository: `allenai/ScienceWorld`, commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`.
- Gold-path archive SHA256: `bec9711010a57240ebc0cf263b7ec1c88bce44b953984c934e709baf0f7ea02d`.
- Protocol revision: `skillgen-scienceworld-context-audit-v1`.

## Data flow

The official ScienceWorld `train` fold is used only for SkillGen's original
skill verification/generation protocol. The official `test` fold is held out
for routing and paired scoring. Task variation IDs are retained and no train
or test instance may overlap.

The intervention is the first skill artifact whose original SkillGen
verification gate is active under the recorded model/configuration. Once that
artifact and its verification receipt are written, the skill body is
immutable for the audit. ACRE consumes paired verification outcomes and only
execution-time public context fields. Held-out success, judge output,
trajectory, and gold-path fields are scoring-only and are joined after all
policy decisions are serialized.

## Policies

The audit compares global SkillGen reuse, SkillGen's public semantic router,
positive-only support reuse, CatBoost-LTT, and Full ACRE. All policies use the
same held-out instances, model snapshot, task order, execution budget, and
three execution seeds when the provider budget permits. Router calls use
temperature `0.0`; the exact model and usage receipts are recorded in the
result manifest. Cache keys include model, intervention text, and canonical
public context.

## Metrics and claim boundary

The artifact reports coverage, repairs retained, regressions/deployed,
non-beneficial/deployed, net success change, support-decision regret, and
per-instance decision reasons. The bootstrap unit is task instance with
execution seed nested within instance and task-type stratification.

This is a public external fixed-skill deployment audit. A complete result can
support an external context-indexed support claim. It is not by itself a
prospective hosted-agent efficacy result; an unavailable provider, incomplete
skill verification, or missing held-out split produces `status=blocked` and no
efficacy claim.

## Calibration artifact (2026-09-14)

The first provider-backed calibration uses DeepSeek `deepseek-v4-flash` with
temperature `0` and provider thinking disabled so that the plan is returned in
the public response channel. It uses the fixed intervention in
`docs/skillgen_scienceworld_fixed_skill_v1.md`, 12 train instances, and 12
disjoint test instances. The resulting artifact is
`results/skillgen_external_calibration_20260914_r2.json` with its companion
manifest. It is a protocol-readiness/calibration result, not the original
SkillGen verification protocol, not a sealed treatment comparison, and not an
efficacy claim. Its scoring uses deterministic key-action coverage; gold
actions and outcomes are joined only after routing and paired plan generation.
