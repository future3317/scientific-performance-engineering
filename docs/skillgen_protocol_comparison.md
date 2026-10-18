# SkillGen protocol comparison

This note records why the current ScienceWorld train-side run must not be
described as a faithful reproduction of the released AAAI 2026 SkillGen
ScienceWorld protocol. It is a protocol audit, not an efficacy result.

## Upstream released path (repository revision `816c91f`)

- Uses the repository's ScienceWorld wrapper and label file with task-specific
  subgoals and modified goals.
- Loads the official simulator with `generateGoldPath=True` and uses the
  wrapper's subgoal/progress bookkeeping.
- Uses the released ScienceWorld prompt template, including a long worked
  example and the public action grammar.
- Uses the upstream sampling, skill-extraction, graph/retrieval, and
  skill-prompting stages. The repository scripts use six sampling trajectories
  per fold and four folds; the reported paper protocol is the source for its
  sample-size/minimum-sample settings (120 / 20).
- The released scripts target Qwen2.5-7B-Instruct locally, with a 20-step
  inference loop and the upstream skill retrieval settings.
- Reports the upstream grounding/progress/success/AUPC family of metrics.

The released code does step through the interactive simulator. The important
distinction is that its end-to-end protocol is built around its own labels,
gold-path-enabled wrapper, trajectory sampling, and graph-based skill
extraction; it is not the same as the current single-call provider skill
generator below.

## Current native train-side protocol

- Uses the pinned ScienceWorld 1.2.3 simulator and a project-generated
  train/dev manifest.
- Gives the provider public task description, initial observation, and action
  examples; it does not provide the upstream label file, subgoal annotations,
  or upstream trajectory-extraction artifacts.
- Generates one free-form skill text per generation seed through the current
  provider model, then injects that frozen text into the project's paired
  baseline/skill executor.
- Parses a JSON action plan from the provider response and executes at most 20
  actions with the project's native adapter.
- Uses the project's global verification gate and currently stores only the
  aggregate `VerificationReport` for each candidate.
- Has not run the upstream graph extraction/retrieval procedure and therefore
  cannot be used to claim SkillGen-SDM reproduction or to compare the
  upstream Grounding Rate / Progress Rate / Success Rate / AUPC protocol.

## Consequence for claims

The frozen 3-candidate, 30-pair result is evidence about the project's
train-side generated fixed-skill feasibility path only. It is not evidence
that the upstream SkillGen method fails on native ScienceWorld. A valid
upstream reproduction requires a separate train/dev-only revision that uses
the released data/wrapper, sampling, extraction, retrieval, inference, and
metric definitions, followed by an explicit environment/model compatibility
audit.

No final-test, ACRE acquisition, C/D, or held-out treatment claim is authorized
by this comparison.
