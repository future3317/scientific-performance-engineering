# Governed Rule Integrity and Utility Implementation Plan

> **For agentic workers:** Execute this plan inline with test-first changes; keep one production path and run the repository contract suite before publishing.

**Goal:** Make experience-to-rule evolution auditable, leakage-resistant, replay-backed, calibrated, attributable, and maintainable without enlarging the runtime `SKILL.md`.

**Architecture:** Extend the existing JSON contracts rather than adding a parallel store. A replay runner produces a paired-intervention manifest with digests and Bayesian admission statistics; evolution validation checks provenance, held-out separation, graph integrity, and human review. A usage record and a small MDL/rate-distortion scorer close the runtime-telemetry and library-maintenance loop.

**Tech Stack:** Python 3 standard library, JSON Schema-shaped contracts, existing repository scripts, Git revision metadata.

## Global Constraints

- Experience records are evidence only; runtime collection cannot promote or edit canonical rules.
- Canonical promotion requires a real replay manifest, held-out admission cases, passing regression cases, scientific gates, and explicit human review provenance.
- No new dependency, daemon, hook, automatic PR, or automatic canonical mutation.
- Hashes are used only for replay/artifact provenance where the digest changes acceptance or reproducibility decisions.
- Preserve `SKILL.md` as a small router; detailed policy stays in references and scripts.

### Task 1: Lock the missing contracts with failing fixtures

Add assertions for source-case existence/status, disjoint admission/regression lineage, structured artifact provenance, review provenance, graph cycles/dangling edges, replay-manifest verification, usage telemetry, Bayesian evidence, and MDL scoring. Run the focused fixtures and confirm they fail against the current implementation.

### Task 2: Implement provenance and replay-gated promotion

Update schemas/templates and `validate_evolution.py`; add `run_rule_replay.py`. The runner reads paired on/off outcomes, checks scientific gates, computes paired utility effect and a confidence lower bound, computes Beta-posterior admission probability, and writes a digest-attested manifest. The validator accepts canonical cards only when that manifest exists and its content matches the card.

### Task 3: Implement usage telemetry and library maintenance scoring

Add a validated usage-record schema/template and capture/validation script. Add `score_rule_library.py` to calculate description length, replay utility distortion, conflict cost, and actionable merge/retire/specialize recommendations without mutating rules.

### Task 4: Route documentation and update regression fixtures

Move stale generic algorithmic/multitask gates out of `SKILL.md` into routed references, document the three innovations and the exact commands, and update README plus all contract fixtures/templates.

### Task 5: Verify, review diff, commit, and push

Run focused red-green tests, all validators/self-tests, compile checks, `git diff --check`, inspect the final diff, commit on `main`, and push to `origin/main`.

## Completion Record

- [x] Contract fixtures were made to fail before implementation and pass after it.
- [x] Replay, provenance, usage, Bayesian, graph, and rate-distortion paths are runnable without a new dependency.
- [x] Full repository checks passed locally; remote CI is run after the `main` push.
