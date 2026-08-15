# EVOL-EPISODE-POISON-10 — sequential evolution episode

## Goal

`solution.py` implements the `episode_v1` API. It is given a task workspace, a
skill view, and a budget, and must run the sequential episode defined in
`../episodes/poison_episode.yaml`. The current baseline runs the episode under
condition **C** (append-only); your job is to make it run under condition **D**
(governed, replay-grounded) so that poisoned experiences are filtered out and
the `poisoning_survival_rate` increases.

## API (fixed — `episode_v1`)

- `run_episode_task(task_workspace: str, skill_view: dict, budget: dict) -> dict`
  - Must return a dict containing at least:
    - `episode_score` (float, 0..1): the metric used by the verifier.
    - `episode_metrics` (dict): the raw metrics from the episode run.
  - `task_workspace` is the path to this `workspace/` directory.
  - `skill_view` carries the condition-specific skill view.
  - `budget` contains execution limits.

## Constraints

- stdlib + torch only. No PyG, no network, no file I/O outside the workspace.
- Do not touch timing, randomness, or torch internals; do not spawn threads or
  processes.
- The episode YAML and poison records must not be modified by the agent; the
  harness verifies the package integrity.

## Time budget

600 s total for your session; each verification run of the task finishes in
well under a minute.

## Required diagnosis

Before you finish, state the **bottleneck mechanism** you identified (one of:
`scalar_sync`, `h2d_blocking`, `repeated_compute`, `launch_fragmentation`,
`graph_rebuild`, `compile_break`, `memory_pressure`) and justify it with the
evidence you collected. The mechanism name is scored.

## Smoke test

```
python public_tests/smoke_test.py
```

runs your workspace `solution.py` and checks that it returns a valid episode
result.
