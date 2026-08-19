# Authoring candidate bundle: tasks 21–30

This directory is an intake archive, not part of the canonical benchmark
population. The bundle is based on `abd36b96d0d2c9f79a3966ae45a680ef16afdd93`
and must be calibrated in an isolated worktree before any task can be admitted.

- `spe-evobench-candidate-21-30.zip` is the harness-ready package. Its task
  YAML is serialized in the repository's restricted `miniyaml` subset; task
  semantics and source files are unchanged.
- `spe-evobench-candidate-21-30.source.zip` is the submitted source package,
  retained for provenance.
- `validation_report.source.json` is the submitted package-level validation
  report.

The ten tasks remain authoring candidates. They are not copied into
`benchmark/tasks/`, are not added to `benchmark/manifests/v1.0-50-slots.json`,
and do not contribute formal or efficacy results. The package's candidate task
IDs are intentionally not registered as canonical FamilySpec anchors yet.

Before admission, run each task on the target environment with independent
baseline/oracle/noise calibration. Task 23 requires CUDA validation; task 30
requires the full evolution harness. A public-dev candidate must later produce
a disjoint sealed lineage (new seed/fixture), rather than reusing this bundle.
