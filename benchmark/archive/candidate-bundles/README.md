# Authoring candidate bundle: tasks 21–30

This directory preserves the intake provenance for the ten packages now
materialized in the canonical v1.0-30 benchmark population. The bundle is based
on `abd36b96d0d2c9f79a3966ae45a680ef16afdd93`; the production copies live under
`benchmark/tasks/` and are governed by the current FamilySpec catalog.

- `spe-evobench-candidate-21-30.zip` is the harness-ready package. Its task
  YAML is serialized in the repository's restricted `miniyaml` subset; task
  semantics and source files are unchanged.
- `spe-evobench-candidate-21-30.source.zip` is the submitted source package,
  retained for provenance.
- `validation_report.source.json` is the submitted package-level validation
  report.

The ten tasks are now registered as explicit FamilySpec anchors and included in
the v1.0-30 population report. They remain calibration candidates: no sealed
formal-50 content or efficacy claim is implied, and CUDA/evolution runtime
calibration remains an explicit gate.

Before formal/sealed use, run each task on the target environment with
independent baseline/oracle/noise calibration. Task 23 requires CUDA validation;
task 30 requires the full evolution harness. A public-dev candidate must later
produce a disjoint sealed lineage (new seed/fixture), rather than reusing this
bundle.
