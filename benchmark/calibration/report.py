"""Deterministic calibration report projections."""

from __future__ import annotations

from pathlib import Path

from benchmark.taskgen.validate_population import build_pilot_calibration, build_report


def rebuild_calibration_views(
    *, tasks_root: str | Path, empirical_path: str | Path,
    manifest_path: str | Path | None = None,
):
    """Build report and pilot from task packages, manifest, and empirical evidence."""
    report, errors = build_report(tasks_root, empirical_path, manifest_path)
    pilot = build_pilot_calibration(report, tasks_root)
    return report, pilot, errors

