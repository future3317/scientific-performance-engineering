#!/usr/bin/env python3
"""SPE-EvoBench harness CLI (BENCHMARK_DESIGN.md section 11).

Run from the repository root (the directory containing ``benchmark/``)::

    python -m benchmark.harness.cli validate-task tasks/<id>
    python -m benchmark.harness.cli run-task tasks/<id> --solution DIR --out result.json
    python -m benchmark.harness.cli materialize-condition {A,B,C,C_STRESS,D} --snapshot DIR --out DIR
    python -m benchmark.harness.cli run-episode episodes/<id>.yaml --condition {C,C_STRESS,D} --out DIR
    python -m benchmark.harness.cli check-leakage split/sequential.yaml
    python -m benchmark.harness.cli score-run RUN_DIR --out scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cmd_validate_task(args: argparse.Namespace) -> int:
    from . import verifier

    errors = verifier.validate_task(args.task_dir, check_fixtures=not args.no_fixture_check)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validate-task: {args.task_dir} OK")
    return 0


def _cmd_run_task(args: argparse.Namespace) -> int:
    from . import verifier

    predicted = None
    if args.predict_mechanism:
        predicted = [item.strip() for item in args.predict_mechanism.split(",") if item.strip()]
    result = verifier.verify_task(
        args.task_dir,
        args.solution,
        out_path=args.out,
        predicted_mechanism=predicted,
        seed=args.seed,
        condition=args.condition,
        context_mode=args.context_mode,
        noise_control_path=args.noise_control,
        noise_control_required=args.noise_control_required,
        noise_control_expected={
            key: value for key, value in {
                "outer_trial_id": args.outer_trial_id,
                "benchmark_revision": args.benchmark_revision,
                "task_manifest_digest": args.task_manifest_digest,
            }.items() if value is not None
        },
    )
    verdict = result["verdict"]
    speedup = result.get("verified_speedup", {})
    print(
        f"run-task: verdict={verdict} "
        f"median_speedup={speedup.get('median_speedup')} verified={speedup.get('verified')} "
        f"-> {args.out}"
    )
    return {"pass": 0, "fail": 1, "inconclusive": 3, "error": 2}.get(verdict, 2)


def _cmd_calibrate_noise_control(args: argparse.Namespace) -> int:
    from . import verifier

    verifier.calibrate_noise_control(
        args.task_dir,
        args.solution,
        args.out,
        task_id=args.task_id,
        outer_trial_id=args.outer_trial_id,
        benchmark_revision=args.benchmark_revision,
        task_manifest_digest=args.task_manifest_digest,
        compiler_cache_policy=args.compiler_cache_policy,
        seed=args.seed,
    )
    print(f"calibrate-noise-control: {args.task_id}/{args.outer_trial_id} -> {args.out}")
    return 0


def _cmd_materialize_condition(args: argparse.Namespace) -> int:
    from . import conditions

    manifest = conditions.materialize_condition(args.condition, args.snapshot, args.out, context_mode=args.context_mode)
    print(
        f"materialize-condition: {args.condition} -> {args.out} "
        f"({len(manifest['files'])} files attested)"
    )
    return 0


def _cmd_run_episode(args: argparse.Namespace) -> int:
    from . import evolution

    result = evolution.run_episode(
        args.episode,
        args.condition,
        args.out,
        snapshot_dir=args.snapshot,
        core_repo=args.core_repo,
        context_mode=args.context_mode,
    )
    print(f"run-episode: {result['episode_id']} condition={result['condition']} -> {args.out}")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value}")
    return 0


def _cmd_check_leakage(args: argparse.Namespace) -> int:
    from . import split

    tasks_root = args.tasks_root or (_repo_root() / "benchmark" / "tasks")
    errors = split.check_leakage(args.manifest, tasks_root)
    if errors:
        for error in errors:
            print(f"LEAK: {error}", file=sys.stderr)
        return 1
    print(f"check-leakage: {args.manifest} clean (tasks_root={tasks_root})")
    return 0


def _cmd_score_run(args: argparse.Namespace) -> int:
    from . import scoring

    scores = scoring.score_run(args.run_dir)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    overall = scores["overall"]
    print(
        f"score-run: {args.run_dir} tasks={overall['num_tasks']} "
        f"pass_rate={overall['pass_rate']} composite={overall['composite']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.harness.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-task", help="schema + self-consistency checks for a task package")
    p.add_argument("task_dir", type=Path)
    p.add_argument("--no-fixture-check", action="store_true", help="skip the fixture determinism check")
    p.set_defaults(func=_cmd_validate_task)

    p = sub.add_parser("run-task", help="run the S0-S6 verification pipeline on a candidate solution")
    p.add_argument("task_dir", type=Path)
    p.add_argument("--solution", type=Path, required=True, help="candidate workspace directory")
    p.add_argument("--out", type=Path, required=True, help="result.json output path")
    p.add_argument("--predict-mechanism", default=None, help="comma-separated predicted mechanism ids (diagnosis)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="standalone", choices=["A", "B", "C", "D", "standalone"])
    p.add_argument("--context-mode", default="reset", choices=["reset", "carry"])
    p.add_argument("--noise-control", type=Path, default=None)
    p.add_argument("--noise-control-required", action="store_true")
    p.add_argument("--outer-trial-id", default=None)
    p.add_argument("--benchmark-revision", default=None)
    p.add_argument("--task-manifest-digest", default=None)
    p.set_defaults(func=_cmd_run_task)

    p = sub.add_parser("calibrate-noise-control", help="run one same-host baseline-vs-baseline calibration")
    p.add_argument("task_dir", type=Path)
    p.add_argument("--solution", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--outer-trial-id", required=True)
    p.add_argument("--benchmark-revision", required=True)
    p.add_argument("--task-manifest-digest", required=True)
    p.add_argument("--compiler-cache-policy", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_calibrate_noise_control)

    p = sub.add_parser("materialize-condition", help="build an A/B/C/D condition store from a snapshot")
    p.add_argument("condition", choices=["A", "B", "C", "C_STRESS", "D"])
    p.add_argument("--snapshot", type=Path, default=None, help="pinned skill snapshot (required for B/C/D)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-mode", choices=["reset", "carry"], default="reset")
    p.set_defaults(func=_cmd_materialize_condition)

    p = sub.add_parser("run-episode", help="run an evolution episode under condition C or D")
    p.add_argument("episode", type=Path)
    p.add_argument("--condition", choices=["C", "C_STRESS", "D"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--core-repo", type=Path, default=None)
    p.add_argument("--context-mode", choices=["reset", "carry"], default="reset")
    p.set_defaults(func=_cmd_run_episode)

    p = sub.add_parser("check-leakage", help="verify the sequential split has no group leakage")
    p.add_argument("manifest", type=Path, help="split/sequential.yaml")
    p.add_argument("--tasks-root", type=Path, default=None)
    p.set_defaults(func=_cmd_check_leakage)

    p = sub.add_parser("score-run", help="aggregate per-task and track scores for a run directory")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_score_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
