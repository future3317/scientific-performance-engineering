"""Retrospective, provider-free Soft-ACRE replay on frozen LogicBench receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
from typing import Any, Mapping, Sequence

from benchmark.boundary.soft_acre import SoftAcrePolicy, fit_soft_acre


STUDY_ID = "EVOSPE_SOFT_ACRE_LOGICBENCH_MARGIN_REANALYSIS_20261102_03"
EXTENSION_STUDY_ID = "EVOSPE_SOFT_ACRE_LOGICBENCH_EXTENSION_20261102_03"
EXTENSION_SKILLS = ("logicbench_007", "logicbench_011", "logicbench_017", "logicbench_010")
RISK_WEIGHT = 1.0
COMPLEXITY_PENALTY = 0.002
MAX_LITERALS = 3
SWITCHING_MARGIN_SE = 1.0


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(rows: Sequence[Mapping[str, Any]], selected: Sequence[bool]) -> dict[str, Any]:
    deployed = sum(selected)
    y0 = [bool(row["y0"]) for row in rows]
    y1 = [bool(row["y1"]) for row in rows]
    chosen = [treatment if use else control for treatment, control, use in zip(y1, y0, selected)]
    return {
        "n": len(rows),
        "deployed": deployed,
        "coverage": deployed / len(rows) if rows else None,
        "accuracy_count": sum(chosen),
        "accuracy": sum(chosen) / len(rows) if rows else None,
        "base_accuracy_count": sum(y0),
        "treatment_accuracy_count": sum(y1),
        "incremental_correct": sum((treatment - control) for treatment, control, use in zip(y1, y0, selected) if use),
        "repairs_deployed": sum(use and not control and treatment for treatment, control, use in zip(y1, y0, selected)),
        "regressions_deployed": sum(use and control and not treatment for treatment, control, use in zip(y1, y0, selected)),
    }


def _context(features: Mapping[str, Any], cutpoints: Mapping[str, float]) -> dict[str, Any]:
    context: dict[str, Any] = {"scientific_gate": True}
    for name, value in features.items():
        if name in cutpoints:
            context[name] = "low" if float(value) <= cutpoints[name] else "high"
        else:
            context[name] = value
    return context


def _fit_policy(
    calibration: Sequence[Mapping[str, Any]], *, risk_weight: float, complexity_penalty: float,
    max_literals: int, switching_margin_se: float = SWITCHING_MARGIN_SE,
) -> tuple[SoftAcrePolicy, dict[str, float]]:
    numeric = ("question_length", "bm25_score", "bm25_margin")
    cutpoints = {
        name: median(float(row["features"][name]) for row in calibration)
        for name in numeric
    }
    rows = [
        {"id": row["id"], "family": row["family"], "context": _context(row["features"], cutpoints)}
        for row in calibration
    ]
    scores = {
        row["id"]: {"tau": row["y1"] - row["y0"], "scientific_gate": True}
        for row in calibration
    }
    policy = fit_soft_acre(
        rows, scores,
        feature_names=tuple(key for key in rows[0]["context"] if key != "scientific_gate"),
        risk_weight=risk_weight, complexity_penalty=complexity_penalty,
        max_literals=max_literals, switching_margin_se=switching_margin_se,
    )
    return policy, cutpoints


def _stage_rows(instances: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any],
                y0: Mapping[str, int], all_features: Sequence[str], skill: str,
                stage: str) -> list[dict[str, Any]]:
    outcomes = {row["instance_id"]: int(bool(row["correct"])) for row in evaluation["details"]}
    if set(outcomes) != {row["instance_id"] for row in instances}:
        raise ValueError(f"{skill} {stage} outcome/instance IDs differ")
    rows = []
    for item in instances:
        if item["confirmatory_stage"] != stage:
            raise ValueError(f"unexpected stage for {item['instance_id']}")
        identifier = item["instance_id"]
        if identifier not in y0:
            raise ValueError(f"missing shared Y0 result for {identifier}")
        features = dict(item["confirmatory_features"])
        features.update({name: item[name] for name in all_features if name not in features})
        if set(features) != set(all_features):
            raise ValueError(f"feature fields differ from protocol for {identifier}")
        rows.append({"id": identifier, "family": skill, "features": features,
                     "y0": y0[identifier], "y1": outcomes[identifier]})
    return rows


def _validate_raw_stage(raw_dir: Path, skill: str, stage: str, expected_ids: set[str]) -> None:
    path = raw_dir / f"confirmatory_y1_{skill}_{stage}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["instance_id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != expected_ids:
        raise ValueError(f"{skill} {stage} raw receipts are incomplete or duplicated")
    failures = [identifier for identifier, row in by_id.items() if row.get("error")]
    if failures:
        raise ValueError(f"{skill} {stage} contains provider/executor errors: {failures}")


def freeze_extension_policies(input_dir: Path, calibration_eval_dir: Path,
                              calibration_raw_dir: Path,
                              final_instance_ids: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Fit only on calibration outcomes and freeze decisions for final IDs."""
    protocol = _read(input_dir / "confirmatory_protocol.json")
    y0_eval = _read(input_dir / "confirmatory_y0_eval.json")
    y0 = {row["instance_id"]: int(bool(row["correct"])) for row in y0_eval["details"]}
    all_features = tuple(protocol["allowed_features"])
    gates = {"repairs": 0, "regressions": 0}
    per_skill = []
    for skill in EXTENSION_SKILLS:
        verify_instances = _read(input_dir / f"confirmatory_y1_{skill}_verification_instances.json")
        verify_eval = _read(input_dir / f"confirmatory_y1_{skill}_verification_eval_disabled.json")
        verification = _stage_rows(verify_instances, verify_eval, y0, all_features, skill, "verification")
        repairs = sum(row["y0"] == 0 and row["y1"] == 1 for row in verification)
        regressions = sum(row["y0"] == 1 and row["y1"] == 0 for row in verification)
        if not (repairs > regressions and repairs >= 2):
            raise ValueError(f"{skill} does not pass the frozen verification gate")
        gates["repairs"] += repairs
        gates["regressions"] += regressions

        calibration_instances = _read(input_dir / f"confirmatory_y1_{skill}_calibration_instances.json")
        _validate_raw_stage(calibration_raw_dir, skill, "calibration",
                            {row["instance_id"] for row in calibration_instances})
        calibration_eval = _read(calibration_eval_dir / f"confirmatory_y1_{skill}_calibration_eval_disabled.json")
        calibration = _stage_rows(calibration_instances, calibration_eval, y0, all_features, skill, "calibration")
        final_instances = _read(input_dir / f"confirmatory_y1_{skill}_final_instances.json")
        selected_final_ids = set(final_instance_ids[skill]) if final_instance_ids is not None else None
        final_rows = []
        for item in final_instances:
            if selected_final_ids is not None and item["instance_id"] not in selected_final_ids:
                continue
            if item["confirmatory_stage"] != "final":
                raise ValueError(f"unexpected final stage for {item['instance_id']}")
            features = dict(item["confirmatory_features"])
            features.update({name: item[name] for name in all_features if name not in features})
            if set(features) != set(all_features):
                raise ValueError(f"feature fields differ from protocol for {item['instance_id']}")
            final_rows.append({"id": item["instance_id"], "features": features})

        soft, cutpoints = _fit_policy(
            calibration, risk_weight=RISK_WEIGHT,
            complexity_penalty=COMPLEXITY_PENALTY, max_literals=MAX_LITERALS,
        )
        simple, _ = _fit_policy(
            calibration, risk_weight=0.0, complexity_penalty=0.0, max_literals=1,
            switching_margin_se=0.0,
        )
        decisions = []
        for row in final_rows:
            context = _context(row["features"], cutpoints)
            decisions.append({
                "instance_id": row["id"],
                "features": row["features"],
                "soft_acre_reuse": soft.decide(context).action == "reuse",
                "single_feature_reuse": simple.decide(context).action == "reuse",
                "global_reuse": True,
            })
        per_skill.append({
            "skill": skill,
            "verification_n": len(verification), "verification_repairs": repairs,
            "verification_regressions": regressions,
            "calibration_n": len(calibration), "final_n": len(final_rows),
            "cutpoints": cutpoints,
            "soft_route": soft.route,
            "soft_predicate": [[key, value] for key, value in soft.literals],
            "soft_calibration_objective": soft.calibration_utility,
            "soft_switching_standard_error": soft.switching_standard_error,
            "single_feature_route": simple.route,
            "single_feature_predicate": [[key, value] for key, value in simple.literals],
            "final_decisions": decisions,
        })
    return {
        "study_id": EXTENSION_STUDY_ID,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "original_protocol_status": protocol["status"],
        "model": protocol["model"],
        "skills": list(EXTENSION_SKILLS),
        "n_independent_skills": len(EXTENSION_SKILLS),
        "verification_gate": protocol["verification_gate"],
        "verification_totals": gates,
        "calibration_only_fit": True,
        "final_outcomes_read": False,
        "policy_parameters": {
            "risk_weight": RISK_WEIGHT, "complexity_penalty": COMPLEXITY_PENALTY,
            "max_literals": MAX_LITERALS, "switching_margin_se": SWITCHING_MARGIN_SE,
            "features": list(all_features),
            "numeric_features_binned_at_calibration_median": ["question_length", "bm25_score", "bm25_margin"],
        },
        "per_skill": per_skill,
    }


def score_extension_final(input_dir: Path, final_eval_dir: Path, final_raw_dir: Path,
                          policy_freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Score frozen actions on final model outputs without refitting."""
    y0_eval = _read(input_dir / "confirmatory_y0_eval.json")
    y0 = {row["instance_id"]: int(bool(row["correct"])) for row in y0_eval["details"]}
    aggregate_rows = {name: [] for name in ("soft_acre", "strongest_single_feature", "global_reuse", "base")}
    per_skill = []
    for frozen in policy_freeze["per_skill"]:
        skill = frozen["skill"]
        instances = _read(input_dir / f"confirmatory_y1_{skill}_final_instances.json")
        decisions = {item["instance_id"]: item for item in frozen["final_decisions"]}
        selected_ids = set(decisions)
        instances = [row for row in instances if row["instance_id"] in selected_ids]
        _validate_raw_stage(final_raw_dir, skill, "final", selected_ids)
        evaluation = _read(final_eval_dir / f"confirmatory_y1_{skill}_final_eval_disabled.json")
        rows = _stage_rows(instances, evaluation, y0,
                           tuple(policy_freeze["policy_parameters"]["features"]), skill, "final")
        if set(decisions) != {row["id"] for row in rows}:
            raise ValueError(f"{skill} frozen decisions do not match final outcomes")
        selected = {
            "soft_acre": [bool(decisions[row["id"]]["soft_acre_reuse"]) for row in rows],
            "strongest_single_feature": [bool(decisions[row["id"]]["single_feature_reuse"]) for row in rows],
            "global_reuse": [True] * len(rows),
            "base": [False] * len(rows),
        }
        metrics = {name: _metrics(rows, actions) for name, actions in selected.items()}
        per_skill.append({"skill": skill, "routes": {
            "soft_acre": frozen["soft_route"],
            "strongest_single_feature": frozen["single_feature_route"],
        }, "metrics": metrics})
        for name, actions in selected.items():
            aggregate_rows[name].extend((row, action) for row, action in zip(rows, actions))

    aggregate = {
        name: _metrics([row for row, _ in values], [action for _, action in values])
        for name, values in aggregate_rows.items()
    }
    return {
        "study_id": policy_freeze["study_id"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_class": "new_paid_llm_paired_final_evaluation",
        "provider_calls_expected": sum(row["final_n"] for row in policy_freeze["per_skill"]),
        "n_independent_skills": policy_freeze["n_independent_skills"],
        "per_skill": per_skill,
        "final_cohort_metrics": aggregate,
        "claim_boundary": "Four-skill prospective final split under a fixed DeepSeek Flash and retrieval setup; report as a small benchmark extension with skill-cluster uncertainty, not broad model generalization.",
    }


def run(input_dir: Path) -> dict[str, Any]:
    protocol = _read(input_dir / "confirmatory_protocol.json")
    historical = _read(input_dir / "confirmatory_results.json")
    y0_eval = _read(input_dir / "confirmatory_y0_eval.json")
    y0 = {row["instance_id"]: int(bool(row["correct"])) for row in y0_eval["details"]}
    skills = tuple(historical["skills"])
    if skills != tuple(protocol["skills"][:len(skills)]):
        raise ValueError("final-skill cohort differs from the frozen protocol order")
    if historical["final_outcomes_used_for_policy"] is not False:
        raise ValueError("historical final outcomes were used to freeze the original policies")

    per_skill = []
    final_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    all_features = tuple(protocol["allowed_features"])
    for skill in skills:
        calibration_instances = _read(input_dir / f"confirmatory_y1_{skill}_calibration_instances.json")
        calibration_eval = _read(input_dir / f"confirmatory_y1_{skill}_calibration_eval_disabled.json")
        final_instances = _read(input_dir / f"confirmatory_y1_{skill}_final_instances.json")
        final_eval = _read(input_dir / f"confirmatory_y1_{skill}_final_eval_disabled.json")

        def make_rows(instances, evaluation, expected_stage):
            outcomes = {row["instance_id"]: int(bool(row["correct"])) for row in evaluation["details"]}
            if set(outcomes) != {row["instance_id"] for row in instances}:
                raise ValueError(f"{skill} {expected_stage} outcome/instance IDs differ")
            rows = []
            for item in instances:
                if item["confirmatory_stage"] != expected_stage:
                    raise ValueError(f"unexpected stage for {item['instance_id']}")
                identifier = item["instance_id"]
                if identifier not in y0:
                    raise ValueError(f"missing shared Y0 result for {identifier}")
                features = dict(item["confirmatory_features"])
                features.update({name: item[name] for name in all_features if name not in features})
                if set(features) != set(all_features):
                    raise ValueError(f"feature fields differ from protocol for {identifier}")
                rows.append({
                    "id": identifier, "family": skill, "features": features,
                    "y0": y0[identifier], "y1": outcomes[identifier],
                })
            return rows

        calibration = make_rows(calibration_instances, calibration_eval, "calibration")
        final = make_rows(final_instances, final_eval, "final")
        soft, cutpoints = _fit_policy(
            calibration, risk_weight=RISK_WEIGHT,
            complexity_penalty=COMPLEXITY_PENALTY, max_literals=MAX_LITERALS,
        )
        simple, _ = _fit_policy(
            calibration, risk_weight=0.0, complexity_penalty=0.0, max_literals=1,
            switching_margin_se=0.0,
        )
        soft_selected = [soft.decide(_context(row["features"], cutpoints)).action == "reuse" for row in final]
        simple_selected = [simple.decide(_context(row["features"], cutpoints)).action == "reuse" for row in final]
        per_skill.append({
            "skill": skill,
            "calibration_n": len(calibration),
            "final_n": len(final),
            "calibration_feature_cutpoints": cutpoints,
            "soft_route": soft.route,
            "soft_predicate": [[key, value] for key, value in soft.literals],
            "soft_calibration_objective": soft.calibration_utility,
            "soft_switching_standard_error": soft.switching_standard_error,
            "single_feature_route": simple.route,
            "single_feature_predicate": [[key, value] for key, value in simple.literals],
            "soft_metrics": _metrics(final, soft_selected),
            "single_feature_metrics": _metrics(final, simple_selected),
            "global_metrics": _metrics(final, [True] * len(final)),
            "base_metrics": _metrics(final, [False] * len(final)),
        })
        for row, use_soft, use_simple in zip(final, soft_selected, simple_selected):
            final_rows.append({**row, "soft_selected": use_soft})
            single_rows.append({**row, "single_selected": use_simple})

    soft_all = _metrics(final_rows, [row["soft_selected"] for row in final_rows])
    simple_all = _metrics(single_rows, [row["single_selected"] for row in single_rows])
    global_all = _metrics(final_rows, [True] * len(final_rows))
    base_all = _metrics(final_rows, [False] * len(final_rows))
    historical_full_acre = []
    for line in (input_dir / "confirmatory_policy_decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            decision = json.loads(line)
            if decision.get("policy") == "full_acre":
                historical_full_acre.append(decision)
    full_acre_by_id = {row["instance_id"]: bool(row["deploy"]) for row in historical_full_acre}
    ids = {row["id"] for row in final_rows}
    if set(full_acre_by_id) != ids:
        raise ValueError("historical Full ACRE decisions do not match the final cohort")
    full_acre = _metrics(final_rows, [full_acre_by_id[row["id"]] for row in final_rows])

    return {
        "study_id": STUDY_ID,
        "source_class": "retrospective_provider_free_policy_replay",
        "provider_calls": 0,
        "model": protocol["model"],
        "protocol_skills": list(skills),
        "n_independent_skills": len(skills),
        "policy_parameters": {
            "risk_weight": RISK_WEIGHT,
            "complexity_penalty": COMPLEXITY_PENALTY,
            "max_literals": MAX_LITERALS,
            "switching_margin_se": SWITCHING_MARGIN_SE,
            "features": list(all_features),
            "numeric_features_binned_at_calibration_median": ["question_length", "bm25_score", "bm25_margin"],
        },
        "per_skill": per_skill,
        "final_cohort_metrics": {
            "soft_acre": soft_all,
            "strongest_single_feature": simple_all,
            "global_reuse": global_all,
            "full_acre_frozen": full_acre,
            "base": base_all,
        },
        "claim_boundary": "Exploratory retrospective replay on the already opened historical final cohort; policy parameters are fit on calibration outcomes only, but this is not a prospective or independent efficacy estimate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "freeze", "score"), default="replay")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--calibration-eval-dir", type=Path)
    parser.add_argument("--calibration-raw-dir", type=Path)
    parser.add_argument("--final-eval-dir", type=Path)
    parser.add_argument("--final-raw-dir", type=Path)
    parser.add_argument("--policy-freeze", type=Path)
    parser.add_argument("--study-manifest", type=Path, help="use its predeclared final-ID subset during freeze")
    args = parser.parse_args()
    if args.mode == "replay":
        result = run(args.input_dir)
    elif args.mode == "freeze":
        if args.calibration_eval_dir is None or args.calibration_raw_dir is None:
            parser.error("--mode freeze requires calibration eval and raw receipt directories")
        final_ids = _read(args.study_manifest)["stage_instance_ids"]["final"] if args.study_manifest else None
        by_skill = None
        if final_ids is not None:
            by_skill = {skill: [] for skill in EXTENSION_SKILLS}
            instance_skill = {}
            for skill in EXTENSION_SKILLS:
                items = _read(args.input_dir / f"confirmatory_y1_{skill}_final_instances.json")
                instance_skill.update({item["instance_id"]: skill for item in items})
            for identifier in final_ids:
                skill = instance_skill.get(identifier)
                if skill is None:
                    raise ValueError("study manifest contains unknown final instance IDs")
                by_skill[skill].append(identifier)
        result = freeze_extension_policies(args.input_dir, args.calibration_eval_dir,
                                           args.calibration_raw_dir, by_skill)
    else:
        if args.final_eval_dir is None or args.final_raw_dir is None or args.policy_freeze is None:
            parser.error("--mode score requires final eval/raw directories and --policy-freeze")
        result = score_extension_final(args.input_dir, args.final_eval_dir,
                                       args.final_raw_dir, _read(args.policy_freeze))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
