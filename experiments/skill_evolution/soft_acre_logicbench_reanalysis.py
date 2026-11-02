"""Retrospective, provider-free Soft-ACRE replay on frozen LogicBench receipts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from benchmark.boundary.soft_acre import SoftAcrePolicy, fit_soft_acre


STUDY_ID = "EVOSPE_SOFT_ACRE_LOGICBENCH_REANALYSIS_20261102_01"
RISK_WEIGHT = 1.0
COMPLEXITY_PENALTY = 0.002
MAX_LITERALS = 3


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
    max_literals: int,
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
        max_literals=max_literals,
    )
    return policy, cutpoints


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
        simple, _ = _fit_policy(calibration, risk_weight=0.0, complexity_penalty=0.0, max_literals=1)
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
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
