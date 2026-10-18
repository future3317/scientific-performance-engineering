"""Exercise trace mining, bounded patching, paired verification and rollback with synthetic outcomes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.skills import PairReceipt, Patch, SkillEvolution, SkillEvolutionStore, SkillPatchOptimizer, TraceContrastMiner, TrialTrace
from core.skills.evolution import approve_outer_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "evidence_kind": "synthetic_implementation_replay", "provider_calls": 0,
        "inner_pairs": 64, "outer_pairs": 64, "outer_alpha": 0.025,
        "inner_scope": "train-only candidate screening; not a joint confidence claim",
        "outer_scope": "independent synthetic calibration tasks; not real-task evidence",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    traces = [
        TrialTrace("source-good", "source-task", "source-group", "train", True, "verified", None,
                   ("go to cabinet 1", "take apple 1 from cabinet 1", "clean apple 1 with sinkbasin 1", "put apple 1 in/on table 1"), "clean_place"),
        TrialTrace("source-bad", "source-task", "source-group", "train", False, "verified", "incomplete",
                   ("go to cabinet 1", "take apple 1 from cabinet 1", "put apple 1 in/on table 1"), "clean_place"),
    ]
    parent = TraceContrastMiner().extract_subskill("alfworld.clean-place", traces)
    patch = Patch("ADD_STEP", 1, "Inspect visible state before selecting the next action.")
    rows = {}
    for name, count, off, on in (
        ("positive", 64, False, True), ("regression", 64, True, False),
        ("low_power", 1, False, True), ("outer_regression", 64, False, True),
        ("outer_low_power", 64, False, True),
    ):
        optimizer = SkillPatchOptimizer(max_skill_characters=3000)
        preview, errors = optimizer.apply(parent, [patch])
        if preview is None:
            raise ValueError(errors)
        receipts = []
        for index in range(count):
            for arm, success in (("off", off), ("on", on)):
                receipts.append(PairReceipt(
                    f"{name}-{index}-{arm}", f"pair-{index}", f"task-{index}", index, arm,
                    "train", preview.digest if arm == "on" else None,
                    "synthetic-agent", "synthetic-environment", success, float(success),
                ))
        store = SkillEvolutionStore(args.output / name)
        store.save_candidate(parent)
        result = SkillEvolution(SkillPatchOptimizer(max_skill_characters=3000), store).run_round(parent, [patch], receipts)
        outer = None
        if result.accepted:
            outer_receipts = []
            for index in range(1 if name == "outer_low_power" else 64):
                for arm, success in (("off", name == "outer_regression"), ("on", name != "outer_regression")):
                    outer_receipts.append(PairReceipt(
                        f"{name}-outer-{index}-{arm}", f"outer-pair-{index}", f"outer-task-{index}",
                        index, arm, "calibration", result.candidate.digest if arm == "on" else None,
                        "synthetic-agent", "synthetic-environment", success, float(success),
                    ))
            outer = approve_outer_validation(
                result.candidate, preview.digest, result.evidence, outer_receipts,
                split="calibration", alpha=config["outer_alpha"],
            )
            if outer.assessment_status == "CANDIDATE_ACCEPTED":
                store.record_acceptance(result.candidate, outer)
            else:
                store.record_evidence(result.candidate, outer)
                if outer.assessment_status == "REJECTED":
                    store.rollback(result.candidate, parent, "independent_outer_regression",
                                   evidence_ids=outer.evidence_trial_ids)
        store.journal.verify_against_store_diff(store.root)
        rows[name] = asdict(result)
        rows[name]["outer_validation"] = asdict(outer) if outer is not None else None
        rows[name]["final_candidate_accepted"] = bool(outer and outer.assessment_status == "CANDIDATE_ACCEPTED")
    result = {"evidence_kind": "synthetic_implementation_replay", "provider_calls": 0,
              "cases": rows, "real_task_efficacy": "NOT_ESTABLISHED"}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {
        "inner_assessment": row["evidence"]["assessment_status"],
        "outer_assessment": row["outer_validation"]["assessment_status"] if row["outer_validation"] else "NOT_RUN",
        "final_candidate_accepted": row["final_candidate_accepted"],
    } for name, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
