from benchmark.boundary.soft_acre import fit_soft_acre, select_soft_acre_candidate


def _rows(kind):
    rows = []
    scores = {}
    for family in ("f1", "f2", "f3"):
        for task_type, tau in (("matched", 0.2), ("distractor", -0.2)):
            identifier = f"{family}-{task_type}"
            rows.append({"id": identifier, "family": family, "context": {"task_type": task_type, "scientific_gate": True}})
            scores[identifier] = {"tau": tau, "scientific_gate": True}
    if kind == "broad":
        for score in scores.values():
            score["tau"] = 0.1
    elif kind == "null":
        for score in scores.values():
            score["tau"] = 0.0
    return rows, scores


def test_soft_acre_audits_broad_and_selective_support_on_calibration():
    rows, scores = _rows("selective")
    selective = fit_soft_acre(rows, scores, feature_names=("task_type",))
    assert selective.route == "selective"
    assert selective.literals == (("task_type", "matched"),)

    rows, scores = _rows("broad")
    broad = fit_soft_acre(rows, scores, feature_names=("task_type",))
    assert broad.route == "global"


def test_soft_acre_abstains_without_positive_calibration_utility():
    rows, scores = _rows("null")
    policy = fit_soft_acre(rows, scores, feature_names=("task_type",))
    assert policy.route == "abstain"


def test_scientific_gate_remains_hard_for_soft_policy():
    rows, scores = _rows("broad")
    scores["f1-matched"]["scientific_gate"] = False
    rows[0]["context"]["scientific_gate"] = False
    policy = fit_soft_acre(rows, scores, feature_names=("task_type",))
    assert policy.route == "global"
    assert policy.decide(rows[0]["context"]).action == "abstain"


def test_soft_acre_recovers_a_three_field_interaction():
    rows = []
    scores = {}
    for family in ("f1", "f2", "f3", "f4"):
        for task_type in ("matched", "adjacent"):
            for scale in ("small", "large"):
                for compatible in (0, 1):
                    identifier = f"{family}-{task_type}-{scale}-{compatible}"
                    context = {
                        "task_type": task_type,
                        "scale_regime": scale,
                        "constraint_compatibility": compatible,
                        "scientific_gate": True,
                    }
                    complete_match = task_type == "matched" and scale == "large" and compatible == 1
                    partial_match = sum((task_type == "matched", scale == "large", compatible == 1)) == 2
                    rows.append({"id": identifier, "family": family, "context": context})
                    scores[identifier] = {"tau": 0.25 if complete_match else -0.20 if partial_match else 0.0,
                                          "scientific_gate": True}

    policy = fit_soft_acre(
        rows,
        scores,
        feature_names=("task_type", "scale_regime", "constraint_compatibility"),
        risk_weight=1.0,
        complexity_penalty=0.002,
        max_literals=3,
    )
    simple = fit_soft_acre(
        rows,
        scores,
        feature_names=("task_type", "scale_regime", "constraint_compatibility"),
        risk_weight=0.0,
        complexity_penalty=0.0,
        max_literals=1,
    )
    assert policy.literals == (
        ("task_type", "matched"),
        ("scale_regime", "large"),
        ("constraint_compatibility", 1),
    )
    assert policy.calibration_utility > simple.calibration_utility


def test_soft_acre_requires_a_one_se_gain_to_leave_global_reuse():
    rows = [{"id": f"x{i}", "family": "skill", "context": {"type": "A" if i < 8 else "B"}}
            for i in range(10)]
    scores = {row["id"]: {"tau": 0.01 if i < 9 else -0.01, "scientific_gate": True}
              for i, row in enumerate(rows)}
    mask = {row["id"]: i < 8 for i, row in enumerate(rows)}

    selected = select_soft_acre_candidate(
        rows, scores,
        {"global": {row["id"]: True for row in rows},
         "abstain": {row["id"]: False for row in rows},
         "cegis": mask},
    )

    assert selected.name == "global"
    assert selected.baseline_name == "global"
    assert selected.switching_standard_error > 0


def test_soft_acre_selects_a_candidate_with_a_clear_calibration_gain():
    rows, scores = _rows("selective")
    masks = {
        "global": {row["id"]: True for row in rows},
        "abstain": {row["id"]: False for row in rows},
        "cegis": {row["id"]: row["context"]["task_type"] == "matched" for row in rows},
    }
    selected = select_soft_acre_candidate(rows, scores, masks)
    assert selected.name == "cegis"
