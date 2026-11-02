from benchmark.boundary.soft_acre import fit_soft_acre


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
