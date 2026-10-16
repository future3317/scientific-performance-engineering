"""Provider-free behavioral contracts for the registered three-arm attribution."""
from dataclasses import replace
import json
import pytest

from core.acre.resource_budget import manifest_digest
from core.skills import SkillContract
from experiments.skill_evolution.attribution import (
    ARMS, LENGTH_CONTROL, arm_fingerprints, arm_texts, diagnose_failure,
    episode_metrics, main, summarize_attribution, validate_manifest,
)


def skill():
    return SkillContract(skill_id="fixture.skill",version=1,task_family="pick_and_place_simple",subtask="search",
        intent="Search and verify.",applicability=("Visible goal.",),procedure=("Inspect an unvisited location.",),
        procedure_step_ids=("inspect",),required_procedure_step_ids=("inspect",),preconditions=("Visible goal.",),
        postconditions=("Official completion.",),protected_semantics=("Admissible actions only.",),source_trial_ids=("source-1",))


PRICING={"input_hit_per_million":"0.006","input_miss_per_million":"0.3","output_per_million":"1.2",
         "off_peak":{"input_hit_per_million":"0.003","input_miss_per_million":"0.15","output_per_million":"0.6"}}


def task(i=0):
    return {"cluster_id":f"c{i}","task_id":f"t{i}","template_id":f"p{i}","gamefile":f"/fixture/g{i}",
            "seed":67,"arm_order":list(ARMS)}


def receipt(arm="OFF",outcome=True,i=0):
    return {**task(i),"arm":arm,"trial_id":f"trial-{i}-{arm}","status":"VERIFIED_SUCCESS"if outcome is True else
            "VERIFIED_FAILURE"if outcome is False else "PROVIDER_FAILURE", "success":outcome,
            "environment_won":outcome,"environment_done":True if outcome is not None else None,
            "termination_reason":"success"if outcome is True else "environment_done"if outcome is False else "policy_error",
            "env_steps":1,"skill_injected":arm=="SKILL","skill_rendered":arm=="SKILL",
            "trajectory":[{"selected_action":"look","observation":"Visible room.","step":0,"invalid_action":False,
                           "admissible_actions":["look"],"completed_subgoals":[],"input_tokens":100,"output_tokens":60,
                           "reasoning_tokens":50,"cache_hit_tokens":20,"cache_miss_tokens":80,
                           "estimated_provider_usd":"0.00009612"}],"progress_events":[]}


def test_arms_change_only_injected_content_and_no_hidden_skill():
    fp=arm_fingerprints(skill())
    assert len({v["prompt_sha256"]for v in fp.values()})==3
    assert len({v["messages"][0]["content"]for v in fp.values()})==1
    assert skill().body not in fp["OFF"]["messages"][1]["content"]
    assert skill().body not in fp["LENGTH"]["messages"][1]["content"]
    assert LENGTH_CONTROL=="Be concise. Do not produce unnecessary explanation. Return the next valid action efficiently."
    assert all(word not in LENGTH_CONTROL.lower()for word in("search","container","place","inventory","object"))
    for arm,text in arm_texts(skill()).items():
        normalized=fp[arm]["messages"][1]["content"].replace(text or "No reusable skill is provided.","<TREATMENT>")
        assert normalized==fp["OFF"]["messages"][1]["content"].replace("No reusable skill is provided.","<TREATMENT>")
    with pytest.raises(ValueError,match="length-control"):
        arm_texts(replace(skill(),procedure=("Be concise.",)))


def test_metrics_keep_reported_usage_distinct_and_price_windows():
    m=episode_metrics(receipt(),PRICING)
    assert m["final_content_tokens"]==10 and m["total_tokens"]==160
    assert m["peak_usd_estimate"]==pytest.approx(.00009612)
    assert m["off_peak_usd_estimate"]==pytest.approx(.00004806)
    assert m["actual_billed_usd"]is None and m["skill_usage_confidence"]=="UNKNOWN"
    row=receipt(); row["trajectory"][0]["reasoning_tokens"]=None
    assert episode_metrics(row,PRICING)["reasoning_tokens"]is None
    assert episode_metrics(row,PRICING)["final_content_tokens"]is None


def test_shared_control_counts_pairs_and_missing_not_failure():
    rows=[receipt("OFF",False),receipt("LENGTH",None),receipt("SKILL",True)]
    result=summarize_attribution([task()],rows,PRICING)
    assert result["planned_independent_clusters"]==1
    assert result["comparisons"]["OFF_vs_SKILL"]["repair"]==1
    assert result["comparisons"]["OFF_vs_LENGTH"]["missing"]==1
    assert result["comparisons"]["OFF_vs_LENGTH"]["00"]==0
    assert result["comparisons"]["OFF_vs_SKILL"]["resource_savings"]["provider_usd_estimate"]["n"]==0
    assert result["interaction_effect"]is None
    with pytest.raises(ValueError,match="duplicate"):
        summarize_attribution([task()],rows+[rows[0]],PRICING)
    rows[2]["seed"]=999
    with pytest.raises(ValueError,match="seed"):
        summarize_attribution([task()],rows,PRICING)


def test_both_success_efficiency_and_no_discordance_is_not_equivalence():
    a,b=receipt(),receipt("SKILL")
    b["trajectory"][0]["output_tokens"]=40
    b["trajectory"][0]["reasoning_tokens"]=30
    result=summarize_attribution([task()],[a,b],PRICING)["comparisons"]["OFF_vs_SKILL"]
    assert result["11"]==1 and result["exact_mcnemar_p"]is None
    assert result["resource_savings"]["reasoning_tokens"]["median"]==20
    assert result["delta_success"]==0


def test_public_failure_diagnosis_does_not_invent_distance_or_hidden_location():
    row=receipt(outcome=False)
    row["trajectory"]=[{**row["trajectory"][0],"step":i,"selected_action":"go to cabinet 1",
        "admissible_actions":["go to cabinet 1","go to cabinet 2"]}for i in range(3)]
    d=diagnose_failure(row)
    assert d["valid_actions"]==3 and d["repeated_actions"]==2
    assert d["repeated_location_ratio"]==pytest.approx(2/3)
    assert d["unvisited_initial_locations"]==["cabinet 2"]
    assert d["goal_state_distance"]is None and d["longest_no_progress_window"]==3


def test_final_outcome_and_exposed_tasks_cannot_enter_inner_manifest(tmp_path,monkeypatch):
    member={**task(),"split":"train_inner","confirmatory_eligible":True}
    parent={"task_membership":[member]}
    s=skill();manifest={"stage":"skill_attribution","split":"train_inner","split_manifest_sha256":manifest_digest(parent),
        "skill_digest":s.digest,"skill_body_sha256":s.body_digest,"arms":list(ARMS),"treatments":arm_texts(s),
        "excluded_clusters":[],"excluded_templates":[],"plan":[task()],"seed":67,"thinking":"enabled","max_retries":0}
    validate_manifest(manifest,parent,s)
    manifest["excluded_clusters"]=["c0"]
    with pytest.raises(ValueError,match="exposed"):
        validate_manifest(manifest,parent,s)
    manifest["excluded_clusters"]=[]; parent["task_membership"][0]["split"]="train_outer"
    manifest["split_manifest_sha256"]=manifest_digest(parent)
    with pytest.raises(ValueError,match="eligible inner"):
        validate_manifest(manifest,parent,s)
    parent["task_membership"][0]["split"]="train_inner";manifest["split_manifest_sha256"]=manifest_digest(parent)
    for name,value in (("manifest",manifest),("split",parent),("skill",s.to_dict())):
        (tmp_path/name).write_text(json.dumps(value))
    import experiments.external_real.alfworld_stateful_runner as runner
    monkeypatch.setattr(runner.DeepSeekStatefulClient,"__init__",lambda *a,**k:pytest.fail("provider constructed"))
    assert main(["--manifest",str(tmp_path/"manifest"),"--split-manifest",str(tmp_path/"split"),"--skill",str(tmp_path/"skill"),
                 "--approval",str(tmp_path/"missing-approval"),"--pricing",str(tmp_path/"missing-pricing")])==0
