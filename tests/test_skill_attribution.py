"""Provider-free behavioral contracts for the registered three-arm attribution."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import pytest

from core.acre.resource_budget import manifest_digest
from core.skills import SkillContract
from experiments.skill_evolution.attribution import (
    ARMS, E1_ARMS, LENGTH_CONTROL, arm_fingerprints, arm_texts, diagnose_failure, conditional_increment_decision,
    episode_metrics, main, run_attribution, summarize_attribution, summarize_spooled, validate_manifest, select_conditional_plan,
)


def test_conditional_composer_preserves_exact_v1_and_base_messages():
    texts=arm_texts(skill(),design="conditional_increment")
    assert texts=={"LENGTH":LENGTH_CONTROL,"V1_PLUS_LENGTH":LENGTH_CONTROL+"\n\n"+skill().body}
    fp=arm_fingerprints(skill(),design="conditional_increment")
    a,b=(fp[arm]["messages"]for arm in E1_ARMS)
    assert a[0]==b[0]
    assert a[1]["content"].replace(texts["LENGTH"],"<TREATMENT>")==b[1]["content"].replace(texts["V1_PLUS_LENGTH"],"<TREATMENT>")
    assert fp["V1_PLUS_LENGTH"]["skill_content"] is True


def test_e1_decision_quality_missing_and_cache_sensitivity():
    plan=[task(i) for i in range(12)]
    rows=[]
    for i in range(12):
        a,b=receipt("LENGTH",i=i),receipt("V1_PLUS_LENGTH",i=i)
        b["trajectory"][0].update(output_tokens=40,reasoning_tokens=30)
        rows.extend([a,b])
    d=conditional_increment_decision(plan,rows,PRICING)
    assert d["status"]=="E1_COMPLETED_POSITIVE_EXPLORATORY"
    assert d["signs"]["positive"]==12 and len(d["leave_one_cluster_out"])==12
    assert d["no_cache_savings"]["mean"]==pytest.approx(.000024)
    assert conditional_increment_decision(plan,rows[:-1],PRICING)["status"]=="E1_COMPLETED_INCONCLUSIVE"
    rows[-1]=receipt("V1_PLUS_LENGTH",False,i=11)
    assert conditional_increment_decision(plan,rows,PRICING)["status"]=="E1_COMPLETED_NEGATIVE_EXPLORATORY"
    rows[-1]=receipt("V1_PLUS_LENGTH",True,i=11);rows[-2]=receipt("LENGTH",False,i=11)
    assert conditional_increment_decision(plan,rows,PRICING)["quality_gain_cost_ranking_suspended"]


def test_e1_selection_freshness_balance_and_validation():
    from experiments.external_real.alfworld_runner import TASK_TYPE_NAMES
    members=[{**task(i*3+j),"task_type":f,"split":"train_inner","confirmatory_eligible":True,"identity_status":"IDENTIFIED"}
             for i,f in enumerate(TASK_TYPE_NAMES) for j in range(3)]
    parent={"task_membership":members}
    plan=select_conditional_plan(parent,{"c0"},{"p4"},seed=1)
    assert plan==select_conditional_plan(parent,{"c0"},{"p4"},seed=1)
    assert len(plan)==12 and not {r["cluster_id"]for r in plan}&{"c0","c4"}
    s=skill();m={"stage":"skill_attribution","design":"conditional_increment","split":"train_inner",
        "split_manifest_sha256":manifest_digest(parent),"skill_digest":s.digest,"skill_body_sha256":s.body_digest,
        "arms":list(E1_ARMS),"treatments":arm_texts(s,design="conditional_increment"),
        "excluded_clusters":["c0"],"excluded_templates":["p4"],"plan":plan,"seed":1,"thinking":"enabled","max_retries":0}
    validate_manifest(m,parent,s)
    m["plan"][1]["arm_order"]=m["plan"][0]["arm_order"]
    with pytest.raises(ValueError,match="balanced"):
        validate_manifest(m,parent,s)


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


def live_fixture(tmp_path):
    t=task();s=skill();parent={"task_membership":[{**t,"split":"train_inner","confirmatory_eligible":True}]}
    manifest={"stage":"skill_attribution","split":"train_inner","split_manifest_sha256":manifest_digest(parent),
        "skill_digest":s.digest,"skill_body_sha256":s.body_digest,"arms":list(ARMS),"treatments":arm_texts(s),
        "excluded_clusters":[],"excluded_templates":[],"plan":[t],"seed":67,"thinking":"enabled","max_retries":0,
        "controller_commit":"fixture","study_id":"fixture","run_id":"fixture","model":"deepseek-flash",
        "base_url":"https://api.deepseek.com","output_dir":str(tmp_path/"output"),"spool_dir":str(tmp_path/"spool"),
        "max_completion_per_request":100,"max_input_per_request":2000,"max_steps":1,
        "agent_sha":"fixture-agent","configuration_sha":"fixture-config","prompt_template_sha256":"fixture-prompt",
        "max_usd_per_episode":"0.1","max_calls_per_episode":1,"max_episode_wall_seconds":60}
    now=datetime.now(timezone.utc)
    approval={"provider_enabled":True,"execute":True,"controller_commit":"fixture","model":manifest["model"],
        "study_id":"fixture","run_id":"fixture","approved_manifest_sha256":manifest_digest(manifest),
        "approved_split_manifest_sha256":manifest["split_manifest_sha256"],"permitted_stages":[manifest["stage"]],
        "api_base_allowlist":[manifest["base_url"]],"approved_by":"SYNTHETIC_FIXTURE","approval_reference":"SYNTHETIC_FIXTURE",
        "approved_at_utc":now.isoformat(),"expiration_utc":(now+timedelta(hours=1)).isoformat(),
        "max_provider_calls":3,"max_input_tokens":6000,"max_completion_tokens":300,"per_episode_max_steps":1,
        "per_request_max_completion_tokens":100,"max_wall_seconds":3600,"max_total_usd":"1","max_failed_retries":0,
        "max_usd_per_episode":"0.1","max_calls_per_episode":1,"max_episode_wall_seconds":60}
    pricing={**PRICING,"currency":"USD","model":manifest["model"],"checked_at_utc":now.isoformat(),
             "pricing_source_url":"SYNTHETIC_FIXTURE","price_period":"SYNTHETIC_FIXTURE","estimate_quality":"SYNTHETIC_FIXTURE"}
    return manifest,parent,s,approval,pricing


def test_complete_arm_resume_after_summary_failure_never_constructs_provider_or_resets(tmp_path,monkeypatch):
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
    import experiments.skill_evolution.attribution as module
    manifest,parent,s,approval,pricing=live_fixture(tmp_path)
    calls,resets=[],[]
    def fake_factory(_key,base,model,*,budget,max_completion_tokens,persist_budget):
        client=DeepSeekStatefulClient.__new__(DeepSeekStatefulClient)
        def provider(**_request):
            calls.append(1)
            return SimpleNamespace(id=f"fake-{len(calls)}",model=model,system_fingerprint="fixture",
                usage=SimpleNamespace(prompt_tokens=31,completion_tokens=9,prompt_cache_hit_tokens=4,prompt_cache_miss_tokens=27,
                                      completion_tokens_details=SimpleNamespace(reasoning_tokens=6)),
                choices=[SimpleNamespace(message=SimpleNamespace(content="Action: look"),finish_reason="stop")])
        client.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider)))
        client.base_url=base;client.model=model;client.temperature=0;client.budget=budget
        client.max_completion_tokens=max_completion_tokens;client.persist_budget=persist_budget
        client.response_identity=None;client.response_ids=set();client.receipt_journal=None
        client._episode_context=None;client._episode_turn=0;client.evidence_origin="SYNTHETIC_FIXTURE"
        return client
    class Env:
        def reset(self):
            resets.append(1)
            return ["Your task is to: inspect the target."],{"admissible_commands":[["look"]],"extra.gamefile":["/fixture/g0"]}
        def step(self,actions):
            assert actions==["look"]
            return ["Done."],[1],[True],{"won":[True],"admissible_commands":[["look"]]}
        def close(self):pass
    actual=module._save_summary
    monkeypatch.setattr(module,"_save_summary",lambda *_:(_ for _ in ()).throw(OSError("summary disk full")))
    with pytest.raises(OSError,match="disk full"):
        run_attribution(manifest,parent,s,approval,pricing,output=tmp_path/"output",revision="fixture",
            client_factory=fake_factory,environment_factory=lambda *_a,**_k:Env())
    assert len(calls)==len(resets)==3
    monkeypatch.setattr(module,"_save_summary",actual)
    def never(*_a,**_k):pytest.fail("resume constructed provider or reset completed environment")
    result=run_attribution(manifest,parent,s,approval,pricing,output=tmp_path/"output",revision="fixture",
        client_factory=never,environment_factory=never)
    assert result["provider_budget"]["calls_reserved"]==3
    assert result["comparisons"]["OFF_vs_SKILL"]["11"]==1
    assert len(calls)==len(resets)==3
    events=[json.loads(x)for x in(tmp_path/"spool/journal.jsonl").read_text().splitlines()]
    assert sum(e["event"]=="usage_ledger"for e in events)==3
    episodes=[json.loads(x)["episode"] for x in (tmp_path/"spool/episodes.jsonl").read_text().splitlines()
              if json.loads(x)["event"]=="COMPLETED"]
    assert all(r["agent_sha"]=="fixture-agent"and r["configuration_sha"]=="fixture-config"for r in episodes)
    assert all(manifest_digest({k:v for k,v in r.items()if k!="receipt_sha256"})==r["receipt_sha256"]for r in episodes)
    assert all(r["actual_new_model_rollout"]is False for r in episodes)


def test_explicit_approval_denial_precedes_sdk_and_environment(tmp_path):
    from core.acre.resource_budget import ProviderBudgetExceeded
    manifest,parent,s,approval,pricing=live_fixture(tmp_path)
    approval["provider_enabled"]=False
    def never(*a,**k):pytest.fail("provider/environment reached without approval")
    with pytest.raises(ProviderBudgetExceeded,match="APPROVAL_REQUIRED"):
        run_attribution(manifest,parent,s,approval,pricing,output=tmp_path/"output",revision="fixture",
                        client_factory=never,environment_factory=never)


def test_offline_partial_stop_summary_keeps_missing_outcomes_without_provider(tmp_path):
    manifest,_,_,_,pricing=live_fixture(tmp_path)
    (tmp_path/"spool").mkdir();(tmp_path/"output").mkdir()
    events=[{"event":"COMPLETED","trial_id":"trial-0-OFF","episode":receipt()},
            {"event":"STOP","reason":"UNKNOWN_ARM"}]
    (tmp_path/"spool/episodes.jsonl").write_text("\n".join(json.dumps(e)for e in events)+"\n")
    result=summarize_spooled(manifest,pricing)
    assert result["status"]=="STOP_UNKNOWN_ARM"and result["missing_arms"]==2
    assert result["comparisons"]["OFF_vs_SKILL"]["missing"]==1
    assert result["comparisons"]["OFF_vs_SKILL"]["00"]==0


def test_changed_task_bytes_rejected_before_real_client_construction(tmp_path,monkeypatch):
    manifest,parent,s,approval,pricing=live_fixture(tmp_path)
    game=tmp_path/"game";game.write_text("changed bytes")
    manifest["plan"][0].update(gamefile=str(game),gamefile_sha256="0"*64)
    parent["task_membership"][0]["gamefile"]=str(game)
    manifest["split_manifest_sha256"]=manifest_digest(parent)
    manifest["environment_packages"]={}
    approval["approved_manifest_sha256"]=manifest_digest(manifest)
    approval["approved_split_manifest_sha256"]=manifest["split_manifest_sha256"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setenv("DEEPSEEK_API_KEY","SYNTHETIC_FIXTURE_NOT_A_KEY")
    import experiments.external_real.alfworld_stateful_runner as runner
    monkeypatch.setattr(runner.DeepSeekStatefulClient,"__init__",lambda *_a,**_k:pytest.fail("SDK reached"))
    with pytest.raises(ValueError,match="task content changed"):
        run_attribution(manifest,parent,s,approval,pricing,output=tmp_path/"output",revision="fixture")
