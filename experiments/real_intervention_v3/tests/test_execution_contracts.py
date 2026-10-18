from pathlib import Path

import pytest

from experiments.real_intervention_v3.model_split import split_families
from experiments.real_intervention_v3.release_final import eligibility
from experiments.real_intervention_v3.paired_study import classify_pair, measure
from experiments.real_intervention_v3.run_state import dump, execute, directory_bytes, digest
from experiments.real_intervention_v3.run_all import baseline_pool


def test_repeat_horizon_not_counted_as_independent_family():
    rows=[{'family':'same','model':str(i)} for i in range(18)]
    assert len(split_families(rows))==1


def test_model_family_split_disjoint_and_versioned():
    rows=[{'family':str(i)} for i in range(20)]
    split=split_families(rows)
    assert split==split_families(rows)
    assert len([k for k,v in split.items() if v=='calibration'])==5
    assert set(split.values())=={'development','calibration','final'}


def test_final_stays_closed_without_five_valid_calibration_families():
    m={'family_split':{str(i):'final' for i in range(20)},'minimum_total_families':18,
       'minimum_calibration_families':5}
    p=[{'family':str(i),'split':'calibration','status':'VALID_PAIR'} for i in range(4)]
    p+=[{'family':'dev','split':'development','status':'VALID_PAIR'}]
    assert not eligibility(p,m)['qualified']
    assert eligibility(p,m)['missing_valid_calibration_families']==1


def test_policy_cannot_read_candidate_final_before_freeze(tmp_path):
    with pytest.raises(RuntimeError,match='closed'):
        measure(tmp_path,{'contexts':[]},'final',{})


def test_missing_cuda_utils_marks_platform_invalid_not_candidate_harm():
    assert classify_pair({'status':'valid'},{'status':'error','error':'cuda_utils.so: No such file'})=='PLATFORM_LINK_OR_INSTALL'


def test_model_specific_compilation_failure_after_healthy_canary_is_recorded():
    assert classify_pair({'status':'valid'},{'status':'error','error':'Unsupported op'})=='INTERVENTION_MODEL_UNSUPPORTED'


def test_both_arm_failure_not_converted_to_zero_seconds():
    assert classify_pair({'status':'error'},{'status':'error'})=='BASELINE_INVALID'


def test_two_arm_same_weights_and_inputs_sha256():
    baseline={'status':'valid','weights_sha256':'a','inputs_sha256':'b'}
    assert classify_pair(baseline,dict(baseline))=='VALID_PAIR'
    with pytest.raises(RuntimeError,match='identity'):
        classify_pair(baseline,dict(baseline,weights_sha256='c'))


def test_resume_same_identity_no_duplicate_arm_processes(tmp_path):
    dump(tmp_path/'raw'/'x'/'intent.json',{'spec_digest':'wrong'})
    with pytest.raises(RuntimeError,match='identity'):
        execute(tmp_path,'x','unused',{},'unused',1)


def test_live_cache_meter_tolerates_triton_removing_temporary_directory(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import run_state
    (tmp_path/'kept').write_bytes(b'12345')
    temporary=tmp_path/'tmp.pid_1';temporary.mkdir()
    (temporary/'kernel').write_bytes(b'temporary')
    def live_walk(root):
        yield str(root),['tmp.pid_1'],['kept']
        (temporary/'kernel').unlink();temporary.rmdir()
        yield str(temporary),[],['kernel']
    monkeypatch.setattr(run_state.os,'walk',live_walk)
    assert directory_bytes(tmp_path)==5


def test_inherited_unknown_attempt_never_launches_again(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import run_state
    spec={'arm':'S1','reference_path':'prior/reference.pt'}
    dump(tmp_path/'raw'/'x'/'intent.json',{'spec_digest':digest(spec)})
    dump(tmp_path/'INHERITED_ATTEMPTS.json',{'unknown_attempts':{'x':'prior/raw/x/intent.json'}})
    def forbidden(*args,**kwargs):
        raise AssertionError('must not launch a previously started unknown arm')
    monkeypatch.setattr(run_state.subprocess,'Popen',forbidden)
    result=execute(tmp_path,'x','unused',spec,'unused',1)
    assert result['status']=='unknown'
    assert 'total_cold_inclusive_s' not in result
    assert classify_pair({'status':'valid'},result)=='SESSION_NOISE_UNRESOLVED'


def test_reboot_inherits_closed_pairs_and_both_unknowns_without_fabricating_exit(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import run_state
    prior=tmp_path/'prior';current=tmp_path/'current'
    dump(prior/'RUN_IDENTITY.json',{'source_commit':'old'})
    dump(prior/'STUDY_LAUNCH.json',{'started_unix':10})
    dump(prior/'FROZEN_MANIFEST.json',{'contexts':[]})
    dump(prior/'pairs'/'closed.json',{'status':'VALID_PAIR'})
    for name in ['old_unknown','reboot_unknown']:
        dump(prior/'raw'/name/'intent.json',{'started_unix':20,'spec_digest':digest({})})
    dump(prior/'INHERITED_ATTEMPTS.json',{'unknown_attempts':{'old_unknown':'original/intent.json'}})
    original_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'btime 100\n' if str(p)=='/proc/stat' else original_read(p,*a,**k))
    monkeypatch.setattr(run_state,'source_identity',lambda:'new')
    monkeypatch.setattr(run_state.subprocess,'check_output',lambda *a,**k:b'')
    run_state.inherit_interrupted_study(prior,current)
    inherited=run_state.read(current/'INHERITED_ATTEMPTS.json')
    assert inherited['prior_exit'] is None
    assert inherited['interruption']['reason']=='server_reboot'
    assert set(inherited['unknown_attempts'])=={'old_unknown','reboot_unknown'}
    assert run_state.read(current/'pairs'/'closed.json')['status']=='VALID_PAIR'
    assert not (prior/'RUN_EXIT.json').exists()
    for name in inherited['unknown_attempts']:
        assert execute(current,name,'unused',{},'unused',1)['status']=='unknown'


def test_missing_exit_without_subsequent_reboot_cannot_resume(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import run_state
    prior=tmp_path/'prior'
    dump(prior/'STUDY_LAUNCH.json',{'started_unix':100})
    original_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'btime 50\n' if str(p)=='/proc/stat' else original_read(p,*a,**k))
    with pytest.raises(RuntimeError,match='no subsequent server reboot'):
        run_state.inherit_interrupted_study(prior,tmp_path/'new')


def test_no_model_selected_using_s1_outcome():
    universe=[{'source':'torchvision','model':'resnet18','family':'resnet','observed_speedup':-10},
              {'source':'torchvision','model':'resnet50','family':'resnet','observed_speedup':100}]
    assert [r['model'] for r in baseline_pool(universe)]==['resnet18']


def test_inductor_canary_cannot_be_satisfied_by_eager_aa():
    from experiments.real_intervention_v3.backend_canary import healthy
    assert not healthy([{'status':'valid','correctness':True}]*3)
    rows=[{'status':'valid','compiled_graphs':[{'generated_kernel_count':1,'numeric_valid':True}]*2,
           'identity':{'cache':str(i)}} for i in range(3)]
    assert healthy(rows)
    rows[1]['compiled_graphs'][0]['generated_kernel_count']=0
    assert not healthy(rows)


def test_gpu_migration_never_compares_arms_on_different_devices():
    a={'status':'valid','weights_sha256':'a','inputs_sha256':'b',
       'runtime_identity':{'device_uuid':'old-gpu'}}
    b=dict(a,runtime_identity={'device_uuid':'new-gpu'})
    assert classify_pair(a,b)=='SESSION_NOISE_UNRESOLVED'
    assert classify_pair(a,dict(a))=='VALID_PAIR'


def test_gpu_migration_requires_real_compiler_canary_not_inherited_health(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import run_all
    calls=[]
    def execute_canary(root,key,module,spec,gpu,timeout,environment):
        calls.append((key,gpu))
        return {'status':'valid','compiled_graphs':[{'generated_kernel_count':1,'numeric_valid':True}]*2,
                'identity':{'cache':key}}
    monkeypatch.setattr(run_all,'execute',execute_canary)
    result=run_all.qualify_migrated_gpu(tmp_path,'gpu3','source',{})
    assert len(result)==3
    assert calls==[('canary-runtime-'+str(i),'gpu3') for i in range(3)]
    run_all.qualify_migrated_gpu(tmp_path,'gpu3','source',{})
    assert len(calls)==3
    dump(tmp_path/'CANARY_RUNTIME.json',[{'status':'error'}]*3)
    with pytest.raises(RuntimeError,match='stop without changing intervention'):
        run_all.qualify_migrated_gpu(tmp_path,'gpu3','source',{})


def test_runtime_gpu_override_preserves_frozen_cohort_and_completed_pair(tmp_path,monkeypatch):
    from experiments.real_intervention_v3 import paired_study
    contexts=[{'context_id':name,'split':'development','order_bit':0,'family':name,'model':name,
               'source':'torchvision','horizon':1,'batch':1} for name in ['old','new']]
    frozen={'contexts':contexts,'sessions':1,'gpu_uuid':'old-gpu','selected_intervention':'compile'}
    dump(tmp_path/'FROZEN_MANIFEST.json',frozen)
    before=(tmp_path/'FROZEN_MANIFEST.json').read_bytes()
    dump(tmp_path/'pairs'/'old-session-0.json',{'pair_id':'old-session-0','status':'VALID_PAIR'})
    calls=[]
    def run(root,key,module,spec,gpu,timeout,env):
        calls.append((spec['arm'],gpu))
        return {'status':'valid','inputs_sha256':'input','weights_sha256':'weights',
                'runtime_identity':{'device_uuid':gpu}}
    monkeypatch.setattr(paired_study,'execute',run)
    result=paired_study.measure(tmp_path,dict(frozen,gpu_uuid='new-gpu',arm_timeout_seconds=900),'development',{})
    assert calls==[('S0','new-gpu'),('S1','new-gpu')]
    assert result[0]=={'pair_id':'old-session-0','status':'VALID_PAIR'}
    assert result[1]['status']=='VALID_PAIR'
    assert result[1]['runtime_gpu_uuid']=='new-gpu'
    assert (tmp_path/'FROZEN_MANIFEST.json').read_bytes()==before
