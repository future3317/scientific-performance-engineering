from pathlib import Path

import pytest

from experiments.real_intervention_v3.model_split import split_families
from experiments.real_intervention_v3.release_final import eligibility
from experiments.real_intervention_v3.paired_study import classify_pair
from experiments.real_intervention_v3.run_state import dump, execute
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
