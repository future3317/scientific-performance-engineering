import json

from experiments.real_intervention_v3.loader_diagnosis import diagnose
from experiments.real_intervention_v3.run_state import dump


def test_source_digests_current_and_frozen_evidence_unchanged(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    error='ImportError: /missing/cuda_utils.so: cannot open shared object file: No such file or directory'
    dump(old/'arms'/'a.json',{'status':'compile_error','error_type':'BackendCompilerFailed','error':error})
    dump(old/'RUN_MANIFEST.json',{'source_commit':'frozen-source'})
    dump(old/'METRICS.json',{'valid_pairs':8})
    (old/'PAIRS.jsonl').write_text(json.dumps({'status':'invalid'})+'\n')
    before={p.relative_to(old):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    result=diagnose(old,new)
    after={p.relative_to(old):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    assert before==after
    assert result['cuda_utils_failure_count']==1
    assert result['old_source']=='frozen-source'
    assert result['evidence'][0]['full_traceback']==error
    assert result['evidence'][0]['artifact_exists_now'] is False
    assert result['python'].endswith('python') or 'python3' in result['python']
