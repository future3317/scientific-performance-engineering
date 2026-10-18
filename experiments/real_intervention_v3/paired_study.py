from __future__ import annotations

from pathlib import Path
import random

from .run_state import dump, execute, read


def classify_pair(s0: dict, s1: dict) -> str:
    if s0.get('status') in {'unknown','not_started'} or s1.get('status') in {'unknown','not_started'}:
        return 'SESSION_NOISE_UNRESOLVED'
    if s0.get('status') != 'valid':
        return 'BASELINE_INVALID'
    if 'cuda_utils.so' in s1.get('error','') and 'No such file' in s1['error']:
        return 'PLATFORM_LINK_OR_INSTALL'
    if s1.get('status') == 'numeric_mismatch':
        return 'INTERVENTION_NUMERIC_MISMATCH'
    if s1.get('status') != 'valid':
        return 'INTERVENTION_MODEL_UNSUPPORTED'
    for field in ['inputs_sha256','weights_sha256']:
        if s0.get(field) != s1.get(field):
            raise RuntimeError('paired identity differs: '+field)
    return 'VALID_PAIR'


def measure(root: Path, manifest: dict, split: str, environment: dict) -> list[dict]:
    pairs = []
    for context in manifest['contexts']:
        if context['split'] != split:
            continue
        for session in range(manifest['sessions']):
            key = context['context_id'] + '-session-' + str(session)
            path = root/'pairs'/(key+'.json')
            if path.exists():
                pairs.append(read(path)); continue
            order = ['S0','S1'] if (int(context['order_bit'])+session)%2==0 else ['S1','S0']
            arms={}
            for arm in order:
                spec = dict(context, arm=arm, intervention=manifest['selected_intervention'])
                arms[arm] = execute(root,key+'-'+arm,'experiments.real_intervention_v3.arm_worker',
                                    spec,manifest['gpu_uuid'],manifest['arm_timeout_seconds'],environment)
            status=classify_pair(arms['S0'],arms['S1'])
            pair={'pair_id':key,'family':context['family'],'model':context['model'],
                  'source':context['source'],'split':split,'horizon':context['horizon'],
                  'session':session,'order':order,'status':status,'S0':arms['S0'],'S1':arms['S1'],
                  'features':{'horizon':context['horizon'],'batch':context['batch'],
                              'parameter_count':context.get('parameter_count',0)},
                  'context_id':context['context_id']}
            dump(path,pair);pairs.append(pair)
            print('PAIR',key,context['family'],status,flush=True)
            if status=='PLATFORM_LINK_OR_INSTALL':
                raise RuntimeError('compiler platform error returned after canary; stop batch')
    return pairs
