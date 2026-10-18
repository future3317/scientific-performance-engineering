from __future__ import annotations

from pathlib import Path
import random
import math

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
    devices=[r.get('runtime_identity',{}).get('device_uuid') for r in [s0,s1]]
    if not all(devices) or devices[0] != devices[1]:
        return 'SESSION_NOISE_UNRESOLVED'
    for receipt in [s0,s1]:
        for key in ['gpu_telemetry_before','gpu_telemetry_after']:
            snapshot=receipt.get(key)
            if snapshot and ('error' in snapshot or snapshot.get('compute_processes')):
                return 'SESSION_NOISE_UNRESOLVED'
    return 'VALID_PAIR'


def measure(root: Path, manifest: dict, split: str, environment: dict) -> list[dict]:
    if split=='final':
        policy=root/'POLICY_FREEZE.json'
        release=root/'FINAL_RELEASE.json'
        if not policy.exists() or not release.exists() or not read(release)['qualified']:
            raise RuntimeError('final candidate remains closed before qualified policy freeze')
        if read(policy).get('manifest_digest') != manifest['manifest_digest']:
            raise RuntimeError('final policy belongs to a different frozen cohort')
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
            shapes=list(context.get('actual_input_shapes',{}).values())
            image=next((shape for shape in shapes if len(shape)==4),None)
            pair={'pair_id':key,'family':context['family'],'model':context['model'],
                  'source':context['source'],'split':split,'horizon':context['horizon'],
                  'session':session,'order':order,'status':status,'S0':arms['S0'],'S1':arms['S1'],
                  'features':{'horizon':context['horizon'],'batch':context['batch'],
                              'parameter_count':context.get('parameter_count',0),
                              'input_height':image[-2] if image else 0,'input_width':image[-1] if image else 0,
                              'input_channels':image[1] if image else 0,
                              'input_numel':sum(math.prod(shape) for shape in shapes)},
                  'context_id':context['context_id']}
            pair['runtime_gpu_uuid']=manifest['gpu_uuid']
            dump(path,pair);pairs.append(pair)
            print('PAIR',key,context['family'],status,flush=True)
            if status=='PLATFORM_LINK_OR_INSTALL':
                raise RuntimeError('compiler platform error returned after canary; stop batch')
    return pairs
