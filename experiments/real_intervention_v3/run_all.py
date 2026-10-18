"""One resumable entry: diagnosis, canary, admission, pairing, freeze, evaluation."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

from .run_state import digest, dump, execute, read, source_identity, directory_bytes, inherit_interrupted_study


def baseline_pool(universe: list[dict]) -> list[dict]:
    # Literal representatives are frozen independently of any S1 outcome.
    tv = {'alexnet','convnext_tiny','densenet121','efficientnet_b0','inception_v3','maxvit_t',
          'mnasnet0_5','mobilenet_v3_small','regnet_y_400mf','resnet18',
          'shufflenet_v2_x0_5','squeezenet1_0','swin_t','vgg11','vit_b_16'}
    tb = {'pytorch_unet','pytorch_CycleGAN_and_pix2pix','Super_SloMo','dlrm','fastNLP_Bert','maml'}
    return [r for r in universe if (r['source']=='torchvision' and r['model'] in tv) or
            (r['source']=='torchbench' and r['model'] in tb)]


def main_run(config: Path, root: Path, resume: bool) -> None:
    from .loader_diagnosis import diagnose
    from .backend_canary import healthy
    from .paired_study import measure
    from .model_split import split_families
    from .source_adapters import model_universe
    from .release_final import eligibility
    from .analyze_real import analyze
    from .policy_bench import freeze_policies, evaluate_policies
    from .report import write_report

    cfg=read(config)
    sha=source_identity()
    start=time.time()
    binding={'source_commit':sha,'execution_id':root.name,'config_digest':digest(cfg),'python':sys.executable,'gpu_uuid':cfg['gpu_uuid'],
             'packages':{k:importlib.metadata.version(k) for k in ['torch','torchvision','triton']}}
    binding['environment_digest']=digest({k:v for k,v in binding.items() if k!='source_commit'})
    if (root/'RUN_IDENTITY.json').exists():
        if read(root/'RUN_IDENTITY.json')!=binding:
            raise RuntimeError('resume source/config/environment identity differs')
    else: dump(root/'RUN_IDENTITY.json',binding)
    if (root/'RUN_COMPLETE.json').exists():
        print('ALREADY_COMPLETE',json.dumps(read(root/'RUN_COMPLETE.json')),flush=True)
        return
    protocol=dict(cfg,source_commit=sha,registered_unix=start)
    if not (root/'EXECUTION_PROTOCOL.json').exists():
        dump(root/'EXECUTION_PROTOCOL.json',protocol)
    if not (root/'RESOURCE_CONTRACT.json').exists():
        dump(root/'RESOURCE_CONTRACT.json',dict(started_unix=start,wall_cap_seconds=cfg['wall_cap_seconds'],
                                               disk_cap_bytes=cfg['disk_cap_bytes'],concurrency=1,cpu_threads=1,
                                               gpu_uuid=cfg['gpu_uuid']))
    if not (root/'SOURCE_PATCH.diff').exists():
        patch=subprocess.check_output(['git','diff','57a93a941d7b435fb5389ec2291892a591c9996b',sha])
        (root/'SOURCE_PATCH.diff').write_bytes(patch)
    os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    # Model code is preinstalled; prevent accidental dataset/weight downloads.
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    if not (root/'LOADER_DIAGNOSIS.json').exists():
        diagnose(Path(cfg['old_results']),root)
    if not (root/'STUDY_DECISION_TREE.json').exists():
        dump(root/'STUDY_DECISION_TREE.json',dict(cfg,source_commit=sha))
    environment={}
    original=[]
    if (root/'CANARY_ORIGINAL.json').exists():
        original=read(root/'CANARY_ORIGINAL.json')
    else:
        for i in range(3):
            original.append(execute(root,'canary-original-'+str(i),'experiments.real_intervention_v3.backend_canary',
                                    {'graphs':['elementwise_reduction','conv_linear'],'source':sha},cfg['gpu_uuid'],180))
        dump(root/'CANARY_ORIGINAL.json',original)
    canaries=original
    if not healthy(original):
        text=' '.join(r.get('traceback',r.get('error','')) for r in original)
        # The single correction targets the observed asynchronous cache-loader path.
        if 'cuda_utils' in text or 'SubprocException' in text:
            environment=cfg['repair']
            canaries=[execute(root,'canary-isolated-'+str(i),'experiments.real_intervention_v3.backend_canary',
                              {'graphs':['elementwise_reduction','conv_linear'],'source':sha},cfg['gpu_uuid'],180,environment)
                      for i in range(3)]
            dump(root/'CANARY_ISOLATED.json',canaries)
    primary='compile' if healthy(canaries) else 'channels_last'
    dump(root/'BACKEND_SELECTION.json',{'selected_intervention':primary,'environment':environment,
                                      'canary_passed':sum(r['status']=='valid' for r in canaries),
                                      'fallback_reason':None if primary=='compile' else 'compiler platform canary failed'})
    (root/'BACKEND_CANARY_REPORT.md').write_text(
        '# Backend canary\n\nOriginal independent processes: '+str(sum(r['status']=='valid' for r in original))+
        '/3 passed. Two GPU graphs per process; kernel generation, execution, synchronization and numerical checks required.\n\n'+
        'Selected intervention: '+primary+'. See raw/canary-* and LOADER_DIAGNOSIS.json for full logs.\n')
    # Reuse the already-installed workload-only optional dependencies read-only.
    # Torch/Triton still resolve from equivcompiler, as demonstrated by canaries.
    if cfg.get('workload_dependency_overlay'):
        overlay=Path(cfg['workload_dependency_overlay'])
        if any((overlay/name).exists() for name in ['torch','torchvision','triton']):
            raise RuntimeError('workload overlay must not replace compiler stack')
        environment=dict(environment,PYTHONPATH=str(overlay)+os.pathsep+str(Path.cwd()))
    actual_tb=subprocess.check_output(['git','-C',cfg['torchbench_root'],'rev-parse','HEAD'],text=True).strip()
    if actual_tb!=cfg['torchbench_commit']:
        raise RuntimeError('TorchBench pinned identity differs')
    if (root/'MODEL_UNIVERSE.json').exists():
        universe=read(root/'MODEL_UNIVERSE.json')
    else:
        universe=model_universe(cfg['torchbench_root']);dump(root/'MODEL_UNIVERSE.json',universe)
    pool=baseline_pool(universe)
    if primary=='channels_last':
        pool=[r for r in pool if r['source']=='torchvision' and r['family'] not in {'vit','swin','maxvit'}]
    dump(root/'BASELINE_POOL.json',pool)
    admitted=[];admission=[]
    inherited=(root/'INHERITED_ATTEMPTS.json').exists()
    if inherited:
        admission=read(root/'MODEL_ADMISSION.json')
        frozen_manifest=read(root/'FROZEN_MANIFEST.json')
        seen=set()
        for context in frozen_manifest['contexts']:
            key=(context['source'],context['model'])
            if key not in seen:
                admitted.append(context);seen.add(key)
    for row in ([] if inherited else pool):
        key=row['source']+'-'+row['model']
        spec=dict(row,batch=1,shape=[3,224,224],seed=cfg['seed'],torchbench_root=cfg['torchbench_root'],
                  horizon=1,correctness=cfg['correctness'],reference_path=str(root/'references'/(key+'.pt')))
        if row.get('availability') not in {'available',None}:
            result={'status':'unavailable','admission_status':row.get('availability'),'reason':row.get('reason')}
        else:
            result=execute(root,'admission-'+key,'experiments.real_intervention_v3.model_admission',spec,
                           cfg['gpu_uuid'],cfg['admission_timeout_seconds'],environment)
        admission.append(dict(row,result=result))
        if result.get('status')=='valid' and result.get('admission_status')=='ADMITTED_BASELINE_ONLY':
            admitted.append(dict(spec,parameter_count=result.get('parameter_count',0),
                                 actual_input_shapes=result.get('actual_input_shapes',{})))
        print('ADMISSION',key,result.get('admission_status',result['status']),flush=True)
    dump(root/'MODEL_ADMISSION.json',admission)
    with (root/'MODEL_ADMISSION_MATRIX.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','model','family','status','reason']);w.writeheader()
        for r in admission: w.writerow({k:r.get(k) for k in ['source','model','family']}|
                                       {'status':r['result'].get('admission_status',r['result']['status']),
                                        'reason':r['result'].get('error',r['result'].get('reason',''))})
    if not admitted: raise RuntimeError('no baseline-valid model families')
    if (root/'FROZEN_MANIFEST.json').exists():
        manifest=read(root/'FROZEN_MANIFEST.json')
    else:
        manifest=dict(cfg,source_commit=sha,selected_intervention=primary,family_split=split_families(admitted,cfg['seed']))
        contexts=[]
        rng=random.Random(cfg['seed'])
        for spec in admitted:
            for h in cfg['horizons']:
                c=dict(spec,horizon=h,split=manifest['family_split'][spec['family']],order_bit=rng.randrange(2))
                c['context_id']=spec['source']+'-'+spec['model']+'-H'+str(h)
                contexts.append(c)
        manifest['contexts']=contexts;manifest['manifest_digest']=digest(manifest)
        dump(root/'FROZEN_MANIFEST.json',manifest)
    pairs=[]
    for split in ['development','calibration']:
        pairs+=measure(root,manifest,split,environment)
    release=eligibility(pairs,manifest);dump(root/'FINAL_RELEASE.json',release)
    dev=[p for p in pairs if p['split']=='development'];cal=[p for p in pairs if p['split']=='calibration']
    frozen=freeze_policies(dev,cal,root)
    dump(root/'POLICY_FREEZE.json',dict(frozen,release=release,manifest_digest=manifest['manifest_digest']))
    if release['qualified']:
        pairs+=measure(root,manifest,'final',environment)
    metrics=analyze(pairs)
    policies=evaluate_policies(pairs,frozen)
    raw=[read(p) for p in (root/'raw').glob('*/receipt.json')]
    resources={'provider_calls':0,'parent_wall_seconds':time.time()-start,
               'child_cpu_seconds':sum(r.get('child_cpu_s',0) for r in raw),
               'gpu_session_window_seconds':sum(r.get('parent_wall_s',0) for r in raw),
               'downloads_bytes':0,'disk_bytes':directory_bytes(root),
               'inherited_unknown_attempts':read(root/'INHERITED_ATTEMPTS.json') if inherited else None}
    extra={'status':release['status'],'release':release,'identity':binding,'selected_intervention':primary,
           'canary_passed':sum(r['status']=='valid' for r in canaries),'baseline_pool':len(pool),
           'baseline_valid':len(admitted),'family_split':manifest['family_split'],
           'resources':resources,'historical_files_modified':0,'final_opened':release['qualified'],
           'policy_freeze':frozen,
           'input_scope':'REAL_MODEL_MATERIALIZED_INPUT / TORCHBENCH_MINIATURE_INPUT',
           'claim_scope':'fixed intervention system performance; no scientific dataset accuracy or automatic Skill efficacy'}
    write_report(root,metrics,pairs,policies,extra)
    dump(root/'RUN_COMPLETE.json',dict(extra,finished_unix=time.time()))
    print('COMPLETE',json.dumps(extra),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study-config',type=Path,required=True);p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--mode',choices=['full'],default='full');p.add_argument('--resume-safe',action='store_true')
    p.add_argument('--prior-results',type=Path)
    args=p.parse_args();args.results_root.mkdir(parents=True,exist_ok=True)
    with (args.results_root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.prior_results:
                inherit_interrupted_study(args.prior_results,args.results_root)
            main_run(args.study_config,args.results_root,args.resume_safe)
        except Exception as exc:
            dump(args.results_root/'FATAL_CAUSE.json',{'error':str(exc),'traceback':traceback.format_exc(),
                                                      'action':'inspect exact failed receipt; never repeat an unknown attempt'})
            raise


if __name__=='__main__': main()
