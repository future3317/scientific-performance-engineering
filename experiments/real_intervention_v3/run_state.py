"""Durable subprocess receipts; unknown attempts are never retried."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time

from experiments.real_intervention_v2.paired_executor import _gpu_snapshot


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read(path: Path):
    return json.loads(path.read_text())


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def directory_bytes(root: Path) -> int:
    """Count a live compiler cache without requiring its temporary files to persist."""
    total=0
    for parent, directories, files in os.walk(root):
        for name in files:
            try:
                total+=(Path(parent)/name).stat().st_size
            except FileNotFoundError:
                # Triton publishes by rename and removes the temporary directory.
                continue
    return total


def source_identity() -> str:
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('execution requires a clean worktree')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def inherit_interrupted_study(prior: Path, root: Path) -> None:
    """Keep the frozen cohort and completed evidence after a monitor-only correction."""
    if (root/'INHERITED_ATTEMPTS.json').exists():
        return
    if (root/'RUN_IDENTITY.json').exists():
        raise RuntimeError('inheritance must precede the new execution identity')
    if read(prior/'RUN_EXIT.json')['returncode']==0:
        raise RuntimeError('completed studies are not continued')
    cause=read(prior/'FATAL_CAUSE.json')
    if 'FileNotFoundError' not in cause['traceback'] or 'rglob' not in cause['traceback']:
        raise RuntimeError('this continuation only corrects the observed live-cache monitor failure')
    old_source=read(prior/'RUN_IDENTITY.json')['source_commit']
    unchanged=['arm_worker.py','source_adapters.py','model_admission.py','numerical_checks.py','backend_canary.py']
    paths=['experiments/real_intervention_v3/'+name for name in unchanged]
    if subprocess.check_output(['git','diff',old_source,'HEAD','--',*paths]):
        raise RuntimeError('qualification or measurement semantics changed; cannot inherit evidence')
    names=['LOADER_DIAGNOSIS.json','LOADER_TRACEBACKS.txt','CANARY_ORIGINAL.json','CANARY_ISOLATED.json',
           'STUDY_DECISION_TREE.json','MODEL_UNIVERSE.json','BASELINE_POOL.json','MODEL_ADMISSION.json',
           'MODEL_ADMISSION_MATRIX.csv','FROZEN_MANIFEST.json','BACKEND_CANARY_REPORT.md','ENVIRONMENT.json']
    root.mkdir(parents=True,exist_ok=True)
    for name in names:
        if (prior/name).exists(): shutil.copy2(prior/name,root/name)
    prior_evidence=root/'prior_execution';prior_evidence.mkdir()
    for name in ['FATAL_CAUSE.json','RUN_EXIT.json','RUN_IDENTITY.json','RUN.log','SOURCE_PATCH.diff',
                 'FULL_PYTEST.log','FULL_PYTEST_EXIT.json']:
        if (prior/name).exists(): shutil.copy2(prior/name,prior_evidence/name)
    if (prior/'validation-complete-topology').exists():
        shutil.copytree(prior/'validation-complete-topology',prior_evidence/'validation-complete-topology')
    shutil.copytree(prior/'raw',root/'raw')
    if (prior/'pairs').exists(): shutil.copytree(prior/'pairs',root/'pairs')
    unknown={p.parent.name:str(p) for p in (prior/'raw').glob('*/intent.json')
             if not (p.parent/'receipt.json').exists()}
    dump(root/'INHERITED_ATTEMPTS.json',{'prior_root':str(prior),'prior_source_commit':old_source,
                                       'execution_source_commit':source_identity(),
                                       'unknown_attempts':unknown,
                                       'unknown_cpu_seconds':None,'prior_exit':read(prior/'RUN_EXIT.json'),
                                       'policy':'unknown attempts remain unresolved and are never executed again'})


def execute(root: Path, key: str, module: str, spec: dict, gpu: str,
            timeout: int, environment: dict | None = None) -> dict:
    if (root/'RESOURCE_CONTRACT.json').exists():
        contract=read(root/'RESOURCE_CONTRACT.json')
        elapsed=time.time()-contract['started_unix']
        if elapsed+timeout > contract['wall_cap_seconds']:
            raise RuntimeError('study wall cap cannot accommodate next arm')
        used=directory_bytes(root)
        if used > contract['disk_cap_bytes']:
            raise RuntimeError('study disk cap reached')
    folder = root / 'raw' / key
    folder.mkdir(parents=True, exist_ok=True)
    spec_digest = digest(spec)
    if (folder / 'intent.json').exists():
        intent = read(folder / 'intent.json')
        if intent['spec_digest'] != spec_digest:
            raise RuntimeError('resume identity differs: ' + key)
        if (folder / 'receipt.json').exists():
            return read(folder / 'receipt.json')
        inherited=root/'INHERITED_ATTEMPTS.json'
        if inherited.exists() and key in read(inherited)['unknown_attempts']:
            return {'status':'unknown','reason':'prior_monitor_interruption_no_retry',
                    'inherited_intent':read(inherited)['unknown_attempts'][key],
                    'derived_status_only':True}
        raise RuntimeError('unknown in-flight attempt; inspect, do not retry: ' + key)
    before = _gpu_snapshot(gpu)
    if before is None or 'error' in before:
        raise RuntimeError('GPU telemetry unavailable before worker')
    dump(folder / 'spec.json', spec)
    dump(folder / 'intent.json', {'spec_digest': spec_digest, 'source': source_identity(),
                                'started_unix': time.time(), 'pid': os.getpid()})
    cache = root / 'caches' / key
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                'TORCHINDUCTOR_CACHE_DIR': str(cache / 'inductor'),
                'TRITON_CACHE_DIR': str(cache / 'triton'), 'CUDA_CACHE_PATH': str(cache / 'cuda'),
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    env.update(environment or {})
    command = [sys.executable, '-m', module, '--spec', str(folder / 'spec.json'),
               '--output', str(folder / 'result.json')]
    start = time.perf_counter()
    samples=[dict(unix=time.time(),snapshot=before)]
    cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True,start_new_session=True)
    timed_out=False;limit_reason=None
    try:
        while True:
            try:
                out,err=process.communicate(timeout=min(5,timeout))
                break
            except subprocess.TimeoutExpired:
                samples.append({'unix':time.time(),'snapshot':_gpu_snapshot(gpu)})
                if time.perf_counter()-start>=timeout:
                    limit_reason='worker_timeout'
                if (root/'RESOURCE_CONTRACT.json').exists():
                    contract=read(root/'RESOURCE_CONTRACT.json')
                    used=directory_bytes(root)
                    if used>=contract['disk_cap_bytes']:
                        limit_reason='study_storage_cap'
                if limit_reason:
                    timed_out=True
                    os.killpg(process.pid,signal.SIGTERM)
                    try: out,err=process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        out,err=process.communicate()
                    break
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.communicate()
        raise
    code=process.returncode
    cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime
    (folder / 'stdout.log').write_text(out)
    (folder / 'stderr.log').write_text(err)
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {
        'status': 'error' if timed_out else 'unknown', 'error': limit_reason if timed_out else 'missing_receipt'}
    if timed_out:
        result.update(status='error',error=limit_reason)
    result.update({'returncode': code, 'timed_out': timed_out, 'parent_wall_s': time.perf_counter()-start,
                   'child_cpu_s': cpu, 'gpu_telemetry_before': before,
                   'gpu_telemetry_after': _gpu_snapshot(gpu), 'spec_digest': spec_digest,
                   'gpu_telemetry_samples':samples,
                   'cache_root': str(cache), 'environment_overrides': environment or {}})
    dump(folder / 'receipt.json', result)
    if limit_reason=='study_storage_cap':
        raise RuntimeError('storage cap stopped worker; inspect durable receipt')
    return result
