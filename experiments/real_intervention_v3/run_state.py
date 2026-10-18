"""Durable subprocess receipts; unknown attempts are never retried."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
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


def source_identity() -> str:
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('execution requires a clean worktree')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def execute(root: Path, key: str, module: str, spec: dict, gpu: str,
            timeout: int, environment: dict | None = None) -> dict:
    if (root/'RESOURCE_CONTRACT.json').exists():
        contract=read(root/'RESOURCE_CONTRACT.json')
        elapsed=time.time()-contract['started_unix']
        if elapsed+timeout > contract['wall_cap_seconds']:
            raise RuntimeError('study wall cap cannot accommodate next arm')
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
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
                    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
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
