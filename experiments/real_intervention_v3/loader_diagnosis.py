"""Read historical failures and inspect their exact loader artifacts."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys

from .run_state import dump, read


def diagnose(old: Path, root: Path) -> dict:
    rows = [(p, read(p)) for p in sorted((old / 'arms').glob('*.json'))]
    errors = [(p, row) for p, row in rows if 'cuda_utils.so' in row.get('error', '')]
    evidence = []
    for path, row in errors:
        text = row['error']
        paths = re.findall(r'(/[^\s:]+/cuda_utils\.so)', text)
        so = Path(paths[-1]) if paths else None
        commands = {}
        if so and so.exists():
            for args in [['ldd', '-r', str(so)], ['readelf', '-d', str(so)], ['readelf', '-Ws', str(so)]]:
                p = subprocess.run(args, capture_output=True, text=True)
                commands[' '.join(args[:2])] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        evidence.append({'receipt': str(path), 'full_traceback': text, 'error_type': row.get('error_type'),
                         'so_path': str(so), 'artifact_exists_now': bool(so and so.exists()),
                         'linker': commands})
    manifest = read(old / 'RUN_MANIFEST.json')
    result = {'old_status': 'V2_EXPLORATORY_PLATFORM_LIMITED',
              'arm_counts': dict(Counter(r['status'] for _, r in rows)),
              'cuda_utils_failure_count': len(errors), 'evidence': evidence,
              'old_source': manifest['source_commit'],
              'python': sys.executable,
              'packages': {k: importlib.metadata.version(k) for k in ['torch', 'torchvision', 'triton']},
              'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
              'cause': 'loader ENOENT at recorded cache artifact; later file existence alone cannot establish historical cause'}
    dump(root / 'LOADER_DIAGNOSIS.json', result)
    (root / 'LOADER_TRACEBACKS.txt').write_text('\n\n'.join(e['full_traceback'] for e in evidence))
    return result
