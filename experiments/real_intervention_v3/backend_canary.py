"""Execute two real Inductor GPU graphs; eager correctness alone never passes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from .run_state import dump, read


def canary(spec: dict) -> dict:
    import torch
    import triton
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(31)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import metrics
    identity = {'python': sys.executable, 'torch': torch.__version__, 'torch_file': torch.__file__,
                'triton': triton.__version__, 'triton_file': triton.__file__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                'cache': os.environ['TORCHINDUCTOR_CACHE_DIR'],
                'triton_cache': os.environ['TRITON_CACHE_DIR']}
    x = torch.randn(64, 64, device='cuda')
    image = torch.randn(2, 3, 16, 16, device='cuda')
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(),
                              torch.nn.Flatten(), torch.nn.Linear(8*14*14, 4)).cuda().eval()
    graphs = [(lambda x: (torch.sin(x) + x*x).sum(dim=1), x), (net, image)]
    checks = []
    start = time.perf_counter()
    with torch.inference_mode():
        for fn, value in graphs:
            expected = fn(value)
            torch.cuda.synchronize()
            metrics.reset()
            compiled = torch.compile(fn, backend='inductor', mode='default', dynamic=None, fullgraph=True)
            actual = compiled(value)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            kernels = metrics.generated_kernel_count
            if kernels < 1:
                raise RuntimeError('no generated Inductor kernel; eager is insufficient')
            checks.append({'generated_kernel_count': kernels, 'numeric_valid': True})
    libraries = list(Path(os.environ['TRITON_CACHE_DIR']).rglob('cuda_utils.so'))
    # PyTorch can add a per-device Triton cache root under the Inductor cache.
    libraries += list(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('cuda_utils.so'))
    if not libraries:
        raise RuntimeError('no CUDA driver helper artifact after compiled execution')
    return {'status': 'valid', 'compiled_graphs': checks, 'cuda_utils_paths': list(map(str,libraries)),
            'identity': identity, 'gpu_window_s': time.perf_counter()-start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        result = canary(read(Path(args.spec)))
    except Exception as exc:
        result = {'status': 'error', 'error': str(exc), 'error_type': type(exc).__name__,
                  'traceback': traceback.format_exc()}
    dump(Path(args.output), result)


if __name__ == '__main__':
    main()
