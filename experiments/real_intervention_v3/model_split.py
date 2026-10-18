from __future__ import annotations

import math
import random


def split_families(admitted: list[dict], seed: int = 31) -> dict:
    families = sorted({r['family'] for r in admitted})
    random.Random(seed).shuffle(families)
    n = len(families)
    ncal = max(2, math.ceil(n * .25)) if n >= 6 else max(0, n // 3)
    nfinal = max(1, math.floor(n * .2)) if n >= 3 else 0
    return {family: ('calibration' if i < ncal else 'final' if i < ncal+nfinal else 'development')
            for i, family in enumerate(families)}
