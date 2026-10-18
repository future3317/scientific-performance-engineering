from __future__ import annotations


def eligibility(pairs: list[dict], manifest: dict) -> dict:
    dev = {p['family'] for p in pairs if p['split']=='development' and p['status']=='VALID_PAIR'}
    cal = {p['family'] for p in pairs if p['split']=='calibration' and p['status']=='VALID_PAIR'}
    total = len(manifest['family_split'])
    needed = manifest['minimum_total_families']
    ncal = manifest['minimum_calibration_families']
    qualified = total >= needed and len(cal) >= ncal and bool(dev)
    return {'qualified': qualified, 'status': 'QUALIFIED_FOR_POLICY_FREEZE' if qualified else 'UNDERPOWERED_NOT_FINAL',
            'valid_development_families': len(dev), 'valid_calibration_families': len(cal),
            'admitted_families': total, 'missing_total_families': max(0, needed-total),
            'missing_valid_calibration_families': max(0, ncal-len(cal)),
            'minimum_calibration_families': ncal, 'minimum_total_families': needed}
