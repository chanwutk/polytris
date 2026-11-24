"""Resolve the explicitly selected LEAP results without replacing archived runs."""
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'measurements-results/leap_canonical.json'


def canonical_leap_result(dataset):
    # Existing caches remain the fallback for datasets without a promoted result.
    if not MANIFEST.exists():
        return None
    selected = json.loads(MANIFEST.read_text())['datasets'].get(dataset)
    if selected is None:
        return None
    assert selected['param_id'] == 0
    assert math.isfinite(selected['compute_s']) and selected['compute_s'] > 0
    assert selected['frames'] > 0
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in selected['hota'].values())
    # A registry entry may reference only experiment artifacts inside this checkout.
    resolved = dict(selected)
    for field in ('tracking_root', 'native_intervals', 'source_comparison'):
        path = Path(selected[field])
        assert not path.is_absolute() and '..' not in path.parts, f'Invalid canonical LEAP path: {path}'
        resolved[field] = ROOT / path
    return resolved
