"""Select LEAP HOTA consistently across paper figures and numeric exports."""
import json
import math
from pathlib import Path


def reconstruction_scores(data: Path, datasets: dict) -> dict:
    anchored = json.loads((data / 'leap_anchored_hota.json').read_text())['datasets']
    assert anchored.keys() == datasets.keys()
    scores = {}
    for label, entry in datasets.items():
        values = {method: entry['hota'][method]['HOTA'] for method in ('linear', 'pattern')}
        # The canonical B3D records store four decimal places.
        tolerance = 5e-5 if label.startswith('B3D') else 1e-12
        assert all(math.isclose(value, anchored[label][method], rel_tol=0, abs_tol=tolerance)
                   for method, value in values.items()), f'Stale anchored LEAP scores for {label}'
        # Compare the new method with the full-precision control from the same scoring run.
        values = anchored[label]
        assert all(0 <= value <= 1 for value in values.values()), label
        scores[label] = values
    return scores
