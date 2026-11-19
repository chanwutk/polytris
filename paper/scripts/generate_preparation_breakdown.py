"""Export R9.O2 per-dataset preparation breakdown rows for Tetris from the saved preparation-total snapshot.

Source: scripts/data/revision_evidence/preparation_total.json (hashed in sources.json; the same components that
generate_revision_figures.py sums into the revisionTetrisPreparation* macros). Hours per dataset in four categories
plus the total. Run on the host: python scripts/generate_preparation_breakdown.py
"""
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = root / 'scripts/data/revision_evidence/preparation_total.json'
out = root / 'figures/generated'
# Column order and the short labels used in the response's other tables.
LABELS = {'ams-y05': 'Ams.', 'caldot1-y05': 'CDoT1', 'caldot2-y05': 'CDoT2',
          'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
# Row label -> (phase, component prefixes); every Tetris component must fall in exactly one row.
ROWS = {'Detector pass (training split)': ('training', ('p011', 'detector loading')),
        'Classifier training': ('training', ('p012', 'p013', 'p014', 'p015')),
        'Mistrack profiling': ('training', ('p016',)),
        'Validation sweep': ('validation', ('',))}

data = json.loads(source.read_text())
comps = [c for c in data['components'] if c['system'] == 'tetris']
hours = {(row, ds): 0.0 for row in ROWS for ds in LABELS}
for c in comps:
    rows = [row for row, (phase, prefixes) in ROWS.items()
            if c['phase'] == phase and any(c['component'].startswith(p) for p in prefixes)]
    assert len(rows) == 1, c
    hours[rows[0], c['dataset']] += c['seconds'] / 3600
# The rows must add up to the totals the paper reports.
totals = {ds: sum(t['seconds'] for t in data['totals'] if t['system'] == 'tetris' and t['dataset'] == ds) / 3600
          for ds in LABELS}
for ds in LABELS:
    assert abs(sum(hours[row, ds] for row in ROWS) - totals[ds]) < 1e-6, ds

def fmt(h):
    # One decimal; durations that round to zero print as "<0.1" rather than an exact zero.
    return f'{h:.1f}' if h >= 0.05 else '$<$0.1'


lines = [' & '.join([row] + [fmt(hours[row, ds]) for ds in LABELS]) + r' \\' for row in ROWS]
lines.append(r'\midrule')
lines.append(' & '.join(['Total'] + [f'{totals[ds]:.1f}' for ds in LABELS]) + r' \\')
(out / 'revision_preparation_breakdown_rows.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
