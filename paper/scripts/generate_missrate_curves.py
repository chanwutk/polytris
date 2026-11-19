#!/usr/bin/env python3
"""Plot each tile's measured miss rates against its candidate sampling gaps.

Uses the cached B3D4 BYTETrack profile and the same 2-by-3 crop as Figure 7.
Regenerate with ./scripts/generate_missrate_figure.sh or run this script directly.
"""
import argparse
import json
from pathlib import Path

import altair as alt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TOLERANCE = .6
CHART_HEIGHT = 55


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / 'scripts/data/missrate_figure_cache')
    parser.add_argument('--output', '-o', type=Path, default=ROOT / 'figures/generated/missrate_tile_curves.pdf')
    parser.add_argument('--width', type=int, default=205)
    parser.add_argument('--height', type=int, default=CHART_HEIGHT)
    parser.add_argument('--row-start', type=int, default=9)
    parser.add_argument('--col-start', type=int, default=10)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.width > 0 and args.height > 0
    gaps = np.asarray(json.loads((args.artifact_dir / 'sample_rates.json').read_text()))
    accuracy = np.load(args.artifact_dir / 'accuracy.npy', allow_pickle=False)
    assert accuracy.ndim == 3 and accuracy.shape[0] == len(gaps)
    assert np.all(gaps > 0) and np.all(np.diff(gaps) > 0)
    r0, c0, rows, cols = args.row_start, args.col_start, args.rows, args.cols
    assert rows > 0 and cols > 0 and 0 <= r0 <= accuracy.shape[1] - rows
    assert 0 <= c0 <= accuracy.shape[2] - cols
    # This illustrative selection uses the same two-decimal rates as the percentage matrices.
    missrates = np.round(1 - accuracy[:, r0:r0 + rows, c0:c0 + cols].astype(np.float64), 2)
    assert np.isfinite(missrates).all() and np.all((0 <= missrates) & (missrates <= 1))
    tiles = [f'({r0 + row},{c0 + col})' for row in range(rows) for col in range(cols)]
    records = [dict(tile=tiles[row * cols + col], gamma=int(gap), missrate=float(missrates[k, row, col]))
               for row in range(rows) for col in range(cols) for k, gap in enumerate(gaps)]
    assert len(records) == rows * cols * len(gaps)
    # Highlight only the measured segment crossing the example tolerance, retaining its actual endpoints.
    crossing, selected = [], []
    for left, right in zip(records, records[1:]):
        if left['tile'] == right['tile'] and left['missrate'] <= EXAMPLE_TOLERANCE < right['missrate']:
            crossing.extend([left, right])
            selected.append(left)
    x = alt.X('missrate:Q', title='Miss rate', scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(values=[0, .2, .4, .6, .8, 1], format='.1f',
                            titleAnchor='end', titleAlign='right', titleBaseline='bottom', titleY=-2))
    y = alt.Y('gamma:Q', title='γ (frames)', scale=alt.Scale(type='log', base=2),
              axis=alt.Axis(values=gaps.tolist(), labelBaseline=alt.ExprRef(
                  expr=f"datum.value === {int(gaps[-1])} ? 'top' : 'middle'")))
    base = alt.Chart(alt.Data(values=records))
    lines = base.mark_line(strokeWidth=1, opacity=.4).encode(
        x=x, y=y, detail='tile:N',
        # Connect in gamma order, preserving any non-monotonicity in the measured miss rates.
        order=alt.Order('gamma:Q'),
    )
    highlighted = alt.Chart(alt.Data(values=crossing)).mark_line(color='#2ca02c', strokeWidth=1.5).encode(
        x=x, y=y, detail='tile:N', order='gamma:Q',
    )
    points = base.mark_point(filled=True, size=15, opacity=1).encode(x=x, y=y)
    circles = alt.Chart(alt.Data(values=selected)).mark_point(
        filled=False, color='#2ca02c', size=65, strokeWidth=1.2, opacity=1,
    ).encode(x=x, y=y)
    threshold = alt.Chart(alt.Data(values=[dict(missrate=EXAMPLE_TOLERANCE)]))
    rule = threshold.mark_rule(color='#d62728', strokeWidth=1).encode(x=x)
    label = threshold.mark_text(text='M̄', color='#d62728', align='left', baseline='bottom', dx=4, fontSize=8).encode(
        x=x, y=alt.value(args.height - 2),
    )
    chart = alt.layer(lines, highlighted, rule, circles, label, points).properties(width=args.width, height=args.height).configure(
        font='Arial',
    ).configure_axis(labelFontSize=8, titleFontSize=8, titleFontWeight='normal', gridColor='#dddddd').configure_view(stroke=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.output))
    print(f'wrote {args.output} ({rows * cols} tiles, {len(gaps)} points per tile)')


if __name__ == '__main__':
    main()
