"""Stack fixed-configuration operator time relative to the reference pipeline's total time."""
import json
import sys
from pathlib import Path

import altair as alt
import duckdb

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent))
from evaluation.colors import categorical_color_scale

out = root / 'figures/generated'
out.mkdir(parents=True, exist_ok=True)
source = root.parent / 'measurements-results/fixed_test_s1_square_m02/results'
labels = ['CalDoT1', 'CalDoT2', 'Amsterdam', 'B3D1', 'B3D2', 'B3D3', 'B3D4']
con = duckdb.connect(config={'threads': 4})
con.execute('CREATE TABLE measurements AS SELECT * FROM read_json_auto(?, union_by_name=true)', [str(source / '*.json')])
assert con.execute('SELECT count(*),sum(clips) FROM measurements').fetchone() == (7, 200)
assert con.execute("SELECT count(*) FROM measurements WHERE config <> 'ShuffleNet05_60_1_080_r050_square_s100_sortcython'").fetchone()[0] == 0
con.execute('CREATE TABLE stage_names(stage VARCHAR, operator VARCHAR, ordinal INTEGER)')
con.executemany('INSERT INTO stage_names VALUES (?,?,?)', [
    (stage, label, index) for index, (stage, label) in enumerate(zip(
        ['020_exec_classify', '022_exec_prune_polyominoes', '030_exec_compress',
         '040_exec_detect', '050_exec_uncompress', '060_exec_track'],
        ['Classification', 'ILP', 'Packing', 'Detection', 'Unpacking', 'Tracking']))])

# The reference throughput gives its counted total time on the same test frames.
con.execute('''CREATE TABLE breakdown AS SELECT b.*,b.seconds/b.naive_seconds AS runtime_fraction FROM (
    SELECT m.dataset,m.reference.label AS label,m.config,
    s.operator,s.ordinal,
    json_extract(to_json(m.seconds_by_stage), '$.' || s.stage)::DOUBLE AS seconds,
    m.seconds AS expected,m.frames/m.reference.reference_fps AS naive_seconds,m.speedup
    FROM measurements m CROSS JOIN stage_names s) b''')
assert con.execute('SELECT count(*) FROM breakdown WHERE seconds IS NULL OR seconds<0 OR naive_seconds<=0').fetchone()[0] == 0
totals = json.loads(con.execute('''SELECT to_json(list(t)) FROM (
    SELECT dataset,any_value(config) AS config,sum(seconds) AS seconds,
    any_value(expected) AS expected,any_value(naive_seconds) AS naive_seconds,
    sum(runtime_fraction) AS runtime_fraction,1/any_value(speedup) AS inverse_speedup
    FROM breakdown GROUP BY dataset ORDER BY dataset) t''').fetchone()[0])
assert len(totals) == 7
assert all(abs(row['seconds']-row['expected']) <= 1e-6 * row['expected'] for row in totals), totals
assert all(abs(row['runtime_fraction']-row['inverse_speedup']) < 1e-10 for row in totals), totals
rows = json.loads(con.execute('SELECT to_json(list(t)) FROM (SELECT * FROM breakdown ORDER BY dataset,ordinal) t').fetchone()[0])
assert len(rows) == 42
bars = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
    y=alt.Y('label:N', title=None, sort=labels),
    x=alt.X('runtime_fraction:Q', stack='zero', title='Runtime relative to reference pipeline', axis=alt.Axis(format='.0%')),
    color=alt.Color('operator:N', title=None,
                    scale=categorical_color_scale(['Classification', 'ILP', 'Packing', 'Detection', 'Unpacking', 'Tracking']), legend=alt.Legend(
                                        orient='none', legendX=200, legendY=4, columns=3,
                                        values=['ILP', 'Packing', 'Tracking', 'Detection', 'Unpacking', 'Classification'],
                                        columnPadding=8, rowPadding=1, padding=3, fillColor='white')),
    order=alt.Order('ordinal:Q'),
    tooltip=['label:N', 'operator:N', alt.Tooltip('runtime_fraction:Q', title='Relative runtime', format='.2%'),
             alt.Tooltip('seconds:Q', title='Operator time (s)', format='.3f'),
             alt.Tooltip('naive_seconds:Q', title='Reference time (s)', format='.3f'), 'config:N'],
)
chart = bars.properties(width=440, height=80).configure(font='sans-serif').configure_axis(
    labelFontSize=12, titleFontSize=13,
).configure_legend(labelFontSize=12)
chart.save(str(out / 'revision_runtime_breakdown.pdf'))
chart.save(str(out / 'revision_runtime_breakdown.html'))
chart.save(str(out / 'revision_runtime_breakdown.vl.json'))
(out / 'revision_runtime_breakdown.json').write_text(json.dumps(dict(
    method='Fixed s=1, Tr=.5, M=.2, square padding, SORT on 200 test clips. Each operator uses p121 counted time divided by reference total time on the same clips. Bar totals equal inverse speedup; no normalization to Tetris runtime.',
    source=str(source),
    totals=totals, rows=rows), indent=2) + '\n')
print(json.dumps(totals, indent=2))
