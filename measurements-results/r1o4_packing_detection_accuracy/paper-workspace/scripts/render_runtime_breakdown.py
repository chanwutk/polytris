"""Stack the cached runtime components counted in Figure 8 at its 5% and 10% points."""
import json
import sys
from pathlib import Path

import altair as alt
import duckdb

sys.path.insert(0, '/polyis')
from measurements import r9o1_classifier_cost as costs
from measurements.common import DATASET_LABELS, query_dicts
from polyis.io import cache


root = Path(__file__).resolve().parents[1]
out = root / 'figures/generated'
out.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(config={'threads': 4})
costs.load_configs(con, list(DATASET_LABELS))
con.execute('CREATE TABLE breakdown(dataset VARCHAR, bound VARCHAR, config VARCHAR, operator VARCHAR, seconds DOUBLE, expected DOUBLE)')
con.execute('CREATE TABLE stage_names(stage VARCHAR, operator VARCHAR, ordinal INTEGER)')
con.executemany('INSERT INTO stage_names VALUES (?,?,?)', [
    (stage, label, index) for index, (stage, label) in enumerate(zip(costs.STAGE_OPERATORS,
        ['Classification', 'ILP', 'Packing', 'Detection', 'Unpacking', 'Tracking']))])
for loss in (0.05, 0.10):
    costs.choose_operating_points(con, loss)
    costs.load_operator_ops(con, {ds: str(cache.eval(ds, 'tp', 'measurements', 'query_execution_per_op.csv')) for ds in DATASET_LABELS})
    assert not query_dicts(con, '''SELECT DISTINCT stage FROM operator_ops
        WHERE system='tetris' AND stage NOT IN (SELECT stage FROM stage_names)''')
    con.execute('''INSERT INTO breakdown
        SELECT o.dataset, ?, c.config, s.operator, sum(o.seconds), c.seconds
        FROM operator_ops o JOIN chosen c USING(dataset) JOIN stage_names s USING(stage)
        WHERE o.system='tetris' GROUP BY ALL''', [f'{loss:.0%}'])

# This is the Figure 8 denominator, with no additional timing fields or remeasurements.
totals = query_dicts(con, '''SELECT dataset,bound,any_value(config) AS config,
    sum(seconds) AS seconds,any_value(expected) AS expected,
    sum(seconds)-any_value(expected) AS difference
    FROM breakdown GROUP BY dataset,bound ORDER BY dataset,bound''')
assert len(totals) == 14
assert all(abs(row['difference']) <= 1e-6 * row['expected'] for row in totals), totals
con.execute('CREATE TABLE labels(dataset VARCHAR,label VARCHAR)')
con.executemany('INSERT INTO labels VALUES (?,?)', list(DATASET_LABELS.items()))
rows = query_dicts(con, '''SELECT b.*, l.label, s.ordinal
    FROM breakdown b JOIN labels l USING(dataset) JOIN stage_names s USING(operator)
    ORDER BY dataset,bound,ordinal''')
bars = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
    y=alt.Y('label:N', title=None, sort=list(DATASET_LABELS.values()), axis=alt.Axis(labelPadding=30)),
    yOffset=alt.YOffset('bound:N', sort=['5%', '10%'], scale=alt.Scale(paddingInner=0.15)),
    x=alt.X('seconds:Q', stack='normalize', title='Share of extraction time', axis=alt.Axis(format='.0%')),
    color=alt.Color('operator:N', title=None,
                    scale=alt.Scale(domain=['Classification', 'ILP', 'Packing', 'Detection', 'Unpacking', 'Tracking'],
                                    scheme='tableau10'), legend=alt.Legend(orient='top', columns=3)),
    order=alt.Order('ordinal:Q'),
    tooltip=['label:N', 'bound:N', 'operator:N', alt.Tooltip('seconds:Q', format='.3f'), 'config:N'],
)
labels = alt.Chart(alt.Data(values=rows)).transform_aggregate(groupby=['label', 'bound']).mark_text(
    align='right', dx=-4, fontSize=9,
).encode(
    y=bars.encoding.y, yOffset=bars.encoding.yOffset, x=alt.value(0), text='bound:N',
)
chart = alt.layer(bars, labels).properties(width=300, height=240).configure(font='sans-serif').configure_axis(
    labelFontSize=11, titleFontSize=12,
).configure_legend(labelFontSize=11)
chart.save(str(out / 'revision_runtime_breakdown.pdf'))
chart.save(str(out / 'revision_runtime_breakdown.html'))
chart.save(str(out / 'revision_runtime_breakdown.vl.json'))
(out / 'revision_runtime_breakdown.json').write_text(json.dumps(dict(
    method='Existing p121 counted per-operator time; same selection and denominator as Figure 8. Each bar normalized to 100%; upper bar 5%, lower bar 10%.',
    totals=totals, rows=rows), indent=2) + '\n')
print(json.dumps(totals, indent=2))
