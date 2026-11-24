"""Measure complete vehicle-visibility durations from saved annotation-track diagnostics."""
import hashlib
import json
from pathlib import Path

import duckdb

OUT = Path(__file__).resolve().parent
saved = json.loads((OUT / 'all_datasets.json').read_text())['datasets']
datasets = {'caldot1-y05': 'CalDoT1', 'caldot2-y05': 'CalDoT2', 'ams-y05': 'Amsterdam',
            'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
sources = []
for dataset, label in datasets.items():
    directory = OUT if dataset.endswith('-y05') else OUT.parent / 'r1o2_leap_sampling_diagnostics'
    path = directory / f'{dataset}-tracks.parquet'
    sources.append(dict(dataset=dataset, label=label, filename=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

with duckdb.connect() as con:
    con.execute('SET threads=4')
    con.execute('CREATE TABLE sources AS SELECT unnest(?::VARCHAR[]) filename, unnest(?::VARCHAR[]) label',
                [[r['filename'] for r in sources], [r['label'] for r in sources]])
    con.execute('''CREATE TABLE durations AS SELECT label, (p.e-p.s+1)/15.0 AS duration_s
        FROM read_parquet(?,filename=true) p JOIN sources f ON p.filename=f.filename
        WHERE NOT censored AND p.e-p.s+1>=15''', [[r['filename'] for r in sources]])
    result = con.execute('''WITH grouped AS (
        SELECT label,duration_s FROM durations UNION ALL
        SELECT 'B3D pooled',duration_s FROM durations WHERE starts_with(label,'B3D'))
        SELECT label,count(*) tracks,avg(duration_s) mean_s,stddev_pop(duration_s) stddev_s,
               var_pop(duration_s) variance_s2,stddev_pop(duration_s)/avg(duration_s) coefficient_of_variation,
               quantile_cont(duration_s,0.1) p10_s,median(duration_s) p50_s,quantile_cont(duration_s,0.9) p90_s
        FROM grouped GROUP BY label ORDER BY label''')
    rows = [dict(zip([column[0] for column in result.description], row)) for row in result.fetchall()]
    for row in rows:
        if row['label'] != 'B3D pooled':
            previous = saved[row['label']]['duration_uncensored_at_least_one_second']
            assert row['tracks'] == previous['tracks']
            assert abs(row['mean_s'] - previous['mean_s']) < 1e-10
            assert abs(row['coefficient_of_variation'] - previous['coefficient_of_variation']) < 1e-10
    report = dict(method='Population variance of (last-first+1)/15 seconds; tracks exclude both clip boundaries and last at least one second. B3D pooled combines B3D1-4; B3D labels are provisional.',
                  sources=sources, rows=rows)
    (OUT / 'duration_variability.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
