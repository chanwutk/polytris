"""Refresh fixed-run reference denominators from p121 without rerunning extraction."""
import json
from pathlib import Path

import duckdb

from polyis.io import cache


def refresh(root: Path):
    inputs = json.loads((root / 'inputs.json').read_text())
    paths = [str(cache.eval(dataset, 'tp', 'measurements', 'query_execution_per_op.csv')) for dataset in inputs]
    con = duckdb.connect(config={'threads': 4})
    # Restrict every denominator to the fixed experiment's original test clips.
    con.execute('CREATE TABLE expected(dataset VARCHAR, video VARCHAR)')
    con.executemany('INSERT INTO expected VALUES (?, ?)',
                    [(dataset, video) for dataset, info in inputs.items() for video in info['videos']])
    con.execute('''CREATE TABLE reference_ops AS SELECT p.*
        FROM read_csv_auto(?, union_by_name=true) p JOIN expected e USING(dataset,video)
        WHERE videoset='test' AND variant='naive' ''', [paths])
    missing = con.execute('''SELECT * FROM expected EXCEPT
        SELECT DISTINCT dataset,video FROM reference_ops''').fetchall()
    assert not missing, f'Missing reference timing clips: {missing}'
    assert con.execute('''SELECT count(*) FROM reference_ops
        WHERE stage='001_preprocess_groundtruth_detection' AND op='read' ''').fetchone()[0] == 0
    rows = con.execute('''SELECT dataset,sum(time) AS naive_seconds,
        sum(time) FILTER(WHERE stage='001_preprocess_groundtruth_detection' AND op='detect') AS detector_seconds
        FROM reference_ops GROUP BY dataset ORDER BY dataset''').fetchall()
    totals = {dataset: dict(dataset=dataset, naive_seconds=seconds, detector_seconds=detector)
              for dataset, seconds, detector in rows}
    assert set(totals) == set(inputs)
    pending = {}
    # Only derived reference FPS and speedup move; measured times, accuracy and launch manifests stay intact.
    for dataset, info in inputs.items():
        path = root / 'results' / f'{dataset}.json'
        result = json.loads(path.read_text())
        reference = totals[dataset]
        assert result['clips'] == len(info['videos'])
        assert 0 < reference['detector_seconds'] <= reference['naive_seconds']
        assert result['frames'] > 0 and result['seconds'] > 0
        reference['frames'] = result['frames']
        result['reference']['reference_fps'] = result['frames'] / reference['naive_seconds']
        result['speedup'] = reference['naive_seconds'] / result['seconds']
        pending[path] = result
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    by_label = {inputs[result['dataset']]['label']: result for result in pending.values()}
    assert {row['dataset'] for row in summary} == set(by_label)
    for row in summary:
        result = by_label[row['dataset']]
        assert row['frames'] == result['frames'] and row['seconds'] > 0
        # Historical summary rows can describe older runs; keep each row's own measured duration.
        row['speedup'] = totals[result['dataset']]['naive_seconds'] / row['seconds']
    pending[summary_path] = summary
    pending[root / 'naive_operator_times.json'] = list(totals.values())
    # Validate all inputs before replacing any derived summary.
    for path, value in pending.items():
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(list(totals.values()), indent=2))


if __name__ == '__main__':
    refresh(Path('/polyis/measurements-results/fixed_test_s1_square_m02'))
