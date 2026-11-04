#!/usr/local/bin/python
"""Compare the original 0.1-second ILP outputs with the 60-second rerun."""
import json
from pathlib import Path

import duckdb
import numpy as np

from measurements.common import decode_grids
from polyis.io import cache


ORIGINAL = Path('/polyis/measurements-results/fixed_test_s1_square_m02')
RERUN = Path('/polyis/measurements-results/fixed_ilp_60s')
OUTPUT = RERUN / 'comparison'
CONFIG = 'ShuffleNet05_60_1_080_r050_sortcython'


def main():
    original = {(row['dataset'], row['video']): row for row in
                json.loads((ORIGINAL / 'operator_details.json').read_text())}
    rerun = [json.loads(line) for line in (RERUN / 'results.jsonl').read_text().splitlines()]
    records = []
    for new in rerun:
        old = original[(new['dataset'], new['video'])]
        old_solve_ms = next(op['time'] for op in old['runtime']['runtime'] if op['op'] == 'solve_ilp')
        if old_solve_ms < 100:
            continue
        cache.CACHE_DIR = f"/runs/{new['dataset']}/polyis-cache"
        path = cache.exec(new['dataset'], 'pruned-polyominoes', new['video'], CONFIG, 'score/score.jsonl')
        grids, _ = decode_grids([json.loads(line) for line in path.read_text().splitlines()])
        old_tiles = int(np.count_nonzero(grids))
        check = old['solution_check']
        old_feasible = check['forced_dropped'] == 0 and check['window_violations'] == 0
        records.append({
            'dataset': new['dataset'], 'video': new['video'],
            'old_solve_ms': old_solve_ms, 'new_solve_ms': new['solve_ms'],
            'hit_60s': new['solve_ms'] >= 60_000,
            'old_feasible': old_feasible,
            'old_tiles': old_tiles, 'new_tiles': new['objective_tiles'],
            'tiles_reduced': old_tiles - new['objective_tiles'],
        })

    assert len(records) == 103
    assert all(row['old_feasible'] for row in records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT / 'instances.jsonl'
    records_path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in records))

    con = duckdb.connect(config={'threads': 1})
    source = str(records_path).replace("'", "''")
    result = con.execute(f"""SELECT coalesce(dataset, 'all') dataset, count(*) instances,
        count(*) FILTER (new_solve_ms < 1000) within_1s,
        count(*) FILTER (new_solve_ms < 10000) within_10s,
        count(*) within_60s_including_cap,
        count(*) FILTER (hit_60s) hit_60s,
        count(*) FILTER (old_feasible) comparable_instances,
        sum(old_tiles) FILTER (old_feasible) old_tiles,
        sum(new_tiles) FILTER (old_feasible) new_tiles,
        sum(tiles_reduced) FILTER (old_feasible) tiles_reduced,
        round(100 * sum(tiles_reduced) FILTER (old_feasible) /
              sum(old_tiles) FILTER (old_feasible), 3) tiles_reduced_pct,
        count(*) FILTER (old_feasible AND tiles_reduced > 0) improved,
        count(*) FILTER (old_feasible AND tiles_reduced = 0) unchanged,
        count(*) FILTER (old_feasible AND tiles_reduced < 0) worsened
      FROM read_json_auto('{source}')
      GROUP BY GROUPING SETS ((dataset), ()) ORDER BY dataset""")
    columns = [column[0] for column in result.description]
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    summary = {
        'population': 'original fixed-test instances with composite solve time >= 100 ms',
        'threshold_note': '60-second cap hits count as complete; solve_ms includes extraction and cleanup',
        'tile_note': 'all canonical 0.1-second outputs have feasible incumbents',
        'by_dataset_and_all': rows,
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
