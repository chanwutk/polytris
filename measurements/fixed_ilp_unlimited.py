#!/usr/local/bin/python
"""Solve the 200 fixed-test ILPs sequentially with Gurobi's infinite time limit."""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import duckdb
import gurobipy
import numpy as np

from measurements.common import decode_grids
from measurements import r9o3_ilp_instance_sizes as sizes
from polyis.io import cache
from polyis.sample.ilp.c.gurobi import solve_ilp


EXPERIMENT = Path('/polyis/measurements-results/fixed_test_s1_square_m02')
OUTPUT = Path('/polyis/measurements-results/fixed_ilp_unlimited')
TIME_LIMIT_SECONDS = 1e100  # Gurobi's documented infinity value and TimeLimit maximum.


def selected_stats(structure, selected):
    """Return objective and assert that a returned selection satisfies every ILP row."""
    kept = np.zeros(len(structure['lengths']), dtype=bool)
    for frame, polyomino in selected:
        kept[structure['offsets'][frame] + polyomino] = True
    assert kept[structure['forced']].all()
    entry_kept = kept[structure['var']]
    cumulative = np.concatenate([[0], np.cumsum(entry_kept)])
    hits = cumulative[structure['window_end']] - cumulative[structure['window'] + 1]
    assert not (entry_kept[structure['window']] & (hits == 0)).any()
    return int(np.dot(kept, structure['lengths']))


def write_summary(results_path):
    con = duckdb.connect(config={'threads': 1})
    source = str(results_path).replace("'", "''")
    result = con.execute(f"""SELECT dataset, count(*) clips,
        round(min(solve_ms), 3) min_solve_ms,
        round(median(solve_ms), 3) median_solve_ms,
        round(quantile_cont(solve_ms, .9), 3) p90_solve_ms,
        round(quantile_cont(solve_ms, .95), 3) p95_solve_ms,
        round(max(solve_ms), 3) max_solve_ms,
        round(avg(solve_ms), 3) mean_solve_ms,
        round(median(build_ms), 3) median_build_ms,
        round(max(total_ms), 3) max_total_ms
      FROM read_json_auto('{source}') GROUP BY GROUPING SETS ((dataset), ())
      ORDER BY dataset NULLS LAST""")
    columns = [column[0] for column in result.description]
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    for row in rows:
        row['dataset'] = row['dataset'] or 'all'
    summary = {
        'measurement': f'same fixed-test ILPs, sequential, Gurobi TimeLimit={TIME_LIMIT_SECONDS}s',
        'clips': sum(row['clips'] for row in rows if row['dataset'] != 'all'),
        'by_dataset_and_all': rows,
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((EXPERIMENT / 'inputs.json').read_text())
    old = {(row['dataset'], row['video']): row for row in
           json.loads((EXPERIMENT / 'operator_details.json').read_text())}
    results_path = OUTPUT / 'results.jsonl'
    done = set()
    if results_path.exists():
        done = {(row['dataset'], row['video']) for row in
                map(json.loads, results_path.read_text().splitlines())}

    metadata = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'host': platform.node(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'gurobi_version': gurobipy.gurobi.version(),
        'time_limit': TIME_LIMIT_SECONDS,
        'execution': 'one process; clips solved sequentially',
        'source_inputs': str(EXPERIMENT / 'inputs.json'),
    }
    (OUTPUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    total = sum(len(info['videos']) for info in inputs.values())
    with results_path.open('a', buffering=1) as output:
        for dataset, info in inputs.items():
            cache.CACHE_DIR = f'/runs/{dataset}/polyis-cache'
            distance = sizes.max_distance(dataset, 'sortcython', 60, .8, 1)
            for video in info['videos']:
                if (dataset, video) in done:
                    continue
                score_path = cache.exec(dataset, 'relevancy', video, 'ShuffleNet05_60_1', 'score/score.jsonl')
                score_bytes = score_path.read_bytes()
                grids, indices = decode_grids([json.loads(line) for line in score_bytes.splitlines()])
                tile_ids, lengths = sizes.group((grids >= np.floor(.5 * 255)).astype(np.uint8))
                structure = sizes.ilp_structure(tile_ids, lengths, distance)
                expected = old[(dataset, video)]
                assert sizes.instance_size(structure)['variables'] == expected['variables']

                wall_start = time.monotonic()
                result = solve_ilp(tile_ids, lengths, distance, tile_ids.shape[1], tile_ids.shape[2],
                                   time_limit_seconds=TIME_LIMIT_SECONDS)
                total_ms = (time.monotonic() - wall_start) * 1000
                objective = selected_stats(structure, result.selected)
                row = {
                    'dataset': dataset,
                    'video': video,
                    'frames': len(indices),
                    'variables': expected['variables'],
                    'constraints': expected['constraints'],
                    'selected': len(result.selected),
                    'objective_tiles': objective,
                    'build_ms': result.build_ms,
                    'solve_ms': result.solve_ms,
                    'total_ms': total_ms,
                    'old_solve_ms': next(op['time'] for op in expected['runtime']['runtime']
                                         if op['op'] == 'solve_ilp'),
                    'score_sha256': hashlib.sha256(score_bytes).hexdigest(),
                }
                output.write(json.dumps(row, separators=(',', ':')) + '\n')
                output.flush()
                os.fsync(output.fileno())
                done.add((dataset, video))
                print(f"{len(done)}/{total} {dataset}/{video}: solve={result.solve_ms:.1f} ms", flush=True)

    assert len(done) == total == 200
    write_summary(results_path)


if __name__ == '__main__':
    main()
