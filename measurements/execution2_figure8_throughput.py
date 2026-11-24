"""
execution2's throughput at every full-Tetris point of the paper's Figure 8 vs the throughput the figure reports.

Inputs:
  measurements-results/execution2-figure8/figure8_points.json             the "Tetris" rows Figure 8 plots
  measurements-results/execution2-figure8/runs/<dataset>/<config>/result.json   execution2.tools.benchmark
                                                                             --skip-scripts, one per point
Throughputs (frames per second over each dataset's test videos):
  reported     Figure 8's throughput_fps (the published p121 boundary: logged op times, Tetris decode not counted)
  execution2   first-to-last result of execution2.main (wall clock, pipelined, decode included, startup excluded)
  process      the execution2.main process's wall clock (startup and model loading included)

Writes measurements-results/execution2-figure8/summary.json (one row per point) and
summary_by_dataset.json, and prints both.  Points without a result yet are listed as missing.
Run in the polyis container:  python measurements/execution2_figure8_throughput.py
"""

import json
from pathlib import Path

import duckdb

from execution.tools.make_figure8_configs import config_name, params_of

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'measurements-results' / 'execution2-figure8'


def main() -> None:
    con = duckdb.connect()
    points = json.loads((BASE / 'figure8_points.json').read_text())['points']
    con.execute('CREATE TABLE reported (dataset VARCHAR, dataset_display VARCHAR, config VARCHAR, '
                'reported_fps DOUBLE, reported_hota DOUBLE)')
    con.executemany('INSERT INTO reported VALUES (?, ?, ?, ?, ?)',
                    [(p['dataset'], p['dataset_display'], config_name(params_of(p)), p['throughput_fps'],
                      p['HOTA_HOTA']) for p in points])
    files = sorted(str(f) for f in (BASE / 'runs').glob('*/*/result.json'))
    if files:
        con.execute(f"""
            CREATE TABLE runs AS
            SELECT dataset, regexp_extract(config, '/([^/]+)\\.yaml$', 1) AS config, len(videos) AS videos,
                   execution2.configured.frames AS frames, execution2.configured.fps AS execution2_fps,
                   execution2.configured.process_fps AS process_fps,
                   execution2.configured.outputs_same_up_to_id_renaming AS same_up_to_renaming,
                   execution2.configured.outputs_identical AS identical
            FROM read_json_auto({files})
        """)
    else:
        con.execute('CREATE TABLE runs (dataset VARCHAR, config VARCHAR, videos BIGINT, frames BIGINT, '
                    'execution2_fps DOUBLE, process_fps DOUBLE, same_up_to_renaming BIGINT, identical BIGINT)')
    per_point = con.execute("""
        SELECT r.dataset, r.dataset_display, r.config, round(r.reported_fps, 1) AS reported_fps,
               round(x.execution2_fps, 1) AS execution2_fps, round(x.process_fps, 1) AS process_fps,
               round(x.execution2_fps / r.reported_fps, 2) AS x_reported, x.videos, x.frames,
               x.same_up_to_renaming, x.identical, round(r.reported_hota, 4) AS reported_hota
        FROM reported r LEFT JOIN runs x USING (dataset, config)
        ORDER BY r.dataset, r.reported_fps
    """)
    columns = [d[0] for d in per_point.description]
    rows = [dict(zip(columns, row)) for row in per_point.fetchall()]
    by_dataset = con.execute("""
        SELECT r.dataset, any_value(r.dataset_display) AS dataset_display, count(*) AS points,
               count(x.config) AS run, sum(CASE WHEN x.execution2_fps > r.reported_fps THEN 1 ELSE 0 END) AS faster,
               round(median(x.execution2_fps / r.reported_fps), 2) AS median_x,
               round(min(x.execution2_fps / r.reported_fps), 2) AS min_x,
               round(max(x.execution2_fps / r.reported_fps), 2) AS max_x,
               sum(CASE WHEN x.same_up_to_renaming = x.videos THEN 1 ELSE 0 END) AS outputs_match
        FROM reported r LEFT JOIN runs x USING (dataset, config)
        GROUP BY r.dataset ORDER BY r.dataset
    """)
    dcols = [d[0] for d in by_dataset.description]
    drows = [dict(zip(dcols, row)) for row in by_dataset.fetchall()]
    (BASE / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (BASE / 'summary_by_dataset.json').write_text(json.dumps(drows, indent=2) + '\n')
    for r in drows:
        print(f"{r['dataset_display']:11s} points {r['points']:3d} run {r['run']:3d} faster {r['faster']:3d} "
              f"x median {r['median_x']} [{r['min_x']}, {r['max_x']}] outputs match {r['outputs_match']}")
    missing = [f"{r['dataset']}/{r['config']}" for r in rows if r['execution2_fps'] is None]
    print(f'{len(rows) - len(missing)} of {len(rows)} points have results; missing: {missing[:10]}{" ..." if len(missing) > 10 else ""}')


if __name__ == '__main__':
    main()
