"""
execution's throughput at the paper's operating points vs the throughput the paper reports.

Inputs:
  measurements-results/execution-paper/<dataset>-<loss>/result.json   execution.tools.benchmark --skip-scripts
  measurements-results/r1o3_clean_extraction_timing/throughput.json     the paper side and the clean script reruns
  measurements-results/r1o3_clean_extraction_timing/raw/plan.json       dataset ids <-> paper labels

Throughput boundaries compared (frames / seconds over the same test videos):
  execution      first-to-last result of execution.main (wall clock, pipelined, one decode, startup excluded)
  execution_proc the execution.main process's wall clock (startup and model loading included)
  paper_p121      the paper's published Tetris throughput (sum of logged op times, Tetris decode not counted)
  paper_one_dec   paper_p121's ops plus one decode of every video (R1.O3's matched boundary)
  clean_steady    the staged scripts' wall clock on idle bombe GPUs, per-stage process startup removed

Writes <runs>/summary.json and prints the table.  Run in the polyis container:
  python measurements/execution_paper_throughput.py \
      --runs measurements-results/execution-paper-current --baseline measurements-results/execution-paper
--baseline adds an earlier set of runs' execution throughput for comparison.
"""

import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
R1O3 = ROOT / 'measurements-results' / 'r1o3_clean_extraction_timing'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=ROOT / 'measurements-results' / 'execution-paper')
    parser.add_argument('--baseline', type=Path, default=None)
    return parser.parse_args()


def load_runs(con: duckdb.DuckDBPyConnection, name: str, runs: Path) -> None:
    """Table `name`: one row per operating point's result.json under runs."""
    files = sorted(str(p) for p in runs.glob('*-0.??/result.json'))
    assert files, f'no execution results under {runs}'
    con.execute(f"""
        CREATE TABLE {name} AS
        SELECT dataset, regexp_extract(config, '-(0\\.\\d+)\\.yaml$', 1) AS loss,
               len(videos) AS videos, execution.configured.frames AS frames,
               execution.configured.fps AS execution, execution.configured.process_fps AS execution_proc,
               execution.configured.outputs_same_up_to_id_renaming AS same_up_to_renaming,
               execution.configured.outputs_identical AS identical
        FROM read_json_auto({files})
    """)


def main() -> None:
    args = parse_args()
    con = duckdb.connect()
    # Paper and clean throughput per (label, loss), flattened from throughput.json's nesting.
    throughput = json.loads((R1O3 / 'throughput.json').read_text())
    labels = {k: v['label'] for k, v in json.loads((R1O3 / 'raw' / 'plan.json').read_text())['datasets'].items()}
    paper_rows = [
        {'label': label, 'loss': loss, 'config': rec['config'], 'frames': rec['frames'], 'videos': rec['videos'],
         'paper_p121': rec['paper']['p121']['tetris_fps'], 'paper_one_dec': rec['paper']['one_decode']['tetris_fps'],
         'clean_steady': (rec['clean'].get('wall_clock_steady') or {}).get('tetris_fps')}
        for label, levels in throughput['datasets'].items() for loss, rec in levels.items()
        if isinstance(rec, dict) and 'config' in rec]
    con.execute('CREATE TABLE paper (label VARCHAR, loss VARCHAR, config VARCHAR, frames BIGINT, videos BIGINT, '
                'paper_p121 DOUBLE, paper_one_dec DOUBLE, clean_steady DOUBLE)')
    con.executemany('INSERT INTO paper VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [list(r.values()) for r in paper_rows])
    # execution runs (and optionally an earlier set): one result.json per operating point.
    load_runs(con, 'runs', args.runs)
    if args.baseline is not None:
        load_runs(con, 'base', args.baseline)
    else:
        con.execute('CREATE TABLE base AS SELECT * FROM runs WHERE false')
    con.execute('CREATE TABLE labels (dataset VARCHAR, label VARCHAR)')
    con.executemany('INSERT INTO labels VALUES (?, ?)', list(labels.items()))
    table = con.execute("""
        SELECT p.label, r.dataset, p.loss, p.config, r.videos, r.frames, p.frames AS paper_frames,
               round(r.execution, 1) AS execution, round(r.execution_proc, 1) AS execution_proc,
               round(p.paper_p121, 1) AS paper_p121, round(p.paper_one_dec, 1) AS paper_one_dec,
               round(p.clean_steady, 1) AS clean_steady,
               round(r.execution / p.paper_p121, 2) AS x_paper_p121,
               round(r.execution / p.paper_one_dec, 2) AS x_paper_one_dec,
               round(r.execution / p.clean_steady, 2) AS x_clean_steady,
               r.same_up_to_renaming, r.identical,
               round(b.execution, 1) AS baseline_execution, round(r.execution / b.execution, 2) AS x_baseline
        FROM runs r JOIN labels l USING (dataset) JOIN paper p ON p.label = l.label AND p.loss = r.loss
        LEFT JOIN base b ON b.dataset = r.dataset AND b.loss = r.loss
        ORDER BY p.label, p.loss
    """)
    columns = [d[0] for d in table.description]
    rows = [dict(zip(columns, row)) for row in table.fetchall()]
    # Every run must cover the same videos and frames as the paper's number it is compared with.
    mismatched = [r for r in rows if r['frames'] != r['paper_frames']]
    assert not mismatched, f'frame counts differ from the paper side: {mismatched}'
    (args.runs / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(f"{r['label']:9s} {r['loss']} {r['config']:50s} exec {r['execution']:7.1f} fps "
              f"(proc {r['execution_proc']:6.1f}) | paper p121 {r['paper_p121']:7.1f} one_dec {r['paper_one_dec']:6.1f} "
              f"clean {r['clean_steady']:6.1f} | x{r['x_paper_p121']:.2f} x{r['x_paper_one_dec']:.2f} "
              f"x{r['x_clean_steady']:.2f} | tracks {r['same_up_to_renaming']}/{r['videos']}"
              + (f" | baseline {r['baseline_execution']:.1f} x{r['x_baseline']:.2f}" if r['baseline_execution'] else ''))


if __name__ == '__main__':
    main()
