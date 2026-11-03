"""Regression gates and old-vs-new report after re-scoring B3D on human ground truth.

Compares every rescored table with its <path>.auto-gt snapshot (measurements/snapshot_outputs.py):
- Gates (any failure exits non-zero): each table keeps the same configurations and every non-accuracy
  column; non-B3D tables are byte-identical to the checksums recorded before the run.
- Report: reference, OTIF and LEAP HOTA old vs new, the Tetris HOTA shift, and the fastest Tetris
  configuration within each HOTA-loss bound (HOTA >= reference x (1 - loss), as in r7o1) old vs new.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb

from measurements.common import query_dicts


# Columns a ground-truth change may alter; every other column must stay identical.
ACCURACY_COLUMNS = ('HOTA_HOTA', 'HOTA_AssA', 'HOTA_DetA', 'Count_DetsMAPE', 'Count_TracksMAPE')
# Columns that identify a row in each kind of table.
TETRIS_KEY = ('variant', 'variant_id')
PRIOR_KEY = ('param_id',)


def parse_args():
    parser = argparse.ArgumentParser(description='Gates and old-vs-new report for the B3D human ground truth')
    parser.add_argument('--cache', default='/polyis-cache')
    parser.add_argument('--datasets', nargs='+', default=['jnc0', 'jnc2', 'jnc6', 'jnc7'])
    parser.add_argument('--untouched-checksums', required=True)
    parser.add_argument('--losses', nargs='+', type=float, default=[0.05, 0.10])
    parser.add_argument('--suffix', default='.auto-gt')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    # Hash a file's bytes.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_table(con: duckdb.DuckDBPyConnection, old: Path, new: Path, key: tuple[str, ...]) -> list[str]:
    """Gate failures between a snapshot and its rescored table: changed rows or non-accuracy values."""
    # Load both versions as text so any formatting change in non-accuracy columns is caught too.
    con.execute('CREATE OR REPLACE TABLE old AS SELECT * FROM read_csv(?, all_varchar=true)', [str(old)])
    con.execute('CREATE OR REPLACE TABLE new AS SELECT * FROM read_csv(?, all_varchar=true)', [str(new)])
    columns = [r[0] for r in con.execute('DESCRIBE old').fetchall()]
    if columns != [r[0] for r in con.execute('DESCRIBE new').fetchall()]:
        return [f'{new}: columns changed']
    stable = [c for c in columns if c not in ACCURACY_COLUMNS]
    keys = ', '.join(f'"{c}"' for c in key)
    failures = []
    # The same configurations must be scored, once each.
    for side in ('old', 'new'):
        dupes = con.execute(f'SELECT count(*) FROM (SELECT {keys} FROM {side} GROUP BY ALL HAVING count(*) > 1)').fetchone()[0]
        if dupes:
            failures.append(f'{new}: {dupes} duplicate keys in {side}')
    missing = con.execute(f'SELECT count(*) FROM (SELECT {keys} FROM old EXCEPT SELECT {keys} FROM new)').fetchone()[0]
    added = con.execute(f'SELECT count(*) FROM (SELECT {keys} FROM new EXCEPT SELECT {keys} FROM old)').fetchone()[0]
    if missing or added:
        failures.append(f'{new}: {missing} configurations dropped, {added} added')
    # Runtime and configuration columns must not move.
    cols = ', '.join(f'"{c}"' for c in stable)
    changed = con.execute(f'SELECT count(*) FROM (SELECT {cols} FROM old EXCEPT SELECT {cols} FROM new)').fetchone()[0]
    if changed:
        failures.append(f'{new}: {changed} rows changed outside the accuracy columns')
    return failures


def tetris_report(con: duckdb.DuckDBPyConnection, old: Path, new: Path, losses: list[float]) -> dict:
    """Reference HOTA, Tetris HOTA shift, and the operating point per loss bound, old vs new."""
    con.execute('CREATE OR REPLACE TABLE t AS SELECT \'old\' side, * FROM read_csv(?) UNION ALL BY NAME '
                'SELECT \'new\' side, * FROM read_csv(?)', [str(old), str(new)])
    reference = {r['side']: r['HOTA_HOTA'] for r in query_dicts(con, "SELECT side, HOTA_HOTA FROM t WHERE variant = 'naive'")}
    shift = query_dicts(con, '''SELECT count(*) configs, avg(n.HOTA_HOTA - o.HOTA_HOTA) mean_delta,
            min(n.HOTA_HOTA - o.HOTA_HOTA) min_delta, max(n.HOTA_HOTA - o.HOTA_HOTA) max_delta
        FROM t o JOIN t n USING (variant_id) WHERE o.side = 'old' AND n.side = 'new' AND o.variant = 'polytris' ''')[0]
    points = {}
    for loss in losses:
        # Fastest configuration meeting the bound on each side; ties broken by name for determinism.
        rows = query_dicts(con, '''SELECT p.side, p.variant_id, p.throughput_fps, p.HOTA_HOTA
            FROM t p JOIN t r ON r.side = p.side AND r.variant = 'naive'
            WHERE p.variant = 'polytris' AND p.HOTA_HOTA >= r.HOTA_HOTA * (1 - ?)
            QUALIFY row_number() OVER (PARTITION BY p.side ORDER BY p.throughput_fps DESC, p.variant_id) = 1''', [loss])
        chosen = {r['side']: r for r in rows}
        points[f'{loss:g}'] = {**chosen, 'moved': chosen.get('old', {}).get('variant_id') != chosen.get('new', {}).get('variant_id')}
    return {'reference_hota': reference, 'tetris_hota_shift': shift, 'operating_points': points}


def prior_report(con: duckdb.DuckDBPyConnection, old: Path, new: Path) -> list[dict]:
    # HOTA of every prior-system parameter set, old vs new.
    return query_dicts(con, '''SELECT o.param_id, o.HOTA_HOTA old_hota, n.HOTA_HOTA new_hota
        FROM read_csv(?) o JOIN read_csv(?) n USING (param_id) ORDER BY o.param_id''', [str(old), str(new)])


def main():
    args = parse_args()
    cache = Path(args.cache)
    snap = lambda p: p.with_name(p.name + args.suffix)
    failures: list[str] = []
    report: dict = {'datasets': {}}
    with duckdb.connect() as con:
        for dataset in args.datasets:
            tradeoff = cache / dataset / 'evaluation' / '090_tradeoff' / 'tradeoff.csv'
            failures += compare_table(con, snap(tradeoff.parent) / 'tradeoff.csv', tradeoff, TETRIS_KEY)
            entry = {'tetris': tetris_report(con, snap(tradeoff.parent) / 'tradeoff.csv', tradeoff, args.losses)}
            for system in ('otif', 'leap'):
                prior = cache / 'SOTA' / system / dataset / 'tradeoff.csv'
                failures += compare_table(con, snap(prior), prior, PRIOR_KEY)
                entry[system] = prior_report(con, snap(prior), prior)
            report['datasets'][dataset] = entry
    # Tables outside B3D must be untouched, byte for byte.
    recorded = json.loads(Path(args.untouched_checksums).read_text())
    report['untouched_checked'] = len(recorded)
    failures += [f'{path}: changed (non-B3D table)' for path, digest in recorded.items() if sha256(Path(path)) != digest]
    report['gate_failures'] = failures
    Path(args.output).write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))
    if failures:
        raise SystemExit(f'{len(failures)} gate failure(s); see {args.output}')


if __name__ == '__main__':
    main()
