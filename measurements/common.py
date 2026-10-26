"""Helpers shared by the revision measurement scripts (see REVISION.md)."""
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from evaluation.p200_compare_compute import load_sota_tradeoff_data
from polyis.io import cache
from polyis.utilities import load_tradeoff_data, split_tradeoff_variants


# Prior systems compared with Tetris, as named in the SOTA cache.
PRIOR_SYSTEMS = ('otif', 'leap')
# Every dataset in the paper's evaluation, with the labels used in the manuscript.
DATASET_LABELS = {'caldot1-y05': 'CalDoT1', 'caldot2-y05': 'CalDoT2', 'ams-y05': 'Amsterdam',
                  'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
# Root of the per-script result directories, next to this package.
RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'measurements-results'


def results_dir(script_path: str) -> Path:
    """measurements-results/<script name>/ for measurements/<script name>.py, created if missing."""
    # Name the directory after the script so outputs always match their producer.
    path = RESULTS_ROOT / Path(script_path).stem
    # Create it on first use.
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(directory: Path, filename: str, data: Any) -> Path:
    """Write one JSON result file and report where it went."""
    # Resolve the output file inside the script's result directory.
    path = directory / filename
    # Write with stable key order so reruns produce clean diffs.
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    # Report the output path for the run log.
    print(f'wrote {path}')
    return path


def query_dicts(con: duckdb.DuckDBPyConnection, sql: str, params: list | dict | None = None) -> list[dict]:
    """Run a DuckDB query and return its rows as dicts keyed by column name."""
    # Execute with bound parameters.
    cursor = con.execute(sql, params or [])
    # Column names from the result description.
    names = [d[0] for d in cursor.description]
    # Pair each row with the column names.
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def load_test_tradeoff(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Tetris test-split configurations (validated Pareto set) and the reference pipeline's test row.

    Reuses the pipeline's own loader for the small tradeoff table; analysis happens in DuckDB.
    """
    # Load the canonical split-level tradeoff table written by p130.
    tradeoff = load_tradeoff_data(dataset)
    # Keep only the test split.
    tradeoff = tradeoff[tradeoff['videoset'] == 'test']
    # Separate Tetris configurations from the reference (naive) pipeline.
    tetris, naive = split_tradeoff_variants(tradeoff)
    # The reference pipeline must appear exactly once on the test split.
    assert len(naive) == 1, f'{dataset}: expected one naive test row, found {len(naive)}'
    # Every Tetris configuration must have a positive throughput to be comparable.
    assert (tetris['throughput_fps'] > 0).all(), f'{dataset}: non-positive Tetris throughput'
    return tetris, naive.iloc[0]


def prior_system_test_rows(dataset: str, system: str) -> pd.DataFrame:
    """A prior system's (OTIF or LEAP) test-split configurations with param_id, time, and HOTA_HOTA."""
    # Load the transformed SOTA tradeoff table written by p142.
    rows = load_sota_tradeoff_data([dataset], system)
    # A missing table is an error: the comparison would silently drop a system.
    assert not rows.empty, f'{dataset}: no {system} tradeoff table'
    # Keep only the test split.
    rows = rows[rows['videoset'] == 'test']
    # Every configuration needs a positive runtime to derive throughput.
    assert (rows['time'] > 0).all(), f'{dataset}: non-positive {system} runtime'
    return rows


def fastest_within(con: duckdb.DuckDBPyConnection, table: str, loss: float | None, systems: list[str],
                   bounds: dict[str, float] | None = None) -> list[dict]:
    """Per dataset and system, the fastest configuration whose accuracy meets that dataset's bound.

    The bound is `loss` relative to the reference pipeline's accuracy (reference x (1 - loss)),
    or, when `bounds` is given instead, an explicit accuracy per dataset (for example the
    accuracy of the weakest system, so that it qualifies). Exactly one of the two must be set.

    `table` needs columns dataset, system, config, fps, accuracy, with exactly one row per
    dataset whose system is 'reference'. The bound is inclusive; throughput ties go to the
    more accurate configuration, then to the smaller config name. Systems that miss the bound
    are returned with meets_bound false and their best accuracy. Each row also reports the
    bound as a relative loss from the reference (1 - bound / reference accuracy).
    """
    # Exactly one way of setting the bound.
    assert (loss is None) != (bounds is None), 'set exactly one of loss and bounds'
    # Every dataset needs exactly one positive reference accuracy to define a relative bound.
    bad = query_dicts(con, f"""
        SELECT dataset, count(*) AS n, min(accuracy) AS accuracy
        FROM {table} WHERE system = 'reference' GROUP BY dataset
        HAVING n <> 1 OR min(accuracy) <= 0
    """)
    assert not bad, f'invalid reference rows: {bad}'
    missing = query_dicts(con, f"""
        SELECT DISTINCT dataset FROM {table}
        WHERE dataset NOT IN (SELECT dataset FROM {table} WHERE system = 'reference')
    """)
    assert not missing, f'datasets without a reference row: {missing}'
    if bounds is None:
        # Relative bound: the same loss for every dataset.
        bound_sql = 'SELECT dataset, reference_accuracy * (1 - $loss) AS bound FROM reference'
        params = {'loss': loss, 'systems': systems}
    else:
        # Explicit bounds: every dataset in the table needs one.
        datasets = {r['dataset'] for r in query_dicts(con, f'SELECT DISTINCT dataset FROM {table}')}
        assert datasets <= set(bounds), f'no bound for datasets {sorted(datasets - set(bounds))}'
        con.execute('CREATE OR REPLACE TEMP TABLE explicit_bounds (dataset VARCHAR, bound DOUBLE)')
        con.executemany('INSERT INTO explicit_bounds VALUES (?, ?)', list(bounds.items()))
        bound_sql = 'SELECT dataset, bound FROM explicit_bounds'
        params = {'systems': systems}
    # Select, per dataset and system, the fastest configuration inside the bound.
    return query_dicts(con, f"""
        WITH reference AS (
            SELECT dataset, accuracy AS reference_accuracy, fps AS reference_fps
            FROM {table} WHERE system = 'reference'
        ),
        bounds AS ({bound_sql}),
        candidates AS (
            SELECT t.dataset, t.system, t.config, t.fps, t.accuracy,
                   r.reference_accuracy, r.reference_fps, b.bound
            FROM {table} t JOIN reference r USING (dataset) JOIN bounds b USING (dataset)
            WHERE list_contains($systems, t.system)
        ),
        best AS (
            SELECT dataset, system, max(accuracy) AS best_accuracy, any_value(bound) AS bound,
                   any_value(reference_accuracy) AS reference_accuracy, any_value(reference_fps) AS reference_fps
            FROM candidates GROUP BY dataset, system
        ),
        chosen AS (
            SELECT dataset, system, config, fps, accuracy FROM candidates
            WHERE accuracy >= bound
            QUALIFY row_number() OVER (PARTITION BY dataset, system ORDER BY fps DESC, accuracy DESC, config) = 1
        )
        SELECT b.dataset, b.system, b.bound, 1 - b.bound / b.reference_accuracy AS loss,
               b.reference_accuracy, b.reference_fps, b.best_accuracy,
               c.config IS NOT NULL AS meets_bound, c.config, c.fps, c.accuracy
        FROM best b LEFT JOIN chosen c USING (dataset, system)
        ORDER BY b.dataset, b.system
    """, params)


def best_accuracy_bounds(con: duckdb.DuckDBPyConnection, table: str, system: str) -> dict[str, float]:
    """Per dataset, the best accuracy any configuration of `system` reaches: the bound at which it just qualifies."""
    rows = query_dicts(con, f"""
        SELECT dataset, max(accuracy) AS bound FROM {table} WHERE system = $system GROUP BY dataset
    """, {'system': system})
    # Every dataset in the table must have at least one configuration of the system.
    datasets = {r['dataset'] for r in query_dicts(con, f'SELECT DISTINCT dataset FROM {table}')}
    found = {r['dataset']: r['bound'] for r in rows}
    assert set(found) == datasets, f'{system} has no configuration on {sorted(datasets - set(found))}'
    return found


def operating_points(con: duckdb.DuckDBPyConnection, table: str, systems: list[str], loss: float | None = None,
                     bounds: dict[str, float] | None = None, primary: str = 'tetris') -> dict:
    """Each system's fastest configuration within the bound, per dataset, arranged for a JSON report.

    `table` needs columns dataset, system, config, fps, hota, accuracy (see fastest_within for the
    bound). Every system that meets the bound gets its speedup over the reference pipeline, and
    `primary` gets its speedup over every other system that also meets the bound.
    """
    chosen = fastest_within(con, table, loss, systems, bounds)
    # HOTA of each configuration, to show what accuracy-selected points cost in tracking fidelity.
    hota = {(r['dataset'], r['system'], r['config']): r['hota']
            for r in query_dicts(con, f'SELECT dataset, system, config, hota FROM {table}')}
    out: dict = {}
    # Arrange the selected rows as {dataset: {bound, loss, reference, systems, ratios}}.
    for r in chosen:
        entry = out.setdefault(r['dataset'], dict(
            bound=round(r['bound'], 4), loss=round(r['loss'], 4), systems={},
            reference=dict(accuracy=r['reference_accuracy'], fps=round(r['reference_fps'], 1))))
        entry['systems'][r['system']] = (
            dict(meets_bound=True, config=r['config'], accuracy=r['accuracy'], fps=round(r['fps'], 1),
                 hota=hota[(r['dataset'], r['system'], r['config'])],
                 speedup_vs_reference=round(r['fps'] / r['reference_fps'], 1))
            if r['meets_bound'] else dict(meets_bound=False, best_accuracy=r['best_accuracy']))
    # The primary system against each other system that also meets the bound.
    for entry in out.values():
        lead = entry['systems'].get(primary)
        entry.update({f'{primary}_speedup_vs_{s}': round(lead['fps'] / v['fps'], 1)
                      for s, v in entry['systems'].items()
                      if s != primary and lead and lead['meets_bound'] and v['meets_bound']})
    return out


def print_operating_points(points: dict, metric: str, primary: str = 'tetris'):
    """Print {bound name: {dataset: operating points}} as one line per dataset."""
    for name, by_dataset in points.items():
        print(f'\n{metric}: bound {name}')
        for dataset, p in by_dataset.items():
            # One cell per system: speedup over the reference, or the best accuracy it reached.
            cells = [f"{s} {v['speedup_vs_reference']}x" if v['meets_bound'] else f"{s} misses (best {v['best_accuracy']})"
                     for s, v in p['systems'].items()]
            ratios = ', '.join(f"vs {k.removeprefix(primary + '_speedup_vs_')} {v}x" for k, v in p.items()
                               if k.startswith(primary + '_speedup_vs_'))
            print(f"  {dataset:10s} ref {p['reference']['accuracy']:.3f} bound {p['bound']} (loss {p['loss']:.1%}) | "
                  + ' | '.join(cells) + f' | {primary} {ratios}')


def system_configs(dataset: str) -> list[dict]:
    """Every test-split configuration of the reference pipeline, Tetris, OTIF, and LEAP on a dataset.

    Each row has system, config, fps (throughput over the test split), and hota. Tetris rows are
    its validated configurations from the tradeoff table; prior systems' throughput is the test
    split's frame count over their runtime, as in evaluation/p200_compare_compute.py.
    """
    # Tetris configurations and the reference pipeline from the tradeoff table.
    tetris, naive = load_test_tradeoff(dataset)
    # Test-split frame count, used to turn prior systems' runtimes into throughput.
    frames = float(naive['frame_count'])
    rows = [dict(system='reference', config='naive', fps=float(naive['throughput_fps']), hota=float(naive['HOTA_HOTA']))]
    rows += [dict(system='tetris', config=str(c), fps=float(f), hota=float(h))
             for c, f, h in zip(tetris['variant_id'], tetris['throughput_fps'], tetris['HOTA_HOTA'])]
    for system in PRIOR_SYSTEMS:
        prior = prior_system_test_rows(dataset, system)
        rows += [dict(system=system, config=f'param {int(p)}', fps=frames / float(t), hota=float(h))
                 for p, t, h in zip(prior['param_id'], prior['time'], prior['HOTA_HOTA'])]
    return rows


def output_path(dataset: str, system: str, config: str, video: str, kind: str = 'tracking') -> Path:
    """A configuration's tracking.jsonl (kind 'tracking') or detection.jsonl (kind 'detection') for one test video.

    system is 'ground truth' (003_groundtruth), 'reference' (002_naive), 'tetris' (060/050 stages),
    or a prior system ('otif', 'leap'; config 'param N'). Prior systems only have tracking output.
    """
    # The file name inside the stage directory.
    name = f'{kind}.jsonl'
    if system == 'ground truth':
        return Path(cache.exec(dataset, 'groundtruth', video, name))
    if system == 'reference':
        return Path(cache.exec(dataset, 'naive', video, name))
    if system == 'tetris':
        # Tracks are unpacked in 060; detections mapped back from canvases are in 050.
        stage = 'ucomp-tracks' if kind == 'tracking' else 'ucomp-dets'
        return Path(cache.exec(dataset, stage, video, config, name))
    # Prior systems store one directory per parameter id.
    assert system in PRIOR_SYSTEMS and kind == 'tracking', f'no {kind} output for system {system!r}'
    param = int(config.removeprefix('param '))
    return Path(cache.sota(system, dataset, video, 'tracking_results', f'{param:03d}', name))
