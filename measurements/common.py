"""Helpers shared by the revision measurement scripts (see REVISION.md)."""
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from evaluation.p200_compare_compute import load_sota_tradeoff_data
from polyis.utilities import load_tradeoff_data, split_tradeoff_variants


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


def fastest_within(con: duckdb.DuckDBPyConnection, table: str, loss: float, systems: list[str]) -> list[dict]:
    """Per dataset and system, the fastest configuration within `loss` of the reference pipeline's accuracy.

    `table` needs columns dataset, system, config, fps, accuracy, with exactly one row per
    dataset whose system is 'reference'. The bound is inclusive; throughput ties go to the
    more accurate configuration, then to the smaller config name. Systems that miss the bound
    are returned with meets_bound false and their best accuracy.
    """
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
    # Select, per dataset and system, the fastest configuration inside the bound.
    return query_dicts(con, f"""
        WITH reference AS (
            SELECT dataset, accuracy AS reference_accuracy, fps AS reference_fps
            FROM {table} WHERE system = 'reference'
        ),
        candidates AS (
            SELECT t.dataset, t.system, t.config, t.fps, t.accuracy,
                   r.reference_accuracy, r.reference_fps, r.reference_accuracy * (1 - $loss) AS bound
            FROM {table} t JOIN reference r USING (dataset)
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
        SELECT b.dataset, b.system, b.bound, b.reference_accuracy, b.reference_fps, b.best_accuracy,
               c.config IS NOT NULL AS meets_bound, c.config, c.fps, c.accuracy
        FROM best b LEFT JOIN chosen c USING (dataset, system)
        ORDER BY b.dataset, b.system
    """, {'loss': loss, 'systems': systems})
