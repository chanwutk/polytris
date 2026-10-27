"""Helpers shared by the revision measurement scripts (see REVISION.md).

Measurement scripts import shared code from here only, never from each other.
"""
import hashlib
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from evaluation.p200_compare_compute import load_sota_tradeoff_data
from polyis.io import cache
from polyis.utilities import dataset_root_name, load_tradeoff_data, split_tradeoff_variants


# Prior systems compared with Tetris, as named in the SOTA cache.
PRIOR_SYSTEMS = ('otif', 'leap')
# Every dataset in the paper's evaluation, with the labels used in the manuscript.
DATASET_LABELS = {'caldot1-y05': 'CalDoT1', 'caldot2-y05': 'CalDoT2', 'ams-y05': 'Amsterdam',
                  'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
# Root of the per-script result directories, next to this package.
RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'measurements-results'
# Schema of a tracking.jsonl line (ground truth, reference, Tetris, and converted SOTA tracks): a frame index and
# boxes [track_id, x1, y1, x2, y2].
TRACK_COLUMNS = "{'frame_idx': 'BIGINT', 'tracks': 'DOUBLE[][]'}"
# Where each stage's runtime-log lines keep their op list, as JSON paths mirroring
# evaluation/p121_throughput_compute.py's INDEX/QUERY_DATA_ACCESSORS.
OP_LIST_PATHS = {
    '011_tune_detect': '$[#-1]',
    '012_tune_create_training_data': '$',
    '013_tune_train_classifier': '$',
    '001_preprocess_groundtruth_detection': '$',
    '002_preprocess_groundtruth_tracking': '$.runtime',
    '020_exec_classify': '$',
    '022_exec_prune_polyominoes': '$.runtime',
    '030_exec_compress': '$.runtime',
    '040_exec_detect': '$',
    '050_exec_uncompress': '$',
    '060_exec_track': '$.runtime',
}
# JSON types DuckDB reports for numeric op times; anything else (a string such as p040's image_file name, a
# boolean, null) is not a duration.
NUMERIC_JSON_TYPES = ('DOUBLE', 'BIGINT', 'UBIGINT')
# The pruning stage (scripts/p022_exec_prune_polyominoes.py): its directory, its name in the throughput tables, and
# its source, whose call to solve_ilp sets the default solve budget.
PRUNE_DIR = cache.EXEC_STAGE_MAP['pruned-polyominoes']
PRUNE_STAGE = '022_exec_prune_polyominoes'
P022_SOURCE = Path(__file__).resolve().parent.parent / 'scripts' / 'p022_exec_prune_polyominoes.py'
# Dataset, video, and pruning parameter string from a p022 runtime.jsonl path.
PRUNE_PATH_PATTERN = rf'([^/]+)/execution/([^/]+)/{PRUNE_DIR}/([^/]+)/score/runtime\.jsonl$'
# p022's parameter string: classifier, tile size, sampling rate, accuracy threshold, relevance threshold, tracker.
PRUNE_PARAM_PATTERN = r'^(.+)_([0-9]+)_([0-9]+)_([0-9]{3})_r([0-9]{3})_([^_]+)$'
# A throughput variant id adds tile padding and canvas scale before the tracker; dropping them gives p022's string.
VARIANT_PATTERN = r'^(.+_[0-9]{3}_r[0-9]{3})_[^_]+_s[0-9]+_([^_]+)$'
# Video-name prefix of each split p022 runs on.
SPLIT_PREFIXES = {'va': 'valid', 'te': 'test'}
# Relevance binarization rules for p022's input (020 scores, 0-255, at relevance threshold r). 'current' is p022
# as committed since 2026-05-22 (score > r * 255). 'floor' (score >= floor(r * 255)) is what the April 2026 outputs
# in the cache match: they keep tiles scoring exactly 63, 127, or 191 at r = 0.25, 0.5, 0.75, which 'current' drops.
RELEVANCE_RULES = {'current': lambda scores, r: scores > r * 255,
                   'floor': lambda scores, r: scores >= np.floor(r * 255)}


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


def rounded(value, digits: int = 4):
    """`value` with every float rounded to `digits` decimals, through nested dicts and lists (for JSON output).

    Integers, booleans, strings, and None pass through; tuples become lists, as JSON writes them anyway.
    """
    # Floats (NumPy float64 included, a float subclass) are rounded; containers are rebuilt; anything else is kept.
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    return value


def plain(value) -> object:
    """A pandas or NumPy scalar as a Python value, with NA as None."""
    # Missing values of any kind become None.
    if value is None or pd.isna(value):
        return None
    # NumPy scalars become Python ints and floats.
    return value.item() if hasattr(value, 'item') else value


def assert_files_exist(paths, what: str):
    """Fail when any of `paths` is missing, saying how many and naming a few, instead of a bare I/O error later."""
    # Every missing path, in input order.
    missing = [str(p) for p in paths if not os.path.exists(p)]
    assert not missing, f'{len(missing)} {what} missing, e.g. {missing[:3]}'


def iso_utc(seconds: float | None) -> str | None:
    """An epoch time as an ISO-8601 UTC string to the second, for provenance; None stays None."""
    # Unknown times stay unknown.
    if seconds is None:
        return None
    # Second precision is enough for provenance and keeps the strings comparable.
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec='seconds')


def file_record(path: str | Path) -> dict:
    """Path, size, modification time, and SHA-256 of one input file, so a result says which version it read.

    A missing file is recorded with exists False rather than failing; callers check required inputs themselves.
    """
    # Missing files are recorded, not fatal.
    p = Path(path)
    if not p.exists():
        return dict(path=str(p), exists=False)
    # Size, modification time, and content hash identify the version read (inputs are small tables and arrays).
    stat = p.stat()
    return dict(path=str(p), exists=True, bytes=stat.st_size, mtime=iso_utc(stat.st_mtime),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def host_info() -> dict:
    """The host running the measurement: name, CPU count, and GPU models with their counts (nvidia-smi, if present)."""
    # GPU names from nvidia-smi, which only queries the driver and runs nothing on the GPUs.
    try:
        gpus = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split('\n')
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus = []
    # One entry per distinct GPU model, with how many the host has.
    names = [g.strip() for g in gpus if g.strip()]
    # Host name, CPU count, and GPU models with counts.
    return dict(hostname=socket.gethostname(), cpus=os.cpu_count(), gpus={g: names.count(g) for g in sorted(set(names))})


def decode_grids(entries: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Frames of a 020 or 022 score.jsonl as uint8 [frames, rows, cols] and their frame indices."""
    # p022 cannot build an instance from a file without frames.
    assert entries, 'score file has no frames'
    # Every frame must share one tile grid.
    sizes = {tuple(e['classification_size']) for e in entries}
    assert len(sizes) == 1, f'inconsistent grid sizes {sorted(sizes)}'
    (rows, cols), = sizes
    # Decode all frames' hex at once; each frame is rows * cols bytes in row-major order.
    grids = np.frombuffer(bytes.fromhex(''.join(e['classification_hex'] for e in entries)), dtype=np.uint8)
    # A truncated or padded hex string does not fill the declared grids exactly.
    assert grids.size == len(entries) * rows * cols, 'hex length does not match the grid size'
    return grids.reshape(len(entries), rows, cols), [e['idx'] for e in entries]


def runtime_items_sql(op_path: str, numbered: bool = True) -> str:
    """SQL for one row per op item of the runtime logs bound to parameter $paths (a list of JSONL files).

    Each non-blank line is parsed and its op list found at JSON path `op_path` (OP_LIST_PATHS has each stage's).
    Columns: filename, last_modified, line (1-based, in file order), text (the line), list_type (JSON type of the
    op list: ARRAY, OBJECT, NULL for a JSON null, SQL NULL when the path is absent), items (ops in the list; an
    object counts as one, anything else as none), item (the op as JSON), op, time_type (JSON type of its time),
    and time (milliseconds; NULL unless the JSON type is numeric). A line whose op list is empty, null, absent, or
    not a list or object yields one row with item, op, and time NULL, so it does not vanish from checks.
    Malformed JSON raises duckdb.InvalidInputException. Every caller applies its own checks of lines, ops, and times.

    numbered=False reads the lines with DuckDB's parallel read_json_objects instead of read_text, for sums over
    gigabytes of logs where line order does not matter; line and last_modified are then NULL.
    """
    # The path is embedded as a SQL literal, so it must not close the literal.
    assert "'" not in op_path, f'bad JSON path {op_path!r}'
    # Non-blank lines in file order with each file's modification time, or, unnumbered, from the parallel JSON reader.
    lines = ("""
        SELECT filename, last_modified, line, text FROM (
            SELECT filename, last_modified, unnest(lines) AS text, generate_subscripts(lines, 1) AS line
            FROM (SELECT filename, last_modified, string_split(trim(content, chr(10)), chr(10)) AS lines
                  FROM read_text($paths))
        ) WHERE trim(text) <> ''""" if numbered else """
        SELECT filename, NULL::TIMESTAMPTZ AS last_modified, NULL::BIGINT AS line, json AS text
        FROM read_json_objects($paths, format = 'newline_delimited', filename = true)""")
    return f"""
        WITH runtime_lists AS (
            SELECT filename, last_modified, line, text, json_extract(text, '{op_path}') AS ops FROM ({lines})
        ),
        runtime_items AS (
            SELECT filename, last_modified, line, text, json_type(ops) AS list_type,
                   CASE json_type(ops) WHEN 'ARRAY' THEN json_array_length(ops) WHEN 'OBJECT' THEN 1 ELSE 0 END AS items,
                   unnest(CASE WHEN json_type(ops) = 'ARRAY' AND json_array_length(ops) > 0 THEN from_json(ops, '["json"]')
                               WHEN json_type(ops) = 'OBJECT' THEN [ops] ELSE [NULL::JSON] END) AS item
            FROM runtime_lists
        )
        SELECT filename, last_modified, line, text, list_type, items, item, item->>'op' AS op,
               json_type(item->'time') AS time_type,
               CASE WHEN list_contains({list(NUMERIC_JSON_TYPES)}, json_type(item->'time'))
                    THEN CAST(item->>'time' AS DOUBLE) END AS time
        FROM runtime_items
    """


def default_time_limit(source: Path = P022_SOURCE) -> float:
    """The solve budget in seconds that p022 applies when --time-limit is not given, read from its source."""
    # p022 passes `time_limit_seconds=time_limit or <default>` to solve_ilp.
    matches = re.findall(r'time_limit_seconds\s*=\s*time_limit\s+or\s+([0-9]*\.?[0-9]+)', source.read_text())
    # Exactly one call site, or the default is ambiguous.
    assert len(matches) == 1, f'{source}: expected one `time_limit or <default>` call, found {len(matches)}'
    return float(matches[0])


def load_prune_runs(con: duckdb.DuckDBPyConnection, files: list[str], default_limit: float):
    """Table runs: one row per p022 run (one clip under one pruning configuration) from its score/runtime.jsonl.

    Columns: dataset, video, param (from the path), split (from the video name), sample_rate, tilesize, time_limit
    (as logged), budget_s (p022's `time_limit or default_limit`), build_ms, solve_ms, build_plus_solve_ms, stage_ms
    (every logged op), last_modified, filename. Strict: each file must hold exactly one record with a time_limit
    (number or null) and its ops under 'runtime', exactly one build_ilp and one solve_ilp, and no negative or
    non-numeric time; every path must name a dataset, video, and parseable parameter string of a known split.
    """
    # Reading zero files would fail inside DuckDB with a less clear message.
    assert files, 'no runtime files to read'
    # Every op of every file, with the record's time_limit (SQL NULL when the key is absent, JSON null when null).
    con.execute(f"""
        CREATE OR REPLACE TABLE run_ops AS
        SELECT filename, last_modified, line, list_type, op, time, json_extract(text, '$.time_limit') AS time_limit
        FROM ({runtime_items_sql(OP_LIST_PATHS[PRUNE_STAGE])})
    """, {'paths': files})
    # p022 writes one record per file, so a file with no record or several is not one run.
    bad = query_dicts(con, """
        SELECT f.filename, count(DISTINCT o.line) AS records
        FROM (SELECT unnest($files) AS filename) f LEFT JOIN run_ops o USING (filename)
        GROUP BY f.filename HAVING records <> 1 LIMIT 5
    """, {'files': files})
    assert not bad, f'runtime files without exactly one record: {bad}'
    # The budget is unknown without a numeric or null time_limit.
    bad = query_dicts(con, f"""
        SELECT DISTINCT filename FROM run_ops
        WHERE time_limit IS NULL OR NOT list_contains({['NULL', *NUMERIC_JSON_TYPES]}, json_type(time_limit)) LIMIT 5
    """)
    assert not bad, f'runtime files without a numeric or null time_limit: {bad}'
    # Every path must name a dataset, video, and parameter string.
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_runs AS
        SELECT filename, any_value(last_modified) AS last_modified, CAST(any_value(time_limit) AS DOUBLE) AS time_limit,
               regexp_extract(filename, '{PRUNE_PATH_PATTERN}', ['dataset', 'video', 'param']) AS loc
        FROM run_ops GROUP BY filename
    """)
    bad = query_dicts(con, "SELECT filename FROM raw_runs WHERE loc.param = '' LIMIT 5")
    assert not bad, f'unrecognized runtime paths: {bad}'
    # Each run must log build_ilp and solve_ilp exactly once, under an op list, and no negative or non-numeric time.
    bad = query_dicts(con, """
        SELECT filename, count(*) FILTER (op = 'build_ilp') AS builds, count(*) FILTER (op = 'solve_ilp') AS solves,
               count(*) FILTER (time < 0 OR (op IS NOT NULL AND time IS NULL)
                                OR list_type IS DISTINCT FROM 'ARRAY') AS invalid
        FROM run_ops GROUP BY filename HAVING builds <> 1 OR solves <> 1 OR invalid > 0 LIMIT 5
    """)
    assert not bad, f'runs without exactly one valid build_ilp and solve_ilp: {bad}'
    # Parse the parameter string and split, apply p022's `time_limit or default`, and sum the stage.
    con.execute(f"""
        CREATE OR REPLACE TABLE runs AS
        WITH params AS (
            SELECT filename, last_modified, loc.dataset, loc.video, loc.param, time_limit,
                   regexp_extract(loc.param, '{PRUNE_PARAM_PATTERN}',
                                  ['classifier', 'tilesize', 'sample_rate', 'threshold', 'relevance', 'tracker']) AS p
            FROM raw_runs
        ),
        times AS (
            SELECT filename, sum(time) FILTER (op = 'build_ilp') AS build_ms, sum(time) FILTER (op = 'solve_ilp') AS solve_ms,
                   sum(time) AS stage_ms
            FROM run_ops GROUP BY filename
        )
        SELECT dataset, video, param,
               CASE {' '.join(f"WHEN video LIKE '{k}%' THEN '{v}'" for k, v in SPLIT_PREFIXES.items())} END AS split,
               TRY_CAST(p.sample_rate AS INTEGER) AS sample_rate, TRY_CAST(p.tilesize AS INTEGER) AS tilesize,
               time_limit, coalesce(nullif(time_limit, 0), $default_limit) AS budget_s,
               build_ms, solve_ms, build_ms + solve_ms AS build_plus_solve_ms, stage_ms, last_modified, filename
        FROM params JOIN times USING (filename)
    """, {'default_limit': default_limit})
    # Every run must belong to a known split and carry a parseable sampling rate.
    bad = query_dicts(con, 'SELECT filename FROM runs WHERE split IS NULL OR sample_rate IS NULL LIMIT 5')
    assert not bad, f'runs with an unknown split or unparseable parameter string: {bad}'


def load_tradeoff_points(con: duckdb.DuckDBPyConnection, table: str, tradeoffs: dict[str, tuple[pd.DataFrame, pd.Series]],
                         columns: dict[str, str] | None = None):
    """Table <table>(dataset, system, config, fps, accuracy, *columns) from {dataset: load_test_tradeoff(dataset)}.

    Per dataset, the reference pipeline's test row (system 'reference', config 'naive') and every validated Tetris
    configuration (system 'tetris', config its variant id); fps is throughput and accuracy test HOTA, the shape
    fastest_within reads. `columns` maps further tradeoff columns to their DuckDB types; the reference row takes
    them from its own row (NULL where it has none or NaN), Tetris rows from theirs (pandas NaN arrives as NULL).
    """
    # An empty mapping has no dataset to report.
    assert tradeoffs, 'no tradeoff tables'
    # The table with the fixed columns and the requested ones.
    columns = columns or {}
    extra = ''.join(f', "{c}" {t}' for c, t in columns.items())
    con.execute(f'CREATE OR REPLACE TABLE {table} (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, '
                f'accuracy DOUBLE{extra})')
    # One reference row and one insert of the Tetris rows per dataset (a handful of tables, not data rows).
    for dataset, (tetris, naive) in tradeoffs.items():
        # The reference row, its extra values as plain Python values (NaN as NULL).
        con.execute(f"INSERT INTO {table} VALUES (?, 'reference', 'naive', ?, ?{', ?' * len(columns)})",
                    [dataset, float(naive['throughput_fps']), float(naive['HOTA_HOTA']),
                     *[plain(naive.get(c)) for c in columns]])
        # Every Tetris configuration, read straight from the pandas frame and cast to the table's types.
        con.register('tradeoff_frame', tetris)
        con.execute(f"""
            INSERT INTO {table} SELECT $dataset, 'tetris', variant_id, throughput_fps, HOTA_HOTA
                   {''.join(f', "{c}"' for c in columns)}
            FROM tradeoff_frame
        """, {'dataset': dataset})
        # Release the view so the next dataset can register its own frame.
        con.unregister('tradeoff_frame')


def load_pruning_points(con: duckdb.DuckDBPyConnection, tradeoffs: dict[str, tuple[pd.DataFrame, pd.Series]],
                        table: str = 'points'):
    """Table <table>(dataset, system, config, fps, accuracy, time_s, prunes) from {dataset: load_test_tradeoff(dataset)}.

    load_tradeoff_points with each row's runtime (time_s) and whether it prunes (a Tetris configuration with a
    tracking-accuracy threshold; the reference never does), plus the pruning configurations again as system
    'tetris_pruned', so fastest_within can select among all and among pruning configurations at once.
    """
    # Reference and Tetris rows with their runtime and pruning threshold.
    load_tradeoff_points(con, table, tradeoffs, {'time': 'DOUBLE', 'tracking_accuracy_threshold': 'DOUBLE'})
    # The pruning flag, and the pruning configurations again under their own system name.
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH p AS (
            SELECT dataset, system, config, fps, accuracy, "time" AS time_s,
                   system = 'tetris' AND tracking_accuracy_threshold IS NOT NULL AS prunes
            FROM {table}
        )
        SELECT * FROM p
        UNION ALL SELECT * REPLACE ('tetris_pruned' AS system) FROM p WHERE system = 'tetris' AND prunes
    """)


# The two selections the R9.O3 reports make over load_pruning_points' systems: the fastest configuration within the
# bound, and the fastest one that prunes.
PRUNING_POINTS = {'tetris': 'operating_point', 'tetris_pruned': 'fastest_pruning_point'}


def pruning_operating_points(con: duckdb.DuckDBPyConnection, loss: float, table: str = 'points') -> list[dict]:
    """fastest_within over load_pruning_points' `table` among all and among pruning configurations, per dataset.

    Each row also carries `point`, its name in PRUNING_POINTS; table chosen_points(dataset, point, config) holds the
    selections that meet the bound.
    """
    # Fastest configuration within the bound, among all and among pruning configurations, named by its role.
    chosen = [dict(r, point=PRUNING_POINTS[r['system']]) for r in fastest_within(con, table, loss, list(PRUNING_POINTS))]
    # The selections that meet the bound, as a table to join with per-configuration data.
    con.execute('CREATE OR REPLACE TABLE chosen_points (dataset VARCHAR, point VARCHAR, config VARCHAR)')
    selected = [(r['dataset'], r['point'], r['config']) for r in chosen if r['meets_bound']]
    # Insert them only when there are any (executemany needs at least one row).
    if selected:
        con.executemany('INSERT INTO chosen_points VALUES (?, ?, ?)', selected)
    return chosen


def fastest_tetris(con: duckdb.DuckDBPyConnection, table: str, loss: float) -> dict[str, dict]:
    """Per dataset of `table` (load_tradeoff_points' shape), the fastest Tetris configuration whose HOTA is within
    `loss` of the reference pipeline's, as a fastest_within row; fails when some dataset has none."""
    # The shared selection rule (inclusive bound, ties to the more accurate configuration, then the smaller name).
    met = {r['dataset']: r for r in fastest_within(con, table, loss, ['tetris']) if r['meets_bound']}
    # Every dataset of the table needs an operating point, including one without any Tetris row.
    missed = sorted({r['dataset'] for r in query_dicts(con, f'SELECT DISTINCT dataset FROM {table}')} - set(met))
    assert not missed, f'no Tetris configuration within {loss:.0%} HOTA loss on {missed}'
    return met


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
    # Keep only the test split.
    rows = rows[rows['videoset'] == 'test']
    # A missing table or one without test rows is an error: the comparison would silently drop a system.
    assert not rows.empty, f'{dataset}: no {system} test-split tradeoff rows'
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
    # Unrounded throughput of every chosen configuration, so ratios are rounded only once.
    fps = {(r['dataset'], r['system']): r['fps'] for r in chosen if r['meets_bound']}
    # The primary system against each other system that also meets the bound.
    for dataset, entry in out.items():
        entry.update({f'{primary}_speedup_vs_{s}': round(fps[(dataset, primary)] / fps[(dataset, s)], 1)
                      for s in entry['systems']
                      if s != primary and (dataset, primary) in fps and (dataset, s) in fps})
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

    system is 'ground truth' (003_groundtruth), 'reference' (002_naive), 'tetris' (060 tracks only),
    or a prior system ('otif', 'leap'; config 'param N'). Prior systems only have tracking output.
    Ground truth is read from the dataset's root (caldot1-y05 -> caldot1, ams-y05 -> ams), the
    tracks p110 and p141 score every system against; the detector variants' own 003_groundtruth
    copies are stale.
    """
    # The file name inside the stage directory.
    name = f'{kind}.jsonl'
    if system == 'ground truth':
        return Path(cache.exec(dataset_root_name(dataset), 'groundtruth', video, name))
    if system == 'reference':
        return Path(cache.exec(dataset, 'naive', video, name))
    if system == 'tetris':
        # 050 directories drop the tracker for unpruned configurations and hold detections.jsonl,
        # so they are not addressable by the variant id; use detection_config in
        # measurements/r1o4_packing_detection_accuracy.py for unpacked detections.
        assert kind == 'tracking', 'Tetris detections are named by detection configuration, not variant id'
        return Path(cache.exec(dataset, 'ucomp-tracks', video, config, name))
    # Prior systems store one directory per parameter id.
    assert system in PRIOR_SYSTEMS and kind == 'tracking', f'no {kind} output for system {system!r}'
    param = int(config.removeprefix('param '))
    return Path(cache.sota(system, dataset, video, 'tracking_results', f'{param:03d}', name))
