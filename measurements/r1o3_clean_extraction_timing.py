#!/usr/local/bin/python
"""R1.O3: extraction throughput of the reference pipeline and Tetris from clean reruns, at seven timing boundaries.

Aggregates the raw outputs of measurements-gpu/r1o3_clean_extraction_timing.py (measurements-results/
r1o3_clean_extraction_timing/raw/: plan.json and one directory per unit attempt; see that script for what ran, how,
and where), and fails with a clear error when they are missing. Runs on ace, whose cache holds the paper's timing
logs and p121's per-op tables.

Units and phases: per dataset, a reference unit (p002 detection and tracking) and one unit per Tetris operating point
(the fastest configuration within 5% and 10% HOTA loss), on all test videos, once while the job's three GPUs ran
datasets at the same time on one NUMA node ('parallel') and once one unit at a time on an idle node ('serial', the
headline numbers); and the reference and the 5% point once more on a few calibration videos, one unit at a time on an
idle node ('solo').

Attempt choice (ATTEMPT_RULE): per unit, among the attempts whose job finished, the one the launcher did not flag as
contaminated (another process on its GPU), then the one other processes left its node idle for (foreign_on_node mean
over its timed steps at most FOREIGN_CPU_MAX), then the lowest foreign share, then the latest. foreign_on_node is the
node's busy share minus, per second, the own share (cgroup accounting) of every attempt of this job pinned to the same
CPUs, so it is other tracks' and users' use. conditions.json lists every attempt with its share.

Boundaries, per system, in seconds over a set of test videos (throughput = frames / seconds; speedup = Tetris
throughput / reference throughput on the same videos):
  p121                       evaluation/p121_throughput_compute.py's sum over the stages' runtime logs: every op except
                             p121's ignored ops (common.P121_IGNORED_OPS) and its excluded ops (p020 read and retrieve,
                             p030 save_collage and pack_all_total), parsed by common.load_p121_runtime_ops. It counts the reference's per-frame decode (p002 'read')
                             and Tetris's canvas reads (p040 'read'), but none of Tetris's decodes (p020's 'read' is
                             excluded, p030's decode is untimed).
  decode_excluded            p121 minus the counted video-decode ops (DECODE_OPS); Tetris has none counted.
  one_decode                 decode_excluded plus one decode of every video at its measured cost, on both sides: for
                             the clean runs the GPU job's separate cv2 decode of the same videos (open plus every
                             cap.read(), in the reference unit of the same phase), the median of its passes (each pass
                             decodes every video once; the fastest and slowest pass give speedup_range); for the paper
                             the reference's own logged 'read' (the decode the paper's run measured).
  one_decode_no_canvas_read  one_decode minus Tetris's canvas reads (CANVAS_READ_OPS), with the same range.
  one_decode_logged          decode_excluded plus one decode of every video at the reference's own logged 'read' of the
                             same side and phase (p002's cap.read() interleaved with detection), on both sides; equal
                             to one_decode on the paper side, so it compares the clean and paper runs like for like.
  wall_clock                 clean runs only: the stage processes' wall clocks, summed. The staged prototype ran one
                             worker process at a time on one GPU (a Tetris stage is one process over all videos; p002
                             starts one worker process per video), so this includes each process's startup
                             (interpreter, imports, CUDA, model loading), file I/O between stages, Tetris's untimed
                             decodes (p020 reads every frame, p030 decodes every video again), and p020's channel copy.
                             Startup is paid per process, not per frame, so this boundary depends on how many videos
                             one process handles (5 on B3D, 60 elsewhere); conditions.json gives each stage's time to
                             its first video's last runtime write. A unit whose p022 could not run (expired Gurobi
                             license) uses p022's logged p121 seconds in place of its process and is flagged
                             (p022_wall_from_cache).
  wall_clock_steady          wall_clock with each Tetris stage's process startup taken out: per stage, the time from
                             its first video's last runtime write to its last video's, times n / (n - 1) for n videos
                             (the first video counted as an average one; every stage writes a video's runtime file when
                             it finishes that video, in sorted order). p002 starts a worker process per video, so the
                             reference keeps its per-video startup, as the prototype pays it. Units recorded before the
                             serial phase have no write times, so only the serial phase has this boundary. It is the
                             counterpart of REVISION.md finding 3's wall-clock row, which is r9o2_preparation_cost's
                             process boundary (per-video file-timestamp spans on ace, which leave out process startup).
The paper's side is the same accounting over ace's cached runtime logs of the same configurations and videos
('paper'); a port check requires those logs' p121 sums to equal p121's per-op table (query_execution_per_op.csv) per
op, video, stage, and configuration, and the resulting throughput to equal the tradeoff table's.

Reports (measurements-results/r1o3_clean_extraction_timing/):
  throughput.json    the serial phase over all test videos at every boundary, next to the paper's logs.
  per_op.json        every op of every stage, serial against the paper's logs: seconds, items, and their ratio, with
                     whether p121 counts the op (p020's uncounted 'read' is Tetris's decode and copy). The paper's runs
                     on ace ran up to 8 (p002, p020, p040, p060), 20 (p022, p030), or 40 (p050) tasks of a stage at once
                     with every Pareto configuration, while the serial runs ran one task at a time, so the ratios
                     reflect that concurrency as well as other users' load on ace.
  interference.json  COMPARISONS: parallel over serial on all test videos (the cost of running three GPUs' datasets
                     at once on one node) and solo over serial on the calibration videos (two clean runs of the same
                     videos: repeatability), as ratios of each boundary's seconds and speedups, each op's seconds, and
                     the decode; plus the spread of the decode passes per phase and dataset.
  conditions.json    per unit: attempts and the chosen one, GPU, CPU set, contamination, the unit GPU's utilization per
                     stage, the other GPUs' peaks, the load average, the node's busy share split into this unit's, the
                     rest (others_on_node), and other tracks' and users' (foreign_on_node), and each stage's wall clock,
                     logged op seconds, and time to its first output; plus checks: the port check, frames per video,
                     and output digests against ace's cached outputs (byte-identical; a different row order or float
                     last digit counts as different, which is what the 040-060 differences checked by hand were, so
                     the check that both runs did the same work is per_op.json's item counts, equal for every op).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import duckdb

from measurements.common import (DATASET_LABELS, assert_files_exist, load_p121_per_op, load_p121_runtime_ops,
                                 query_dicts, results_dir, rounded, write_json)
from polyis.io import cache


# The GPU job, loaded from its file (measurements-gpu is not a package; its top level needs only the standard
# library). Its constants define the raw layout, the stages, the phases, and the idle-node threshold.
GPU_PATH = Path(__file__).resolve().parent.parent / 'measurements-gpu' / 'r1o3_clean_extraction_timing.py'
_spec = importlib.util.spec_from_file_location('r1o3_clean_extraction_timing_gpu', GPU_PATH)
gpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu)
# Counted ops that decode the input video, per stage (p121 already drops p020's 'read'; p030's decode is untimed).
DECODE_OPS = {'001_preprocess_groundtruth_detection': ('read',)}
# Counted ops that read Tetris's packed canvases back from disk.
CANVAS_READ_OPS = {'040_exec_detect': ('read',)}
# Boundaries in report order, and those that add the measured decode (reported with the passes' range).
BOUNDARIES = ('p121', 'decode_excluded', 'one_decode', 'one_decode_no_canvas_read', 'one_decode_logged', 'wall_clock',
              'wall_clock_steady')
DECODE_BAND_BOUNDARIES = ('one_decode', 'one_decode_no_canvas_read')
# The aggregated phases, and the one whose units give the headline numbers (and carry the paper's logs).
PHASES = ('parallel', 'serial', 'solo')
HEADLINE = 'serial'
# Phase comparisons (a over b) and the videos they cover: 'unit' all of a unit's videos, 'calibration' the solo
# phase's videos.
COMPARISONS = (('parallel', 'serial', 'unit'), ('solo', 'serial', 'calibration'))
# The rule choose_attempts applies, as reported.
ATTEMPT_RULE = ('per unit, among finished attempts: not contaminated first, then foreign_on_node mean over the timed '
                f'steps at most {gpu.FOREIGN_CPU_MAX}, then the lowest foreign_on_node mean, then the latest attempt')
# The reference's name in p121's tables.
NAIVE = 'naive'
# Relative and absolute tolerance of the port check (p121 sums in file order, DuckDB in parallel).
PORT_RTOL, PORT_ATOL = 1e-9, 1e-6
# Columns of the launcher's per-second samples, of the job's CPU samples, and of the measured decode (attempts before
# the serial phase decoded once and wrote no pass or thread count).
SAMPLE_COLUMNS = "{'t': 'TIMESTAMPTZ', 'util': 'DOUBLE', 'mem_mib': 'DOUBLE', 'foreign': 'BIGINT', 'load_1m': 'DOUBLE'}"
CPU_COLUMNS = ("{'t_ns': 'BIGINT', 'step': 'VARCHAR', 'unit_cpus_busy': 'DOUBLE', 'other_cpus_busy': 'DOUBLE', "
               "'own_busy': 'DOUBLE'}")
DECODE_COLUMNS = ("{'video': 'VARCHAR', 'decode_pass': 'BIGINT', 'frames': 'BIGINT', 'read_s': 'DOUBLE', "
                  "'open_s': 'DOUBLE', 'total_s': 'DOUBLE', 'threads': 'DOUBLE'}")


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O3: clean extraction throughput at seven timing boundaries')
    parser.add_argument('--datasets', nargs='+', default=None)
    return parser.parse_args()


def load_plan(raw: Path, datasets: list[str] | None) -> tuple[dict, list[dict]]:
    """The GPU job's plan restricted to `datasets` (all planned datasets when None), and its units of every
    aggregated phase (the serial units are derived from the parallel ones, as the GPU job derives them)."""
    # Without the plan nothing ran; say which job writes it.
    path = raw / gpu.PLAN_FILE
    assert path.is_file(), f'{path} missing: run {gpu.SCRIPT} plan, stage, and run first (see its docstring)'
    plan = json.loads(path.read_text())
    datasets = datasets or list(plan['datasets'])
    unknown = sorted(set(datasets) - set(plan['datasets']))
    assert not unknown, f'datasets not in the plan: {unknown}'
    # Keep the requested datasets.
    plan['datasets'] = {d: plan['datasets'][d] for d in datasets}
    units = [u for phase in PHASES for u in gpu.phase_units(plan, phase, datasets)]
    return plan, units


def load_attempts(con: duckdb.DuckDBPyConnection, raw: Path, units: list[dict]):
    """Tables attempts (every attempt of every unit: status, contamination, CPU set), cpu_all (the CPU samples of
    every attempt, with foreign_busy: the node's busy share minus, per second, every attempt's own share on the same
    CPUs), and attempt_cpu (per attempt, the shares over its timed steps)."""
    con.execute('CREATE OR REPLACE TABLE attempts (unit VARCHAR, attempt VARCHAR, attempt_no BIGINT, path VARCHAR, '
                'status VARCHAR, launched BOOLEAN, contaminated BOOLEAN, cpus VARCHAR)')
    con.execute('CREATE OR REPLACE TABLE cpu_all (path VARCHAR, cpus VARCHAR, t_ns BIGINT, step VARCHAR, '
                'unit_cpus_busy DOUBLE, other_cpus_busy DOUBLE, own_busy DOUBLE)')
    # A few dozen units with one to three attempts each; their records are small JSON files.
    for u in units:
        for path in gpu.attempts(u['unit'], raw):
            manifest = json.loads((path / gpu.MANIFEST).read_text()) if (path / gpu.MANIFEST).exists() else None
            summary = json.loads((path / gpu.LAUNCHER_SUMMARY).read_text()) if (path / gpu.LAUNCHER_SUMMARY).exists() else None
            con.execute('INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        [u['unit'], path.name, int(path.name[len(gpu.ATTEMPT_PREFIX):]), str(path),
                         manifest['status'] if manifest else None, summary is not None,
                         summary['contaminated'] if summary else None, summary['cpus'] if summary else None])
            # Every attempt that ran in a container on known CPUs counts as a sibling, finished or not.
            if summary is not None and (path / gpu.CPU_SAMPLES).exists():
                con.execute(f"""
                    INSERT INTO cpu_all SELECT $path, $cpus, t_ns, step, unit_cpus_busy, other_cpus_busy, own_busy
                    FROM read_json($file, format = 'newline_delimited', columns = {CPU_COLUMNS})
                """, dict(path=str(path), cpus=summary['cpus'], file=str(path / gpu.CPU_SAMPLES)))
    # Foreign use: the node's busy share minus the share every attempt pinned to the same CPUs used in that second
    # (each attempt's mean own share per second, summed). An attempt with no sample in a second counts as idle then,
    # so this slightly overestimates foreign use while siblings run.
    con.execute("""
        CREATE OR REPLACE TABLE cpu_all AS
        WITH per_second AS (
            SELECT path, cpus, t_ns // 1000000000 AS sec, avg(own_busy) AS own FROM cpu_all GROUP BY ALL
        ),
        job AS (SELECT cpus, sec, fsum(own) AS job_busy FROM per_second GROUP BY ALL)
        SELECT c.*, c.unit_cpus_busy - j.job_busy AS foreign_busy
        FROM cpu_all c LEFT JOIN job j ON j.cpus = c.cpus AND j.sec = c.t_ns // 1000000000
    """)
    con.execute("""
        CREATE OR REPLACE TABLE attempt_cpu AS
        SELECT path, count(*) AS samples, avg(unit_cpus_busy - own_busy) AS others_on_node_mean,
               avg(foreign_busy) AS foreign_on_node_mean, max(foreign_busy) AS foreign_on_node_max
        FROM cpu_all WHERE NOT list_contains($untimed, step) GROUP BY path
    """, dict(untimed=list(gpu.UNTIMED_STEPS)))


def choose_attempts(con: duckdb.DuckDBPyConnection, units: list[dict]) -> list[dict]:
    """Every unit with the attempt ATTEMPT_RULE picks, its manifest and launcher summary, and its CPU flags."""
    rows = {r['unit']: r for r in query_dicts(con, """
        SELECT a.unit, a.attempt, a.path, a.contaminated, c.foreign_on_node_mean,
               coalesce(c.foreign_on_node_mean <= $limit, false) AS cpu_clean
        FROM attempts a LEFT JOIN attempt_cpu c USING (path)
        WHERE a.status = 'ok' AND a.launched
        QUALIFY row_number() OVER (PARTITION BY a.unit ORDER BY a.contaminated, cpu_clean DESC,
                                   c.foreign_on_node_mean NULLS LAST, a.attempt_no DESC) = 1
    """, dict(limit=gpu.FOREIGN_CPU_MAX))}
    chosen = []
    # A few dozen units.
    for unit in units:
        assert unit['unit'] in rows, f'unit {unit["unit"]} has no successful attempt: run {gpu.SCRIPT} run (see its log)'
        best = rows[unit['unit']]
        path = Path(best['path'])
        manifest = json.loads((path / gpu.MANIFEST).read_text())
        # The manifest must describe the planned unit (same videos and configuration).
        m = manifest['unit']
        assert (m['videos'], m['variant_id'], m['system']) == (unit['videos'], unit['variant_id'], unit['system']), \
            f'{path} ran a different unit than planned'
        chosen.append(dict(unit, path=path, manifest=manifest,
                           launcher=json.loads((path / gpu.LAUNCHER_SUMMARY).read_text()), attempt=best['attempt'],
                           contaminated=bool(best['contaminated']), cpu_clean=best['cpu_clean'],
                           foreign_on_node_mean=best['foreign_on_node_mean']))
    # The chosen attempts, for the per-unit queries.
    con.execute('CREATE OR REPLACE TABLE chosen (unit VARCHAR, path VARCHAR, phase VARCHAR, dataset VARCHAR, '
                'system VARCHAR, variant_id VARCHAR, cpu_clean BOOLEAN)')
    con.executemany('INSERT INTO chosen VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(u['unit'], str(u['path']), u['phase'], u['dataset'], u['system'], u['variant_id'] or NAIVE,
                      u['cpu_clean']) for u in chosen])
    return chosen


def runtime_files(units: list[dict], cache_root: Path) -> list[dict]:
    """The runtime logs to read: each unit's harvested logs ('clean') and, for headline units, ace's cached logs at the
    same cache-relative paths ('paper'). Each row: side, unit, phase, dataset, system, variant_id, video, stage, path."""
    rows = []
    # One harvested log per unit, stage, and video, named by the manifest.
    for u in units:
        variant = u['variant_id'] if u['system'] == 'tetris' else NAIVE
        for stage, by_video in u['manifest']['outputs'].items():
            for video, out in by_video.items():
                base = dict(unit=u['unit'], phase=u['phase'], dataset=u['dataset'], system=u['system'],
                            variant_id=variant, video=video, stage=stage)
                rows.append(dict(base, side='clean', path=str(u['path'] / gpu.LOGS_DIR / f'{out["runtime"]}.gz')))
                if u['phase'] == HEADLINE:
                    rows.append(dict(base, side='paper', path=str(cache_root / out['runtime'])))
    assert_files_exist([r['path'] for r in rows], 'runtime logs (clean: rerun the unit; paper: ace cache)')
    return rows


def load_runtime_ops(con: duckdb.DuckDBPyConnection, files: list[dict]):
    """Table ops: per runtime file and op, seconds and items, with whether p121 counts the op, and whether it decodes the
    video or reads canvases, attributed to each row of `files` (side, unit, phase, dataset, system, variant_id, video,
    stage). Parsed and checked as p121 parses runtime logs (common.load_p121_runtime_ops, which raises on a line without
    an op list, a timed item without an op, a boolean time, or a negative time of an op p121 does not ignore); ops
    without numeric times (p040's image_file) are dropped."""
    # The files, keyed by path; one path can serve several rows (two configurations sharing p020's output).
    con.execute('CREATE OR REPLACE TABLE runtime_files (side VARCHAR, unit VARCHAR, phase VARCHAR, dataset VARCHAR, '
                'system VARCHAR, variant_id VARCHAR, video VARCHAR, stage VARCHAR, path VARCHAR)')
    con.executemany('INSERT INTO runtime_files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [(f['side'], f['unit'], f['phase'], f['dataset'], f['system'], f['variant_id'], f['video'],
                      f['stage'], f['path']) for f in files])
    # Every file's ops once, with the decode and canvas-read flags.
    load_p121_runtime_ops(con, 'runtime_ops', [(f['stage'], f['path']) for f in files],
                          {'decode': DECODE_OPS, 'canvas_read': CANVAS_READ_OPS})
    # Each file's ops for every row that names the file.
    con.execute("""
        CREATE OR REPLACE TABLE ops AS
        SELECT f.side, f.unit, f.phase, f.dataset, f.system, f.variant_id, f.video, f.stage, r.op, r.seconds,
               r.records AS items, r.counted, r.decode, r.canvas_read
        FROM runtime_ops r JOIN runtime_files f ON f.path = r.runtime_file AND f.stage = r.stage
    """)


def load_paper_table(con: duckdb.DuckDBPyConnection, plan: dict, cache_root: Path):
    """Table paper_table(dataset, variant_id, video, stage, op, seconds, items): p121's per-op rows of the planned
    configurations and the reference, test split (ace's query_execution_per_op.csv, common.load_p121_per_op)."""
    paths = {d: cache_root / d / 'evaluation' / cache.EVAL_STAGE_MAP['tp'] / 'measurements' / 'query_execution_per_op.csv'
             for d in plan['datasets']}
    load_p121_per_op(con, 'p121_per_op', paths)
    # The planned variants per dataset.
    con.execute('CREATE OR REPLACE TABLE planned (dataset VARCHAR, variant_id VARCHAR)')
    con.executemany('INSERT INTO planned VALUES (?, ?)',
                    [(d, v) for d, info in plan['datasets'].items() for v in [NAIVE, *info['configs']]])
    # The planned rows of the test split, summed exactly per configuration, video, stage, and op.
    con.execute("""
        CREATE OR REPLACE TABLE paper_table AS
        SELECT dataset, variant_id, video, stage, op, fsum(seconds) AS seconds, sum(records) AS items
        FROM p121_per_op SEMI JOIN planned USING (dataset, variant_id)
        WHERE videoset = 'test'
        GROUP BY ALL
    """)


def port_check(con: duckdb.DuckDBPyConnection) -> dict:
    """Fail unless the paper side's counted ops equal p121's per-op table exactly (within float summation order)."""
    # Every (configuration, video, stage, op) on either side, compared.
    rows = query_dicts(con, f"""
        WITH logs AS (
            SELECT dataset, variant_id, video, stage, op, seconds, items FROM ops WHERE side = 'paper' AND counted
        )
        SELECT dataset, variant_id, video, stage, op, l.seconds AS log_s, t.seconds AS table_s, l.items AS log_items,
               t.items AS table_items
        FROM logs l FULL JOIN paper_table t USING (dataset, variant_id, video, stage, op)
        WHERE l.seconds IS NULL OR t.seconds IS NULL OR l.items <> t.items
           OR abs(l.seconds - t.seconds) > {PORT_ATOL} + {PORT_RTOL} * abs(t.seconds)
        LIMIT 5
    """)
    assert not rows, f'the cached logs do not reproduce p121\'s per-op table: {rows}'
    n = con.execute("SELECT count(*) FROM paper_table").fetchone()[0]
    # Tolerances as text, so the output's rounding to 4 decimals does not print them as 0.
    return dict(rows_compared=n, mismatches=0, rtol=f'{PORT_RTOL:g}', atol_s=f'{PORT_ATOL:g}')


def load_frames_and_decode(con: duckdb.DuckDBPyConnection, plan: dict, units: list[dict]):
    """Tables frames(dataset, video, frames) from the plan and decode(phase, dataset, video, decode_pass, frames,
    read_s, open_s, total_s, threads) from the reference units' measured decode, checked against the plan."""
    con.execute('CREATE OR REPLACE TABLE frames (dataset VARCHAR, video VARCHAR, frames BIGINT)')
    con.executemany('INSERT INTO frames VALUES (?, ?, ?)',
                    [(d, v, n) for d, info in plan['datasets'].items() for v, n in info['videos'].items()])
    con.execute('CREATE OR REPLACE TABLE decode (phase VARCHAR, dataset VARCHAR, video VARCHAR, decode_pass BIGINT, '
                'frames BIGINT, read_s DOUBLE, open_s DOUBLE, total_s DOUBLE, threads DOUBLE)')
    # One decode file per reference unit; a file without passes is one pass (0).
    for u in units:
        if u['system'] != 'reference':
            continue
        path = u['path'] / u['manifest']['files']['decode']
        con.execute(f"""
            INSERT INTO decode SELECT $phase, $dataset, video, coalesce(decode_pass, 0), frames, read_s, open_s, total_s,
                                      threads
            FROM read_json($path, format = 'newline_delimited', columns = {DECODE_COLUMNS})
        """, dict(phase=u['phase'], dataset=u['dataset'], path=str(path)))
    # Every reference unit's videos decoded once per pass, with the plan's frame count and a positive time.
    bad = query_dicts(con, """
        SELECT d.phase, d.dataset, d.video, d.decode_pass, d.frames, f.frames AS planned,
               count(*) OVER (PARTITION BY d.phase, d.dataset, d.video, d.decode_pass) AS n
        FROM decode d LEFT JOIN frames f USING (dataset, video)
        QUALIFY d.frames IS DISTINCT FROM f.frames OR n <> 1 OR d.total_s IS NULL OR d.total_s <= 0
        LIMIT 5
    """)
    assert not bad, f'decoded videos missing from the plan, repeated, with other frame counts, or without a time: {bad}'
    # Every video of a unit decoded in the same passes.
    bad = query_dicts(con, """
        SELECT phase, dataset, count(DISTINCT video) AS videos, count(DISTINCT decode_pass) AS passes, count(*) AS records
        FROM decode GROUP BY ALL HAVING count(*) <> count(DISTINCT video) * count(DISTINCT decode_pass) LIMIT 5
    """)
    assert not bad, f'reference units whose videos were not all decoded in every pass: {bad}'


def unit_walls(con: duckdb.DuckDBPyConnection, units: list[dict]):
    """Tables walls(unit, stage, ran, start_s, end_s, wall_s, source): each stage process from the manifests; a p022
    taken from the cache gets its p121 seconds over the unit's videos (from table ops) as wall_s, with source 'cache';
    and outputs(unit, stage, video, mtime_ns): each runtime file's last write (attempts made from the serial phase on)."""
    con.execute('CREATE OR REPLACE TABLE walls (unit VARCHAR, stage VARCHAR, ran BOOLEAN, start_s DOUBLE, end_s DOUBLE, '
                'wall_s DOUBLE, source VARCHAR)')
    rows = [(u['unit'], s['stage'], s['ran'], s['start_ns'] / 1e9 if s['ran'] else None,
             s['end_ns'] / 1e9 if s['ran'] else None, s['wall_s'] if s['ran'] else None, s.get('source', 'rerun'))
            for u in units for s in u['manifest']['stages']]
    con.executemany('INSERT INTO walls VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
    # Stages not run stand in with their logged op time (p121's boundary) over the unit's videos.
    con.execute("""
        UPDATE walls SET wall_s = s.seconds
        FROM (SELECT unit, stage, fsum(seconds) FILTER (WHERE counted) AS seconds FROM ops WHERE side = 'clean' GROUP BY ALL) s
        WHERE walls.unit = s.unit AND walls.stage = s.stage AND NOT walls.ran
    """)
    bad = query_dicts(con, 'SELECT unit, stage FROM walls WHERE wall_s IS NULL LIMIT 5')
    assert not bad, f'stages without a wall clock or logged time: {bad}'
    con.execute('CREATE OR REPLACE TABLE outputs (unit VARCHAR, stage VARCHAR, video VARCHAR, mtime_ns BIGINT)')
    con.executemany('INSERT INTO outputs VALUES (?, ?, ?, ?)',
                    [(u['unit'], stage, video, out.get('mtime_ns')) for u in units
                     for stage, by_video in u['manifest']['outputs'].items() for video, out in by_video.items()])


def boundary_totals(con: duckdb.DuckDBPyConnection, plan: dict):
    """Table totals(side, phase, scope, dataset, system, variant_id, videos, frames, <boundary>_s ...): each boundary's
    seconds per unit over all of the unit's videos (scope 'unit') and over its calibration videos (scope
    'calibration'), with the decode passes' range for the decode boundaries. Wall clock exists only for whole units of
    the clean side."""
    # The calibration videos per dataset.
    con.execute('CREATE OR REPLACE TABLE calibration (dataset VARCHAR, video VARCHAR)')
    con.executemany('INSERT INTO calibration VALUES (?, ?)',
                    [(d, v) for d, info in plan['datasets'].items() for v in info['calibration_videos']])
    # Counted seconds, counted decode ops, and canvas reads per unit and video, in both scopes.
    con.execute("""
        CREATE OR REPLACE TABLE scoped AS
        WITH per_video AS (
            SELECT side, unit, phase, dataset, system, variant_id, video,
                   fsum(seconds) FILTER (WHERE counted) AS p121_s,
                   coalesce(fsum(seconds) FILTER (WHERE counted AND decode), 0) AS decode_ops_s,
                   coalesce(fsum(seconds) FILTER (WHERE counted AND canvas_read), 0) AS canvas_read_s
            FROM ops GROUP BY ALL
        )
        SELECT *, 'unit' AS scope FROM per_video
        UNION ALL
        SELECT *, 'calibration' AS scope FROM per_video v
        WHERE EXISTS (SELECT 1 FROM calibration c WHERE c.dataset = v.dataset AND c.video = v.video)
    """)
    # The measured decode per phase, scope, and dataset: each pass's total over the reference unit's videos in scope,
    # then the median, fastest, and slowest pass.
    con.execute("""
        CREATE OR REPLACE TABLE decode_stats AS
        WITH videos AS (SELECT DISTINCT phase, scope, dataset, video FROM scoped WHERE side = 'clean' AND system = 'reference'),
        passes AS (
            SELECT v.phase, v.scope, v.dataset, d.decode_pass, count(*) AS videos, fsum(d.total_s) AS seconds,
                   fsum(d.frames) AS frames
            FROM videos v JOIN decode d USING (phase, dataset, video) GROUP BY ALL
        )
        SELECT phase, scope, dataset, count(*) AS passes, min(videos) AS min_videos, max(videos) AS max_videos,
               any_value(frames) AS frames, median(seconds) AS decode_s, min(seconds) AS decode_min_s,
               max(seconds) AS decode_max_s
        FROM passes GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE totals AS
        WITH t AS (
            SELECT side, phase, scope, dataset, system, variant_id, any_value(unit) AS unit, count(*) AS videos,
                   list(video ORDER BY video) AS video_list, sum(f.frames) AS frames, fsum(p121_s) AS p121_s,
                   fsum(decode_ops_s) AS decode_ops_s, fsum(canvas_read_s) AS canvas_read_s
            FROM scoped s JOIN frames f USING (dataset, video)
            GROUP BY side, phase, scope, dataset, system, variant_id
        ),
        stage_writes AS (
            -- Each stage's videos' last runtime writes (stages that ran, attempts from the serial phase on).
            SELECT w.unit, w.stage, w.ran, w.wall_s, count(m.mtime_ns) AS n, min(m.mtime_ns) AS first_ns, max(m.mtime_ns) AS last_ns
            FROM walls w LEFT JOIN outputs m ON m.unit = w.unit AND m.stage = w.stage AND w.ran
            GROUP BY w.unit, w.stage, w.ran, w.wall_s
        ),
        wall AS (
            -- Stage wall clocks, and the same with startup taken out (every stage that ran needs two writes).
            SELECT unit, fsum(wall_s) AS wall_s, bool_or(NOT ran) AS p022_wall_from_cache,
                   CASE WHEN bool_and(NOT ran OR n >= 2)
                        THEN fsum(CASE WHEN ran THEN (last_ns - first_ns) / 1e9 * n / (n - 1) ELSE wall_s END) END AS steady_s
            FROM stage_writes GROUP BY unit
        ),
        with_decode AS (
            -- One decode per video: the clean side's measured decode of the same phase, the paper's logged read; and
            -- the reference's logged read of the same side and phase on both sides.
            SELECT t.*, r.video_list AS reference_videos, r.decode_ops_s AS logged_decode_s,
                   CASE WHEN t.side = 'clean' THEN d.decode_s ELSE r.decode_ops_s END AS decode_s,
                   CASE WHEN t.side = 'clean' THEN d.decode_min_s ELSE r.decode_ops_s END AS decode_min_s,
                   CASE WHEN t.side = 'clean' THEN d.decode_max_s ELSE r.decode_ops_s END AS decode_max_s,
                   CASE WHEN t.side = 'clean' THEN d.passes ELSE 1 END AS decode_passes,
                   d.min_videos AS decode_min_videos, d.max_videos AS decode_max_videos,
                   CASE WHEN t.side = 'clean' AND t.scope = 'unit' THEN w.wall_s END AS wall_clock_s,
                   CASE WHEN t.side = 'clean' AND t.scope = 'unit' THEN w.steady_s END AS wall_clock_steady_s,
                   CASE WHEN t.side = 'clean' AND t.scope = 'unit' THEN w.p022_wall_from_cache END AS p022_wall_from_cache
            FROM t
            LEFT JOIN t r ON r.side = t.side AND r.phase = t.phase AND r.scope = t.scope AND r.dataset = t.dataset
                         AND r.system = 'reference'
            LEFT JOIN decode_stats d ON t.side = 'clean' AND d.phase = t.phase AND d.scope = t.scope AND d.dataset = t.dataset
            LEFT JOIN wall w ON w.unit = t.unit
        )
        SELECT *, p121_s - decode_ops_s AS decode_excluded_s,
               p121_s - decode_ops_s + decode_s AS one_decode_s,
               p121_s - decode_ops_s + decode_min_s AS one_decode_min_s,
               p121_s - decode_ops_s + decode_max_s AS one_decode_max_s,
               p121_s - decode_ops_s + decode_s - canvas_read_s AS one_decode_no_canvas_read_s,
               p121_s - decode_ops_s + decode_min_s - canvas_read_s AS one_decode_no_canvas_read_min_s,
               p121_s - decode_ops_s + decode_max_s - canvas_read_s AS one_decode_no_canvas_read_max_s,
               p121_s - decode_ops_s + logged_decode_s AS one_decode_logged_s
        FROM with_decode
    """)
    # Both systems on the same videos, so throughputs compare.
    bad = query_dicts(con, """
        SELECT side, phase, scope, dataset, system, variant_id FROM totals
        WHERE reference_videos IS DISTINCT FROM video_list LIMIT 5
    """)
    assert not bad, f'Tetris and the reference timed on different videos (or no reference): {bad}'
    # The clean side's measured decode covers every video in every pass.
    bad = query_dicts(con, """
        SELECT side, phase, scope, dataset, system, videos, decode_min_videos, decode_max_videos FROM totals
        WHERE side = 'clean' AND (decode_s IS NULL OR decode_min_videos <> videos OR decode_max_videos <> videos) LIMIT 5
    """)
    assert not bad, f'videos without a measured decode in every pass (missing reference unit or decode): {bad}'


def boundary_struct(b: str) -> str:
    """SQL for one boundary's struct of a Tetris row t joined to its reference row r: seconds, throughput, speedup,
    and, for the decode boundaries, the speedup at the fastest and the slowest decode pass."""
    r, t = f'r.{b}_s', f't.{b}_s'
    fields = (f'reference_s := {r}, tetris_s := {t}, reference_fps := t.frames / {r}, tetris_fps := t.frames / {t}, '
              f'speedup := {r} / {t}')
    # Adding the same decode to both sides: the speedup moves monotonically with it.
    if b in DECODE_BAND_BOUNDARIES:
        low, high = f'r.{b}_min_s / t.{b}_min_s', f'r.{b}_max_s / t.{b}_max_s'
        fields += f', speedup_range := [least({low}, {high}), greatest({low}, {high})]'
    return f'CASE WHEN {r} IS NOT NULL AND {t} IS NOT NULL THEN struct_pack({fields}) END AS {b}'


def speedups(con: duckdb.DuckDBPyConnection, plan: dict) -> list[dict]:
    """Table speedup_rows: per side, phase, scope, dataset, and bound, both systems' seconds and throughput at every
    boundary and Tetris's speedup (structs named by boundary), from table totals."""
    # Each dataset's operating point per bound.
    con.execute('CREATE OR REPLACE TABLE bound_points (dataset VARCHAR, bound VARCHAR, variant_id VARCHAR)')
    con.executemany('INSERT INTO bound_points VALUES (?, ?, ?)',
                    [(d, b, p['variant_id']) for d, info in plan['datasets'].items() for b, p in info['bounds'].items()])
    con.execute(f"""
        CREATE OR REPLACE TABLE speedup_rows AS
        SELECT t.side, t.phase, t.scope, t.dataset, b.bound, t.variant_id, t.unit, r.unit AS reference_unit, t.videos,
               t.frames, t.p022_wall_from_cache, t.decode_passes, {', '.join(boundary_struct(b) for b in BOUNDARIES)}
        FROM totals t JOIN bound_points b ON b.dataset = t.dataset AND b.variant_id = t.variant_id
        JOIN totals r ON r.side = t.side AND r.phase = t.phase AND r.scope = t.scope AND r.dataset = t.dataset
                     AND r.system = 'reference'
        WHERE t.system = 'tetris'
    """)
    return query_dicts(con, 'SELECT * FROM speedup_rows ORDER BY side, phase, scope, dataset, bound')


def check_paper_throughput(con: duckdb.DuckDBPyConnection, plan: dict) -> dict:
    """Fail unless the paper side's p121 throughput equals the tradeoff table's (the plan's) for both systems."""
    con.execute('CREATE OR REPLACE TABLE plan_fps (dataset VARCHAR, bound VARCHAR, reference_fps DOUBLE, tetris_fps DOUBLE)')
    con.executemany('INSERT INTO plan_fps VALUES (?, ?, ?, ?)',
                    [(d, b, info['reference']['fps'], p['fps']) for d, info in plan['datasets'].items()
                     for b, p in info['bounds'].items()])
    # Every planned (dataset, bound) on the paper side, over all test videos.
    rows, worst = con.execute(f"""
        SELECT count(s.dataset), max(greatest(abs(s.p121.reference_fps / p.reference_fps - 1),
                                            abs(s.p121.tetris_fps / p.tetris_fps - 1)))
        FROM plan_fps p LEFT JOIN speedup_rows s
          ON s.dataset = p.dataset AND s.bound = p.bound AND s.side = 'paper' AND s.phase = '{HEADLINE}' AND s.scope = 'unit'
    """).fetchone()
    planned = con.execute('SELECT count(*) FROM plan_fps').fetchone()[0]
    assert rows == planned, f'paper throughput for {rows} of {planned} planned operating points'
    assert worst < 1e-6, f'paper throughput differs from the tradeoff table by up to {worst:.2e}'
    return dict(max_relative_difference=worst)


def per_op_comparison(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Every op of every headline unit, clean against the paper's logs, summed over the test videos."""
    return query_dicts(con, f"""
        WITH s AS (
            SELECT dataset, system, variant_id, stage, op, any_value(counted) AS counted,
                   fsum(seconds) FILTER (WHERE side = 'clean') AS clean_s, fsum(seconds) FILTER (WHERE side = 'paper') AS paper_s,
                   sum(items) FILTER (WHERE side = 'clean') AS clean_items, sum(items) FILTER (WHERE side = 'paper') AS paper_items
            FROM ops WHERE phase = '{HEADLINE}' GROUP BY ALL
        )
        SELECT *, clean_s / nullif(paper_s, 0) AS clean_over_paper, clean_items IS NOT DISTINCT FROM paper_items AS items_match
        FROM s ORDER BY dataset, system, variant_id, stage, op
    """)


def compare_phases(con: duckdb.DuckDBPyConnection, a: str, b: str, scope: str) -> dict:
    """Phase a against phase b on the same videos (scope): per dataset and bound, each boundary's seconds as a / b for
    both systems and both speedups; per op, seconds a / b; the decode (measured median pass and p002's logged read)
    a / b; and the chosen attempts' foreign CPU shares."""
    ratio = ', '.join(
        f'CASE WHEN x.{k} IS NOT NULL AND y.{k} IS NOT NULL THEN struct_pack('
        f'reference_ratio := x.{k}.reference_s / y.{k}.reference_s, tetris_ratio := x.{k}.tetris_s / y.{k}.tetris_s, '
        f'speedup_a := x.{k}.speedup, speedup_b := y.{k}.speedup) END AS {k}' for k in BOUNDARIES)
    params = dict(a=a, b=b, scope=scope)
    # Each clean row of phase a in the scope, with the same dataset and bound of phase b.
    pairs = """
        FROM speedup_rows x JOIN speedup_rows y ON y.side = 'clean' AND y.phase = $b AND y.scope = x.scope
                                                AND y.dataset = x.dataset AND y.bound = x.bound
        WHERE x.side = 'clean' AND x.phase = $a AND x.scope = $scope"""
    # Both phases must have timed the same frames.
    bad = query_dicts(con, f'SELECT x.dataset, x.bound, x.frames, y.frames AS b_frames {pairs} AND x.frames <> y.frames',
                      params)
    assert not bad, f'{a} and {b} timed on different videos: {bad}'
    boundaries = query_dicts(con, f'SELECT x.dataset, x.bound, x.variant_id, x.videos, x.frames, {ratio} {pairs} '
                                  'ORDER BY x.dataset, x.bound', params)
    # Per op over the scope's videos, for configurations both phases ran.
    ops = query_dicts(con, """
        WITH both_phases AS (
            SELECT dataset, system, variant_id FROM ops WHERE side = 'clean' AND phase IN ($a, $b)
            GROUP BY ALL HAVING count(DISTINCT phase) = 2
        ),
        s AS (
            SELECT o.dataset, o.system, o.variant_id, o.stage, o.op, any_value(o.counted) AS counted,
                   fsum(o.seconds) FILTER (WHERE o.phase = $a) AS a_s, fsum(o.seconds) FILTER (WHERE o.phase = $b) AS b_s,
                   sum(o.items) FILTER (WHERE o.phase = $a) AS a_items, sum(o.items) FILTER (WHERE o.phase = $b) AS b_items
            FROM ops o SEMI JOIN both_phases USING (dataset, system, variant_id)
            WHERE o.side = 'clean' AND o.phase IN ($a, $b)
              AND ($scope = 'unit' OR EXISTS (SELECT 1 FROM calibration c WHERE c.dataset = o.dataset AND c.video = o.video))
            GROUP BY ALL
        )
        SELECT *, a_s / nullif(b_s, 0) AS a_over_b, a_items IS NOT DISTINCT FROM b_items AS items_match
        FROM s ORDER BY dataset, system, variant_id, stage, op
    """, params)
    decode = query_dicts(con, """
        SELECT x.dataset, x.decode_s AS a_measured_s, y.decode_s AS b_measured_s, x.decode_s / y.decode_s AS measured_a_over_b,
               x.logged_decode_s AS a_logged_s, y.logged_decode_s AS b_logged_s,
               x.logged_decode_s / y.logged_decode_s AS logged_a_over_b
        FROM totals x JOIN totals y ON y.side = 'clean' AND y.phase = $b AND y.scope = x.scope AND y.dataset = x.dataset
                                   AND y.system = 'reference'
        WHERE x.side = 'clean' AND x.phase = $a AND x.scope = $scope AND x.system = 'reference' ORDER BY x.dataset
    """, params)
    foreign = query_dicts(con, """
        SELECT h.unit, h.phase, h.dataset, h.system, h.variant_id, c.foreign_on_node_mean, c.foreign_on_node_max, h.cpu_clean
        FROM chosen h JOIN attempt_cpu c USING (path)
        WHERE h.phase IN ($a, $b)
          AND EXISTS (SELECT 1 FROM chosen o WHERE o.phase = $a AND o.dataset = h.dataset AND o.system = h.system
                      AND o.variant_id = h.variant_id)
        ORDER BY h.dataset, h.system, h.variant_id, h.phase
    """, dict(a=a, b=b))
    return dict(a=a, b=b, scope=scope, boundaries=boundaries, ops=ops, decode=decode, foreign_cpu=foreign)


def decode_spread(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per phase and dataset (all of the reference unit's videos): the measured decode's passes, their median,
    fastest, and slowest totals in ms per frame and the spread (slowest - fastest) / median, the decoder's thread
    counts, and p002's logged read of the same unit for comparison."""
    return query_dicts(con, """
        SELECT d.phase, d.dataset, d.passes, d.frames, 1000 * d.decode_s / d.frames AS median_ms_per_frame,
               1000 * d.decode_min_s / d.frames AS fastest_ms_per_frame, 1000 * d.decode_max_s / d.frames AS slowest_ms_per_frame,
               (d.decode_max_s - d.decode_min_s) / d.decode_s AS spread,
               (SELECT list(DISTINCT threads ORDER BY threads) FROM decode x WHERE x.phase = d.phase AND x.dataset = d.dataset)
                   AS threads,
               1000 * t.logged_decode_s / d.frames AS logged_read_ms_per_frame
        FROM decode_stats d JOIN totals t ON t.side = 'clean' AND t.phase = d.phase AND t.scope = d.scope
                                         AND t.dataset = d.dataset AND t.system = 'reference'
        WHERE d.scope = 'unit' ORDER BY d.phase, d.dataset
    """)


def load_conditions(con: duckdb.DuckDBPyConnection, units: list[dict]) -> list[dict]:
    """Per unit: attempts, placement, contamination, the unit GPU per stage, other GPUs' peaks, load, the NUMA node's
    CPUs, and each stage's timing."""
    con.execute('CREATE OR REPLACE TABLE gpu_samples (unit VARCHAR, t_s DOUBLE, util DOUBLE, mem_mib DOUBLE, "foreign" BIGINT, '
                'load_1m DOUBLE)')
    con.execute('CREATE OR REPLACE TABLE all_gpus (unit VARCHAR, gpu BIGINT, util DOUBLE, mem DOUBLE)')
    # Two small files per unit.
    for u in units:
        con.execute(f"""
            INSERT INTO gpu_samples SELECT $unit, epoch(t), util, mem_mib, "foreign", load_1m
            FROM read_json($path, format = 'newline_delimited', columns = {SAMPLE_COLUMNS})
        """, dict(unit=u['unit'], path=str(u['path'] / gpu.LAUNCHER_SAMPLES)))
        # nvidia-smi's CSV pads values with spaces and never quotes; a line cut by the sampler's exit is padded with
        # NULLs (no dialect sniffing, which such a line defeats) and dropped below.
        con.execute("""
            INSERT INTO all_gpus SELECT $unit, TRY_CAST(trim(gpu) AS BIGINT), TRY_CAST(trim(util) AS DOUBLE),
                                        TRY_CAST(trim(mem) AS DOUBLE)
            FROM read_csv($path, auto_detect = false, header = false, delim = ',', quote = '',
                          columns = {'ts': 'VARCHAR', 'gpu': 'VARCHAR', 'util': 'VARCHAR', 'mem': 'VARCHAR'}, null_padding = true)
        """, dict(unit=u['unit'], path=str(u['path'] / gpu.LAUNCHER_GPUS)))
    # The chosen attempts' CPU samples.
    con.execute('CREATE OR REPLACE TABLE cpu_samples AS SELECT h.unit, c.* FROM cpu_all c JOIN chosen h USING (path)')
    per_stage_gpu = {(r['unit'], r['stage']): r for r in query_dicts(con, """
        SELECT w.unit, w.stage, count(g.t_s) AS samples, avg(g.util) AS util_mean, max(g.util) AS util_max,
               max(g.mem_mib) AS mem_max_mib, max(g."foreign") AS foreign_max
        FROM walls w LEFT JOIN gpu_samples g ON g.unit = w.unit AND g.t_s BETWEEN w.start_s AND w.end_s
        WHERE w.ran GROUP BY ALL
    """)}
    per_stage_cpu = {(r['unit'], r['step']): r for r in query_dicts(con, """
        SELECT unit, step, count(*) AS samples, avg(unit_cpus_busy) AS node_busy_mean, max(unit_cpus_busy) AS node_busy_max,
               avg(own_busy) AS own_busy_mean, avg(unit_cpus_busy - own_busy) AS others_on_node_mean,
               max(unit_cpus_busy - own_busy) AS others_on_node_max, avg(foreign_busy) AS foreign_on_node_mean,
               max(foreign_busy) AS foreign_on_node_max, avg(other_cpus_busy) AS other_node_busy_mean
        FROM cpu_samples GROUP BY ALL
    """)}
    # Each stage's process wall clock, its logged op seconds (p121's count) over the unit's videos, and the time from
    # the process start to its first and last video's last runtime write (from the serial phase on).
    per_stage_time = {(r['unit'], r['stage']): r for r in query_dicts(con, """
        SELECT w.unit, w.stage, w.source, w.wall_s, any_value(o.seconds) AS ops_s,
               (min(m.mtime_ns) / 1e9) - w.start_s AS first_output_s, (max(m.mtime_ns) / 1e9) - w.start_s AS last_output_s
        FROM walls w
        LEFT JOIN (SELECT unit, stage, fsum(seconds) FILTER (WHERE counted) AS seconds FROM ops WHERE side = 'clean'
                   GROUP BY ALL) o ON o.unit = w.unit AND o.stage = w.stage
        LEFT JOIN outputs m ON m.unit = w.unit AND m.stage = w.stage AND w.ran
        GROUP BY w.unit, w.stage, w.source, w.wall_s, w.start_s
    """)}
    unit_load = {r['unit']: r for r in query_dicts(con, """
        SELECT unit, count(*) AS samples, avg(load_1m) AS load_1m_mean, max(load_1m) AS load_1m_max,
               sum("foreign") AS foreign_process_samples FROM gpu_samples GROUP BY unit
    """)}
    # The node's CPU shares over a unit's timed steps (its stages and, for the reference, the separate decode).
    unit_cpu = {r['unit']: r for r in query_dicts(con, """
        SELECT h.unit, c.samples, c.others_on_node_mean, c.foreign_on_node_mean, c.foreign_on_node_max
        FROM chosen h JOIN attempt_cpu c USING (path)
    """)}
    # Every attempt of every unit, with its CPU shares and whether it is the chosen one.
    unit_attempts: dict[str, list] = {}
    for r in query_dicts(con, """
        SELECT a.unit, a.attempt, a.status, a.contaminated, c.foreign_on_node_mean, c.foreign_on_node_max,
               EXISTS (SELECT 1 FROM chosen h WHERE h.path = a.path) AS chosen
        FROM attempts a LEFT JOIN attempt_cpu c USING (path) ORDER BY a.unit, a.attempt_no
    """):
        unit_attempts.setdefault(r.pop('unit'), []).append(r)
    other_gpus: dict[str, dict] = {}
    for r in query_dicts(con, """
        SELECT unit, gpu, max(util) AS util_max, avg(util) AS util_mean, max(mem) AS mem_max_mib, count(*) AS samples
        FROM all_gpus WHERE gpu IS NOT NULL AND util IS NOT NULL AND mem IS NOT NULL GROUP BY ALL ORDER BY unit, gpu
    """):
        other_gpus.setdefault(r['unit'], {})[str(r['gpu'])] = {k: r[k] for k in ('util_max', 'util_mean', 'mem_max_mib', 'samples')}
    out = []
    # Arrange per unit.
    for u in units:
        stages = [s['stage'] for s in u['manifest']['stages']]
        out.append(dict(unit=u['unit'], phase=u['phase'], dataset=u['dataset'], attempt=u['attempt'],
                        attempts=unit_attempts[u['unit']], cpu_clean=u['cpu_clean'], gate=u['manifest']['unit'].get('gate'),
                        gpu=u['launcher']['gpu'], numa_node=u['launcher']['numa_node'], cpus=u['launcher']['cpus'],
                        contaminated=u['contaminated'], foreign_samples=u['launcher']['foreign_samples'],
                        start=u['launcher']['start'], end=u['launcher']['end'], p022_source=u['manifest'].get('p022_source'),
                        load=unit_load.get(u['unit']), cpu_timed=unit_cpu.get(u['unit']), gpus=other_gpus.get(u['unit'], {}),
                        stages={s: dict(gpu=per_stage_gpu.get((u['unit'], s)), cpu=per_stage_cpu.get((u['unit'], s)),
                                        time=per_stage_time.get((u['unit'], s)))
                                for s in stages},
                        steps_cpu={k[1]: v for k, v in per_stage_cpu.items() if k[0] == u['unit'] and k[1] not in stages}))
    return out


def output_digests(units: list[dict], cache_root: Path) -> list[dict]:
    """Per headline unit and stage: how many videos' outputs are byte-identical to ace's cached outputs."""
    rows = []
    # A few files per video; hashing ace's cached files is the only cost.
    for u in units:
        if u['phase'] != HEADLINE:
            continue
        for stage, by_video in u['manifest']['outputs'].items():
            for name in gpu.STAGES[stage][3]:
                same = 0
                for video, out in by_video.items():
                    cached = cache_root / Path(out['runtime']).parent / name
                    clean = out['files'].get(name)
                    same += bool(clean and cached.exists() and gpu.sha256(cached) == clean['sha256'])
                rows.append(dict(unit=u['unit'], dataset=u['dataset'], stage=stage, file=name, videos=len(by_video),
                                 identical=same))
    return rows


def frame_checks(con: duckdb.DuckDBPyConnection) -> dict:
    """Per clean reference unit and video, the frames p002 detected and tracked against the plan's count."""
    videos, bad = con.execute("""
        WITH r AS (
            SELECT o.unit, o.video, f.frames,
                   max(o.items) FILTER (WHERE o.stage = '001_preprocess_groundtruth_detection' AND o.op = 'detect') AS detected,
                   max(o.items) FILTER (WHERE o.stage = '002_preprocess_groundtruth_tracking' AND o.op = 'tracker_update') AS tracked
            FROM ops o JOIN frames f USING (dataset, video)
            WHERE o.side = 'clean' AND o.system = 'reference' GROUP BY ALL
        )
        SELECT count(*), list(struct_pack(unit, video, frames, detected, tracked))
                         FILTER (WHERE detected IS DISTINCT FROM frames OR tracked IS DISTINCT FROM frames)
        FROM r
    """).fetchone()
    assert not bad, f'reference units that did not process every frame: {bad[:5]}'
    return dict(videos=videos, all_frames=True)


def label(dataset: str) -> str:
    """The manuscript's name of a dataset (its own name when it has none)."""
    return DATASET_LABELS.get(dataset, dataset)


def arrange_throughput(rows: list[dict], units: list[dict]) -> dict:
    """{datasets: {label: {bound: {config, frames, clean, paper}}}, speedup_table: {bound: {boundary: {label: {clean, paper}}}}}
    for the headline phase over all test videos."""
    by_unit = {u['unit']: u for u in units}
    datasets: dict = {}
    table: dict = {}
    # A handful of rows: each dataset and bound, on the clean side and the paper's.
    for r in rows:
        if r['scope'] != 'unit' or r['phase'] != HEADLINE:
            continue
        tetris, reference = by_unit[r['unit']], by_unit[r['reference_unit']]
        entry = datasets.setdefault(label(r['dataset']), {}).setdefault(r['bound'], dict(
            config=r['variant_id'], videos=r['videos'], frames=r['frames'],
            contaminated=tetris['contaminated'] or reference['contaminated'],
            cpu_clean=tetris['cpu_clean'] and reference['cpu_clean'], p022_source=tetris['manifest'].get('p022_source')))
        entry[r['side']] = {b: r[b] for b in BOUNDARIES}
        if r['side'] == 'clean':
            entry['p022_wall_from_cache'] = r['p022_wall_from_cache']
            entry['decode_passes'] = r['decode_passes']
        for b in BOUNDARIES:
            if r[b] is not None:
                table.setdefault(r['bound'], {}).setdefault(b, {}).setdefault(label(r['dataset']), {})[r['side']] = r[b]['speedup']
    return dict(datasets=datasets, speedup_table=table)


def about(plan: dict, units: list[dict]) -> dict:
    """What every report measured, how, and where its inputs came from."""
    return dict(
        measured=('Throughput of the reference pipeline and of Tetris at its 5% and 10% HOTA-loss operating points, '
                  'from the pipeline scripts rerun unchanged on idle Quadro RTX 6000 GPUs of bombe (NUMA-pinned), at '
                  'seven timing boundaries, next to the same accounting over the paper\'s cached logs.'),
        boundaries={'p121': 'p121 as published: reference decode counted, Tetris decodes not, Tetris canvas reads counted',
                    'decode_excluded': 'p121 minus counted video decoding (the reference\'s p002 read)',
                    'one_decode': ('decode_excluded plus one decode of every video on each side (clean: cv2 decode measured '
                                   'separately on the same videos, median pass, speedup_range over the fastest and slowest '
                                   'pass; paper: the reference\'s logged read)'),
                    'one_decode_no_canvas_read': 'one_decode minus Tetris\'s canvas reads (p040 read)',
                    'one_decode_logged': ('decode_excluded plus one decode of every video on each side at the reference\'s '
                                          'logged p002 read of the same run (equals one_decode on the paper side)'),
                    'wall_clock': ('clean only: the staged prototype\'s stage processes, summed, one worker process at a '
                                   'time on one GPU, including per-process startup, so it depends on the videos per '
                                   'process (5 on B3D, 60 elsewhere); p022 from its log when it could not run'),
                    'wall_clock_steady': ('clean serial only: wall_clock with each stage\'s process startup taken out '
                                          '(first to last video\'s runtime write, times n / (n - 1)); the counterpart of '
                                          'REVISION.md finding 3\'s wall-clock row (r9o2\'s process boundary on ace)')},
        phases=dict(serial='all test videos, one unit at a time on an idle node (headline numbers)',
                    parallel='all test videos while this job\'s three GPUs on one NUMA node ran datasets at once',
                    solo='calibration videos, one unit at a time on an idle node (repeatability against serial)'),
        attempt_choice=ATTEMPT_RULE, foreign_cpu_max=gpu.FOREIGN_CPU_MAX,
        worker_counts=('ProgressBar runs at most num_workers tasks of a stage at once: visible GPUs for p002, p020, '
                       'p040, p060 (1 on bombe, 8 on ace), 20 for p022 and p030, 40 for p050. A Tetris stage has one '
                       'task per configuration (its videos in sorted order), p002 one task per video; the clean runs '
                       'ran one task at a time, the paper\'s runs every Pareto configuration at once.'),
        p022=('Gurobi\'s license had expired, so p022 did not run: its outputs and runtime log are the cached ones '
              '(p022_source cache), and the wall clock uses its logged time (p022_wall_from_cache).'),
        plan=dict(created=plan['created'], units=len(units)), gpu_job=gpu.SCRIPT)


def compute(raw: Path, cache_root: Path, datasets: list[str] | None) -> dict[str, dict]:
    """Every report, keyed by its file name, from the GPU job's raw directory and a cache holding the paper's logs."""
    plan, planned = load_plan(raw, datasets)
    con = duckdb.connect()
    # Attempts and their CPU shares first: the choice of attempt depends on them.
    load_attempts(con, raw, planned)
    units = choose_attempts(con, planned)
    # Runtime logs of both sides, p121's own tables, and the port check that ties them.
    load_runtime_ops(con, runtime_files(units, cache_root))
    load_paper_table(con, plan, cache_root)
    ported = port_check(con)
    load_frames_and_decode(con, plan, units)
    unit_walls(con, units)
    boundary_totals(con, plan)
    rows = speedups(con, plan)
    checks = dict(port_check=ported, paper_throughput_vs_tradeoff=check_paper_throughput(con, plan),
                  frames=frame_checks(con), outputs_identical_to_cache=output_digests(units, cache_root))
    info = about(plan, units)
    return {'throughput.json': dict(about=info, **arrange_throughput(rows, units)),
            'per_op.json': dict(about=info, ops=per_op_comparison(con)),
            'interference.json': dict(about=info, comparisons=[compare_phases(con, a, b, s) for a, b, s in COMPARISONS],
                                      decode_passes=decode_spread(con)),
            'conditions.json': dict(about=info, checks=checks, units=load_conditions(con, units))}


def main(args):
    # Every report from this checkout's raw directory and ace's cache, rounded for clean diffs.
    reports = compute(gpu.RAW, Path(cache.CACHE_DIR), args.datasets)
    out_dir = results_dir(__file__)
    for name, data in reports.items():
        write_json(out_dir, name, rounded(data))


if __name__ == '__main__':
    main(parse_args())
