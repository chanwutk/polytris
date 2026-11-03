#!/usr/local/bin/python
"""R9.O3: ILP construction and solve timings, and the ILP's share of extraction runtime.

Nothing is rerun. Two kinds of saved logs are summarized.

Pruning runs
    scripts/p022_exec_prune_polyominoes.py writes one score/runtime.jsonl per
    (dataset, video, pruning configuration) under
    <dataset>/execution/<video>/022_pruned_polyominoes/<param>/, holding the
    stage's operations (prepare_bitmaps, group_tiles, build_ilp, solve_ilp,
    extract_grids) in milliseconds and the --time-limit it ran with. Every saved
    run (one clip under one configuration) is read, validation and test videos
    alike, and build_ilp, solve_ilp, and their sum are summarized per run
    (median, mean, p95, max), overall and by split, whole-frame sampling rate,
    dataset, p022 invocation, and clip position.

    Timing boundaries, from polyis/sample/ilp/c/gurobi.pyx: build_ilp starts
    before the Gurobi environment is created and started (license check) and
    ends after the last constraint is added; solve_ilp covers GRBoptimize,
    reading the solution back, and freeing the model and environment.

    Solve budget. p022 calls solve_ilp with `time_limit or <default>`, so a run
    logged with time_limit null (or 0) used the default, which is read from
    p022's source rather than assumed. A run with --time-limit set is written to
    the separate 022t stage and is not read here. Gurobi's termination status is
    not logged, so a run "reaches the budget" when its solve_ilp time is at least
    the budget. A time-limited solve cannot read below the budget, so this is an
    upper bound; counts just below and above the budget, and the share at 1.1x
    the budget (a lower bound if readout and cleanup take at most 10% of the
    budget), show how much it can overstate.

    Clip position. p022 starts one process per task (one configuration over one
    dataset split's videos, in sorted order), PRUNE_WORKERS at a time. The first
    PRUNE_WORKERS tasks of an invocation start Gurobi at the same moment, so the
    first clip of each is reported apart from other tasks' first clips and from
    later clips. An invocation is a run of writes without a gap longer than
    INVOCATION_GAP_S; tasks are ranked by their first write (mtimes have 1 s
    resolution, so ties share a rank). A task's first clip is the first video of
    the split's directory listing, as p022 sorts it, not the earliest saved run:
    p022 deletes the pruning outputs of every video it lists when it starts, so
    a later invocation over some videos removes the other configurations' runs
    on them (reported as missing runs in inputs.coverage).

Extraction runtime
    The ILP's share of extraction is taken from the same per-operation runtime
    table the throughput numbers are computed from
    (evaluation/p121_throughput_compute.py writes
    <dataset>/evaluation/080_throughput/measurements/query_execution_per_op.csv,
    whose per-configuration totals evaluation/p130_tradeoff_compute.py turns into
    throughput). Each configuration's total must equal the tradeoff table's
    runtime, and its pruning-stage ILP times must equal the saved p022 logs; both
    are checked. The operating point is the fastest validated Tetris
    configuration within --loss of the reference pipeline's test HOTA. When that
    configuration does not prune (no tracking-accuracy threshold), its ILP share
    is zero, so the fastest pruning configuration within the same bound and the
    spread over every validated pruning configuration are reported as well, each
    selected point with its per-clip ILP timings on the test videos.

Writes measurements-results/r9o3_ilp_timings/:
  ilp_timings.json       inputs and coverage, solve budget, and per-run timing statistics by grouping
  extraction_share.json  the ILP's and the pruning stage's share of extraction runtime per dataset
"""
import argparse
import glob
from pathlib import Path

import duckdb

from evaluation.manifests import list_split_videos
from measurements.common import (DATASET_LABELS, PRUNE_DIR, PRUNE_STAGE, SPLIT_PREFIXES, VARIANT_PATTERN,
                                 default_time_limit, load_p121_per_op, load_prune_runs, load_pruning_points,
                                 load_test_tradeoff, pruning_operating_points, query_dicts, results_dir, write_json)
from polyis.io import cache


# p022 runs mp.cpu_count() // 4 processes at once; the logs were written on ace, which has 80 CPUs.
PRUNE_WORKERS = 20
# Writes further apart than this belong to different p022 invocations (observed: under 1 minute within, 9 hours between).
INVOCATION_GAP_S = 600
# The three per-run quantities summarized.
TIMINGS = ('build_ms', 'solve_ms', 'build_plus_solve_ms')
# Largest allowed difference between the throughput table's ILP time and the p022 log it was parsed from.
MATCH_TOLERANCE_MS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O3: ILP timings and share of extraction runtime')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    return parser.parse_args()


def runtime_files(cache_root: Path, datasets: list[str]) -> list[str]:
    """Every saved pruning runtime log of the datasets, failing on a dataset with none."""
    # One glob per dataset, so a dataset without runs is reported by name.
    found = {d: sorted(glob.glob(str(cache_root / d / 'execution' / '*' / PRUNE_DIR / '*' / 'score' / 'runtime.jsonl')))
             for d in datasets}
    # Datasets whose cache holds no pruning run at all.
    empty = [d for d, files in found.items() if not files]
    assert not empty, f'no pruning runtime logs for datasets {empty} under {cache_root}'
    # One flat list over all datasets.
    return [f for files in found.values() for f in files]


def split_videos(datasets: list[str]) -> list[tuple[str, str, str]]:
    """(dataset, split, video) for every video p022 lists in each dataset's validation and test directories."""
    # p022 lists each split directory and prunes the video files in it in sorted order, as list_split_videos does.
    rows = [(d, split, f) for d in datasets for split in SPLIT_PREFIXES.values() for f in list_split_videos(d, split)]
    # A split without videos would leave its runs without a first clip.
    empty = sorted({(d, s) for d in datasets for s in SPLIT_PREFIXES.values()} - {(d, s) for d, s, _ in rows})
    assert not empty, f'no videos listed for (dataset, split) {empty}'
    # Listing rows in dataset, split, and sorted video order.
    return rows


def load_runs(con: duckdb.DuckDBPyConnection, files: list[str], default_limit: float,
              videos: list[tuple[str, str, str]], workers: int = PRUNE_WORKERS, gap_s: float = INVOCATION_GAP_S):
    """Table runs: one row per pruning run with its parameters, budget, build/solve/stage times in ms, and clip position.

    The runs are read strictly by common.load_prune_runs. `videos` lists (dataset, split, video) as p022 saw each
    split directory (see split_videos); it is kept as table split_videos for the coverage report.
    """
    # Every saved run, parsed and validated, with p022's `time_limit or default` budget.
    load_prune_runs(con, files, default_limit)
    # The split directory listings p022 iterated, one row per video.
    con.execute('CREATE OR REPLACE TABLE split_videos (dataset VARCHAR, split VARCHAR, video VARCHAR)')
    # Insert the listing only when there is one (executemany needs at least one row).
    if videos:
        con.executemany('INSERT INTO split_videos VALUES (?, ?, ?)', videos)
    # A run on an unlisted video means the listing is not the one p022 used, so first clips would be wrong.
    bad = query_dicts(con, """
        SELECT dataset, split, video FROM runs ANTI JOIN split_videos USING (dataset, split, video) LIMIT 5
    """)
    assert not bad, f'runs on videos missing from the split directory listing: {bad}'
    # Tag each run with its p022 invocation, its task's launch rank, and its clip position.
    con.execute("""
        CREATE OR REPLACE TABLE runs AS
        WITH stamps AS (SELECT DISTINCT last_modified FROM runs),
        starts AS (
            SELECT last_modified,
                   coalesce(epoch(last_modified) - epoch(lag(last_modified) OVER (ORDER BY last_modified)) > $gap, true)
                       AS starts_invocation
            FROM stamps
        ),
        invocations AS (
            SELECT last_modified, sum(starts_invocation::INTEGER) OVER (ORDER BY last_modified) AS invocation FROM starts
        ),
        tagged AS (SELECT r.*, i.invocation FROM runs r JOIN invocations i USING (last_modified)),
        tasks AS (
            SELECT invocation, dataset, split, param,
                   rank() OVER (PARTITION BY invocation ORDER BY min(last_modified)) AS launch_rank
            FROM tagged GROUP BY invocation, dataset, split, param
        ),
        firsts AS (SELECT dataset, split, min(video) AS first_video FROM split_videos GROUP BY dataset, split)
        SELECT t.*, k.launch_rank,
               CASE WHEN t.video = f.first_video AND k.launch_rank <= $workers THEN 'launch_first_clip'
                    WHEN t.video = f.first_video THEN 'other_first_clip'
                    ELSE 'later_clip' END AS clip_position
        FROM tagged t JOIN tasks k USING (invocation, dataset, split, param) JOIN firsts f USING (dataset, split)
    """, {'gap': gap_s, 'workers': workers})


def timing_stats(con: duckdb.DuckDBPyConnection, keys: list[str], table: str = 'runs') -> list[dict]:
    """Per group of `keys` (none for all runs) of `table`: per-run timing statistics, budget hits, and the ILP's share of the stage.

    `table` needs columns budget_s, build_ms, solve_ms, build_plus_solve_ms, stage_ms. Percentiles
    interpolate linearly (DuckDB quantile_cont). A run reaches the budget when its solve time is at
    least its budget; the runs_* counts bin solve time around the budget, and
    lower_bound_percent moves the boundary to 1.1x the budget.
    """
    # Median, mean, p95, and max of one per-run timing column, in ms.
    def stats(column: str) -> str:
        return (f"struct_pack(median := round(median({column}), 2), mean := round(avg({column}), 2), "
                f"p95 := round(quantile_cont({column}, 0.95), 2), max := round(max({column}), 2)) AS {column}")
    # Group keys first, then one statistics struct per timing column.
    select = ', '.join(keys + [stats(c) for c in TIMINGS])
    # Grouped and ordered by the keys, or a single row over the whole table.
    group = f"GROUP BY {', '.join(keys)} ORDER BY {', '.join(keys)}" if keys else ''
    # Solve time as a multiple of the run's own budget, so mixed budgets bin consistently.
    ratio = 'solve_ms / (1000 * budget_s)'
    # One row per group: run count, timing statistics, budget hits with bins around the budget, and the stage share.
    return query_dicts(con, f"""
        SELECT {select}, count(*) AS runs,
               struct_pack(
                   runs := count(*) FILTER ({ratio} >= 1),
                   percent := round(100 * avg(({ratio} >= 1)::INTEGER), 2),
                   lower_bound_percent := round(100 * avg(({ratio} >= 1.1)::INTEGER), 2),
                   median_solve_over_budget := round(median({ratio}), 3),
                   runs_0_90_to_1_00x := count(*) FILTER ({ratio} >= 0.9 AND {ratio} < 1),
                   runs_1_00_to_1_05x := count(*) FILTER ({ratio} >= 1 AND {ratio} < 1.05),
                   runs_1_05_to_1_10x := count(*) FILTER ({ratio} >= 1.05 AND {ratio} < 1.1),
                   runs_from_1_10x := count(*) FILTER ({ratio} >= 1.1)
               ) AS solve_reaches_budget,
               struct_pack(
                   median_per_run_percent := round(100 * median(build_plus_solve_ms / nullif(stage_ms, 0)), 2),
                   aggregate_percent := round(100 * sum(build_plus_solve_ms) / nullif(sum(stage_ms), 0), 2)
               ) AS ilp_share_of_pruning_stage
        FROM {table} {group}
    """)


def timing_report(con: duckdb.DuckDBPyConnection, default_limit: float, workers: int = PRUNE_WORKERS) -> dict:
    """Inputs, coverage, budget, and timing statistics overall and by split, sampling rate, dataset, invocation, and clip position."""
    # Per dataset and split: saved runs against configurations x listed videos, and the write window.
    con.execute("""
        CREATE OR REPLACE TABLE coverage AS
        WITH per_config AS (SELECT dataset, split, param, count(*) AS n FROM runs GROUP BY dataset, split, param),
        spread AS (SELECT dataset, split, min(n) AS min_videos_per_config, max(n) AS max_videos_per_config
                   FROM per_config GROUP BY dataset, split),
        listed AS (SELECT dataset, split, count(*) AS listed_videos FROM split_videos GROUP BY dataset, split),
        saved AS (
            SELECT dataset, split, count(*) AS runs, count(DISTINCT video) AS saved_videos, count(DISTINCT param) AS configs,
                   list(DISTINCT tilesize ORDER BY tilesize) AS tile_sizes,
                   CAST(min(last_modified) AS VARCHAR) AS first_written, CAST(max(last_modified) AS VARCHAR) AS last_written
            FROM runs GROUP BY dataset, split
        )
        SELECT s.*, l.listed_videos, s.configs * l.listed_videos AS expected_runs,
               s.configs * l.listed_videos - s.runs AS missing_runs, p.min_videos_per_config, p.max_videos_per_config
        FROM saved s JOIN listed l USING (dataset, split) JOIN spread p USING (dataset, split)
    """)
    # Coverage rows in a stable order.
    coverage = query_dicts(con, 'SELECT * FROM coverage ORDER BY dataset, split')
    # Saved, expected, and missing runs over every dataset and split.
    totals = query_dicts(con, """
        SELECT sum(runs) AS runs, sum(expected_runs) AS expected_runs, sum(missing_runs) AS missing_runs FROM coverage
    """)[0]
    # Per p022 invocation: write window, runs, tasks, launch tasks, and what it covered.
    invocations = query_dicts(con, """
        SELECT invocation, CAST(min(last_modified) AS VARCHAR) AS first_written, CAST(max(last_modified) AS VARCHAR) AS last_written,
               count(*) AS runs, count(DISTINCT (dataset, split, param)) AS tasks,
               count(DISTINCT (dataset, split, param)) FILTER (launch_rank <= $workers) AS launch_tasks,
               count(*) FILTER (clip_position = 'launch_first_clip') AS launch_first_clips,
               list(DISTINCT dataset ORDER BY dataset) AS datasets, list(DISTINCT split ORDER BY split) AS splits
        FROM runs GROUP BY invocation ORDER BY invocation
    """, {'workers': workers})
    # The logged time limits and the budget each implies.
    limits = query_dicts(con, """
        SELECT CAST(time_limit AS VARCHAR) AS logged_time_limit, budget_s, count(*) AS runs
        FROM runs GROUP BY ALL ORDER BY ALL
    """)
    # Statistics per sampling rate, keyed by the rate below.
    by_rate = timing_stats(con, ['sample_rate'])
    # Provenance and definitions, then the statistics per grouping.
    return dict(
        source=f'<cache>/<dataset>/execution/<video>/{PRUNE_DIR}/<param>/score/runtime.jsonl, every saved run '
               '(one clip under one pruning configuration)',
        inputs=dict(coverage=coverage, totals=totals, invocations=invocations,
                    coverage_rule='expected_runs = configurations with any saved run x videos in the split directory'),
        solve_budget=dict(
            default_seconds=default_limit, logged_time_limits=limits,
            rule=f'p022 passes `time_limit or {default_limit}` to solve_ilp; a null or zero time_limit used the default',
            reaches_budget='estimate: solve_ilp time >= budget, since termination status is not logged. A time-limited '
                           'solve cannot read below the budget, so percent is an upper bound. solve_ilp also includes '
                           'solution readout and cleanup, so an optimal solve can read just above the budget; the '
                           'runs_* counts bin solve time in multiples of the budget, lower_bound_percent counts '
                           'solve >= 1.1 x budget (a lower bound if readout and cleanup take at most 0.1 x budget), '
                           'and median_solve_over_budget >= 1 means the median is set by the budget'),
        timing_boundaries=dict(
            build_ms='Gurobi environment creation and start, model creation, and constraint construction',
            solve_ms='GRBoptimize, solution readout, and freeing the model and environment',
            stage_ms='all p022 operations: prepare_bitmaps, group_tiles, build_ilp, solve_ilp, extract_grids',
            load=f'p022 runs {workers} tasks (processes) at once and leaves Gurobi Threads at its default, so these '
                 'are timings under that concurrency, the same logs the throughput numbers sum',
            clip_position=f'launch_first_clip: first clip of the first {workers} tasks a p022 invocation starts '
                          '(ranked by first write, 1 s resolution, ties share a rank), whose processes start Gurobi '
                          'at the same moment; other_first_clip: first clip of any other task; later_clip: the rest. '
                          'A first clip is the first video of the split directory listing, as p022 sorts it. An '
                          f'invocation is a run of writes with no gap over {INVOCATION_GAP_S} s'),
        percentiles='linear interpolation (DuckDB quantile_cont)',
        all=timing_stats(con, [])[0],
        by_split={r['split']: r for r in timing_stats(con, ['split'])},
        by_sample_rate={str(r['sample_rate']): r for r in by_rate},
        by_dataset={DATASET_LABELS.get(r['dataset'], r['dataset']): r for r in timing_stats(con, ['dataset'])},
        by_split_and_sample_rate={f"{r['split']} s={r['sample_rate']}": r
                                  for r in timing_stats(con, ['split', 'sample_rate'])},
        by_invocation={str(r['invocation']): r for r in timing_stats(con, ['invocation'])},
        # Simultaneous process start-up, not first clips in general, sets the overall maximum.
        by_clip_position={r['clip_position']: r for r in timing_stats(con, ['clip_position'])},
        # Per dataset, since datasets with few videos per task contribute many first clips.
        by_dataset_and_clip_position={f"{DATASET_LABELS.get(r['dataset'], r['dataset'])} {r['clip_position']}": r
                                      for r in timing_stats(con, ['dataset', 'clip_position'])},
    )


def load_throughput_ops(con: duckdb.DuckDBPyConnection, paths: dict[str, Path]):
    """Table ops(dataset, videoset, variant_id, video, stage, op, seconds, records) from p121's per-operation tables
    (common.load_p121_per_op), test split only."""
    # Every dataset's table at once (the shared loader fails on a missing one, which would silently drop a dataset).
    load_p121_per_op(con, 'ops', paths)
    # The tradeoff table is the test split, so the runtime table must be too.
    bad = query_dicts(con, "SELECT DISTINCT dataset, videoset FROM ops WHERE videoset IS DISTINCT FROM 'test'")
    assert not bad, f'throughput tables are not the test split: {bad}'
    # Each file must hold only its own dataset.
    found = {r['dataset'] for r in query_dicts(con, 'SELECT DISTINCT dataset FROM ops')}
    assert found == set(paths), f'throughput tables cover {sorted(found)}, expected {sorted(paths)}'


def variant_videos(con: duckdb.DuckDBPyConnection) -> dict:
    """Table variant_video: per throughput configuration and test video, extraction, pruning-stage, and ILP seconds,
    joined to the p022 run its pruning stage was parsed from; fails unless every pruning row matches its run.
    """
    # Sum each (configuration, video) and, for those that prune, look up the run by p022's parameter string.
    con.execute(f"""
        CREATE OR REPLACE TABLE variant_video AS
        WITH per_video AS (
            SELECT dataset, variant_id AS config, video, sum(seconds) AS total_s,
                   coalesce(sum(seconds) FILTER (stage = '{PRUNE_STAGE}'), 0) AS prune_s,
                   coalesce(sum(seconds) FILTER (stage = '{PRUNE_STAGE}' AND op IN ('build_ilp', 'solve_ilp')), 0) AS ilp_s,
                   count(*) FILTER (stage = '{PRUNE_STAGE}') > 0 AS prunes
            FROM ops GROUP BY dataset, variant_id, video
        )
        SELECT v.*, r.param, r.budget_s, r.build_ms, r.solve_ms, r.build_plus_solve_ms, r.stage_ms, r.clip_position
        FROM per_video v LEFT JOIN runs r
          ON v.prunes AND r.dataset = v.dataset AND r.video = v.video
         AND r.param = regexp_replace(v.config, '{VARIANT_PATTERN}', '\\1_\\2')
    """)
    # Pruning rows, how many found their run, the largest ILP time difference, and how many test runs are used.
    check = query_dicts(con, """
        SELECT count(*) AS throughput_rows, count(param) AS matched_runs,
               max(abs(1000 * ilp_s - build_plus_solve_ms)) AS max_abs_diff_ms,
               count(DISTINCT (dataset, video, param)) FILTER (param IS NOT NULL) AS test_runs_used,
               (SELECT count(*) FROM runs WHERE split = 'test') AS test_runs_saved
        FROM variant_video WHERE prunes
    """)[0]
    # A pruning configuration whose run is missing, or whose id does not map to a p022 parameter string.
    assert check['matched_runs'] == check['throughput_rows'], f'throughput rows without a saved p022 run: {check}'
    # A stale throughput table would carry other ILP times than the saved logs.
    assert (check['max_abs_diff_ms'] or 0) <= MATCH_TOLERANCE_MS, f'throughput ILP times differ from the p022 logs: {check}'
    # The check itself goes into the report.
    return check


def variant_runtimes(con: duckdb.DuckDBPyConnection):
    """Table variant_runtime: each configuration's extraction, pruning-stage, and ILP seconds and percents over the test split.

    Every configuration in points must appear with the tradeoff table's runtime, and must prune on
    every test video exactly when it has a tracking-accuracy threshold, so the shares use exactly
    the totals behind the reported throughput.
    """
    # Aggregate the per-video rows; the excluding-launch share drops videos whose pruning run is a launch first clip.
    con.execute("""
        CREATE OR REPLACE TABLE variant_runtime AS
        WITH agg AS (
            SELECT dataset, config, sum(total_s) AS total_s, sum(prune_s) AS prune_s, sum(ilp_s) AS ilp_s,
                   count(*) AS videos, count(*) FILTER (prunes) AS pruned_videos, bool_or(prunes) AS prunes,
                   count(*) FILTER (clip_position = 'launch_first_clip') AS launch_clips,
                   sum(ilp_s) FILTER (clip_position IS DISTINCT FROM 'launch_first_clip') AS steady_ilp_s,
                   sum(total_s) FILTER (clip_position IS DISTINCT FROM 'launch_first_clip') AS steady_total_s
            FROM variant_video GROUP BY dataset, config
        )
        SELECT * EXCLUDE (steady_ilp_s, steady_total_s),
               100 * ilp_s / nullif(total_s, 0) AS ilp_percent,
               100 * prune_s / nullif(total_s, 0) AS pruning_stage_percent,
               100 * steady_ilp_s / nullif(steady_total_s, 0) AS ilp_percent_excluding_launch_clips
        FROM agg
    """)
    # Configurations whose runtime is missing, differs from the tradeoff table's, or prunes when it should not (or the reverse).
    bad = query_dicts(con, """
        SELECT p.dataset, p.config, p.time_s, v.total_s, p.prunes AS should_prune, v.prunes, v.pruned_videos, v.videos
        FROM points p LEFT JOIN variant_runtime v USING (dataset, config)
        WHERE p.system = 'tetris'
          AND (v.total_s IS NULL OR abs(v.total_s - p.time_s) > 1e-6 * p.time_s
               OR v.prunes IS DISTINCT FROM p.prunes OR v.pruned_videos NOT IN (0, v.videos))
        LIMIT 5
    """)
    assert not bad, f'throughput table disagrees with the tradeoff table: {bad}'


def point_entry(row: dict, runtime: dict, clips: dict | None) -> dict:
    """One selected configuration: accuracy, throughput, rounded runtime shares, and its per-clip ILP timings."""
    # Rounded copies of the configuration's runtime row from variant_runtime.
    shares = {k: (round(runtime[k], 3) if runtime[k] is not None else None)
              for k in ('ilp_percent', 'pruning_stage_percent', 'ilp_percent_excluding_launch_clips')}
    # Identity, accuracy and throughput, runtime and shares, and the clip-level statistics.
    return dict(config=row['config'], hota=row['accuracy'], fps=round(row['fps'], 1), prunes=runtime['prunes'],
                videos=runtime['videos'], launch_clips=runtime['launch_clips'],
                extraction_s=round(runtime['total_s'], 3), ilp_s=round(runtime['ilp_s'], 3), **shares,
                clip_timings=clips)


def extraction_report(con: duckdb.DuckDBPyConnection, loss: float) -> dict:
    """Per dataset: the ILP's share of extraction at the operating point, at the fastest pruning point, and over all pruning points."""
    # Fastest configuration within the bound, among all and among pruning configurations (table chosen_points).
    chosen = pruning_operating_points(con, loss)
    # Every dataset in points needs an operating-point row, or it would silently drop out of the report.
    missing = ({r['dataset'] for r in query_dicts(con, 'SELECT DISTINCT dataset FROM points')}
               - {r['dataset'] for r in chosen if r['system'] == 'tetris'})
    assert not missing, f'no Tetris configuration for datasets {sorted(missing)}'
    # Each configuration's runtime row, keyed by (dataset, config).
    runtime = {(r['dataset'], r['config']): r for r in query_dicts(con, 'SELECT * FROM variant_runtime')}
    # The saved p022 runs behind each selected pruning configuration's test videos.
    con.execute("""
        CREATE OR REPLACE TABLE point_runs AS
        SELECT c.dataset, c.point, v.budget_s, v.build_ms, v.solve_ms, v.build_plus_solve_ms, v.stage_ms
        FROM chosen_points c JOIN variant_video v USING (dataset, config) WHERE v.prunes
    """)
    # Per-clip timings per selected point, with the same statistics as the whole sweep.
    clips = {(r['dataset'], r['point']): r for r in timing_stats(con, ['dataset', 'point'], 'point_runs')}
    # Spread of the shares over every validated pruning configuration.
    spread = {r['dataset']: r for r in query_dicts(con, """
        SELECT v.dataset, count(*) AS configs,
               round(median(ilp_percent), 3) AS median_ilp_percent,
               round(min(ilp_percent), 3) AS min_ilp_percent, round(max(ilp_percent), 3) AS max_ilp_percent,
               round(median(pruning_stage_percent), 3) AS median_pruning_stage_percent,
               round(max(pruning_stage_percent), 3) AS max_pruning_stage_percent
        FROM variant_runtime v JOIN points p ON p.dataset = v.dataset AND p.config = v.config AND p.system = 'tetris_pruned'
        GROUP BY v.dataset
    """)}
    # Per-dataset entries, filled from the selections below.
    out = {}
    # Arrange the two selections and the spread per dataset.
    for r in chosen:
        entry = out.setdefault(DATASET_LABELS.get(r['dataset'], r['dataset']), dict(
            dataset=r['dataset'], hota_bound=round(r['bound'], 4), reference_hota=r['reference_accuracy'],
            all_pruning_configs=spread.get(r['dataset']),
            # Replaced below unless the dataset has no validated pruning configuration at all.
            fastest_pruning_point=dict(meets_bound=False, best_hota=None)))
        # The configuration's shares and clip timings, or the best accuracy it reached when none meets the bound.
        entry[r['point']] = (point_entry(r, runtime[(r['dataset'], r['config'])], clips.get((r['dataset'], r['point'])))
                             if r['meets_bound'] else dict(meets_bound=False, best_hota=r['best_accuracy']))
    # Keyed by the manuscript's dataset label.
    return out


def main():
    args = parse_args()
    # Restrict to the paper's datasets, failing on unknown names.
    unknown = set(args.datasets) - set(DATASET_LABELS)
    assert not unknown, f'unknown datasets {sorted(unknown)}; choose from {sorted(DATASET_LABELS)}'
    # Output directory for this script's JSON files.
    out_dir = results_dir(__file__)
    # One in-memory DuckDB database for every table below.
    con = duckdb.connect()

    # Every saved pruning run, with p022's default budget read from its source and the split listings p022 iterated.
    limit = default_time_limit()
    load_runs(con, runtime_files(Path(cache.CACHE_DIR), args.datasets), limit, split_videos(args.datasets))
    timings = timing_report(con, limit)
    write_json(out_dir, 'ilp_timings.json', timings)

    # Extraction runtime per configuration from the throughput tables, checked against the p022 logs and the tradeoff tables.
    load_throughput_ops(con, {d: cache.eval(d, 'tp', 'measurements', 'query_execution_per_op.csv') for d in args.datasets})
    matches = variant_videos(con)
    load_pruning_points(con, {d: load_test_tradeoff(d) for d in args.datasets})
    variant_runtimes(con)
    extraction = dict(
        source='<cache>/<dataset>/evaluation/080_throughput/measurements/query_execution_per_op.csv (test split), '
               'the per-operation runtimes behind the tradeoff table throughput',
        selection=f'fastest validated Tetris configuration within {args.loss:.0%} of the reference test HOTA; '
                  'fastest_pruning_point restricts to configurations with a tracking-accuracy threshold',
        launch_clips='ilp_percent_excluding_launch_clips drops the (configuration, video) pairs whose pruning run is '
                     'a launch first clip (see ilp_timings.json timing_boundaries.clip_position) from both sides',
        throughput_matches_p022_logs=matches,
        datasets=extraction_report(con, args.loss))
    write_json(out_dir, 'extraction_share.json', extraction)

    # Short summary for the run log.
    a = timings['all']
    print(f"{a['runs']} runs ({timings['inputs']['totals']['missing_runs']} missing); build median "
          f"{a['build_ms']['median']} ms, solve median {a['solve_ms']['median']} ms, both p95 "
          f"{a['build_plus_solve_ms']['p95']} max {a['build_plus_solve_ms']['max']} ms; solve >= {limit} s on "
          f"{a['solve_reaches_budget']['percent']}% (>= 1.1x on {a['solve_reaches_budget']['lower_bound_percent']}%)")
    # One line per clip position.
    for name, s in timings['by_clip_position'].items():
        print(f"  {name}: {s['runs']} runs, build median {s['build_ms']['median']} max {s['build_ms']['max']}, "
              f"build+solve max {s['build_plus_solve_ms']['max']} ms")
    # One line per sampling rate.
    for rate, s in timings['by_sample_rate'].items():
        print(f"  s={rate:>2}: {s['runs']:5d} runs, build+solve median {s['build_plus_solve_ms']['median']} ms, "
              f"at budget {s['solve_reaches_budget']['percent']}%")
    # The throughput cross-check.
    print(f"  throughput vs p022 logs: {matches}")
    # One line per dataset.
    for label, d in extraction['datasets'].items():
        op, fp = d['operating_point'], d['fastest_pruning_point']
        print(f"  {label:10s} op {op.get('config')} ILP {op.get('ilp_percent')}% | pruning point "
              f"{fp.get('config')} ILP {fp.get('ilp_percent')}% (steady {fp.get('ilp_percent_excluding_launch_clips')}%) "
              f"stage {fp.get('pruning_stage_percent')}%")


if __name__ == '__main__':
    main()
