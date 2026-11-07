#!/usr/local/bin/python
"""R7.O1: Tetris, OTIF, and LEAP at matched HOTA, and the operator ablation the response quotes.

Port of tetris-paper-feedback/measurements/matched_accuracy_speedup.py (--metric hota),
extended from OTIF to LEAP, over all seven datasets.

Operating points (primary, symmetric rule)
  Every system is held to one bound per dataset and contributes its fastest test-split
  configuration whose HOTA is at least that bound. The bounds are a relative HOTA loss of
  --losses from the reference pipeline (HOTA >= reference x (1 - loss)) and "leap_fit", LEAP's
  best HOTA on the dataset. Each qualifying system gets its speedup over the reference, and
  Tetris its speedup over every qualifying prior system.

Accuracy-matched rule (p203, the rule the manuscript quotes)
  The abstract, introduction, Section 7, and conclusion quote compareAccuracyMatchedSpeedup*,
  which pairs Tetris's point at the bound with the fastest prior point meeting the bound when
  that point is not more accurate than Tetris's, and otherwise with the most accurate
  Pareto-optimal prior point whose HOTA is not above Tetris's, even when it misses the bound.

Dominance rule (secondary, the feedback script's original rule)
  For each prior system's chosen point, the fastest Tetris configuration at least as
  accurate as that point.

Operator ablation (ablation_speedup.json)
  The R7.O1 sentence "adding polyomino pruning on top of relevance classification and packing
  raises the average speedup over the reference pipeline from 6.7x to 11.1x (within the 5%
  HOTA-loss bound)" quotes the Section 7.2 macros ablationOneSpeedupAvg and
  ablationTwoSpeedupAvg, written by evaluation/p204_ablation_stats.py. That script restricts
  Tetris's test configurations by evaluation.ablation.ABLATION_CONDITIONS (no_both: s = 1 and no
  polyomino pruning; no_sampling: s = 1 with pruning; full: every knob), takes each dataset's
  fastest configuration within the bound, and averages its speedup over datasets.

LEAP (provisional)
  LEAP's CalDoT1 and CalDoT2 stat CSVs list every test video twice; p140 counts each video's
  mean run (since 2026-10-26), and the cached runtime is checked here against the raw stat CSV.
  LEAP's converted outputs are longer than the clips and hold same-frame duplicate boxes, so
  its HOTA, and every LEAP and leap_fit number, is provisional until R1.O2 fixes the conversion.

Checks
  Before writing, the port is run against p203's and p204's own functions on the same inputs
  (port_check); a disagreement stops the run. The submitted manuscript's macro values are
  compared as well (manuscript_check), which will differ by design once B3D is re-scored.

Inputs are the test-split tradeoff tables (Tetris and the reference, from p130), the SOTA
tables (OTIF and LEAP, from p142) through measurements.common.system_configs, and the raw LEAP
stat CSVs (checks only) and SOTA outputs under --sota-dir. HOTA is scored against the cached ground truth.
For B3D1-B3D4 that ground truth is automatic, so every B3D number here must be recomputed
once R1.O1's human-verified labels replace it.

Writes measurements-results/r7o1_hota_matched_speedup/:
  matched_speedup.json   operating points, ranges, the accuracy-matched and dominance rules,
                         LEAP checks, and the p203 port and manuscript checks
  ablation_speedup.json  per bound and ablation condition, per-dataset speedups, mean and
                         maximum, candidate counts, and the p204 port and manuscript checks
"""
import argparse
import math
import os
from datetime import datetime, timezone

import duckdb
import pandas as pd

from evaluation import p203_compare_stats as p203
from evaluation import p204_ablation_stats as p204
from evaluation.ablation import ABLATION_CONDITIONS
from evaluation.p140_otif_transform import build_expected_test_video_manifest, dataset_name_in_sota
from polyis.leap_results import canonical_leap_result
from measurements.common import (DATASET_LABELS, PRIOR_SYSTEMS, accuracy_matched, arrange_matched,
                                 assert_files_exist, bound_specs, choose, file_record, load_test_tradeoff,
                                 matched_stats, operating_points, plain, print_operating_points, query_dicts,
                                 results_dir, rounded, system_configs, write_json)
from polyis.io import cache
from polyis.utilities import get_video_frame_count


# Systems compared at every bound, Tetris first.
SYSTEMS = ['tetris', *PRIOR_SYSTEMS]
# Submitted-manuscript values in tetris-paper/figures/generated/p203_compare_stats.tex, per relative bound.
MANUSCRIPT_COMPARE = {
    '0.05': {'comparePolytrisMeetFivePct': 7, 'comparePriorMeetFivePct': 4, 'comparePriorFailDatasetsFivePct': 3,
             'compareAccuracyMatchedPriorCountFivePct': 7,
             'compareSpeedupMinFivePct': 2.9, 'compareSpeedupMaxFivePct': 16.5,
             'compareAccuracyMatchedSpeedupMinFivePct': 1.5, 'compareAccuracyMatchedSpeedupMaxFivePct': 17.4,
             'compareNaiveSpeedupMinFivePct': 4.7, 'compareNaiveSpeedupMaxFivePct': 68.8},
    '0.10': {'comparePolytrisMeetTenPct': 7, 'comparePriorMeetTenPct': 7, 'comparePriorFailDatasetsTenPct': 0,
             'compareAccuracyMatchedPriorCountTenPct': 7,
             'compareSpeedupMinTenPct': 3.4, 'compareSpeedupMaxTenPct': 25.9,
             'compareAccuracyMatchedSpeedupMinTenPct': 1.6, 'compareAccuracyMatchedSpeedupMaxTenPct': 24.7,
             'compareNaiveSpeedupMinTenPct': 10.4, 'compareNaiveSpeedupMaxTenPct': 91.2},
}
# p203's summary columns and the macro prefix each is written under (p203.save_tex_macros keeps this list local).
P203_FIELDS = {'polytris_meet_count': 'comparePolytrisMeet', 'prior_meet_count': 'comparePriorMeet',
               'prior_fail_count': 'comparePriorFailDatasets', 'accuracy_matched_prior_count': 'compareAccuracyMatchedPriorCount',
               'speedup_min_x': 'compareSpeedupMin', 'speedup_max_x': 'compareSpeedupMax',
               'accuracy_matched_speedup_min_x': 'compareAccuracyMatchedSpeedupMin',
               'accuracy_matched_speedup_max_x': 'compareAccuracyMatchedSpeedupMax',
               'naive_speedup_min_x': 'compareNaiveSpeedupMin', 'naive_speedup_max_x': 'compareNaiveSpeedupMax'}
# Which rule each live manuscript sentence quotes.
MANUSCRIPT_RULES = {
    'compareAccuracyMatchedSpeedup{Min,Max}{FivePct,TenPct}': (
        'accuracy_matched (abstract l.25, introduction l.106-110 and l.138, Section 7 l.208-211, conclusion l.25). '
        'Its matched prior point may miss the bound (accuracy_matched.<bound>.per_dataset.*.prior_meets_bound).'),
    'compareNaiveSpeedup{Min,Max}{FivePct,TenPct}': 'ranges.<bound>.systems.tetris (same sentences).',
    'comparePriorFailDatasetsFivePct, comparePriorMeetFivePct': (
        'ranges.<bound>.prior_fail and prior_meets (abstract l.21, introduction l.104 and l.137, Section 7 l.202, '
        'conclusion l.24).'),
    'compareSpeedup{Min,Max}{FivePct,TenPct}': ('ranges.<bound>.tetris_vs["fastest prior"], the shared-bound rule; '
                                                'no live sentence quotes it.'),
}
# Submitted-manuscript values in tetris-paper/figures/generated/p204_ablation_stats.tex (5% bound).
# p204 prints "CalDoT 1"; it is written here with this script's label.
MANUSCRIPT_ABLATION = {
    'ablationOneSpeedupAvg': 6.7, 'ablationOneSpeedupMax': 12.2, 'ablationOneSpeedupMaxDataset': 'CalDoT1',
    'ablationTwoSpeedupAvg': 11.1, 'ablationTwoSpeedupMax': 17.9, 'ablationTwoSpeedupMaxDataset': 'Amsterdam',
    'ablationThreeSpeedupAvg': 26.7, 'ablationThreeSpeedupMax': 68.8, 'ablationThreeSpeedupMaxDataset': 'Amsterdam',
}
# p204's macro prefix for each ablation condition.
ABLATION_MACROS = {'no_both': 'ablationOne', 'no_sampling': 'ablationTwo', 'full': 'ablationThree'}
# The ablation's bound in Section 7.2 and in the R7.O1 sentence.
ABLATION_BOUND = '0.05'
# Ground-truth caveat recorded in every output.
B3D_NOTE = ('B3D1-B3D4 HOTA is scored against automatic ground truth (tiled RetinaNet detections tracked by '
            'OC-SORT, Section 7.1). Rerun this script after R1.O1 replaces it with human-verified labels. '
            'CalDoT1, CalDoT2, and Amsterdam use OTIF\'s detection labels (tracks linked by OC-SORT) and are '
            'outside R1.O1.')
# Why LEAP's numbers are provisional.
LEAP_NOTE = ('Provisional. LEAP\'s HOTA comes from p140_otif_transform.transform_tracking_json, which sets frame = list '
             'index; LEAP\'s outputs are longer than the clips and hold many same-frame identical boxes '
             '(leap_checks.output_alignment), so the conversion is unverified (R1.O2, "document the HOTA conversion"). '
             'Do not quote any LEAP or leap_fit number, or "LEAP misses the 5%/10% bound", until the conversion is '
             'fixed and rescored. LEAP\'s runtime is p140\'s per-video mean, checked against the raw stat CSV '
             '(leap_checks.runtime).')
# Schema of one raw SOTA output file: a list of frames, each a list of boxes (other fields are ignored).
OUTPUT_SCHEMA = '[[{"left": "DOUBLE", "top": "DOUBLE", "right": "DOUBLE", "bottom": "DOUBLE"}]]'


def parse_args():
    parser = argparse.ArgumentParser(description='R7.O1: matched-HOTA speedups of Tetris, OTIF, and LEAP, and the operator ablation')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS))
    parser.add_argument('--losses', nargs='+', type=float, default=[0.05, 0.10])
    parser.add_argument('--sota-dir', default='/sota-results')
    return parser.parse_args()


def ordered(datasets: list[str]) -> list[str]:
    """Datasets in p203/p204's order (CONFIG['EXEC']['DATASETS']), then any others in the given order."""
    # p204 breaks ties for the maximum by this order (pandas idxmax keeps the first).
    return [d for d in p203.DATASETS if d in datasets] + [d for d in datasets if d not in p203.DATASETS]


def create_dataset_order(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Table dataset_order(dataset, ord): each dataset label's position in p204's order, for explicit tie-breaking."""
    # Replace any table from a previous call.
    con.execute('CREATE OR REPLACE TABLE dataset_order (dataset VARCHAR, ord INTEGER)')
    # One row per dataset label; labels missing from the table sort last, then by name.
    rows = [(DATASET_LABELS[d], i) for i, d in enumerate(ordered(datasets))]
    # An empty insert is a DuckDB error.
    if rows:
        con.executemany('INSERT INTO dataset_order VALUES (?, ?)', rows)


def load_points(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Table hota_points(dataset, system, config, fps, hota, accuracy): every system's test configurations as p203 reads them."""
    # Fail on dataset names the manuscript does not evaluate.
    unknown = set(datasets) - set(DATASET_LABELS)
    assert not unknown, f'unknown datasets {sorted(unknown)}; choose from {sorted(DATASET_LABELS)}'
    # The tie-breaking order for maxima.
    create_dataset_order(con, datasets)
    # Replace any table from a previous call.
    con.execute('CREATE OR REPLACE TABLE hota_points (dataset VARCHAR, system VARCHAR, config VARCHAR, '
                'fps DOUBLE, hota DOUBLE, accuracy DOUBLE)')
    # One row per configuration of the reference, Tetris, OTIF, and LEAP, keyed by the manuscript's dataset label.
    con.executemany('INSERT INTO hota_points VALUES (?, ?, ?, ?, ?, ?)',
                    [(DATASET_LABELS[d], r['system'], r['config'], r['fps'], r['hota'], r['hota'])
                     for d in datasets for r in system_configs(d)])
    # A NaN HOTA sorts above every number in DuckDB and would win every bound, so every value must be finite.
    bad = query_dicts(con, """
        SELECT dataset, system, config, fps, hota FROM hota_points
        WHERE hota IS NULL OR isnan(hota) OR hota < 0 OR hota > 1 OR fps IS NULL OR isnan(fps) OR isinf(fps) OR fps <= 0
    """)
    assert not bad, f'invalid HOTA or throughput: {bad}'
    # Configuration names identify points in the report, so they must be unique per dataset and system.
    dup = query_dicts(con, 'SELECT dataset, system, config, count(*) AS n FROM hota_points GROUP BY ALL HAVING n > 1')
    assert not dup, f'duplicate configurations: {dup}'
    # Every dataset needs one reference row and at least one configuration of every compared system.
    gaps = query_dicts(con, """
        SELECT d.dataset, s.system, count(p.config) AS n
        FROM (SELECT unnest($datasets) AS dataset) d CROSS JOIN (SELECT unnest($systems) AS system) s
        LEFT JOIN hota_points p ON p.dataset = d.dataset AND p.system = s.system
        GROUP BY ALL HAVING (s.system = 'reference' AND n <> 1) OR n = 0
    """, {'datasets': [DATASET_LABELS[d] for d in datasets], 'systems': ['reference', *SYSTEMS]})
    assert not gaps, f'missing or repeated test rows: {gaps}'


def load_tetris_knobs(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Tables tetris_knobs (Tetris configurations with their classifier, sample rate, and pruning threshold) and frame_counts.

    Fails on Perfect (oracle) classifier rows, which p203 and p204 drop and system_configs does not, and on any
    disagreement with hota_points, which reads the same tradeoff tables.
    """
    # Replace any tables from a previous call.
    con.execute('CREATE OR REPLACE TABLE tetris_knobs (dataset VARCHAR, config VARCHAR, classifier VARCHAR, '
                'sample_rate DOUBLE, threshold DOUBLE, fps DOUBLE, hota DOUBLE)')
    con.execute('CREATE OR REPLACE TABLE frame_counts (dataset VARCHAR, frames DOUBLE)')
    # Load each dataset's Tetris knobs and the test split's frame count.
    for d in datasets:
        # The same Tetris test rows and reference row that system_configs reads.
        tetris, naive = load_test_tradeoff(d)
        # Copy the knob columns through DuckDB (pandas NaN, a configuration without pruning, arrives as NULL).
        con.register('tradeoff_rows', tetris)
        con.execute("""
            INSERT INTO tetris_knobs
            SELECT ?, CAST(variant_id AS VARCHAR), classifier, sample_rate, tracking_accuracy_threshold, throughput_fps, HOTA_HOTA
            FROM tradeoff_rows
        """, [DATASET_LABELS[d]])
        con.unregister('tradeoff_rows')
        # The test frame count, which turns prior systems' runtimes into throughput.
        con.execute('INSERT INTO frame_counts VALUES (?, ?)', [DATASET_LABELS[d], float(naive['frame_count'])])
    # Fail on oracle-classifier rows before anything is written.
    perfect = query_dicts(con, "SELECT dataset, count(*) AS n FROM tetris_knobs WHERE classifier = 'Perfect' GROUP BY dataset")
    assert not perfect, f'Perfect-classifier rows in the test tradeoff tables (p203 and p204 drop them): {perfect}'
    # Rows in one table and not the other; both come from the same file, so this is empty.
    diff = query_dicts(con, """
        (SELECT dataset, config, fps, hota FROM tetris_knobs
         EXCEPT SELECT dataset, config, fps, hota FROM hota_points WHERE system = 'tetris')
        UNION ALL
        (SELECT dataset, config, fps, hota FROM hota_points WHERE system = 'tetris'
         EXCEPT SELECT dataset, config, fps, hota FROM tetris_knobs)
    """)
    assert not diff, f'Tetris rows differ between system_configs and load_test_tradeoff: {diff}'


def clip_manifest(dataset: str) -> list[dict]:
    """The dataset's test clips: the SOTA numeric video id, the file name, and the clip's frame count."""
    # p140's manifest maps te01.mp4 to SOTA video id 1, as the SOTA exports name their files.
    manifest = build_expected_test_video_manifest(dataset)
    # Frame counts come from the clip files themselves (read-only).
    return [dict(video_id=int(i), video=str(v), frames=get_video_frame_count(dataset, str(v)))
            for i, v in zip(manifest['video_id'], manifest['video'])]


def load_clips(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Table clips(dataset, raw, video_id, video, frames): every test clip, with the SOTA export's dataset name."""
    # Replace any table from a previous call.
    con.execute('CREATE OR REPLACE TABLE clips (dataset VARCHAR, raw VARCHAR, video_id INTEGER, video VARCHAR, frames DOUBLE)')
    # One row per test clip.
    con.executemany('INSERT INTO clips VALUES (?, ?, ?, ?, ?)',
                    [(DATASET_LABELS[d], dataset_name_in_sota(d), c['video_id'], c['video'], float(c['frames']))
                     for d in datasets for c in clip_manifest(d)])
    # The clips must add up to the frame count that prior systems' throughput is divided by.
    bad = query_dicts(con, """
        SELECT f.dataset, f.frames, sum(c.frames) AS clip_frames FROM frame_counts f LEFT JOIN clips c USING (dataset)
        GROUP BY ALL HAVING clip_frames IS NULL OR clip_frames <> f.frames
    """)
    assert not bad, f'test clips do not add up to the tradeoff frame count: {bad}'


def check_leap_runtime(con: duckdb.DuckDBPyConnection, sota_dir: str) -> dict:
    """LEAP's cached test runtime checked against its raw stat CSV; returns {dataset: the runtime record} for the JSON.

    Since 2026-10-26, p140 normalize_leap_stat_csv takes each test video's inference_total - decode averaged over
    its rows (CalDoT1 and CalDoT2 list every video twice, two runs) and sums over the videos, instead of the CSV's
    'total' row, which summed both runs. The same sum is recomputed here from the raw CSV, and the cached LEAP
    throughput in hota_points (from p142) must equal the test frame count over it, or the run stops.
    """
    # Replace any table from a previous call.
    con.execute('CREATE OR REPLACE TABLE leap_stat_rows (dataset VARCHAR, video_name VARCHAR, runtime DOUBLE, '
                'frames_sampled DOUBLE, frames_filtered DOUBLE)')
    # Each dataset's raw LEAP stat CSV, the export p140 reads.
    raw = query_dicts(con, 'SELECT DISTINCT dataset, raw FROM clips ORDER BY dataset')
    paths = {r['dataset']: os.path.join(sota_dir, r['raw'], f"leap_{r['raw']}.csv") for r in raw}
    assert_files_exist(paths.values(), 'raw LEAP stat CSVs')
    # Read each dataset's CSV.
    for dataset, path in paths.items():
        # LEAP's runtime per row excludes decoding, as in p140.
        con.execute("""
            INSERT INTO leap_stat_rows
            SELECT ?, CAST(video_name AS VARCHAR), inference_total - decode, frames_sampled, frames_filtered FROM read_csv(?)
        """, [dataset, path])
    # Every row other than 'total' must name a numeric video id.
    bad = query_dicts(con, "SELECT dataset, video_name FROM leap_stat_rows WHERE video_name <> 'total' AND try_cast(video_name AS INTEGER) IS NULL")
    assert not bad, f'LEAP stat rows without a video id: {bad}'
    # Every run needs a positive runtime once decoding is subtracted.
    bad = query_dicts(con, "SELECT dataset, video_name, runtime FROM leap_stat_rows WHERE video_name <> 'total' AND (runtime IS NULL OR runtime <= 0)")
    assert not bad, f'LEAP stat rows with a non-positive runtime: {bad}'
    # One runtime per video: the mean over its repeated runs, and how far the runs disagree.
    con.execute("""
        CREATE OR REPLACE TABLE leap_video_runtime AS
        SELECT dataset, CAST(video_name AS INTEGER) AS video_id, count(*) AS runs, avg(runtime) AS runtime,
               (max(runtime) - min(runtime)) / nullif(min(runtime), 0) AS spread,
               count(DISTINCT frames_sampled) = 1 AND count(DISTINCT frames_filtered) = 1 AS same_frames
        FROM leap_stat_rows WHERE video_name <> 'total' GROUP BY ALL
    """)
    # Exactly the test videos: a missing video would understate the runtime, an extra one overstate it.
    mismatch = query_dicts(con, """
        SELECT coalesce(c.dataset, v.dataset) AS dataset, coalesce(c.video_id, v.video_id) AS video_id,
               CASE WHEN v.video_id IS NULL THEN 'no stat row' ELSE 'not a test video' END AS problem
        FROM clips c FULL JOIN leap_video_runtime v ON c.dataset = v.dataset AND c.video_id = v.video_id
        WHERE c.video_id IS NULL OR v.video_id IS NULL ORDER BY ALL
    """)
    assert not mismatch, f'LEAP stat rows do not match the test videos: {mismatch}'
    # The stat CSV describes LEAP's single configuration.
    multi = query_dicts(con, "SELECT dataset, count(*) AS n FROM hota_points WHERE system = 'leap' GROUP BY dataset HAVING n <> 1")
    assert not multi, f'LEAP has more than one configuration: {multi}'
    # Per dataset: the per-video runtimes summed, the raw 'total' row, and the runtime behind the cached throughput.
    rows = query_dicts(con, """
        WITH v AS (
            SELECT dataset, count(*) AS videos, sum(runs) AS stat_rows, max(runs) AS max_runs_per_video,
                   max(spread) AS max_run_spread, bool_and(same_frames) AS same_frames_across_runs, sum(runtime) AS runtime
            FROM leap_video_runtime GROUP BY dataset
        ),
        t AS (SELECT dataset, sum(runtime) AS total_row_runtime FROM leap_stat_rows WHERE video_name = 'total' GROUP BY dataset)
        SELECT v.*, t.total_row_runtime, p.fps, f.frames / p.fps AS cached_runtime
        FROM v JOIN frame_counts f USING (dataset) LEFT JOIN t USING (dataset)
        JOIN hota_points p ON p.dataset = v.dataset AND p.system = 'leap'
        ORDER BY v.dataset
    """)
    # Datasets whose LEAP was replaced by a promoted rerun (measurements-results/leap_canonical.json) are timed
    # by that rerun, which the cached throughput uses; the raw stat CSV describes the replaced run.
    canonical = {DATASET_LABELS[d]: result['compute_s'] for d in DATASET_LABELS
                 if (result := canonical_leap_result(d)) is not None}
    for r in rows:
        r['runtime_source'] = 'canonical rerun' if r['dataset'] in canonical else 'raw stat CSV'
        r['expected_runtime'] = canonical.get(r['dataset'], r['runtime'])
    # The cached runtime must be the expected one (summation orders differ, so up to 1e-9 relative).
    bad = [r for r in rows if not math.isclose(r['cached_runtime'], r['expected_runtime'], rel_tol=1e-9)]
    assert not bad, f'cached LEAP runtime differs from its source (canonical rerun, else raw stat CSV): {bad}'
    # Arrange for the JSON, with runtimes in seconds and throughput rounded as elsewhere.
    return {r['dataset']: dict(
        videos=r['videos'], stat_rows=r['stat_rows'], max_runs_per_video=r['max_runs_per_video'],
        max_run_spread=round(r['max_run_spread'], 4), same_frames_across_runs=r['same_frames_across_runs'],
        total_row_runtime_s=rounded(r['total_row_runtime'], 4),
        runtime_s=round(r['expected_runtime'], 4), runtime_source=r['runtime_source'],
        raw_stat_runtime_s=round(r['runtime'], 4), fps=round(r['fps'], 1)) for r in rows}


def output_alignment(con: duckdb.DuckDBPyConnection, sota_dir: str) -> dict:
    """Per dataset, prior system, and configuration: raw output length against the clip and same-frame duplicate boxes.

    p140 converts each raw output by list index (frame = index). OTIF's lists hold a null entry per
    skipped frame and end at the last processed frame, so a shorter list stays aligned; a list longer
    than its clip cannot be indexed by clip frame. A duplicate is a detection whose box repeats another
    detection's box in the same frame.
    """
    # Every raw output file the converted HOTA came from: LEAP's flat directory and OTIF's per-parameter ones.
    con.execute("""
        CREATE OR REPLACE TABLE output_files AS
        SELECT c.dataset, p.system, p.config, c.video_id, c.frames,
               CASE WHEN p.system = 'leap' THEN concat($root, '/', c.raw, '/leap/', c.video_id, '.json')
                    ELSE concat($root, '/', c.raw, '/', replace(p.config, 'param ', ''), '/', c.video_id, '.json') END AS path
        FROM clips c JOIN (SELECT DISTINCT dataset, system, config FROM hota_points WHERE system <> 'reference' AND system <> 'tetris') p
        USING (dataset)
    """, {'root': sota_dir.rstrip('/')})
    # The file paths, for one read.
    paths = [r[0] for r in con.execute('SELECT path FROM output_files ORDER BY path').fetchall()]
    # A missing file is an error with a clear message rather than a DuckDB IO error.
    assert_files_exist(paths, 'raw SOTA outputs')
    # Entries, detections, and same-frame duplicate boxes per file; null files and frames count as empty.
    rows = query_dicts(con, f"""
        WITH f AS (SELECT filename, from_json(content, '{OUTPUT_SCHEMA}') AS frames FROM read_text($paths)),
        per_file AS (
            SELECT filename, coalesce(len(frames), 0) AS entries,
                   coalesce(len(list_filter(frames, lambda x: x IS NULL)), 0) AS null_entries,
                   coalesce(list_sum(list_transform(frames, lambda x: coalesce(len(x), 0))), 0) AS detections,
                   coalesce(list_sum(list_transform(frames, lambda x: coalesce(len(x) - len(list_distinct(
                       list_transform(x, lambda d: [d.left, d.top, d.right, d.bottom]))), 0))), 0) AS duplicates
            FROM f
        )
        SELECT o.dataset, o.system, o.config, count(*) AS videos,
               count(*) FILTER (WHERE p.entries > o.frames) AS videos_longer_than_clip,
               count(*) FILTER (WHERE p.entries < o.frames) AS videos_shorter_than_clip,
               max(p.entries - o.frames) AS max_extra_entries, min(p.entries - o.frames) AS min_extra_entries,
               sum(p.entries) AS entries, sum(p.null_entries) AS null_entries,
               sum(p.detections) AS detections, sum(p.duplicates) AS duplicate_boxes,
               sum(p.duplicates) / nullif(sum(p.detections), 0) AS duplicate_share
        FROM output_files o JOIN per_file p ON p.filename = o.path
        GROUP BY ALL ORDER BY ALL
    """, {'paths': paths})
    # Nest as {dataset: {system: {config: counts}}}.
    out: dict = {}
    for r in rows:
        share = r.pop('duplicate_share')
        out.setdefault(r.pop('dataset'), {}).setdefault(r.pop('system'), {})[r.pop('config')] = dict(
            r, duplicate_share=rounded(share, 4))
    return out


def macro_suffix(loss: float | None) -> str | None:
    """p203's macro suffix for a relative loss (FivePct for 0.05), or None when p203 names no macro for it."""
    # Fit bounds have no p203 macro.
    if loss is None:
        return None
    # p203 names whole percentages from 1 to 10.
    percent = round(loss * 100)
    return p203.THRESHOLD_SUFFIX_BY_PERCENT.get(percent) if math.isclose(loss * 100, percent) else None


def speedup_stats(con: duckdb.DuckDBPyConnection, chosen: str) -> list[dict]:
    """Per bound and system, datasets meeting the bound and the min, max, mean, and argmax of speedup over the reference.

    The argmax breaks ties by p204's dataset order (table dataset_order), then by name.
    """
    # Aggregates over qualifying datasets, and the top dataset by an explicit order.
    return query_dicts(con, f"""
        WITH s AS (
            SELECT bound, system, count(*) AS datasets, count(*) FILTER (WHERE meets_bound) AS meets_bound,
                   min(fps / reference_fps) FILTER (WHERE meets_bound) AS min,
                   max(fps / reference_fps) FILTER (WHERE meets_bound) AS max,
                   avg(fps / reference_fps) FILTER (WHERE meets_bound) AS mean
            FROM {chosen} GROUP BY bound, system
        ),
        top AS (
            SELECT c.bound, c.system, c.dataset AS max_dataset FROM {chosen} c LEFT JOIN dataset_order o USING (dataset)
            WHERE c.meets_bound
            QUALIFY row_number() OVER (PARTITION BY c.bound, c.system
                                       ORDER BY c.fps / c.reference_fps DESC, o.ord NULLS LAST, c.dataset) = 1
        )
        SELECT s.*, top.max_dataset FROM s LEFT JOIN top USING (bound, system) ORDER BY s.bound, s.system
    """)


def ratio_stats(con: duckdb.DuckDBPyConnection, chosen: str) -> list[dict]:
    """Per bound, Tetris over each prior system and over the fastest qualifying prior, where both qualify.

    The 'fastest prior' row matches p203's compareSpeedup macros; its prior_meets counts datasets where
    any prior system qualifies, and prior_fail those where none does.
    """
    # Ratios on datasets where Tetris and the prior both qualify; prior counts over every dataset of the bound.
    return query_dicts(con, f"""
        WITH t AS (SELECT bound, dataset, fps FROM {chosen} WHERE system = 'tetris' AND meets_bound),
        p AS (SELECT bound, dataset, system, fps FROM {chosen} WHERE system <> 'tetris' AND meets_bound),
        fastest AS (SELECT bound, dataset, 'fastest prior' AS system, max(fps) AS fps FROM p GROUP BY bound, dataset),
        datasets AS (SELECT bound, count(DISTINCT dataset) AS n FROM {chosen} GROUP BY bound),
        ratios AS (
            -- Tetris over each prior system and over the fastest one, on datasets where both qualify.
            SELECT t.bound, q.system, t.fps / q.fps AS ratio
            FROM t JOIN (SELECT * FROM p UNION ALL SELECT * FROM fastest) q USING (bound, dataset)
        ),
        prior_meets AS (SELECT bound, count(*) AS n FROM fastest GROUP BY bound)
        SELECT d.bound, r.system, count(r.ratio) AS datasets, min(r.ratio) AS min, max(r.ratio) AS max,
               coalesce(any_value(m.n), 0) AS prior_meets, any_value(d.n) - coalesce(any_value(m.n), 0) AS prior_fail
        FROM datasets d LEFT JOIN ratios r USING (bound) LEFT JOIN prior_meets m USING (bound)
        GROUP BY d.bound, r.system ORDER BY d.bound, r.system
    """)


def dominance(con: duckdb.DuckDBPyConnection, points: str, chosen: str) -> list[dict]:
    """Per bound, dataset, and qualifying prior point, the fastest Tetris configuration at least as accurate as it.

    Ties go to the more accurate configuration, then the smaller name; tetris_config is NULL when none qualifies.
    """
    # Each qualifying prior point joined with every Tetris configuration at least as accurate, fastest first.
    return query_dicts(con, f"""
        WITH prior AS (
            SELECT bound, dataset, system, config, fps, accuracy, reference_fps FROM {chosen}
            WHERE system <> 'tetris' AND meets_bound
        ),
        matched AS (
            SELECT p.bound, p.dataset, p.system, t.config, t.fps, t.accuracy
            FROM prior p JOIN {points} t ON t.dataset = p.dataset AND t.system = 'tetris' AND t.accuracy >= p.accuracy
            QUALIFY row_number() OVER (PARTITION BY p.bound, p.dataset, p.system
                                       ORDER BY t.fps DESC, t.accuracy DESC, t.config) = 1
        )
        SELECT p.bound, p.dataset, p.system AS prior, p.config AS prior_config, p.fps AS prior_fps,
               p.accuracy AS prior_hota, p.reference_fps, m.config AS tetris_config, m.fps AS tetris_fps,
               m.accuracy AS tetris_hota
        FROM prior p LEFT JOIN matched m USING (bound, dataset, system)
        ORDER BY p.bound, p.dataset, p.system
    """)


def arrange_ranges(speedups: list[dict], ratios: list[dict]) -> dict:
    """{bound: {systems, tetris_vs, prior_meets, prior_fail}} from speedup_stats and ratio_stats rows."""
    # The report, filled below.
    out: dict = {}
    # Speedup over the reference per system, over the datasets where it qualifies.
    for s in speedups:
        out.setdefault(s['bound'], dict(systems={}, tetris_vs={}))['systems'][s['system']] = dict(
            datasets=s['datasets'], meets_bound=s['meets_bound'], speedup_vs_reference_min=rounded(s['min'], 1),
            speedup_vs_reference_max=rounded(s['max'], 1))
    # Tetris over each prior and over the fastest prior, where both qualify.
    for r in ratios:
        entry = out.setdefault(r['bound'], dict(systems={}, tetris_vs={}))
        entry.update(prior_meets=r['prior_meets'], prior_fail=r['prior_fail'])
        # A bound without any ratio has one row with a NULL system.
        if r['system'] is not None:
            entry['tetris_vs'][r['system']] = dict(datasets=r['datasets'], min=rounded(r['min'], 1), max=rounded(r['max'], 1))
    return out


def arrange_dominance(rows: list[dict]) -> dict:
    """{bound: {dataset: {prior: {prior point, Tetris point or None}}}} from dominance rows."""
    # The report, filled below.
    out: dict = {}
    # One entry per bound, dataset, and qualifying prior point.
    for r in rows:
        # The prior system's chosen point.
        prior = dict(config=r['prior_config'], hota=round(r['prior_hota'], 4), fps=round(r['prior_fps'], 1),
                     speedup_vs_reference=rounded(r['prior_fps'] / r['reference_fps'], 1))
        # The matching Tetris point, or None when no configuration is at least as accurate.
        tetris = None if r['tetris_config'] is None else dict(
            config=r['tetris_config'], hota=round(r['tetris_hota'], 4), fps=round(r['tetris_fps'], 1),
            speedup_vs_reference=rounded(r['tetris_fps'] / r['reference_fps'], 1),
            speedup_vs_prior=rounded(r['tetris_fps'] / r['prior_fps'], 1))
        # Nest under bound and dataset.
        out.setdefault(r['bound'], {}).setdefault(r['dataset'], {})[r['prior']] = dict(prior=prior, tetris=tetris)
    return out


def macro_values(speedups: list[dict], ratios: list[dict], matched: list[dict], specs: dict[str, dict]) -> dict:
    """{bound: {p203 macro: unrounded value}} for every relative bound p203 names a macro for."""
    # Index the three row sets by bound (and system).
    speedup = {(s['bound'], s['system']): s for s in speedups}
    ratio = {(r['bound'], r['system']): r for r in ratios}
    prior = {r['bound']: r for r in ratios}
    match = {m['bound']: m for m in matched}
    # The report, filled below.
    out = {}
    # One macro set per named relative bound.
    for name, spec in specs.items():
        # Skip bounds without a p203 macro name.
        suffix = macro_suffix(spec['loss'])
        if suffix is None:
            continue
        # The rows for this bound; a missing row means nothing qualified.
        tetris, fastest = speedup.get((name, 'tetris'), {}), ratio.get((name, 'fastest prior'), {})
        counts, m = prior.get(name, {}), match.get(name, {})
        out[name] = {
            f'comparePolytrisMeet{suffix}': tetris.get('meets_bound', 0),
            f'comparePriorMeet{suffix}': counts.get('prior_meets', 0),
            f'comparePriorFailDatasets{suffix}': counts.get('prior_fail'),
            f'compareAccuracyMatchedPriorCount{suffix}': m.get('matched', 0),
            f'compareSpeedupMin{suffix}': fastest.get('min'), f'compareSpeedupMax{suffix}': fastest.get('max'),
            f'compareAccuracyMatchedSpeedupMin{suffix}': m.get('min'),
            f'compareAccuracyMatchedSpeedupMax{suffix}': m.get('max'),
            f'compareNaiveSpeedupMin{suffix}': tetris.get('min'), f'compareNaiveSpeedupMax{suffix}': tetris.get('max'),
        }
    return out


def p203_reference(datasets: list[str], specs: dict[str, dict]) -> dict:
    """{bound: {macro: value}} from evaluation/p203_compare_stats.py's own functions on the same inputs."""
    # The relative bounds p203 names a macro for.
    named = {name: spec['loss'] for name, spec in specs.items() if macro_suffix(spec['loss']) is not None}
    # Nothing to compare without one.
    if not named:
        return {}
    # p203's pipeline from its main(): test rows, prior throughput from the shared frame count, Pareto fronts, summary.
    polytris, naive = p203.load_polytris_and_naive_test_data(datasets)
    priors = p203.load_prior_test_data(datasets, p203.build_frame_count_lookup(polytris, naive))
    summary, _ = p203.build_threshold_reports(datasets, p203.filter_pareto_by_dataset(polytris), naive, priors,
                                              list(named.values()))
    # The report, filled below.
    out = {}
    # One macro set per bound, named as p203.save_tex_macros names them.
    for name, loss in named.items():
        row = summary[summary['threshold'] == loss].iloc[0]
        out[name] = {f'{prefix}{macro_suffix(loss)}': plain(row[column]) for column, prefix in P203_FIELDS.items()}
    return out


def p204_reference(datasets: list[str], specs: dict[str, dict]) -> dict:
    """{bound: {ablation macro: value}} from evaluation/p204_ablation_stats.py's own functions on the same inputs."""
    # p204's test rows (Perfect dropped).
    polytris, naive = p204.load_polytris_and_naive_test_data(datasets)
    # The report, filled below.
    out = {}
    # p204 runs at one threshold; run it at every relative bound.
    for name, spec in specs.items():
        # Only relative bounds have a p204 threshold.
        if spec['loss'] is None:
            continue
        # Per dataset and condition, the speedup at the bound (NA when nothing qualifies).
        stats = p204.compute_ablation_stats(datasets, polytris, naive, threshold=spec['loss'])
        out[name] = {}
        # Mean, max, and argmax per condition, as p204.save_tex_macros computes them (idxmax keeps the first dataset).
        for condition, prefix in ABLATION_MACROS.items():
            speedup = pd.to_numeric(stats[f'{condition}_speedup'], errors='coerce')
            has = bool(speedup.notna().any())
            out[name].update({
                f'{prefix}SpeedupAvg': float(speedup.mean()) if has else None,
                f'{prefix}SpeedupMax': float(speedup.max()) if has else None,
                f'{prefix}SpeedupMaxDataset': DATASET_LABELS[stats.loc[speedup.idxmax(), 'dataset']] if has else None,
            })
    return out


def same(a, b) -> bool:
    """Equal values; floats are equal up to 1e-9 relative, since summation orders differ."""
    # Floats compare with a tolerance, everything else exactly.
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def check(expected: dict, measured: dict, label: str) -> dict:
    """{name: {<label>: expected, measured, match}} for every expected name."""
    # A value this run could not compute is measured None and matches only an expected None.
    return {k: {label: v, 'measured': measured.get(k), 'match': same(v, measured.get(k))} for k, v in expected.items()}


def mismatches(checks: dict) -> list[str]:
    """'bound/name' for every failed entry of {bound: check(...)}."""
    # Flatten the nested checks.
    return [f'{b}/{k}' for b, entries in checks.items() for k, v in entries.items() if not v['match']]


def ablation_points(con: duckdb.DuckDBPyConnection):
    """Table ablation_points(dataset, system, config, fps, hota, accuracy), system being an ablation condition.

    A Tetris configuration appears once under every condition whose sample-rate and pruning-threshold
    restriction it satisfies (evaluation.ablation.filter_by_ablation_condition, in SQL); each dataset
    also has its reference row, taken from hota_points.
    """
    # The conditions as data: NULL lists mean unrestricted; include_null admits configurations without pruning.
    con.execute('CREATE OR REPLACE TABLE conditions (name VARCHAR, sample_rates DOUBLE[], thresholds DOUBLE[], include_null BOOLEAN)')
    con.executemany('INSERT INTO conditions VALUES (?, ?, ?, ?)', [
        (c.name, c.sample_rates,
         None if c.tracking_accuracy_thresholds is None else [t for t in c.tracking_accuracy_thresholds if t is not None],
         c.tracking_accuracy_thresholds is not None and None in c.tracking_accuracy_thresholds)
        for c in ABLATION_CONDITIONS])
    # Expand each configuration into the conditions it satisfies, then add the reference rows.
    con.execute("""
        CREATE OR REPLACE TABLE ablation_points AS
        SELECT a.dataset, c.name AS system, a.config, a.fps, a.hota, a.hota AS accuracy
        FROM tetris_knobs a JOIN conditions c
          ON (c.sample_rates IS NULL OR list_contains(c.sample_rates, a.sample_rate))
         AND (c.thresholds IS NULL OR list_contains(c.thresholds, a.threshold) OR (c.include_null AND a.threshold IS NULL))
        UNION ALL
        SELECT dataset, system, config, fps, hota, accuracy FROM hota_points WHERE system = 'reference'
    """)


def ablation_report(con: duckdb.DuckDBPyConnection, losses: list[float]) -> tuple[dict, dict]:
    """Per bound, each condition's per-dataset speedup, summary, and candidate counts; and the unrounded p204 macro values."""
    # Conditions in ABLATION_CONDITIONS order, and every dataset of the run.
    names = [c.name for c in ABLATION_CONDITIONS]
    datasets = [r[0] for r in con.execute("SELECT dataset FROM ablation_points WHERE system = 'reference' ORDER BY dataset").fetchall()]
    # Candidate configurations per condition and dataset (the pool each point is chosen from).
    candidates = {(r['system'], r['dataset']): r['n'] for r in query_dicts(
        con, "SELECT system, dataset, count(*) AS n FROM ablation_points WHERE system <> 'reference' GROUP BY ALL")}
    # Fastest configuration per bound, dataset, and condition.
    specs = bound_specs(con, 'ablation_points', losses, None)
    choose(con, 'ablation_points', specs, names, 'ablation_chosen')
    # Every bound and condition reports every dataset, None where nothing qualifies or no candidate exists.
    out = {b: {n: dict(datasets=len(datasets), meets_bound=0, mean_speedup=None, max_speedup=None, max_dataset=None,
                       candidates={d: candidates.get((n, d), 0) for d in datasets}, per_dataset=dict.fromkeys(datasets))
               for n in names} for b in specs}
    macros: dict = {b: {} for b in specs}
    # Summary per bound and condition.
    for s in speedup_stats(con, 'ablation_chosen'):
        out[s['bound']][s['system']].update(meets_bound=s['meets_bound'], mean_speedup=rounded(s['mean'], 1),
                                            max_speedup=rounded(s['max'], 1), max_dataset=s['max_dataset'])
        # The unrounded values p204 writes as macros.
        prefix = ABLATION_MACROS[s['system']]
        macros[s['bound']].update({f'{prefix}SpeedupAvg': s['mean'], f'{prefix}SpeedupMax': s['max'],
                                   f'{prefix}SpeedupMaxDataset': s['max_dataset']})
    # Conditions without any candidate on any dataset still have macro values (all None).
    for b in specs:
        for n in names:
            prefix = ABLATION_MACROS[n]
            macros[b].setdefault(f'{prefix}SpeedupAvg', None)
            macros[b].setdefault(f'{prefix}SpeedupMax', None)
            macros[b].setdefault(f'{prefix}SpeedupMaxDataset', None)
    # Each dataset's chosen configuration per bound and condition.
    for r in query_dicts(con, 'SELECT * FROM ablation_chosen WHERE meets_bound ORDER BY bound, system, dataset'):
        out[r['bound']][r['system']]['per_dataset'][r['dataset']] = dict(
            config=r['config'], hota=round(r['accuracy'], 4), fps=round(r['fps'], 1),
            speedup_vs_reference=rounded(r['fps'] / r['reference_fps'], 1))
    return out, macros


def config_counts(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    """{dataset: {system: number of test configurations}} for the provenance record."""
    # The report, filled below.
    out: dict = {}
    # Count configurations per dataset and system, then nest by dataset.
    for r in query_dicts(con, f'SELECT dataset, system, count(*) AS n FROM {table} GROUP BY ALL ORDER BY ALL'):
        out.setdefault(r['dataset'], {})[r['system']] = r['n']
    return out


def provenance(con: duckdb.DuckDBPyConnection, datasets: list[str], sota_dir: str) -> dict:
    """Run time, every input table's version, and the test rows read, so a copy shows which ground truth it used."""
    # One record per input table per dataset.
    files = {DATASET_LABELS[d]: dict(
        tradeoff=file_record(cache.root(d, 'evaluation', '090_tradeoff', 'tradeoff.csv')),
        otif=file_record(cache.sota('otif', d, 'tradeoff.csv')), leap=file_record(cache.sota('leap', d, 'tradeoff.csv')),
        leap_stat=file_record(os.path.join(sota_dir, dataset_name_in_sota(d), f'leap_{dataset_name_in_sota(d)}.csv')))
        for d in datasets}
    # The run time and the per-system test-row counts.
    return dict(run_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), files=files,
                test_rows=config_counts(con, 'hota_points'))


def main():
    # Command-line options, the result directory, and an in-memory database.
    args = parse_args()
    out_dir = results_dir(__file__)
    con = duckdb.connect()
    # The manuscript checks only apply to the full seven-dataset run.
    all_datasets = set(args.datasets) == set(DATASET_LABELS)

    # Inputs: every system's test configurations as p203 reads them, Tetris's knobs, and the test clips.
    load_points(con, args.datasets)
    load_tetris_knobs(con, args.datasets)
    load_clips(con, args.datasets)
    # LEAP checks: the cached runtime against the raw stat CSV, and the raw outputs against the clips.
    leap_runtime = check_leap_runtime(con, args.sota_dir)
    alignment = output_alignment(con, args.sota_dir)

    # Bounds, relative and at LEAP's best HOTA.
    specs = bound_specs(con, 'hota_points', args.losses, 'leap')
    # The JSON's chosen points, then the same choices as a table.
    points = {name: operating_points(con, 'hota_points', SYSTEMS, **spec) for name, spec in specs.items()}
    choose(con, 'hota_points', specs, SYSTEMS, 'chosen')
    # Shared-bound ranges, the accuracy-matched rule, and their p203 macro values.
    speedups, ratios = speedup_stats(con, 'chosen'), ratio_stats(con, 'chosen')
    accuracy_matched(con, 'hota_points', 'chosen', 'matched')
    matched = matched_stats(con, 'chosen', 'matched')
    macros = macro_values(speedups, ratios, matched, specs)
    # The port against p203's own functions on the same inputs.
    port = {b: check(v, macros.get(b, {}), 'p203') for b, v in p203_reference(ordered(args.datasets), specs).items()}

    # The operator ablation from the same tradeoff tables, and the port against p204's own functions.
    ablation_points(con)
    ablation_bounds, ablation_macros = ablation_report(con, args.losses)
    relative = {name: spec for name, spec in specs.items() if spec['loss'] is not None}
    ablation_port = {b: check(v, ablation_macros.get(b, {}), 'p204')
                     for b, v in p204_reference(ordered(args.datasets), relative).items()}
    # Stop before writing anything if the port disagrees with p203 or p204.
    failed = mismatches(port) + [f'ablation {m}' for m in mismatches(ablation_port)]
    assert not failed, f'port disagrees with p203/p204 on {failed}: {port} {ablation_port}'

    # The submitted manuscript's macros, reproduced from p203's inputs (they change by design after R1.O1).
    manuscript = ({b: check(v, rounded(macros.get(b, {}), 1), 'manuscript') for b, v in MANUSCRIPT_COMPARE.items()}
                  if all_datasets else 'skipped: not all seven datasets')
    # Provenance shared by both reports.
    inputs = provenance(con, args.datasets, args.sota_dir)
    # About the primary report.
    about = dict(
        measured='Each system\'s fastest test-split configuration whose HOTA meets a per-dataset bound, its speedup '
                 'over the reference pipeline, Tetris\'s speedup over each prior system meeting the same bound, and '
                 'p203\'s accuracy-matched speedup.',
        selection='Operating points are chosen by test-split HOTA among the test runs of each system\'s validated '
                  'Pareto configurations (p135), the manuscript\'s method. Candidate pools differ by system '
                  '(configurations): Tetris has tens of configurations per dataset, OTIF about ten, LEAP one.',
        bounds={name: ('HOTA >= reference HOTA x (1 - loss)' if spec['loss'] is not None
                       else 'HOTA >= LEAP\'s best HOTA on the dataset') for name, spec in specs.items()},
        leap_fit='Provisional with LEAP (see leap_status). LEAP\'s best HOTA is far below the other systems\', so '
                 'this bound admits nearly every Tetris and OTIF configuration; Tetris over OTIF there compares '
                 'points of very different HOTA.',
        leap_status=LEAP_NOTE,
        ties='Bounds are inclusive (HOTA >= bound; p203/p204 test round(loss, 12) <= loss bound, which differs only '
             'within 1e-12 of the bound and is checked by port_check). Throughput ties go to the more accurate '
             'configuration, then the smaller config name. Maxima over datasets break ties by p204\'s dataset '
             'order (CONFIG EXEC.DATASETS), as pandas idxmax does.',
        throughput=dict(
            tetris_and_reference='throughput_fps from p130\'s tradeoff table: test frames over the stage runtimes '
                                 'p121 sums, without the ops total_frame_time, read_frame, save_canvas, and '
                                 'save_mapping_files, without read and retrieve in p020 classification, '
                                 'without reference video read/decode, and without save_collage/pack_all_total in p030.',
            otif='runtime_total from OTIF\'s stat CSV (p140 normalize_otif_stat_csv). OTIF\'s per-stage columns '
                 '(seg, detect, track, decode) sum to more than runtime_total, so stages overlap and its decoding '
                 'time cannot be separated out.',
            leap='runtime from p140 (p142\'s table): inference_total - decode per test video from LEAP\'s raw stat '
                 'CSV, so decoding is excluded, with repeated runs of a video averaged; checked against the raw CSV '
                 '(leap_checks.runtime).',
            prior_fps='Prior systems\' throughput is the test split\'s frame count (our clips) over their runtime, as '
                      'in evaluation/p200_compare_compute.py and p203, not the frames the system decoded.'),
        ratios='Speedups and ratios divide unrounded throughputs, then round to 0.1 (common.operating_points).',
        accuracy_matched='p203\'s rule: the fastest prior point meeting the bound if its HOTA is not above Tetris\'s, '
                         'else the most accurate Pareto-optimal prior point with HOTA not above Tetris\'s, which may '
                         'miss the bound (per_dataset.*.prior.meets_bound).',
        dominance='Secondary rule from the feedback script: per prior system meeting the bound, the fastest Tetris '
                  'configuration with HOTA at least that prior point\'s HOTA.',
        manuscript_rules=MANUSCRIPT_RULES,
        checks='port_check: this script\'s values on p203\'s inputs against p203\'s own functions (must match, or the '
               'run stops). manuscript_check: the same values rounded to 0.1 against the submitted macros. '
               'leap_checks.runtime: the cached LEAP runtime must equal the raw stat CSV\'s per-video means, summed '
               '(or the run stops).',
        configurations=config_counts(con, 'hota_points'), inputs=inputs, ground_truth=B3D_NOTE,
        estimates='LEAP\'s CalDoT1 and CalDoT2 runtimes are the mean of two recorded runs per video (an estimate of '
                  'one run). Every other number is measured.')
    # The primary report.
    report = dict(about=about, provisional=dict(systems=['leap'], bounds=['leap_fit'], reason=LEAP_NOTE),
                  leap_checks=dict(runtime=leap_runtime, output_alignment=alignment),
                  operating_points=points, ranges=arrange_ranges(speedups, ratios),
                  accuracy_matched=arrange_matched(query_dicts(con, 'SELECT * FROM matched ORDER BY bound, dataset'), matched),
                  dominance=arrange_dominance(dominance(con, 'hota_points', 'chosen')),
                  port_check=port, manuscript_check=manuscript)
    # The ablation report.
    ablation = dict(
        about=dict(
            measured='Per ablation condition, each dataset\'s fastest Tetris test configuration within the bound over '
                     'the reference throughput; mean and maximum over the datasets where the condition meets the bound.',
            selection='Chosen by test-split HOTA among the test runs of the union of per-condition validation Pareto '
                      'fronts (p135); candidates gives each pool\'s size. Part of the rise from no_both to no_sampling '
                      'reflects a larger pool.',
            source=('R7.O1 response sentence "6.7x to 11.1x" = ablationOneSpeedupAvg (no_both) and ablationTwoSpeedupAvg '
                    '(no_sampling) from evaluation/p204_ablation_stats.py, 5% bound, Section 7.2 l.375-386.'),
            conditions={c.name: dict(label=c.label, sample_rates=c.sample_rates,
                                     tracking_accuracy_thresholds=c.tracking_accuracy_thresholds) for c in ABLATION_CONDITIONS},
            checks='port_check against p204.compute_ablation_stats at every bound (must match); manuscript_check '
                   'against the submitted 5% macros.',
            inputs=inputs, ground_truth=B3D_NOTE, estimates='None; every number is measured.'),
        bounds=ablation_bounds, port_check=ablation_port,
        manuscript_check=('skipped: not all seven datasets' if not all_datasets
                          else 'skipped: the 5% bound was not computed' if ABLATION_BOUND not in ablation_macros
                          else check(MANUSCRIPT_ABLATION, rounded(ablation_macros[ABLATION_BOUND], 1), 'manuscript')))
    # Write both reports only after every check passed.
    write_json(out_dir, 'matched_speedup.json', report)
    write_json(out_dir, 'ablation_speedup.json', ablation)
    # Print the operating points and checks for the run log.
    print_operating_points(points, 'HOTA')
    print('\nranges', report['ranges'])
    print('accuracy matched', {b: {k: v for k, v in m.items() if k != 'per_dataset'} for b, m in report['accuracy_matched'].items()})
    print('LEAP runtime', leap_runtime)
    print('port check failures', failed, '| manuscript p203 check', manuscript)
    print('ablation', {b: {n: (v['mean_speedup'], v['max_speedup'], v['max_dataset']) for n, v in c.items()}
                       for b, c in ablation_bounds.items()})
    print('manuscript p204 check', ablation['manuscript_check'])


if __name__ == '__main__':
    main()
