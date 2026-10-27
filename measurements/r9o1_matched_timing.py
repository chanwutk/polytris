#!/usr/local/bin/python
"""R9.O1: relevance classifier against the reference detector, timed on the same frames, GPU, and run.

Aggregates the raw timings of measurements-gpu/r9o1_matched_timing.py (measurements-results/
r9o1_matched_timing/raw/, see that script for how and in which order it times each run), and fails with a clear
error when they are missing. The runs and their frames per call come from the GPU job's manifest.

Per dataset and run, over every frame of the timed test videos (a call's time split evenly over its frames, so
per-frame values of batched runs are batch averages): frames, calls, mean, median, and p95 (linear interpolation)
ms per frame, fps (frames over summed time = 1000 / mean), and each video's mean. Every run must cover exactly
the frames the manifest lists, in its own batching.

Comparisons (PAIRS), each the reference side's mean over the Tetris side's (the ratio of means, equal to the
ratio of throughputs; per-frame costs of several Tetris-side runs add up), with the ratio of medians when each
side is one run, and the range of the same ratio over the timed videos:
  as_deployed                 the classifier in p020's 16-frame batches against the detector at p002's batch size 1
  as_deployed_with_rgb_copy   the same with p020's BGR-to-RGB copy charged to Tetris, which alone pays it
  as_deployed_unsynchronized  the same with the classifier synchronized once per video (as p020 runs)
  same_batch_1                both models one frame per call
  same_batch_16               both models 16 frames per call

Conditions: nvidia-smi samples of the timed GPU taken inside each run (arrival and window-start phase both the
run), the other GPUs' peak utilization and memory during it, the host-side sampler's compute processes and load
average inside each dataset's window, and the job's placement (NUMA node, CPUs, memory binding, and where the
decoded frames' pages were).

The comparison puts the as_deployed numbers next to measurements-results/r9o1_classifier_cost/ (pipeline logs,
every test frame, six weeks apart): classifier and detector means per frame, their ratio, the log-derived
ratio's range across p020's runs, and the operating point's classification share of extraction recomputed with
the matched classifier cost (a sensitivity row; the share itself is log-derived). Both must describe the same
operating point.

Writes measurements-results/r9o1_matched_timing/matched_timing.json.
"""
import argparse
import json
from pathlib import Path

import duckdb

from measurements.common import (DATASET_LABELS, RESULTS_ROOT, assert_files_exist, query_dicts, results_dir, rounded,
                                 write_json)


# The GPU job's raw directory inside this measurement's result directory, and its manifest.
RAW_DIR = 'raw'
MANIFEST = 'manifest.json'
# The GPU job that writes the raw files.
GPU_JOB = 'measurements-gpu/r9o1_matched_timing.py'
# Comparisons: name -> (Tetris-side runs, whose per-frame costs add up, and the reference-side run).
PAIRS = {'as_deployed': (('classifier',), 'detector'),
         'as_deployed_with_rgb_copy': (('classifier', 'rgb_copy'), 'detector'),
         'as_deployed_unsynchronized': (('classifier_stream',), 'detector'),
         'same_batch_1': (('classifier_1',), 'detector'),
         'same_batch_16': (('classifier',), 'detector_16')}
# What each comparison means, for the report.
PAIR_DESCRIPTIONS = {
    'as_deployed': "classifier in p020's 16-frame batches against the reference detector at p002's batch size 1",
    'as_deployed_with_rgb_copy': "as_deployed with p020's BGR-to-RGB copy charged to Tetris (the detector reads BGR)",
    'as_deployed_unsynchronized': 'as_deployed with the classifier synchronized once per video instead of after every call',
    'same_batch_1': 'both models one frame per call',
    'same_batch_16': 'both models 16 consecutive frames per call',
}
# The comparison r9o1_classifier_cost's logs describe (p020's batching, p002's batch size 1).
LOGGED_PAIR = 'as_deployed'
# Runs every dataset must have: those the comparisons use, and decoding, reported as a boundary.
REQUIRED_RUNS = sorted({run for tetris, reference in PAIRS.values() for run in (*tetris, reference)} | {'decode'})
# Columns of a raw timing line (ops, p020's host-side intervals, are not read).
CALL_COLUMNS = {'dataset': 'VARCHAR', 'video': 'VARCHAR', 'run': 'VARCHAR', 'frame_idx': 'BIGINT', 'frames': 'BIGINT',
                'ms': 'DOUBLE', 'start_ns': 'BIGINT'}
# Columns of a raw nvidia-smi sample line.
SAMPLE_COLUMNS = {'t_ns': 'BIGINT', 'phase': 'VARCHAR', 'window_phase': 'VARCHAR', 'gpu_index': 'BIGINT',
                  'utilization_gpu_pct': 'DOUBLE', 'utilization_memory_pct': 'DOUBLE', 'memory_used_mib': 'DOUBLE',
                  'sm_clock_mhz': 'DOUBLE', 'memory_clock_mhz': 'DOUBLE', 'power_w': 'DOUBLE', 'temperature_c': 'DOUBLE',
                  'pstate': 'VARCHAR', 'clock_event_reasons': 'VARCHAR'}
# Host-side sampler files (one per launch) and their columns: epoch ms, compute processes on the timed GPU and on
# all GPUs, and the 1-minute load average.
HOST_APPS_GLOB = 'host_compute_apps.*.jsonl.gz'
HOST_APPS_COLUMNS = {'t_ms': 'BIGINT', 'apps': 'BIGINT', 'apps_all': 'BIGINT', 'load_1m': 'DOUBLE'}
# The percentile reported next to the median.
PERCENTILE = 0.95
# The log-derived measurement compared against.
LOG_DERIVED_DIR = RESULTS_ROOT / 'r9o1_classifier_cost'


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O1: matched classifier and reference detector timing')
    parser.add_argument('--datasets', nargs='+', default=None)
    return parser.parse_args()


def load_manifest(raw_dir: Path, datasets: list[str] | None) -> tuple[dict, list[str]]:
    """The GPU job's manifest and the datasets to report (all it finished when `datasets` is None)."""
    # Without the manifest there is nothing to aggregate; say which job produces it.
    path = raw_dir / MANIFEST
    assert path.is_file(), f'{path} missing: run {GPU_JOB} first (see its docstring)'
    manifest = json.loads(path.read_text())
    # Every requested dataset must have finished in the GPU job.
    datasets = datasets or sorted(manifest['datasets'])
    missing = sorted(set(datasets) - set(manifest['datasets']))
    assert datasets and not missing, f'datasets not timed by {GPU_JOB}: {missing or "none timed"}'
    # A dataset timed by an older version of the job lacks runs the comparisons need.
    for d in datasets:
        lacking = sorted(set(REQUIRED_RUNS) - set(manifest['datasets'][d].get('runs', {})))
        assert not lacking, f'{d}: the manifest has no runs {lacking}; rerun {GPU_JOB} for it'
    return manifest, datasets


def load_calls(con: duckdb.DuckDBPyConnection, raw_dir: Path, manifest: dict, datasets: list[str]):
    """Tables calls (one row per timed call), videos and run_frames (from the manifest), and frames, after checking coverage.

    frames(dataset, run, video, frame, ms): one row per frame and run, with an even share of its call's time.
    """
    # One raw timing file per dataset.
    files = [str(raw_dir / f'{d}.jsonl.gz') for d in datasets]
    assert_files_exist(files, f'raw timing files (run {GPU_JOB})')
    # Every timed call, with the file it came from.
    con.execute(f"""
        CREATE OR REPLACE TABLE calls AS
        SELECT *, regexp_extract(filename, '([^/]+)\\.jsonl\\.gz$', 1) AS file_dataset
        FROM read_json($files, format = 'newline_delimited', columns = {CALL_COLUMNS}, filename = true)
    """, {'files': files})
    # The frames the manifest says each timed video has.
    con.execute('CREATE OR REPLACE TABLE videos (dataset VARCHAR, video VARCHAR, frames BIGINT)')
    con.executemany('INSERT INTO videos VALUES (?, ?, ?)',
                    [(d, v, r['frames']) for d in datasets for v, r in manifest['datasets'][d]['videos'].items()])
    # Each dataset's runs and their frames per call (NULL: the whole video in one call).
    con.execute('CREATE OR REPLACE TABLE run_frames (dataset VARCHAR, run VARCHAR, per_call BIGINT)')
    con.executemany('INSERT INTO run_frames VALUES (?, ?, ?)',
                    [(d, run, n) for d in datasets for run, n in manifest['datasets'][d]['runs'].items()])
    # Every call needs its file's dataset, a manifest video, one of the dataset's runs, and a positive, finite time.
    bad = query_dicts(con, """
        SELECT c.file_dataset, c.dataset, c.video, c.run, c.frame_idx, c.frames, c.ms
        FROM calls c LEFT JOIN videos v USING (dataset, video) LEFT JOIN run_frames r USING (dataset, run)
        WHERE c.dataset IS DISTINCT FROM c.file_dataset OR v.frames IS NULL OR r.run IS NULL
           OR c.frame_idx IS NULL OR c.frames IS NULL OR c.ms IS NULL OR isnan(c.ms) OR isinf(c.ms) OR c.ms <= 0
        LIMIT 5
    """)
    assert not bad, f'raw calls with a wrong dataset, unknown video or run, or a missing or non-positive time: {bad}'
    # Each call must start at a multiple of its run's batch and hold min(batch, remaining) frames, as p020 and p002 do.
    bad = query_dicts(con, """
        SELECT c.dataset, c.video, c.run, c.frame_idx, c.frames, v.frames AS video_frames, r.per_call
        FROM calls c JOIN videos v USING (dataset, video) JOIN run_frames r USING (dataset, run)
        WHERE c.frame_idx % coalesce(r.per_call, v.frames) <> 0
           OR c.frames <> least(coalesce(r.per_call, v.frames), v.frames - c.frame_idx)
        LIMIT 5
    """)
    assert not bad, f'calls that do not follow their run\'s batching: {bad}'
    # One row per frame of every call, carrying an even share of the call's time.
    con.execute("""
        CREATE OR REPLACE TABLE frames AS
        SELECT dataset, run, video, frame_idx + unnest(range(frames)) AS frame, ms / frames AS ms FROM calls
    """)
    # Every run must time every manifest frame of every video exactly once.
    bad = query_dicts(con, """
        WITH expected AS (
            SELECT v.dataset, r.run, v.video, v.frames FROM videos v JOIN run_frames r USING (dataset)
        ),
        seen AS (
            SELECT dataset, run, video, count(*) AS rows, count(DISTINCT frame) AS distinct_frames,
                   min(frame) AS first, max(frame) AS last
            FROM frames GROUP BY ALL
        )
        SELECT e.dataset, e.run, e.video, e.frames, s.rows, s.distinct_frames, s.first, s.last
        FROM expected e LEFT JOIN seen s USING (dataset, run, video)
        WHERE s.rows IS NULL OR s.rows <> e.frames OR s.distinct_frames <> e.frames OR s.first <> 0 OR s.last <> e.frames - 1
        LIMIT 5
    """)
    assert not bad, f'runs that do not time every manifest frame exactly once: {bad}'


def run_stats(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Tables run_stats (per dataset and run) and video_stats (per dataset, run, and video), and run_stats' rows.

    run_stats: videos, frames, calls, mean, median, p95, min, and max ms per frame, and fps. Needs calls, frames.
    """
    # Per video: each video's mean, which shows how much the cost moves between videos.
    con.execute("""
        CREATE OR REPLACE TABLE video_stats AS
        SELECT dataset, run, video, count(*) AS frames, sum(ms) / count(*) AS mean_ms FROM frames GROUP BY ALL
    """)
    # Per run: the per-frame distribution over every timed video, and the number of calls behind it.
    con.execute("""
        CREATE OR REPLACE TABLE run_stats AS
        WITH per_frame AS (
            SELECT dataset, run, count(DISTINCT video) AS videos, count(*) AS frames, sum(ms) AS total_ms,
                   median(ms) AS median_ms, quantile_cont(ms, $p) AS p95_ms, min(ms) AS min_ms, max(ms) AS max_ms
            FROM frames GROUP BY ALL
        ),
        per_call AS (SELECT dataset, run, count(*) AS calls FROM calls GROUP BY ALL)
        SELECT dataset, run, videos, frames, calls, total_ms, total_ms / frames AS mean_ms, median_ms, p95_ms, min_ms, max_ms,
               1000 * frames / total_ms AS fps, 1000 / median_ms AS fps_at_median
        FROM per_frame JOIN per_call USING (dataset, run)
    """, {'p': PERCENTILE})
    return query_dicts(con, 'SELECT * FROM run_stats ORDER BY dataset, run')


def ratios(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Tables pairs, video_ratios, and ratios, and ratios' rows: per comparison and dataset, the sides' mean ms per frame,
    the ratio of means, the ratio of medians (one run per side only), and the ratio's range over videos. Needs run_stats,
    video_stats."""
    # The comparisons' runs, each on its side.
    con.execute('CREATE OR REPLACE TABLE pairs (pair VARCHAR, side VARCHAR, run VARCHAR)')
    con.executemany('INSERT INTO pairs VALUES (?, ?, ?)',
                    [(p, 'tetris', run) for p, (tetris, _) in PAIRS.items() for run in tetris]
                    + [(p, 'reference', reference) for p, (_, reference) in PAIRS.items()])
    # Per video, each side's mean (summed over its runs, which cover the same frames) and their ratio.
    con.execute("""
        CREATE OR REPLACE TABLE video_ratios AS
        WITH sides AS (
            SELECT p.pair, v.dataset, v.video, p.side, sum(v.mean_ms) AS mean_ms
            FROM pairs p JOIN video_stats v USING (run) GROUP BY ALL
        )
        SELECT pair, dataset, video,
               any_value(mean_ms) FILTER (side = 'reference') / any_value(mean_ms) FILTER (side = 'tetris') AS ratio_of_means
        FROM sides GROUP BY ALL
    """)
    # Per dataset, each side's mean (summed over its runs) and median (a side of several runs has no single median).
    con.execute("""
        CREATE OR REPLACE TABLE ratios AS
        WITH sides AS (
            SELECT p.pair, s.dataset, p.side, sum(s.mean_ms) AS mean_ms,
                   CASE WHEN count(*) = 1 THEN any_value(s.median_ms) END AS median_ms
            FROM pairs p JOIN run_stats s USING (run) GROUP BY ALL
        ),
        video_range AS (
            SELECT pair, dataset, min(ratio_of_means) AS video_min, max(ratio_of_means) AS video_max
            FROM video_ratios GROUP BY ALL
        )
        SELECT t.pair, t.dataset, t.mean_ms AS tetris_mean_ms, r.mean_ms AS reference_mean_ms,
               r.mean_ms / t.mean_ms AS ratio_of_means, r.median_ms / t.median_ms AS ratio_of_medians,
               v.video_min AS ratio_of_means_video_min, v.video_max AS ratio_of_means_video_max
        FROM sides t
        JOIN sides r ON r.pair = t.pair AND r.dataset = t.dataset AND r.side = 'reference'
        JOIN video_range v ON v.pair = t.pair AND v.dataset = t.dataset
        WHERE t.side = 'tetris'
    """)
    return query_dicts(con, 'SELECT * FROM ratios ORDER BY dataset, pair')


def gpu_conditions(con: duckdb.DuckDBPyConnection, raw_dir: Path, manifest: dict, datasets: list[str]) -> list[dict]:
    """Per dataset and run, the timed GPU's nvidia-smi samples taken inside the run, and the other GPUs' peaks during it."""
    # One sample file per dataset, beside its timings.
    files = [str(raw_dir / f'{d}.gpu.jsonl.gz') for d in datasets]
    assert_files_exist(files, f'GPU sample files (run {GPU_JOB})')
    con.execute(f"""
        CREATE OR REPLACE TABLE samples AS
        SELECT *, regexp_extract(filename, '([^/]+)\\.gpu\\.jsonl\\.gz$', 1) AS dataset
        FROM read_json($files, format = 'newline_delimited', columns = {SAMPLE_COLUMNS}, filename = true)
    """, {'files': files})
    # The GPU each dataset was timed on.
    con.execute('CREATE OR REPLACE TABLE timed_gpus (dataset VARCHAR, gpu_index BIGINT)')
    con.executemany('INSERT INTO timed_gpus VALUES (?, ?)', [(d, manifest['datasets'][d]['environment']['gpu_index']) for d in datasets])
    # Every run of every dataset, with or without samples (a run shorter than the averaging window has none). The timed
    # GPU's samples count when both their phases are the run, since utilization averages over the preceding window;
    # the other GPUs' samples count when either phase is, so their peaks err on the side of more load.
    return query_dicts(con, """
        WITH runs AS (SELECT DISTINCT dataset, run FROM frames),
        own AS (
            SELECT s.* FROM samples s JOIN timed_gpus g ON g.dataset = s.dataset AND g.gpu_index = s.gpu_index
            WHERE s.phase = s.window_phase
        ),
        own_stats AS (
            SELECT r.dataset, r.run, count(o.t_ns) AS samples,
                   median(o.utilization_gpu_pct) AS utilization_median_pct, min(o.utilization_gpu_pct) AS utilization_min_pct,
                   max(o.utilization_gpu_pct) AS utilization_max_pct, max(o.memory_used_mib) AS memory_used_max_mib,
                   median(o.sm_clock_mhz) AS sm_clock_median_mhz, min(o.sm_clock_mhz) AS sm_clock_min_mhz,
                   max(o.sm_clock_mhz) AS sm_clock_max_mhz, median(o.power_w) AS power_median_w,
                   max(o.temperature_c) AS temperature_max_c,
                   list(DISTINCT o.pstate ORDER BY o.pstate) FILTER (o.pstate IS NOT NULL) AS pstates,
                   list(DISTINCT o.clock_event_reasons ORDER BY o.clock_event_reasons)
                       FILTER (o.clock_event_reasons IS NOT NULL) AS clock_event_reasons
            FROM runs r LEFT JOIN own o ON o.dataset = r.dataset AND o.phase = r.run
            GROUP BY ALL
        ),
        other_stats AS (
            SELECT r.dataset, r.run, count(s.t_ns) AS other_gpu_samples,
                   max(s.utilization_gpu_pct) AS other_gpus_utilization_max_pct,
                   max(s.memory_used_mib) AS other_gpus_memory_used_max_mib
            FROM runs r
            LEFT JOIN (SELECT s.* FROM samples s JOIN timed_gpus g ON g.dataset = s.dataset AND g.gpu_index <> s.gpu_index) s
                ON s.dataset = r.dataset AND (s.phase = r.run OR s.window_phase = r.run)
            GROUP BY ALL
        )
        SELECT * FROM own_stats JOIN other_stats USING (dataset, run) ORDER BY dataset, run
    """)


def host_processes(con: duckdb.DuckDBPyConnection, raw_dir: Path, manifest: dict, datasets: list[str]) -> dict | None:
    """Per dataset, compute processes and load average from the host-side samples within its start and end; None without files."""
    # The host-side sampler runs outside the container; on hosts without it there is nothing to report.
    files = sorted(str(p) for p in raw_dir.glob(HOST_APPS_GLOB))
    if not files:
        return None
    con.execute(f"""
        CREATE OR REPLACE TABLE host_apps AS
        SELECT * FROM read_json($files, format = 'newline_delimited', columns = {HOST_APPS_COLUMNS})
    """, {'files': files})
    # Each dataset's window from the manifest, in epoch ms.
    con.execute('CREATE OR REPLACE TABLE windows (dataset VARCHAR, start_ms BIGINT, end_ms BIGINT)')
    con.executemany('INSERT INTO windows VALUES (?, ?, ?)',
                    [(d, manifest['datasets'][d]['started_ns'] // 1_000_000, manifest['datasets'][d]['finished_ns'] // 1_000_000)
                     for d in datasets])
    # Samples inside each window: the job's own process makes one on its GPU; more means another job shared it.
    rows = query_dicts(con, """
        SELECT w.dataset, count(h.t_ms) AS samples, max(h.apps) AS max_processes,
               count(*) FILTER (h.apps > 1) AS samples_with_other_processes, max(h.apps_all) AS max_processes_all_gpus,
               median(h.load_1m) AS load_1m_median, max(h.load_1m) AS load_1m_max
        FROM windows w LEFT JOIN host_apps h ON h.t_ms BETWEEN w.start_ms AND w.end_ms
        GROUP BY ALL ORDER BY w.dataset
    """)
    return {r['dataset']: {k: v for k, v in r.items() if k != 'dataset'} for r in rows}


def placement(entry: dict) -> dict:
    """Where a dataset's job ran: the GPU's NUMA node, the job's CPUs and memory binding, and its pages' node per video."""
    p = entry['environment']['placement']
    # numa_maps names nodes N<node>.
    node = f"N{p['gpu_numa_node']}"
    # Share of the anonymous pages (decoded frames and copies dominate) on the GPU's node, per video.
    shares = {v: r['anon_pages_by_node'].get(node, 0) / sum(r['anon_pages_by_node'].values()) for v, r in entry['videos'].items()}
    return dict(gpu_numa_node=p['gpu_numa_node'], cpus=len(p['cpus']), cpus_local_to_gpu=set(p['cpus']) <= set(p['gpu_local_cpus']),
                membind=p['numactl'].get('membind'), anon_pages_on_gpu_node_share=shares)


def log_derived(datasets: list[str], manifest: dict, directory: Path) -> dict:
    """Per dataset, r9o1_classifier_cost's log-derived costs, ratios, and shares, checked to describe the same operating point."""
    # Both reports of the log-derived measurement.
    paths = [directory / 'classifier_vs_detector.json', directory / 'operator_share.json']
    assert_files_exist(paths, 'r9o1_classifier_cost results (run measurements/r9o1_classifier_cost.py)')
    per_frame, shares = (json.loads(p.read_text())['datasets'] for p in paths)
    out = {}
    for dataset in datasets:
        # Both reports key datasets by their manuscript label.
        label = DATASET_LABELS.get(dataset, dataset)
        assert label in per_frame and label in shares, f'r9o1_classifier_cost has no {label}'
        # The comparison only holds for the same configuration, classifier, and tile size.
        point = manifest['datasets'][dataset]['operating_point']
        logged = shares[label]
        assert (logged['config'], logged['classifier'], logged['tilesize']) == (point['config'], point['classifier'], point['tilesize']), \
            (f'{label}: operating point differs from r9o1_classifier_cost ({logged["config"]}, {logged["classifier"]}, '
             f'{logged["tilesize"]} vs {point["config"]}, {point["classifier"]}, {point["tilesize"]}); rerun one of them')
        r = per_frame[label]
        out[dataset] = dict(
            config=logged['config'], videos=r['videos'], frames=r['frames'],
            classifier_mean_ms=r['classifier']['mean_ms_per_frame'], classifier_median_ms=r['classifier']['median_ms_per_frame'],
            detector_mean_ms=r['detector']['mean_ms_per_frame'], detector_median_ms=r['detector']['median_ms_per_frame'],
            ratio_of_means=r['ratio_of_means'], ratio_of_medians=r['ratio_of_medians'],
            ratio_of_means_across_runs=r['cost_across_runs']['ratio_of_means'],
            classify_seconds=logged['operators']['classify']['seconds'], extraction_seconds=logged['extraction_seconds'],
            classify_share_pct=logged['operators']['classify']['share_pct'])
    return out


def stats_entry(s: dict) -> dict:
    """One run's statistics for the report."""
    # The run_stats row under the report's names.
    return dict(videos=s['videos'], frames=s['frames'], calls=s['calls'], mean_ms_per_frame=s['mean_ms'],
                median_ms_per_frame=s['median_ms'], p95_ms_per_frame=s['p95_ms'], min_ms_per_frame=s['min_ms'],
                max_ms_per_frame=s['max_ms'], fps=s['fps'], fps_at_median=s['fps_at_median'])


def classify_share_with_matched_cost(log: dict, matched_over_log: float) -> float:
    """The operating point's classification share of extraction (%) with its logged time scaled to the matched cost."""
    # Only classification changes; every other operator keeps its logged time.
    scaled = log['classify_seconds'] * matched_over_log
    return 100 * scaled / (log['extraction_seconds'] - log['classify_seconds'] + scaled)


def report(con: duckdb.DuckDBPyConnection, raw_dir: Path, manifest: dict, datasets: list[str], log_dir: Path) -> dict:
    """matched_timing.json content: per dataset (keyed by label), a cross-dataset summary, and the method. Needs calls, frames."""
    # Statistics, per-video means, ratios, GPU conditions, host-side samples, and the log-derived numbers.
    stats = {(s['dataset'], s['run']): s for s in run_stats(con)}
    # Nest the per-video rows as {video: value} under (dataset, run) for the report (assembly only, no analysis).
    per_video = {}
    for r in query_dicts(con, 'SELECT dataset, run, video, mean_ms FROM video_stats ORDER BY video'):
        per_video.setdefault((r['dataset'], r['run']), {})[r['video']] = r['mean_ms']
    ratio = {(r['dataset'], r['pair']): r for r in ratios(con)}
    # The same nesting for the per-video ratios under (dataset, comparison).
    video_ratio = {}
    for r in query_dicts(con, 'SELECT pair, dataset, video, ratio_of_means FROM video_ratios ORDER BY video'):
        video_ratio.setdefault((r['dataset'], r['pair']), {})[r['video']] = r['ratio_of_means']
    gpu = {(g['dataset'], g['run']): g for g in gpu_conditions(con, raw_dir, manifest, datasets)}
    host = host_processes(con, raw_dir, manifest, datasets)
    logs = log_derived(datasets, manifest, log_dir)
    out = {}
    for dataset in datasets:
        m, log = manifest['datasets'][dataset], logs[dataset]
        logged_pair = ratio[(dataset, LOGGED_PAIR)]
        # The matched classifier cost over the logged one, which the share sensitivity row scales by.
        classifier_over_log = logged_pair['tetris_mean_ms'] / log['classifier_mean_ms']
        out[DATASET_LABELS.get(dataset, dataset)] = dict(
            dataset=dataset, operating_point=m['operating_point'],
            classifier=dict(name=m['classifier']['name'], tilesize=m['classifier']['tilesize'],
                            optimization=m['classifier']['optimization'], grid=m['classifier']['grid'],
                            tiles_inferred_at_most=m['classifier']['tiles_inferred_at_most']),
            detector=dict(detector=m['detector']['detector'], weights=m['detector']['weights']['path']),
            videos={v: r['frames'] for v, r in m['videos'].items()},
            runs={run: dict(stats_entry(stats[(dataset, run)]), frames_per_call=n, per_video_mean_ms=per_video[(dataset, run)],
                            gpu={k: v for k, v in gpu[(dataset, run)].items() if k not in ('dataset', 'run')})
                  for run, n in m['runs'].items()},
            ratios={pair: dict({k: v for k, v in ratio[(dataset, pair)].items() if k not in ('dataset', 'pair')},
                               per_video=video_ratio[(dataset, pair)]) for pair in PAIRS},
            host_processes=None if host is None else host[dataset], placement=placement(m),
            idle_before_launch=m['idle_before_launch'], gpu=m['environment']['gpu'], host=m['environment']['hostname'],
            cpu=m['environment']['cpu'], started=m['started'], finished=m['finished'],
            log_derived=log,
            matched_over_log=dict(classifier_mean=classifier_over_log,
                                  detector_mean=logged_pair['reference_mean_ms'] / log['detector_mean_ms'],
                                  ratio_of_means=logged_pair['ratio_of_means'] / log['ratio_of_means']),
            classify_share_pct=dict(log_derived=log['classify_share_pct'],
                                    with_matched_classifier_cost=classify_share_with_matched_cost(log, classifier_over_log)))
    # Across datasets, per comparison: the ratio of means (min, median, max), its extremes over every timed video, and
    # how many datasets.
    summary = query_dicts(con, """
        WITH videos AS (SELECT pair, min(ratio_of_means) AS video_min, max(ratio_of_means) AS video_max FROM video_ratios GROUP BY ALL)
        SELECT pair, min(r.ratio_of_means) AS min, median(r.ratio_of_means) AS median, max(r.ratio_of_means) AS max,
               any_value(v.video_min) AS video_min, any_value(v.video_max) AS video_max, count(*) AS datasets
        FROM ratios r JOIN videos v USING (pair) GROUP BY pair ORDER BY pair
    """)
    return dict(datasets=out, summary=dict(ratio_of_means={r.pop('pair'): r for r in summary}),
                method=dict(manifest['method'], pairs={p: dict(tetris=list(PAIRS[p][0]), reference=PAIRS[p][1],
                                                               description=PAIR_DESCRIPTIONS[p]) for p in PAIRS},
                            aggregation=(
                    f'Per-frame ms (a call\'s time split evenly over its frames); mean = summed time / frames, so '
                    f'fps = 1000 / mean and the ratio of means equals the ratio of throughputs; p{round(100 * PERCENTILE)} '
                    'by linear interpolation. A comparison\'s Tetris side adds its runs\' means (they cover the same '
                    'frames) and has a median only when it is one run. per_video and video_min/max are the same ratio '
                    'per timed video. GPU conditions are the timed GPU\'s samples whose arrival and window-start phases '
                    'are both the run, and the other GPUs\' peaks while either phase is. host_processes counts compute '
                    'processes from the host (null when the host-side sampler did not run); the job itself is one. '
                    'log_derived is r9o1_classifier_cost over every test frame from the pipeline logs; matched_over_log '
                    f'divides the matched {LOGGED_PAIR} value by it; classify_share_pct.with_matched_classifier_cost scales '
                    'the logged classification time by matched_over_log.classifier_mean and keeps every other operator.')))


def main():
    # Command-line options.
    args = parse_args()
    # The GPU job's outputs and the datasets to report.
    out_dir = results_dir(__file__)
    raw_dir = out_dir / RAW_DIR
    manifest, datasets = load_manifest(raw_dir, args.datasets)
    # Timings, checked for coverage, then the report.
    con = duckdb.connect()
    load_calls(con, raw_dir, manifest, datasets)
    result = report(con, raw_dir, manifest, datasets, LOG_DERIVED_DIR)
    write_json(out_dir, 'matched_timing.json', rounded(result))
    # One line per dataset: per-frame means, each comparison's ratio with its per-video range, and the logs' ratio.
    for label, r in result['datasets'].items():
        runs, pairs = r['runs'], r['ratios']
        print(f"{label:10s} classifier {runs['classifier']['mean_ms_per_frame']:.3f} ms, rgb copy "
              f"{runs['rgb_copy']['mean_ms_per_frame']:.3f} ms, detector {runs['detector']['mean_ms_per_frame']:.2f} ms | "
              + ' | '.join(f"{p} {pairs[p]['ratio_of_means']:.1f}x ({pairs[p]['ratio_of_means_video_min']:.1f}-"
                           f"{pairs[p]['ratio_of_means_video_max']:.1f})" for p in PAIRS)
              + f" | logs {r['log_derived']['ratio_of_means']}x | other processes "
              f"{None if r['host_processes'] is None else r['host_processes']['samples_with_other_processes']}")
    print('summary', rounded(result['summary']))


if __name__ == '__main__':
    main()
