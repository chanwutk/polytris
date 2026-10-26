#!/usr/local/bin/python
"""R9.O1: cost of relevance classification against the reference detector, and each operator's share of extraction.

Both views summarize timings the pipeline already logged on the test split. Nothing is rerun.

classifier_vs_detector.json compares, per dataset, classifying one frame's tiles with running the
reference detector on the same frame. Both cover every frame of every test video.

  classifier  <video>/020_relevancy/<classifier>_<tile size>_<s>/score/runtime.jsonl for the operating
              point's classifier and tile size. The comparison uses sample rate s = 1, which classifies
              every frame; the runs at every other sample rate p020 executed show the run-to-run spread.
              scripts/p020_exec_classify.py classifies 16 frames per batch and logs one line per batch
              (send, resize, diff, reshape, mask, inference, collect), then one line per video with read
              (video decoding) and retrieve (copy to CPU and hex encoding). A batch's time is split evenly
              over its frames, so per-frame values are batch averages.
  detector    <video>/002_naive/detection_runtime.jsonl, one line per frame: detect (the reference
              detector on the full frame, batch size 1) and read (video decoding).

Op boundaries follow evaluation/p121_throughput_compute.py, which produces the reported extraction
time: read and retrieve are excluded from the classifier. The reference's read is excluded here so
that both sides are pure computation, although p121 counts it in the reference pipeline's runtime.
The headline ratio is the ratio of means, which equals the ratio of throughputs (frames over time).

operator_share.json gives, per dataset, each operator's share of Tetris's extraction time at the
operating point: the fastest Tetris configuration within --loss of the reference pipeline's HOTA on
the test split (common.fastest_within). Times are summed over the test split from
<dataset>/evaluation/080_throughput/measurements/query_execution_per_op.csv, p121's own accounting,
so the operators add up to the runtime behind the reported throughput (checked against tradeoff.csv),
and its classify time equals the operating point's p020 run (checked against the raw log).

Writes measurements-results/r9o1_classifier_cost/:
  classifier_vs_detector.json  per dataset, per-frame cost, fps, and detector/classifier ratio, per run
  operator_share.json          per dataset, each operator's seconds, ms per original frame, and share of
                               extraction at the operating point, and why the share differs from the ratio
"""
import argparse
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone

import duckdb
import numpy as np

from measurements.common import DATASET_LABELS, fastest_within, load_test_tradeoff, query_dicts, results_dir, write_json
from polyis.io import cache
from polyis.utilities import CONFIG_DIR, get_config


# Tetris extraction operators, keyed by their stage names in the throughput manifest.
STAGE_OPERATORS = {'020_exec_classify': 'classify', '022_exec_prune_polyominoes': 'prune', '030_exec_compress': 'pack',
                   '040_exec_detect': 'detect', '050_exec_uncompress': 'unpack', '060_exec_track': 'track'}
# The classification stage, whose time must equal the operating point's p020 run.
CLASSIFY_STAGE = '020_exec_classify'
# The pruning stage, which runs only for configurations with a tracking-accuracy threshold.
PRUNE_STAGE = '022_exec_prune_polyominoes'
# The reference pipeline's stages in the same manifest: detection on every frame, then tracking.
REFERENCE_STAGES = {'001_preprocess_groundtruth_detection': 'detect', '002_preprocess_groundtruth_tracking': 'track'}
# The manifest stage whose runtime file is the reference detector's per-frame log.
DETECTOR_STAGE = '001_preprocess_groundtruth_detection'
# The RetinaNet configuration under configs/ that names the B3D detector's weights (polyis/models/retinanet_b3d.py).
RETINANET_CONFIG = 'retinanet.json'
# Frames per classifier batch (BATCH_SIZE in scripts/p020_exec_classify.py), checked against every log's line count.
CLASSIFIER_BATCH = 16
# Ops p020 logs per batch, sorted: tile preparation (send, resize, diff, reshape, mask), inference, and collect.
CLASSIFIER_BATCH_OPS = ['collect', 'diff', 'inference', 'mask', 'reshape', 'resize', 'send']
# Ops of p020's last line per video, sorted: video decoding and copy-back, both excluded by p121.
CLASSIFIER_TAIL_OPS = ['read', 'retrieve']
# Ops p002 logs per frame, sorted: reference detection and video decoding.
DETECTOR_OPS = ['detect', 'read']
# Ops left out of the per-frame comparison: decoding (both logs) and the classifier's copy-back.
EXCLUDED_OPS = ['read', 'retrieve']
# The classifier run that classifies every frame, which the per-frame comparison uses.
EVERY_FRAME_RATE = 1
# A batch whose inference interval exceeds this multiple of its run's median is an inference spike.
SPIKE_FACTOR = 10
# Relative tolerance between summed timings that must agree (per-op table, tradeoff table, raw logs).
RUNTIME_TOLERANCE = 1e-6
# What the logs record about the machine that produced them, as checked on 2026-10-26.
LOGGED_HARDWARE = ('Not recorded. No runtime log (p002 detection_runtime.jsonl, p020 runtime.jsonl, the p121 tables) '
                   'and no worker log under /polyis-cache/LOGS names a host or GPU (checked 2026-10-26).')
# The paper's hardware statement, for comparison with the host that read the cache.
PAPER_HARDWARE = 'Section 7 (Hardware): dual Intel Xeon Gold 6248, 504 GB RAM, and one NVIDIA Quadro RTX 6000 (24 GB).'
# Why the per-frame ratio and the share answer different questions.
RATIO_VS_SHARE = (
    'The per-frame ratio compares the mean cost of classifying one frame with the mean cost of detecting one full '
    'frame, both run on every frame (the sample-rate-1 classifier run). The share is measured at the operating '
    'point, where three things differ. Whole-frame sampling (sample_rate s) runs the classifier on 1/sampling_factor '
    'of the frames. The operating point\'s own p020 run costs per_classified_frame_cost_vs_every_frame times the '
    'every-frame run per classified frame (run-to-run variation, see classifier_vs_detector.json classifier_runs). '
    'Sampling and packing shrink detection per original frame by detection_reduction. Classification per original '
    'frame shrinks by classification_reduction = sampling_factor / per_classified_frame_cost_vs_every_frame. The share '
    'also counts packing, unpacking, pruning, and tracking, which the ratio leaves out. 1/(1 + ratio) is the share '
    'classification would have if both ran on every frame and nothing else ran.'
)


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O1: relevance classifier cost against the reference detector')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    return parser.parse_args()


def load_configs(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Table configs(dataset, system, config, fps, accuracy, ...) of the reference and every Tetris test configuration."""
    # One row per configuration, with the parameters the breakdown needs.
    con.execute('CREATE OR REPLACE TABLE configs (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, '
                'accuracy DOUBLE, classifier VARCHAR, tilesize INTEGER, sample_rate INTEGER, '
                'tracking_accuracy_threshold DOUBLE, frame_count BIGINT, seconds DOUBLE)')
    for dataset in datasets:
        # Validated test-split configurations and the reference row, from the pipeline's tradeoff table.
        tetris, naive = load_test_tradeoff(dataset)
        # The reference row: its HOTA bounds every configuration.
        con.execute("INSERT INTO configs VALUES (?, 'reference', 'naive', ?, ?, NULL, NULL, NULL, NULL, ?, ?)",
                    [dataset, float(naive['throughput_fps']), float(naive['HOTA_HOTA']),
                     int(naive['frame_count']), float(naive['time'])])
        # Every Tetris configuration, read straight from the pandas frame (NaN thresholds become NULL).
        con.register('tetris_frame', tetris)
        con.execute("""
            INSERT INTO configs SELECT $dataset, 'tetris', variant_id, throughput_fps, HOTA_HOTA, classifier, tilesize,
                   sample_rate, tracking_accuracy_threshold, frame_count, time
            FROM tetris_frame
        """, {'dataset': dataset})
        # Release the view so the next dataset can register its own frame.
        con.unregister('tetris_frame')


def choose_operating_points(con: duckdb.DuckDBPyConnection, loss: float):
    """Table chosen: per dataset, the fastest Tetris configuration within `loss` of the reference HOTA."""
    # Per dataset, the fastest Tetris configuration whose HOTA is within the bound.
    rows = fastest_within(con, 'configs', loss, ['tetris'])
    # A dataset without a qualifying configuration has no operating point to break down.
    missed = [r['dataset'] for r in rows if not r['meets_bound']]
    assert not missed, f'no Tetris configuration within {loss:.0%} HOTA loss on {missed}'
    # Keep the chosen names in a table so the parameters can be joined in SQL.
    con.execute('CREATE OR REPLACE TABLE chosen_names (dataset VARCHAR, config VARCHAR)')
    con.executemany('INSERT INTO chosen_names VALUES (?, ?)', [(r['dataset'], r['config']) for r in rows])
    # Attach the configuration's parameters and the reference row it is bounded by.
    con.execute("""
        CREATE OR REPLACE TABLE chosen AS
        SELECT t.*, r.accuracy AS reference_accuracy, r.fps AS reference_fps, r.seconds AS reference_seconds
        FROM chosen_names n JOIN configs t ON t.dataset = n.dataset AND t.config = n.config AND t.system = 'tetris'
        JOIN configs r ON r.dataset = n.dataset AND r.system = 'reference'
    """)


def assert_files_exist(paths: list[str], what: str):
    """Fail with the missing paths, instead of a bare DuckDB IOException later."""
    # Every input table must exist before DuckDB reads it.
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f'missing {what}: {missing}'


def classifier_rates(dataset: str, video: str, classifier: str, tilesize: int) -> list[int]:
    """Sample rates p020 ran for one test video with this classifier and tile size, from its 020_relevancy directories."""
    # p020 writes one <classifier>_<tile size>_<sample rate> directory per run; other classifiers and tile sizes differ.
    pattern = re.compile(rf'{re.escape(classifier)}_{tilesize}_(\d+)')
    # The video's relevancy directory.
    root = cache.exec(dataset, 'relevancy', video)
    # Its entries, or none when p020 never ran on the video.
    names = os.listdir(root) if os.path.isdir(root) else []
    # Parse the sample rate of every matching directory.
    return sorted(int(m.group(1)) for m in map(pattern.fullmatch, names) if m)


def runtime_files(con: duckdb.DuckDBPyConnection, manifests: dict[str, str]) -> list[tuple]:
    """(dataset, video, kind, sample_rate, path) for the reference detector's log and every classifier run's log.

    `manifests` maps each dataset to its throughput manifest (query_execution.csv); needs table chosen.
    """
    # A missing manifest would otherwise surface as a bare DuckDB error.
    assert_files_exist(list(manifests.values()), 'throughput manifests (run evaluation/p120 --test)')
    # The manifest lists the reference detector's log of every test video; the classifier logs sit beside it.
    rows = query_dicts(con, """
        SELECT m.dataset, m.video, m.runtime_file, c.classifier, c.tilesize
        FROM read_csv($paths, union_by_name := true, all_varchar := true) m JOIN chosen c ON c.dataset = m.dataset
        WHERE m.variant = 'naive' AND m.videoset = 'test' AND m.stage = $stage
        ORDER BY m.dataset, m.video
    """, {'paths': list(manifests.values()), 'stage': DETECTOR_STAGE})
    # Every requested dataset must list test videos.
    missing = set(manifests) - {r['dataset'] for r in rows}
    assert not missing, f'no test videos in the throughput manifest of {sorted(missing)}'
    # The sample rates p020 ran for each test video, discovered on disk.
    con.execute('CREATE OR REPLACE TABLE classifier_rates (dataset VARCHAR, video VARCHAR, rates INTEGER[])')
    con.executemany('INSERT INTO classifier_rates VALUES (?, ?, ?)',
                    [(r['dataset'], r['video'], classifier_rates(r['dataset'], r['video'], r['classifier'], r['tilesize']))
                     for r in rows])
    # Every run must cover every test video, and the every-frame and operating-point runs must exist.
    bad = query_dicts(con, """
        SELECT r.dataset, list(DISTINCT r.rates) AS rate_sets, any_value(c.sample_rate) AS operating_rate
        FROM classifier_rates r JOIN chosen c USING (dataset) GROUP BY r.dataset
        HAVING count(DISTINCT r.rates) <> 1 OR NOT list_has_all(any_value(r.rates), [$every, any_value(c.sample_rate)])
    """, {'every': EVERY_FRAME_RATE})
    assert not bad, f'classifier runs differ across test videos or lack the every-frame or operating-point rate: {bad}'
    # One (video, sample rate) pair per classifier run, with the operating point's classifier and tile size.
    runs = query_dicts(con, """
        SELECT r.dataset, r.video, c.classifier, c.tilesize, unnest(r.rates) AS rate
        FROM classifier_rates r JOIN chosen c USING (dataset) ORDER BY ALL
    """)
    # One classifier log per video and run, at the path p020 writes.
    files = [(r['dataset'], r['video'], 'classifier', r['rate'],
              str(cache.exec(r['dataset'], 'relevancy', r['video'], f"{r['classifier']}_{r['tilesize']}_{r['rate']}",
                             'score', 'runtime.jsonl')))
             for r in runs]
    # Then the detector logs the manifest lists.
    files += [(r['dataset'], r['video'], 'detector', None, r['runtime_file']) for r in rows]
    # A missing log would silently drop a video from one side of the comparison.
    absent = [f[4] for f in files if not os.path.exists(f[4])]
    assert not absent, f'{len(absent)} runtime logs missing, e.g. {absent[:3]}'
    return files


def load_runtime_lines(con: duckdb.DuckDBPyConnection, files: list[tuple]):
    """Tables runtime_ops(dataset, video, kind, sample_rate, line, op, time), one row per op of every log line, and runtime_logs."""
    # Which dataset, video, side, and run each log belongs to.
    con.execute('CREATE OR REPLACE TABLE runtime_files (dataset VARCHAR, video VARCHAR, kind VARCHAR, sample_rate INTEGER, '
                'filename VARCHAR)')
    con.executemany('INSERT INTO runtime_files VALUES (?, ?, ?, ?, ?)', files)
    # Read each log whole and split it into lines, keeping their order.
    con.execute("""
        CREATE OR REPLACE TABLE runtime_text AS
        SELECT filename, last_modified, string_split(trim(content, chr(10)), chr(10)) AS lines
        FROM read_text($paths)
    """, {'paths': [f[4] for f in files]})
    # Number the lines from 1 and parse each non-blank line as a [{op, time}] list.
    con.execute("""
        CREATE OR REPLACE TABLE runtime_parsed AS
        WITH numbered AS (
            SELECT filename, unnest(lines) AS text, generate_subscripts(lines, 1) AS line FROM runtime_text
        )
        SELECT filename, line, json_transform(text, '[{"op": "VARCHAR", "time": "DOUBLE"}]') AS entries
        FROM numbered WHERE trim(text) <> ''
    """)
    # A line that is not a non-empty list (an object, null, or []) would vanish when unnested.
    bad = query_dicts(con, 'SELECT filename, line FROM runtime_parsed WHERE entries IS NULL OR len(entries) = 0 LIMIT 5')
    assert not bad, f'runtime log lines that are not a non-empty [{{op, time}}] list: {bad}'
    # One row per op, labeled with its log's dataset, video, side, and run.
    con.execute("""
        CREATE OR REPLACE TABLE runtime_ops AS
        WITH ops AS (SELECT filename, line, unnest(entries, recursive := true) FROM runtime_parsed)
        SELECT f.dataset, f.video, f.kind, f.sample_rate, o.line, o.op, o.time FROM ops o JOIN runtime_files f USING (filename)
    """)
    # When each log was written, to show which runs the timings come from.
    con.execute("""
        CREATE OR REPLACE TABLE runtime_logs AS
        SELECT f.dataset, f.video, f.kind, f.sample_rate, t.last_modified FROM runtime_text t JOIN runtime_files f USING (filename)
    """)


def create_sampling_macro(con: duckdb.DuckDBPyConnection):
    """Macro p020_sampled(frames, rate): how many of a video's frames p020 classifies, frames 0, s, 2s, ... and the last."""
    # Ceil(frames / rate) multiples of the rate, plus the last frame when it is not one of them; no frames, no samples.
    con.execute("""
        CREATE OR REPLACE MACRO p020_sampled(frames, rate) AS
        CASE WHEN frames <= 0 THEN 0
             ELSE (frames + rate - 1) // rate + CASE WHEN (frames - 1) % rate <> 0 THEN 1 ELSE 0 END END
    """)


def build_frame_tables(con: duckdb.DuckDBPyConnection):
    """Tables detector_frames, classifier_batches, and classifier_frames, after checking every log.

    detector_frames(dataset, video, frame, ms); classifier_batches(dataset, video, sample_rate, line, ms,
    inference_ms, frames, last); classifier_frames(dataset, video, sample_rate, ms, k).
    """
    # The sampling rule sizes each run's batches.
    create_sampling_macro(con)
    # Parameters shared by the checks and the batch sizing.
    params = {'batch_ops': CLASSIFIER_BATCH_OPS, 'tail_ops': CLASSIFIER_TAIL_OPS, 'batch': CLASSIFIER_BATCH}
    # One row per log line, with its sorted op names, the time of the ops that count, and invalid values.
    con.execute("""
        CREATE OR REPLACE TABLE runtime_lines AS
        SELECT dataset, video, kind, sample_rate, line, list_sort(list(op)) AS ops,
               coalesce(sum(time) FILTER (NOT list_contains($excluded, op)), 0) AS ms,
               coalesce(sum(time) FILTER (op = 'retrieve'), 0) AS retrieve_ms,
               coalesce(sum(time) FILTER (op = 'inference'), 0) AS inference_ms,
               count(*) FILTER (op IS NULL OR time IS NULL OR time < 0 OR isnan(time) OR isinf(time)) AS bad_values
        FROM runtime_ops GROUP BY ALL
    """, {'excluded': EXCLUDED_OPS})
    # Every op needs a name and a finite, non-negative number of milliseconds.
    bad = query_dicts(con, 'SELECT dataset, video, kind, sample_rate, line FROM runtime_lines WHERE bad_values > 0 LIMIT 5')
    assert not bad, f'missing op names or missing, negative, or non-finite times in runtime logs: {bad}'
    # Every detector line must hold exactly one read and one detect.
    bad = query_dicts(con, "SELECT dataset, video, line, ops FROM runtime_lines WHERE kind = 'detector' AND ops <> $ops LIMIT 5",
                      {'ops': DETECTOR_OPS})
    assert not bad, f'unexpected detector log lines: {bad}'
    # Each classifier log must have ceil(sampled frames / batch) batch lines, then one read/retrieve line.
    bad = query_dicts(con, """
        WITH frames AS (
            SELECT dataset, video, count(*) AS frames FROM runtime_lines WHERE kind = 'detector' GROUP BY ALL
        ),
        classifier AS (
            SELECT dataset, video, sample_rate, count(*) FILTER (ops = $batch_ops) AS batches,
                   count(*) FILTER (ops = $tail_ops) AS tails,
                   count(*) FILTER (ops <> $batch_ops AND ops <> $tail_ops) AS unknown,
                   max(line) FILTER (ops = $tail_ops) AS tail_line, max(line) AS last_line
            FROM runtime_lines WHERE kind = 'classifier' GROUP BY ALL
        )
        SELECT dataset, video, c.sample_rate, f.frames, c.batches, c.tails, c.unknown
        FROM frames f FULL JOIN classifier c USING (dataset, video)
        WHERE f.frames IS NULL OR c.batches IS NULL
              OR c.batches <> (p020_sampled(f.frames, c.sample_rate) + $batch - 1) // $batch
              OR c.tails <> 1 OR c.unknown > 0 OR c.tail_line <> c.last_line
        LIMIT 5
    """, params)
    assert not bad, f'classifier logs do not hold one line per {CLASSIFIER_BATCH} sampled frames plus a read/retrieve line: {bad}'
    # One row per detected frame.
    con.execute("""
        CREATE OR REPLACE TABLE detector_frames AS
        SELECT dataset, video, line AS frame, ms FROM runtime_lines WHERE kind = 'detector'
    """)
    # One row per batch: the i-th batch line (from 0) holds min(batch, sampled - batch * i) frames.
    con.execute("""
        CREATE OR REPLACE TABLE classifier_batches AS
        WITH frames AS (SELECT dataset, video, count(*) AS frames FROM detector_frames GROUP BY ALL),
        numbered AS (
            SELECT c.dataset, c.video, c.sample_rate, c.line, c.ms, c.inference_ms,
                   p020_sampled(f.frames, c.sample_rate) AS sampled,
                   row_number() OVER (PARTITION BY c.dataset, c.video, c.sample_rate ORDER BY c.line) - 1 AS i
            FROM runtime_lines c JOIN frames f USING (dataset, video)
            WHERE c.kind = 'classifier' AND c.ops = $batch_ops
        )
        SELECT dataset, video, sample_rate, line, ms, inference_ms, least($batch, sampled - $batch * i) AS frames,
               sampled <= $batch * (i + 1) AS last
        FROM numbered
    """, {'batch': CLASSIFIER_BATCH, 'batch_ops': CLASSIFIER_BATCH_OPS})
    # One row per classified frame, carrying an even share of its batch's time.
    con.execute("""
        CREATE OR REPLACE TABLE classifier_frames AS
        SELECT dataset, video, sample_rate, ms / frames AS ms, unnest(range(frames)) AS k FROM classifier_batches
    """)


def summarize_runs(con: duckdb.DuckDBPyConnection):
    """Table classifier_runs: per dataset and sample rate, cost per classified frame, inference spikes, and log dates."""
    # Per run: per-frame and per-batch costs, batches whose inference exceeds SPIKE_FACTOR x the run's median, the
    # retrieve tail, and when the run's logs were written.
    con.execute("""
        CREATE OR REPLACE TABLE classifier_runs AS
        WITH per_frame AS (
            SELECT dataset, sample_rate, count(*) AS frames, median(ms) AS median_ms, avg(ms) AS mean_ms
            FROM classifier_frames GROUP BY ALL
        ),
        per_batch AS (
            SELECT dataset, sample_rate, count(*) AS batches, count(DISTINCT video) AS videos, sum(ms) AS total_ms,
                   median(inference_ms) AS median_inference_ms
            FROM classifier_batches GROUP BY ALL
        ),
        flagged AS (
            SELECT b.dataset, b.sample_rate, b.last, b.inference_ms - p.median_inference_ms AS excess_ms,
                   p.median_inference_ms > 0 AND b.inference_ms > $factor * p.median_inference_ms AS spike
            FROM classifier_batches b JOIN per_batch p USING (dataset, sample_rate)
        ),
        spikes AS (
            SELECT dataset, sample_rate, count(*) FILTER (spike) AS spike_batches,
                   count(*) FILTER (spike AND last) AS spike_last_batches,
                   coalesce(sum(excess_ms) FILTER (spike), 0) AS spike_excess_ms
            FROM flagged GROUP BY ALL
        ),
        tail AS (
            SELECT dataset, sample_rate, sum(retrieve_ms) AS retrieve_ms FROM runtime_lines WHERE kind = 'classifier' GROUP BY ALL
        ),
        logs AS (
            SELECT dataset, sample_rate, min(last_modified)::VARCHAR AS first_written, max(last_modified)::VARCHAR AS last_written
            FROM runtime_logs WHERE kind = 'classifier' GROUP BY ALL
        )
        SELECT * FROM per_frame JOIN per_batch USING (dataset, sample_rate) JOIN spikes USING (dataset, sample_rate)
        JOIN tail USING (dataset, sample_rate) JOIN logs USING (dataset, sample_rate)
    """, {'factor': SPIKE_FACTOR})


def summarize_frames(con: duckdb.DuckDBPyConnection):
    """Table frame_summary: per dataset, detector cost per frame and the every-frame classifier run over the same frames."""
    # The detector side, joined with the run that classifies every frame.
    con.execute("""
        CREATE OR REPLACE TABLE frame_summary AS
        WITH detector AS (
            SELECT dataset, count(DISTINCT video) AS videos, count(*) AS frames, median(ms) AS median_ms, avg(ms) AS mean_ms
            FROM detector_frames GROUP BY ALL
        ),
        logs AS (
            SELECT dataset, min(last_modified) AS first_written, max(last_modified) AS last_written
            FROM runtime_logs WHERE kind = 'detector' GROUP BY ALL
        )
        SELECT d.dataset, d.videos, d.frames, r.frames AS classifier_frames, r.batches,
               r.median_ms AS classifier_median_ms, r.mean_ms AS classifier_mean_ms, r.retrieve_ms / d.frames AS retrieve_ms,
               r.first_written AS classifier_first_written, r.last_written AS classifier_last_written,
               d.median_ms AS detector_median_ms, d.mean_ms AS detector_mean_ms,
               l.first_written::VARCHAR AS detector_first_written, l.last_written::VARCHAR AS detector_last_written,
               epoch(l.first_written) AS detector_first_epoch
        FROM detector d JOIN classifier_runs r ON r.dataset = d.dataset AND r.sample_rate = $every
        JOIN logs l ON l.dataset = d.dataset
    """, {'every': EVERY_FRAME_RATE})
    # Both sides must cover exactly the frames behind the reported throughput; the classifier half is a guard,
    # since classifier frames are sized from the detector's frame count.
    bad = query_dicts(con, """
        SELECT s.dataset, s.frames, s.classifier_frames, c.frame_count FROM frame_summary s JOIN chosen c USING (dataset)
        WHERE s.frames <> c.frame_count OR s.classifier_frames <> c.frame_count
    """)
    assert not bad, f'classifier, detector, and tradeoff frame counts differ: {bad}'
    # Every chosen dataset must have a summary.
    missing = query_dicts(con, 'SELECT dataset FROM chosen WHERE dataset NOT IN (SELECT dataset FROM frame_summary)')
    assert not missing, f'no per-frame timings for {missing}'


def tile_mask(dataset: str, tilesize: int) -> dict:
    """Tiles per frame and how many the classifier can run inference on (p020's always-relevant mask)."""
    # The mask p020 loads: tiles never relevant in the training split are prepared but skipped at inference.
    mask = np.load(cache.index(dataset, 'never-relevant', f'{tilesize}_all.npy'))
    return dict(grid=list(mask.shape), tiles=int(mask.size), tiles_inferred_at_most=int(np.count_nonzero(mask)))


def detector_model(dataset: str, first_log_epoch: float) -> dict:
    """The reference detector configs/detectors.yaml maps the dataset to now; the logs do not record which model ran."""
    # The detector configuration the pipeline reads.
    config = get_config('detectors.yaml')
    # Map the dataset to its detector entry, as polyis.models.detector.get_detector does.
    entry = config['dataset_detector_mapping'][config['dataset_name_mapping'][dataset]]
    # The fields that identify the model, and where they come from.
    out = {k: entry[k] for k in ('detector', 'model_path') if k in entry}
    out['source'] = 'configs/detectors.yaml when this script ran; not recorded in the logs'
    # RetinaNet (B3D) takes its weights from configs/retinanet.json instead, as polyis.models.retinanet_b3d does.
    if entry.get('detector') == 'retina':
        with open(os.path.join(CONFIG_DIR, RETINANET_CONFIG)) as f:
            out['model_path'] = json.load(f)['weights']
        out['source'] = (f'configs/detectors.yaml and configs/{RETINANET_CONFIG} when this script ran; '
                         'not recorded in the logs')
    # The weights must predate the first detector log to be the model that produced it (necessary, not sufficient).
    path = out.get('model_path')
    out['weights_found'] = bool(path) and os.path.exists(path)
    if out['weights_found']:
        # The weights' modification time, in UTC.
        modified = os.path.getmtime(path)
        out['weights_modified'] = datetime.fromtimestamp(modified, timezone.utc).isoformat(timespec='seconds')
        out['weights_predate_first_log'] = modified < first_log_epoch
    return out


def run_entry(run: dict) -> dict:
    """One classifier run: cost per classified frame, its ratio to the detector, inference spikes, and log dates."""
    # The run's mean cost with its spikes' excess over the median removed, as a sensitivity check.
    without = (run['total_ms'] - run['spike_excess_ms']) / run['frames']
    # Rounded for the report.
    return dict(
        classified_frames=run['frames'], batches=run['batches'], videos=run['videos'],
        mean_ms_per_classified_frame=round(run['mean_ms'], 4), median_ms_per_classified_frame=round(run['median_ms'], 4),
        ratio_of_means=round(run['ratio_of_means'], 1), logs_written=[run['first_written'], run['last_written']],
        inference_spikes=dict(batches=run['spike_batches'], last_batch_of_video=run['spike_last_batches'],
                              excess_ms=round(run['spike_excess_ms'], 2),
                              excess_share_pct=round(100 * run['spike_excess_ms'] / run['total_ms'], 2),
                              mean_ms_per_classified_frame_without_excess=round(without, 4)))


def per_frame_report(con: duckdb.DuckDBPyConnection) -> dict:
    """classifier_vs_detector.json content: per dataset (keyed by label) and a cross-dataset summary."""
    # Per dataset: the every-frame summary, host intervals per op, every run, and the cost range across runs.
    rows = query_dicts(con, """
        WITH host AS (
            SELECT o.dataset, map_from_entries(list((o.op, round(o.ms / s.frames, 4)) ORDER BY o.op)) AS ops
            FROM (SELECT dataset, op, sum(time) AS ms FROM runtime_ops
                  WHERE kind = 'classifier' AND sample_rate = $every AND list_contains($ops, op) GROUP BY ALL) o
            JOIN frame_summary s USING (dataset) GROUP BY o.dataset
        ),
        runs AS (
            SELECT r.dataset, list(struct_pack(sample_rate := r.sample_rate, frames := r.frames, batches := r.batches,
                       videos := r.videos, mean_ms := r.mean_ms, median_ms := r.median_ms, total_ms := r.total_ms,
                       ratio_of_means := s.detector_mean_ms / r.mean_ms, spike_batches := r.spike_batches,
                       spike_last_batches := r.spike_last_batches, spike_excess_ms := r.spike_excess_ms,
                       first_written := r.first_written, last_written := r.last_written) ORDER BY r.sample_rate) AS runs,
                   min(r.mean_ms) AS min_mean_ms, max(r.mean_ms) AS max_mean_ms
            FROM classifier_runs r JOIN frame_summary s USING (dataset) GROUP BY r.dataset
        )
        SELECT s.*, c.classifier, c.tilesize, c.sample_rate AS operating_rate, h.ops AS host_ops, r.runs,
               r.min_mean_ms, r.max_mean_ms, o.mean_ms AS operating_mean_ms
        FROM frame_summary s JOIN chosen c USING (dataset) JOIN host h USING (dataset) JOIN runs r USING (dataset)
        JOIN classifier_runs o ON o.dataset = s.dataset AND o.sample_rate = c.sample_rate
        ORDER BY s.dataset
    """, {'every': EVERY_FRAME_RATE, 'ops': CLASSIFIER_BATCH_OPS})
    # Arrange each dataset's row as the classifier side, the detector side, their ratios, and the runs.
    datasets = {DATASET_LABELS.get(r['dataset'], r['dataset']): dict(
        dataset=r['dataset'], videos=r['videos'], frames=r['frames'],
        classifier=dict(
            log=f"020_relevancy/{r['classifier']}_{r['tilesize']}_{EVERY_FRAME_RATE}/score/runtime.jsonl",
            sample_rate=EVERY_FRAME_RATE, batch_size=CLASSIFIER_BATCH, batches=r['batches'], ops=CLASSIFIER_BATCH_OPS,
            excluded_ops=CLASSIFIER_TAIL_OPS,
            median_ms_per_frame=round(r['classifier_median_ms'], 3), mean_ms_per_frame=round(r['classifier_mean_ms'], 3),
            fps=round(1000 / r['classifier_mean_ms'], 1), fps_at_median=round(1000 / r['classifier_median_ms'], 1),
            host_interval_ms_per_frame_by_op=r['host_ops'], retrieve_ms_per_frame=round(r['retrieve_ms'], 4),
            logs_written=[r['classifier_first_written'], r['classifier_last_written']],
            **tile_mask(r['dataset'], r['tilesize'])),
        detector=dict(
            log='002_naive/detection_runtime.jsonl', batch_size=1, ops=['detect'], excluded_ops=['read'],
            median_ms_per_frame=round(r['detector_median_ms'], 3), mean_ms_per_frame=round(r['detector_mean_ms'], 3),
            fps=round(1000 / r['detector_mean_ms'], 1), fps_at_median=round(1000 / r['detector_median_ms'], 1),
            logs_written=[r['detector_first_written'], r['detector_last_written']],
            model=detector_model(r['dataset'], r['detector_first_epoch'])),
        # Headline: ratio of means, which equals the ratio of throughputs; medians as a robustness check.
        ratio_of_means=round(r['detector_mean_ms'] / r['classifier_mean_ms'], 1),
        ratio_of_medians=round(r['detector_median_ms'] / r['classifier_median_ms'], 1),
        classifier_runs={str(run['sample_rate']): run_entry(run) for run in r['runs']},
        cost_across_runs=dict(
            sample_rates=[run['sample_rate'] for run in r['runs']], operating_point_sample_rate=r['operating_rate'],
            mean_ms_per_classified_frame=dict(min=round(r['min_mean_ms'], 4), max=round(r['max_mean_ms'], 4),
                                              operating_point=round(r['operating_mean_ms'], 4)),
            ratio_of_means=dict(min=round(r['detector_mean_ms'] / r['max_mean_ms'], 1),
                                max=round(r['detector_mean_ms'] / r['min_mean_ms'], 1),
                                operating_point=round(r['detector_mean_ms'] / r['operating_mean_ms'], 1)),
            spread_pct=round(100 * (r['max_mean_ms'] / r['min_mean_ms'] - 1), 1)),
    ) for r in rows}
    # Across datasets: the every-frame ratio, the operating-point runs' ratio, and every run's ratio.
    summary = query_dicts(con, """
        WITH runs AS (
            SELECT r.dataset, s.detector_mean_ms / r.mean_ms AS ratio, r.sample_rate = $every AS every,
                   r.sample_rate = c.sample_rate AS operating
            FROM classifier_runs r JOIN frame_summary s USING (dataset) JOIN chosen c USING (dataset)
        )
        SELECT struct_pack(median := round(median(ratio) FILTER (every), 1), min := round(min(ratio) FILTER (every), 1),
                           max := round(max(ratio) FILTER (every), 1), datasets := count(*) FILTER (every)) AS every_frame_run,
               struct_pack(median := round(median(ratio) FILTER (operating), 1), min := round(min(ratio) FILTER (operating), 1),
                           max := round(max(ratio) FILTER (operating), 1), datasets := count(*) FILTER (operating)) AS operating_point_run,
               struct_pack(min := round(min(ratio), 1), max := round(max(ratio), 1), runs := count(*)) AS all_runs
        FROM runs
    """, {'every': EVERY_FRAME_RATE})[0]
    # Per-dataset entries and the cross-dataset ratio summary.
    return dict(datasets=datasets, summary=dict(ratio_of_means=summary))


def load_operator_ops(con: duckdb.DuckDBPyConnection, per_op_tables: dict[str, str]):
    """Tables operator_rows(dataset, system, stage, op, video, seconds, records) and operator_ops (summed over videos).

    Both hold the chosen configuration and the reference. `per_op_tables` maps each dataset to p121's
    query_execution_per_op.csv; needs table chosen.
    """
    # A missing table would otherwise surface as a bare DuckDB error.
    assert_files_exist(list(per_op_tables.values()), 'per-op throughput tables (run evaluation/p121 --test)')
    # p121's per-op, per-video rows (seconds) for each dataset's chosen configuration and its reference, test split.
    con.execute("""
        CREATE OR REPLACE TABLE operator_rows AS
        WITH wanted AS (
            SELECT dataset, 'tetris' AS system, config FROM chosen
            UNION ALL SELECT dataset, 'reference', 'naive' FROM chosen
        )
        SELECT w.dataset, w.system, p.stage, p.op, p.video, p.time AS seconds, p.count AS records
        FROM read_csv($paths, union_by_name := true,
                      types := {'op': 'VARCHAR', 'time': 'DOUBLE', 'count': 'BIGINT', 'stage': 'VARCHAR', 'dataset': 'VARCHAR',
                                'videoset': 'VARCHAR', 'video': 'VARCHAR', 'variant_id': 'VARCHAR'}) p
        JOIN wanted w ON p.dataset = w.dataset AND p.variant_id = w.config
        WHERE p.videoset = 'test'
    """, {'paths': list(per_op_tables.values())})
    # Per-op totals over the test split.
    con.execute("""
        CREATE OR REPLACE TABLE operator_ops AS
        SELECT dataset, system, stage, op, sum(seconds) AS seconds, sum(records) AS records FROM operator_rows GROUP BY ALL
    """)


def check_operator_ops(con: duckdb.DuckDBPyConnection):
    """Fail unless the per-op table covers every stage and test video and sums to the tradeoff runtime.

    Needs tables chosen, operator_rows, operator_ops, and detector_frames (for the test videos).
    """
    # Stages outside the known pipelines would be silently mislabeled.
    bad = query_dicts(con, """
        SELECT DISTINCT dataset, system, stage FROM operator_ops
        WHERE NOT list_contains(CASE system WHEN 'tetris' THEN $tetris ELSE $reference END, stage)
    """, {'tetris': list(STAGE_OPERATORS), 'reference': list(REFERENCE_STAGES)})
    assert not bad, f'unknown stages in the per-op table: {bad}'
    # Every stage except pruning must run; pruning runs exactly when the configuration sets a threshold.
    bad = query_dicts(con, """
        WITH present AS (SELECT dataset, system, list(DISTINCT stage) AS stages FROM operator_ops GROUP BY ALL),
        expected AS (
            SELECT c.dataset, c.tracking_accuracy_threshold, s.system
            FROM chosen c CROSS JOIN (VALUES ('tetris'), ('reference')) s(system)
        )
        SELECT e.dataset, e.system, p.stages FROM expected e LEFT JOIN present p USING (dataset, system)
        WHERE p.stages IS NULL
           OR (e.system = 'tetris' AND (NOT list_has_all(p.stages, $required)
               OR list_contains(p.stages, $prune) <> (e.tracking_accuracy_threshold IS NOT NULL)))
           OR (e.system = 'reference' AND NOT list_has_all(p.stages, $reference))
    """, {'required': [s for s in STAGE_OPERATORS if s != PRUNE_STAGE], 'prune': PRUNE_STAGE,
          'reference': list(REFERENCE_STAGES)})
    assert not bad, f'operating points or references with missing stages: {bad}'
    # Every stage must cover exactly the test videos of the comparison; an op may be absent from some videos.
    bad = query_dicts(con, """
        WITH videos AS (SELECT DISTINCT dataset, video FROM detector_frames),
        stages AS (
            SELECT dataset, system, stage, count(DISTINCT o.video) AS videos,
                   count(DISTINCT o.video) FILTER (v.video IS NOT NULL) AS known
            FROM operator_rows o LEFT JOIN videos v USING (dataset, video) GROUP BY ALL
        )
        SELECT s.dataset, s.system, s.stage, s.videos, s.known, n.videos AS expected
        FROM stages s JOIN (SELECT dataset, count(*) AS videos FROM videos GROUP BY ALL) n USING (dataset)
        WHERE s.videos <> n.videos OR s.known <> n.videos
    """)
    assert not bad, f'stages that do not cover every test video: {bad}'
    # The operators must add up to the runtime behind the reported throughput.
    bad = query_dicts(con, """
        WITH total AS (SELECT dataset, system, sum(seconds) AS seconds FROM operator_ops GROUP BY ALL),
        compared AS (
            SELECT t.dataset, t.system, t.seconds,
                   CASE t.system WHEN 'tetris' THEN c.seconds ELSE c.reference_seconds END AS expected
            FROM total t JOIN chosen c USING (dataset)
        )
        SELECT * FROM compared WHERE abs(seconds - expected) > $tol * expected
    """, {'tol': RUNTIME_TOLERANCE})
    assert not bad, f'per-op totals differ from the tradeoff runtime: {bad}'


def check_classify_against_runs(con: duckdb.DuckDBPyConnection):
    """Fail unless p121's classify time and batch count at each operating point equal that sample rate's p020 log.

    This ties the classify share to a run in classifier_vs_detector.json, whose batch count already
    follows p020's sampling rule. Needs tables chosen, operator_ops, and classifier_runs.
    """
    # p121's classify seconds and inference count (one per batch) against the operating point's raw p020 log.
    bad = query_dicts(con, """
        WITH p121 AS (
            SELECT dataset, sum(seconds) AS seconds, sum(records) FILTER (op = 'inference') AS batches
            FROM operator_ops WHERE system = 'tetris' AND stage = $stage GROUP BY ALL
        )
        SELECT c.dataset, c.sample_rate, p.seconds, r.total_ms / 1000 AS log_seconds, p.batches, r.batches AS log_batches
        FROM chosen c LEFT JOIN p121 p USING (dataset)
        LEFT JOIN classifier_runs r ON r.dataset = c.dataset AND r.sample_rate = c.sample_rate
        WHERE p.seconds IS NULL OR r.total_ms IS NULL OR p.batches IS DISTINCT FROM r.batches
           OR abs(p.seconds - r.total_ms / 1000) > $tol * p.seconds
    """, {'stage': CLASSIFY_STAGE, 'tol': RUNTIME_TOLERANCE})
    assert not bad, f'classify time or batch count at the operating point differs from its p020 log: {bad}'


def ratio_vs_share(p: dict) -> dict:
    """Why the per-frame ratio and the classifier's share differ at one operating point, from the measured costs.

    `p` is one row of share_report's per-dataset query.
    """
    # Mean cost per frame when every frame is classified or detected, from the per-frame logs.
    every_classify, every_detect = p['classifier_mean_ms'], p['detector_mean_ms']
    # Cost per classified frame in the operating point's own run, from p121's classify time.
    run_classify = 1000 * p['classify_seconds'] / p['classified_frames']
    # Cost per original frame at the operating point, from p121's accounting.
    classify_ms, detect_ms = p['classify_ms_per_frame'], p['detect_ms_per_frame']
    # Sampling cuts classified frames; the operating-point run's own cost per classified frame differs too.
    sampling, run_cost = p['frame_count'] / p['classified_frames'], run_classify / every_classify
    # The per-frame ratio of means, and the classifier's measured share.
    ratio, share = every_detect / every_classify, p['classify_share']
    # The rounded factors and a sentence that walks from the ratio to the share.
    return dict(
        sample_rate=p['sample_rate'], classified_frames=p['classified_frames'], frames=p['frame_count'],
        sampling_factor=round(sampling, 3),
        classifier_ms_per_frame_every_frame=round(every_classify, 4),
        classifier_ms_per_classified_frame=round(run_classify, 4),
        per_classified_frame_cost_vs_every_frame=round(run_cost, 3),
        classify_ms_per_original_frame=round(classify_ms, 4),
        classification_reduction=round(every_classify / classify_ms, 2),
        reference_detector_ms_per_frame=round(every_detect, 3),
        detect_ms_per_original_frame=round(detect_ms, 4),
        detection_reduction=round(every_detect / detect_ms, 1),
        per_frame_ratio_of_means=round(ratio, 1), classify_share_pct=round(100 * share, 1),
        share_if_every_frame_and_nothing_else_pct=round(100 / (1 + ratio), 1),
        explanation=(
            f'Per frame, classifying costs {every_classify:.3f} ms and reference detection {every_detect:.1f} ms '
            f'(means; {ratio:.0f}x). At sample rate {p["sample_rate"]} the classifier runs on {p["classified_frames"]} '
            f'of {p["frame_count"]} frames ({sampling:.2f}x fewer). The operating point\'s run cost {run_classify:.3f} ms '
            f'per classified frame, {100 * (run_cost - 1):+.0f}% against the every-frame run, so classification costs '
            f'{classify_ms:.3f} ms per original frame ({every_classify / classify_ms:.2f}x less than classifying every '
            f'frame). Detection on packed canvases costs {detect_ms:.3f} ms per original frame ({every_detect / detect_ms:.1f}x '
            f'less than reference detection). Classification is {100 * share:.1f}% of extraction, against '
            f'{100 / (1 + ratio):.1f}% if both ran on every frame and nothing else ran.'),
    )


def share_report(con: duckdb.DuckDBPyConnection, loss: float) -> dict:
    """operator_share.json content: per dataset, each operator's time and share, and a cross-dataset summary.

    Needs tables chosen, operator_ops (checked), frame_summary, and classifier_runs.
    """
    # Operator name of every Tetris and reference stage.
    con.execute('CREATE OR REPLACE TABLE stage_names (stage VARCHAR, operator VARCHAR)')
    con.executemany('INSERT INTO stage_names VALUES (?, ?)', list(STAGE_OPERATORS.items()) + list(REFERENCE_STAGES.items()))
    # Seconds, ms per original frame, and share of the pipeline's total, per operator.
    con.execute("""
        CREATE OR REPLACE TABLE operator_totals AS
        WITH stages AS (
            SELECT o.dataset, o.system, n.operator, sum(o.seconds) AS seconds
            FROM operator_ops o JOIN stage_names n USING (stage) GROUP BY ALL
        )
        SELECT s.dataset, s.system, s.operator, s.seconds, 1000 * s.seconds / c.frame_count AS ms_per_frame,
               s.seconds / sum(s.seconds) OVER (PARTITION BY s.dataset, s.system) AS share
        FROM stages s JOIN chosen c USING (dataset)
    """)
    # Per-op seconds and share of the pipeline's total, for audits such as the canvas read inside detect.
    con.execute("""
        CREATE OR REPLACE TABLE operator_op_shares AS
        SELECT o.dataset, o.system, n.operator, o.op, o.seconds,
               o.seconds / sum(o.seconds) OVER (PARTITION BY o.dataset, o.system) AS share
        FROM operator_ops o JOIN stage_names n USING (stage)
    """)
    # Per dataset and system, the operators and their ops as rounded maps for the report.
    con.execute("""
        CREATE OR REPLACE TABLE report_maps AS
        WITH ops AS (
            SELECT dataset, system, operator,
                   map_from_entries(list((op, struct_pack(seconds := round(seconds, 3), share_pct := round(100 * share, 2)))
                                         ORDER BY op)) AS ops
            FROM operator_op_shares GROUP BY ALL
        ),
        ops_by_system AS (
            SELECT dataset, system, map_from_entries(list((operator, ops) ORDER BY operator)) AS ops FROM ops GROUP BY ALL
        ),
        operators AS (
            SELECT dataset, system,
                   map_from_entries(list((operator, struct_pack(seconds := round(seconds, 3), ms_per_frame := round(ms_per_frame, 4),
                                                                share_pct := round(100 * share, 1))) ORDER BY operator)) AS operators
            FROM operator_totals GROUP BY ALL
        )
        SELECT * FROM operators JOIN ops_by_system USING (dataset, system)
    """)
    # One row per dataset: the operating point, both runs' costs, the classify and detect totals, and the maps.
    points = query_dicts(con, """
        SELECT c.*, s.classifier_mean_ms, s.detector_mean_ms, r.frames AS classified_frames,
               r.spike_batches, r.spike_last_batches, r.spike_excess_ms / 1000 AS spike_excess_seconds,
               cl.seconds AS classify_seconds, cl.ms_per_frame AS classify_ms_per_frame, cl.share AS classify_share,
               de.ms_per_frame AS detect_ms_per_frame,
               (cl.seconds - r.spike_excess_ms / 1000) / (c.seconds - r.spike_excess_ms / 1000) AS classify_share_without_spikes,
               tm.operators, tm.ops, rm.operators AS reference_operators, rm.ops AS reference_ops
        FROM chosen c JOIN frame_summary s USING (dataset)
        JOIN classifier_runs r ON r.dataset = c.dataset AND r.sample_rate = c.sample_rate
        JOIN operator_totals cl ON cl.dataset = c.dataset AND cl.system = 'tetris' AND cl.operator = 'classify'
        JOIN operator_totals de ON de.dataset = c.dataset AND de.system = 'tetris' AND de.operator = 'detect'
        JOIN report_maps tm ON tm.dataset = c.dataset AND tm.system = 'tetris'
        JOIN report_maps rm ON rm.dataset = c.dataset AND rm.system = 'reference'
        ORDER BY c.dataset
    """)
    # Arrange each dataset's row; operators that did not run (pruning without a threshold) are null, not zero.
    datasets = {DATASET_LABELS.get(p['dataset'], p['dataset']): dict(
        dataset=p['dataset'], config=p['config'], classifier=p['classifier'], tilesize=p['tilesize'],
        sample_rate=p['sample_rate'], pruning_threshold=p['tracking_accuracy_threshold'],
        hota=round(p['accuracy'], 4), reference_hota=round(p['reference_accuracy'], 4),
        hota_loss_pct=round(100 * (1 - p['accuracy'] / p['reference_accuracy']), 2),
        fps=round(p['fps'], 1), reference_fps=round(p['reference_fps'], 1), frames=p['frame_count'],
        extraction_seconds=round(p['seconds'], 3),
        operators={op: p['operators'].get(op) for op in STAGE_OPERATORS.values()},
        ops=p['ops'],
        detect_canvas_read_share_pct=p['ops']['detect'].get('read', {}).get('share_pct'),
        classify_inference_spikes=dict(
            batches=p['spike_batches'], last_batch_of_video=p['spike_last_batches'],
            excess_seconds=round(p['spike_excess_seconds'], 3),
            share_of_classify_pct=round(100 * p['spike_excess_seconds'] / p['classify_seconds'], 1),
            classify_share_without_excess_pct=round(100 * p['classify_share_without_spikes'], 1)),
        reference=dict(seconds=round(p['reference_seconds'], 3), operators=p['reference_operators'], ops=p['reference_ops'],
                       detect_frame_read_share_pct=p['reference_ops']['detect'].get('read', {}).get('share_pct')),
        per_frame_vs_share=ratio_vs_share(p),
    ) for p in points}
    # Median and range of each operator's share across the datasets where it ran (Tetris only).
    summary = query_dicts(con, """
        SELECT map_from_entries(list((operator, struct_pack(median_pct := round(100 * median, 1), min_pct := round(100 * lo, 1),
                                                            max_pct := round(100 * hi, 1), datasets := n)) ORDER BY operator)) AS m
        FROM (SELECT operator, median(share) AS median, min(share) AS lo, max(share) AS hi, count(*) AS n
              FROM operator_totals WHERE system = 'tetris' GROUP BY ALL)
    """)[0]['m']
    # Per-dataset entries and the cross-dataset share summary.
    return dict(loss=loss, datasets=datasets, summary=summary)


def host_info() -> dict:
    """The host whose cache was read; the runs presumably executed on it, since the cache is local, but no log says so."""
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


def hardware_notes(con: duckdb.DuckDBPyConnection) -> dict:
    """What is and is not known about the machine behind the timings, with log dates derived from the logs. Needs runtime_logs."""
    # Per side, when its logs were written, and how many log sets (dataset x run) overlap another set's write window.
    windows = {r['kind']: r for r in query_dicts(con, """
        WITH sets AS (
            SELECT kind, dataset || '/' || coalesce(sample_rate::VARCHAR, '-') AS name,
                   min(last_modified) AS first, max(last_modified) AS last
            FROM runtime_logs GROUP BY ALL
        )
        SELECT a.kind, min(a.first)::VARCHAR AS first, max(a.last)::VARCHAR AS last, epoch(min(a.first)) AS first_epoch,
               epoch(max(a.last)) AS last_epoch, count(DISTINCT a.name) AS log_sets,
               count(DISTINCT a.name) FILTER (b.name IS NOT NULL) AS overlapping_sets
        FROM sets a LEFT JOIN sets b ON a.kind = b.kind AND a.name <> b.name AND a.first <= b.last AND b.first <= a.last
        GROUP BY a.kind
    """)}
    # The two sides' write windows.
    c, d = windows['classifier'], windows['detector']
    # Whether the classifier and detector logs were written during overlapping periods.
    overlap = c['first_epoch'] <= d['last_epoch'] and d['first_epoch'] <= c['last_epoch']
    # The logged-hardware statement, the host, the write windows, and how to describe the timings.
    return dict(
        recorded_in_logs=LOGGED_HARDWARE,
        host_that_read_the_cache=host_info(),
        paper=PAPER_HARDWARE,
        logs_written={k: dict(first=w['first'], last=w['last'], log_sets=w['log_sets'],
                              sets_overlapping_another_set=w['overlapping_sets']) for k, w in windows.items()},
        note=(f'The classifier logs were written {c["first"]} to {c["last"]} and the detector logs {d["first"]} to '
              f'{d["last"]}; ' + ('the two windows overlap.' if overlap else
                                  'the windows do not overlap, so the two sides come from different runs.') +
              f' {c["overlapping_sets"]} of {c["log_sets"]} classifier log sets (dataset x sample rate) and '
              f'{d["overlapping_sets"]} of {d["log_sets"]} detector log sets were written while another set of the '
              f'same side was being written, so other GPUs were likely busy at the same time, sharing CPUs and PCIe. '
              f'Describe these as timings from the pipeline\'s existing logs on the evaluation server; "same hardware" '
              f'needs a matched, synchronized re-timing of both models on one idle GPU.'),
    )


def method_notes(loss: float, hardware: dict) -> dict:
    """Inputs, boundaries, and caveats, written into both JSON files."""
    # One entry per topic.
    return dict(
        operating_point=f'Fastest Tetris test configuration with HOTA >= reference HOTA x (1 - {loss}) (common.fastest_within).',
        timing_source=('Existing pipeline logs only, nothing rerun. Every test video and frame of each dataset; no sampling. '
                       'Each value comes from a single run: one p002 run for the detector, one p020 run per sample rate.'),
        classifier_boundary=('Ops send, resize, diff, reshape, mask, inference, collect per 16-frame batch, split evenly over '
                             'the batch; read (video decoding) and retrieve (copy to CPU, hex encoding) excluded, as in p121.'),
        classifier_runs=(f'The comparison uses the sample-rate-{EVERY_FRAME_RATE} run, which classifies every frame. p020 also ran '
                         'the same classifier and tile size at every other sample rate on the same test videos; their mean '
                         'cost per classified frame (classifier_runs, cost_across_runs) is the run-to-run spread under '
                         'concurrent multi-GPU load, which is wider than the precision of any single ratio. Quote a rounded '
                         'range (summary.ratio_of_means.all_runs), not one run\'s ratio. The operating point\'s run is the '
                         'one p121 counts (checked: equal seconds and batch counts).'),
        ratio=('Headline ratio_of_means = detector mean / classifier mean = classifier fps / detector fps, consistent with '
               'throughput (frames over total time). ratio_of_medians is a robustness check.'),
        detector_boundary='Op detect per frame (batch size 1); read (video decoding) excluded.',
        gpu_synchronization=(
            'p020 times ops on the host (time.time_ns) without torch.cuda.synchronize. Its frames are pageable numpy arrays, '
            'and a host-to-device copy from pageable memory first synchronizes the stream, so send also waits for the '
            'previous batch\'s queued GPU work; the only other synchronization per batch is nonzero() in mask. inference '
            'and collect mostly launch kernels. host_interval_ms_per_frame_by_op is therefore host-side attribution under '
            'asynchronous execution and does not show where GPU time goes (for example, send against inference). Batch '
            'totals are unaffected except for the last batch of a video, whose GPU tail lands in retrieve '
            '(retrieve_ms_per_frame bounds it).'),
        inference_spikes=(f'A batch whose inference interval exceeds {SPIKE_FACTOR}x its run\'s median is a spike. Spikes are '
                          'counted, as in p121; inference_spikes and classify_inference_spikes give their excess over the '
                          'median as a sensitivity check. last_batch_of_video counts spikes in a video\'s last batch, which '
                          'is often partial, a shape the 16-batch warm-up (the first video\'s last full batch) does not '
                          'cover; that cause is not verified.'),
        share_boundary=('p121 accounting (query_execution_per_op.csv). It excludes classify read/retrieve and pack '
                        'save_canvas/save_mapping_files/save_collage, but counts detect read (cv2.imread of the JPEG '
                        'canvases, detect_canvas_read_share_pct), although Section 7 says disk I/O is excluded. Video '
                        'decoding is counted asymmetrically: the reference pipeline\'s runtime includes it (detect read, '
                        'reference.detect_frame_read_share_pct), while Tetris\'s does not (p121 drops classify read, and '
                        'p030 decodes the video again without timing it), so reported speedups over the reference '
                        'include the reference\'s decoding time.'),
        ratio_vs_share=RATIO_VS_SHARE,
        hardware=hardware,
    )


def main():
    # Command-line options.
    args = parse_args()
    # Fail on unknown dataset names before touching the cache.
    unknown = set(args.datasets) - set(DATASET_LABELS)
    assert not unknown, f'unknown datasets {sorted(unknown)}; choose from {sorted(DATASET_LABELS)}'
    # Results directory named after this script, and an in-memory DuckDB.
    out_dir = results_dir(__file__)
    con = duckdb.connect()

    # Operating points from the tradeoff tables.
    load_configs(con, args.datasets)
    choose_operating_points(con, args.loss)

    # Per-frame classifier (every run) and detector cost on every test frame.
    files = runtime_files(con, {d: str(cache.eval(d, 'tp', 'query_execution.csv')) for d in args.datasets})
    load_runtime_lines(con, files)
    build_frame_tables(con)
    summarize_runs(con)
    summarize_frames(con)
    # Notes shared by both files, including the host and when the logs were written.
    method = method_notes(args.loss, hardware_notes(con))
    per_frame = per_frame_report(con)
    # Write the per-frame report with the method notes.
    write_json(out_dir, 'classifier_vs_detector.json', dict(method=method, **per_frame))
    # One line per dataset: the headline ratio and its spread across runs.
    for label, r in per_frame['datasets'].items():
        print(f"{label:10s} classifier {r['classifier']['mean_ms_per_frame']:6.3f} ms/frame | detector "
              f"{r['detector']['mean_ms_per_frame']:6.2f} | ratio of means {r['ratio_of_means']}x "
              f"(medians {r['ratio_of_medians']}x; runs {r['cost_across_runs']['ratio_of_means']})")
    # The cross-dataset ratio summary.
    print('summary', per_frame['summary'])

    # Operator shares at each operating point, from p121's per-op table, tied to the raw classifier logs.
    load_operator_ops(con, {d: str(cache.eval(d, 'tp', 'measurements', 'query_execution_per_op.csv')) for d in args.datasets})
    check_operator_ops(con)
    check_classify_against_runs(con)
    shares = share_report(con, args.loss)
    # Write the share report with the same method notes.
    write_json(out_dir, 'operator_share.json', dict(method=method, **shares))
    # One line per dataset: each operator's share, the canvas reads, and classify without spikes.
    for label, r in shares['datasets'].items():
        cells = ' '.join(f"{op} {'-' if v is None else v['share_pct']:>5}" for op, v in r['operators'].items())
        print(f"{label:10s} s={r['sample_rate']:<2} {cells} | canvas read {r['detect_canvas_read_share_pct']}% | "
              f"classify w/o spikes {r['classify_inference_spikes']['classify_share_without_excess_pct']}%")
    # The cross-dataset share summary.
    print('summary', shares['summary'])


if __name__ == '__main__':
    main()
