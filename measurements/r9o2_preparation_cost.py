#!/usr/local/bin/python
"""R9.O2: Tetris's per-dataset preparation cost and its break-even video duration against the reference pipeline.

Preparation is everything Tetris runs on a camera's training and validation clips before it can
extract tracks from new video of that camera. Each component, and where its time comes from:

  reference detection, training split (p011)    recorded per frame (read, detect)
  training-data creation (p012)                 recorded per frame (split, save); seeking and decoding are untimed
  training-data pruning (p013)                  NOT RECORDED
  classifier training (p014)                    fine-tuning epochs recorded; both stages from file timestamps
  classifier optimization benchmark (p015)      NOT RECORDED; ESTIMATED from output modification times
  mistrack profiling (p016)                     NOT RECORDED; ESTIMATED from output modification times
  reference outputs, validation split (002)     recorded per frame (read, detect, track)
  validation configuration sweep (p020-p060)    recorded per op; decoding, loading, and file I/O partly untimed
  scoring of the sweep (p110)                   NOT RECORDED on validation; ESTIMATED from the test split's scoring
  tradeoff and Pareto selection (p130, p135)    NOT RECORDED (they read one score table per dataset)

Unit. Every figure is process seconds: the time one worker process spends on a unit of work,
summed over units. Stages ran with many workers in parallel (p012, p022, p030 with 20; p040 one
per GPU on 8 GPUs), so these sums are single-worker equivalents, not elapsed time on the host.
p016's elapsed time with 40 workers is converted to an upper bound (elapsed x min(40, videos)).

Two timing boundaries, both reported, since the paper must name the one it quotes (R1.O3).
  instrumented  The ops evaluation/p121_throughput_compute.py counts for the paper's throughput
                (same op lists, same dropped ops), minus video decoding. p121 already drops p020's
                decoding; this boundary also drops the per-frame 'read' of the reference pipeline
                (001) and of p011, so the reference and Tetris are compared like for like. It leaves
                out work no op times, mainly p030's decoding of the whole video once per
                configuration and video, the loading of upstream outputs in p022-p060, p050's debug
                images, and canvas and file writes.
  process       Whole-process time of each work unit, from file timestamps (below), which includes
                all of that. The reference pipeline's detection (preprocess/p002) gets the same
                own-span rule as Tetris's units: it loads the detector before opening its runtime
                file, as p040 loads its detector before its first unit, so detector loading is
                outside on both sides. p011's detection files have stale creation times on some
                datasets (days on Amsterdam and CalDoT1), so p011 keeps its recorded time, which
                times every frame's read and detect.
Components without op records (classifier training, p015, p016, scoring) enter both boundaries at
their process time. The recorded lower bound sums recorded ops only (p121's boundary as is,
fine-tuning epochs only, no estimates, sweep unscaled).

Process time from timestamps (GNU stat, nanosecond creation and modification times on ZFS). Every
stage from p020 to p060 runs one task per configuration that processes the split's videos one
after another in sorted order, and deletes each output directory before recreating it, so the
directory's creation time is fresh. A work unit (one configuration's output on one video) is
chained to the same task's unit on the previous video when it starts no earlier than that unit
ended, at most CHAIN_GAP_S later, and the span is at least the unit's recorded ops; its process
time is then the time between the two units' last writes, which includes work before its
directory was created (loading upstream outputs). The first video of a task, or a unit that
fails those checks, gets max(own span, recorded ops), a lower bound; an own span over
STALE_SPAN_S is taken as a stale creation time and replaced by the recorded ops. p012 and the
reference pipeline's detection run one task per video, so they are never chained and get
max(own span, recorded ops) from their runtime file's creation to their last write.

Shared work is counted once. Every stage writes its runtime file into a directory keyed by
exactly the parameters it depends on, and later stages read earlier outputs from the cache, so the
sweep costs the sum over distinct runtime files of the manifests evaluation/p120_throughput_gather.py
builds. The cost of charging every configuration its full pipeline is reported alongside. A
validation video whose sweep files are not all present (CalDoT2's va00 to va09 were cleared on
2026-05-29 and only one configuration rerun) is left out, and the sweep is ESTIMATED for the full
split by scaling by videos / complete videos.

Classifier training. p014 trains a frozen stage and then a fine-tuning stage of at most 7 epochs
each, and each stage rewrites throughput_per_epoch.jsonl, so the file holds only the fine-tuning
epochs. The result directory is created when training starts, the throughput file when the frozen
stage ends, and the file's last write is the end of fine-tuning. That wall clock is used when both
stage spans agree with the recorded fine-tuning epochs; otherwise the recorded epochs, a lower
bound. Training ran concurrently with other datasets on other GPUs of the same host.

Estimates from modification times. p016 runs every (dataset, tracker) profile back to back in one
invocation, writing one partial file per training video and then the aggregate outputs; a profile
whose first partial file appears within CHAIN_GAP_S after the previous profile's last output
started when that one ended (the first profile of a run gets its own span, a lower bound). p015
benchmarks one resolution group after another in one process and writes the group's datasets
together; a group's time is the gap since the previous group finished (none for the first group).
Scoring per (configuration, video) is the test split's p110 tasks' spans (output directory creation
to last result file; one process per configuration) over their videos, times the validation
configurations and videos. Only the latest run of each survives.

Relevance. Each dataset's operating point is its fastest Tetris configuration within --loss of
the reference pipeline's test HOTA. Only its classifier and tile size are charged for training and
the configured trackers' profiles, since the configured sweep uses both; the rest (p017's baseline
classifier, OC-SORT profiles) is reported separately. B3D's operating points use the current B3D
test ground truth, which R1.O1 replaces, so B3D break-evens are provisional.

Break-even. The reference pipeline needs no preparation beyond the detector, which is trained
once and shared by every compared system (paper section 7), so all of Tetris's preparation is
additional. Extraction seconds saved per video minute are frames per video minute x
(1 / reference fps - 1 / Tetris fps) on the test split at the same boundary; break-even video
minutes are preparation seconds over that saving, when it is positive.

Not included: detector training (shared), detector loading per p011 and reference video and per
p040 configuration (not recorded), and OTIF's and LEAP's preparation (not recorded in this cache;
measurements-gpu/r9o2_preparation_cost.py prints the re-timing plan, whose timings.jsonl this
script does not read yet).

Writes measurements-results/r9o2_preparation_cost/preparation_cost.json.
"""
import argparse
import glob
import os
import subprocess
from pathlib import Path

import cv2
import duckdb
import pandas as pd

from evaluation.manifests import build_polytris_variant_manifest, build_split_video_manifest, list_split_videos
from evaluation.p120_throughput_gather import (build_index_construction_manifest, build_naive_query_manifest,
                                               build_stage020_manifest, build_stage022_manifest, build_stage030_manifest,
                                               build_stage040_manifest, build_stage050_manifest, build_stage060_manifest)
from evaluation.p121_throughput_compute import excluded_runtime_ops
from measurements.common import (DATASET_LABELS, OP_LIST_PATHS, fastest_tetris, host_info, iso_utc, load_test_tradeoff,
                                 load_tradeoff_points, query_dicts, results_dir, rounded, runtime_items_sql, write_json)
from polyis.io import cache, store
from polyis.utilities import get_config, parse_execution_param_str


CONFIG = get_config()
# Trackers the configured sweep searches over; their mistrack profiles feed p022.
TRACKERS = CONFIG['EXEC']['TRACKERS']
# Ops p121 drops from every runtime file (a local list inside p121's parse_runtime_file, copied here; a test
# checks the copy against p121's parser).
IGNORED_OPS = ('total_frame_time', 'read_frame', 'save_canvas', 'save_mapping_files')
# Per-frame video decoding ops that p121 counts and the instrumented boundary drops (p121 already drops p020's 'read').
VIDEO_DECODE_OPS = {'001_preprocess_groundtruth_detection': ['read'], '011_tune_detect': ['read']}
# Which timestamps bound one work unit, per stage: how many path levels above the runtime file the creation time is
# read (0 = the runtime file itself), the runtime file's sibling files whose latest write ends the unit (the runtime
# file always counts), and whether one task processes the split's videos in order (so units can be chained). Read
# from each stage's code and checked on real files; stages not listed keep their recorded time.
STAGE_SPANS = {
    # The reference pipeline (preprocess/p002): one task per video; the detector is loaded, then 002_naive/ is
    # recreated and the runtime file opened; detection.jsonl is written with every frame.
    '001_preprocess_groundtruth_detection': (0, ['detection.jsonl'], False),
    # One task per video; the runtime file is opened when the video starts and closed when it ends.
    '012_tune_create_training_data': (0, [], False),
    # <params>/ is recreated per video before decoding; score.jsonl is written last.
    '020_exec_classify': (2, ['score.jsonl'], True),
    # <params>/ is created only after the work, so only chaining gives its loading and ILP time.
    '022_exec_prune_polyominoes': (2, ['score.jsonl'], True),
    # <params>/ is recreated after loading the classifications; runtime.jsonl is written after every canvas.
    '030_exec_compress': (1, [], True),
    # <params>/ is recreated per video; detections and runtimes are written until the end.
    '040_exec_detect': (1, ['detections.jsonl'], True),
    # <params>/ is recreated per video; detections.jsonl is written after the runtime file.
    '050_exec_uncompress': (1, ['detections.jsonl'], True),
    # <variant>/ is created after loading the detections; tracking.jsonl is written last.
    '060_exec_track': (1, ['tracking.jsonl'], True),
}
# Largest gap between one unit's last write and the next video's start in the same task for the two to be
# consecutive (observed gaps are under one second); also used for p015 and p016 runs.
CHAIN_GAP_S = 60.0
# An own span longer than this is a stale creation time (single work units take minutes at most).
STALE_SPAN_S = 3600.0
# Slack between timestamps and recorded op times, which are taken with different clocks.
TIMESTAMP_TOLERANCE_S = 1.0
# Classifier-training timestamps are trusted when the fine-tuning wall clock is at least the recorded epochs and
# at most this ratio of them plus slack (per-epoch plotting and checkpointing), and the frozen stage likewise.
FINETUNE_WALL_MAX_RATIO = 1.5
FINETUNE_WALL_SLACK_S = 300.0
# p014 trains each stage for at most this many epochs (n_epochs=7 for both stages).
P014_EPOCHS_PER_STAGE = 7
# p016's default worker count, for converting its elapsed time into process seconds.
P016_WORKERS = 40
# Files p016 writes after aggregating a profile's partial counts.
PROFILE_OUTPUT_GLOBS = ('accuracy.npy', 'counts.npy', 'max_rate_table.npy', 'sample_rates.json', 'max_rate_*.txt')
# p015 writes a resolution group's datasets within moments; files further apart than this are different groups.
P015_GROUP_GAP_S = 1.0
# Datasets whose test ground truth R1.O1 replaces with human-verified labels.
PROVISIONAL_DATASETS = ('jnc0', 'jnc2', 'jnc6', 'jnc7')
# The order of the preparation components in the report.
COMPONENTS = ('reference_detection_train', 'training_data_creation', 'classifier_training', 'classifier_optimization',
              'mistrack_profiling', 'reference_outputs_valid', 'validation_sweep', 'validation_scoring')
# Unit of every component except where a component states otherwise.
PROCESS_UNIT = 'process seconds, summed over work units (single-worker equivalent, not elapsed time)'
# Sweep stages p120 builds manifests for, in pipeline order.
STAGE_BUILDERS = (build_stage020_manifest, build_stage022_manifest, build_stage030_manifest,
                  build_stage040_manifest, build_stage050_manifest, build_stage060_manifest)
# What re-timing each baseline's preparation would take; nothing of it is recorded in this cache.
BASELINE_PREPARATION = {
    'otif': dict(recorded=False, needs=[
        'OTIF environment (_archive/otif Docker image, conda env otif, Go toolchain) with the standardized detector '
        '(YOLOv5 on CalDoT1/2 and Amsterdam, B3D RetinaNet), as in the reported runs',
        'reference detection on the training split (pipeline2/10_detect.go) and IoU tracks for the recurrent-tracker '
        'split, which section 7 sets to the training split (iou-tracker2.go)',
        'proxy training: frame extraction (pipeline/16_extract_frames.py) and segmentation-proxy training at each '
        'proxy resolution (pipeline/20_train_segmentation.py)',
        'recurrent rate-tracker training (rnn/train.py)',
        'parameter tuning on the validation split (pipeline2/30_speed_accuracy.go)',
        'B3D datasets: OTIF-format copies of jnc0/2/6/7 are not under /otif-dataset; locate the converted data used for '
        'the reported runs',
    ]),
    'leap': dict(recorded=False, needs=[
        'LEAP environment (official repository at commit 3f7e440 as used for the reported runs, run_docker.sh)',
        'pseudo labels on the training split from the standardized detector (README step 4)',
        'cross-frame association (re-identification) model: dataset generation (vehicle_reid/generate_reid.py) and '
        'training (vehicle_reid/run.sh)',
        'motion-pattern library from training trajectories (traj_cluster/, track_cluster.py)',
        'per-dataset configuration file (configs/<dataset>.yaml); no distilled detector, per section 7',
        'confirm with the LEAP operator which steps the reported runs executed and on which splits',
    ]),
}


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: preparation cost and break-even video duration per dataset')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS), choices=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    return parser.parse_args()


def load_runtime_ops(con: duckdb.DuckDBPyConnection, files: list[tuple[str, str]]):
    """Table runtime_ops(stage, runtime_file, op, seconds, records, ..., counted, ignored, decode): op times per file.

    `files` are (stage, path) pairs; each stage's op list is found with common.OP_LIST_PATHS and parsed by
    common.runtime_items_sql. An op is counted when p121 would keep it for that stage, and decode marks
    VIDEO_DECODE_OPS. As in p121, items without a numeric time are skipped and blank lines ignored; lines
    without an op list, timed items without an op, negative times of ops p121 does not ignore, and boolean
    times (which p121 would count as 0 or 1 ms) raise.
    """
    # One row per (file, op): seconds, record counts for the checks below, and whether p121 counts the op.
    con.execute('CREATE OR REPLACE TABLE runtime_ops (stage VARCHAR, runtime_file VARCHAR, op VARCHAR, seconds DOUBLE, '
                'records BIGINT, negative_records BIGINT, boolean_records BIGINT, bad_lines BIGINT, counted BOOLEAN, '
                'ignored BOOLEAN, decode BOOLEAN)')
    # p121 parses .json files as one array, which these queries do not.
    not_jsonl = [path for _, path in files if not str(path).endswith('.jsonl')]
    assert not not_jsonl, f'runtime files must be JSONL: {not_jsonl[:5]}'
    # Group paths by stage, since each stage stores its op list at a different place in a line.
    by_stage: dict[str, list[str]] = {}
    for stage, path in files:
        by_stage.setdefault(stage, []).append(str(path))
    # One query per stage.
    for stage, paths in sorted(by_stage.items()):
        # Ops outside p121's boundary for this stage.
        dropped = sorted({*IGNORED_OPS, *excluded_runtime_ops(stage)})
        # One parallel pass over every line's op items (an object counts as a one-item list), summed per file and op.
        con.execute(f"""
            INSERT INTO runtime_ops
            SELECT $stage, filename, op, coalesce(sum(time), 0) / 1000, count(time), count(*) FILTER (WHERE time < 0),
                   count(*) FILTER (WHERE time_type = 'BOOLEAN'),
                   count(*) FILTER (WHERE coalesce(list_type NOT IN ('ARRAY', 'OBJECT'), true)),
                   NOT list_contains($dropped, op), list_contains($ignored, op), list_contains($decode, op)
            FROM ({runtime_items_sql(OP_LIST_PATHS[stage], numbered=False)}) GROUP BY filename, op
        """, dict(paths=sorted(set(paths)), stage=stage, dropped=dropped, ignored=list(IGNORED_OPS),
                  decode=VIDEO_DECODE_OPS.get(stage, [])))
    # A line without an op list means the layout assumption is wrong (p121's accessor would raise).
    bad = query_dicts(con, 'SELECT stage, runtime_file FROM runtime_ops WHERE bad_lines > 0 LIMIT 5')
    assert not bad, f'runtime lines without an op list: {bad}'
    # A timed item without an op cannot be attributed (p121 raises on it too).
    nameless = query_dicts(con, 'SELECT stage, runtime_file FROM runtime_ops WHERE op IS NULL AND records > 0 LIMIT 5')
    assert not nameless, f'timed items without an op: {nameless}'
    # Boolean times are not durations, although Python's isinstance would let p121 count them.
    boolean = query_dicts(con, 'SELECT stage, runtime_file, op FROM runtime_ops WHERE boolean_records > 0 LIMIT 5')
    assert not boolean, f'boolean op times: {boolean}'
    # Negative durations cannot be summed into a cost; p121 checks every op it does not ignore.
    negative = query_dicts(con, 'SELECT stage, runtime_file, op FROM runtime_ops '
                                'WHERE NOT ignored AND negative_records > 0 LIMIT 5')
    assert not negative, f'negative op times: {negative}'
    # Ops that never carry a numeric time (p040's image_file, untimed items) are not timings.
    con.execute('DELETE FROM runtime_ops WHERE records = 0')


def first_frame_seconds(con: duckdb.DuckDBPyConnection, paths: list[str]) -> float:
    """Seconds in each p012 runtime file's first 'split' call, summed; the first CUDA call of a process initializes CUDA."""
    # No files, no time.
    if not paths:
        return 0.0
    # Per file, the split of the lowest frame, and how many frames were split more than once.
    duplicates, seconds = con.execute("""
        WITH splits AS (
            SELECT filename, CAST(json->>'time' AS DOUBLE) AS ms, CAST(json->>'frame' AS BIGINT) AS frame
            FROM read_json_objects($paths, format = 'newline_delimited', filename = true) WHERE json->>'op' = 'split'
        )
        SELECT (SELECT count(*) FROM (SELECT filename, frame FROM splits GROUP BY ALL HAVING count(*) > 1)),
               coalesce((SELECT sum(first_ms) FROM (SELECT arg_min(ms, frame) AS first_ms FROM splits GROUP BY filename)), 0) / 1000
    """, dict(paths=sorted(set(paths)))).fetchall()[0]
    # One split per frame and file (one tile size per file); otherwise the first call is ambiguous.
    assert duplicates == 0, f'{duplicates} frames with more than one split call in one file'
    return seconds


def file_times(paths: list[str]) -> dict[str, tuple[float | None, float]]:
    """{path: (creation time or None when unknown, modification time)} in epoch seconds, for the paths that exist.

    Python's os.stat has no creation time on Linux, so GNU stat reads both (%.9W is 0 when the file
    system keeps none). Missing paths are left out; callers treat them as unknown.
    """
    # Nothing to stat.
    if not paths:
        return {}
    # One xargs-batched stat over NUL-separated paths, so any number of paths and any name without NUL works.
    result = subprocess.run(['xargs', '-0', 'stat', '-c', '%.9W %.9Y %n'], input='\0'.join(paths),
                            capture_output=True, text=True)
    # xargs exits 123 when stat failed on some path (a missing one); anything else is a real failure.
    assert result.returncode in (0, 123), f'stat failed ({result.returncode}): {result.stderr[:500]}'
    # Parse stat's output into {path: (birth, mtime)}.
    times = {}
    # Each line is "<birth> <mtime> <name>"; the name may contain spaces, so split twice only.
    for line in result.stdout.splitlines():
        birth, mtime, name = line.split(' ', 2)
        times[name] = (float(birth) or None, float(mtime))
    return times


def load_units(con: duckdb.DuckDBPyConnection, units: str, videos: str):
    """Table units: per work unit, recorded and decoding seconds and the process-time estimate (module docstring).

    `units` is a table (stage, runtime_file, videoset, video) of distinct work units; `videos` a
    table (videoset, video, rank) giving each split's videos in the pipeline's sorted order.
    Reads runtime_ops and timestamps the files named by STAGE_SPANS.
    """
    # Creation times are read at most two levels up (the runtime file, its directory, or the one above).
    assert all(level in (0, 1, 2) for level, _, _ in STAGE_SPANS.values()), 'STAGE_SPANS levels must be 0, 1, or 2'
    # The span rules as a table, to join with the units.
    con.execute('CREATE OR REPLACE TABLE stage_spans (stage VARCHAR, start_level INTEGER, end_siblings VARCHAR[], '
                'chain BOOLEAN)')
    con.executemany('INSERT INTO stage_spans VALUES (?, ?, ?, ?)',
                    [(s, level, ends, chain) for s, (level, ends, chain) in STAGE_SPANS.items()])
    # The path whose creation starts each unit and the paths whose last write ends it.
    con.execute(f"""
        CREATE OR REPLACE TABLE unit_paths AS
        SELECT u.*, s.start_level IS NOT NULL AS timed, coalesce(s.chain, false) AS chain,
               CASE s.start_level WHEN 0 THEN u.runtime_file
                                  WHEN 1 THEN regexp_replace(u.runtime_file, '/[^/]*$', '')
                                  WHEN 2 THEN regexp_replace(u.runtime_file, '/[^/]*/[^/]*$', '') END AS start_path,
               list_prepend(u.runtime_file, list_transform(s.end_siblings,
                            lambda x: regexp_replace(u.runtime_file, '/[^/]*$', '') || '/' || x)) AS end_paths
        FROM {units} u LEFT JOIN stage_spans s USING (stage)
    """)
    # Every path whose timestamps bound some unit.
    paths = [r[0] for r in con.execute("""
        SELECT start_path FROM unit_paths WHERE timed UNION SELECT unnest(end_paths) FROM unit_paths WHERE timed
    """).fetchall()]
    # Timestamps of those paths, as a table.
    con.execute('CREATE OR REPLACE TABLE file_times (path VARCHAR, birth_s DOUBLE, mtime_s DOUBLE)')
    # Timestamp every path at once, if any stage has spans.
    times = file_times(sorted(paths))
    if times:
        con.executemany('INSERT INTO file_times VALUES (?, ?, ?)', [(p, b, m) for p, (b, m) in times.items()])
    # Each unit's process time: chained to its task's previous video, else its own span or its recorded ops.
    con.execute(f"""
        CREATE OR REPLACE TABLE units AS
        WITH recorded AS (
            SELECT runtime_file, sum(seconds) FILTER (WHERE counted) AS recorded_s,
                   sum(seconds) FILTER (WHERE counted AND decode) AS decode_s
            FROM runtime_ops GROUP BY runtime_file
        ),
        starts AS (
            SELECT p.runtime_file, f.birth_s AS start_s
            FROM unit_paths p JOIN file_times f ON f.path = p.start_path WHERE p.timed
        ),
        ends AS (
            SELECT e.runtime_file, max(f.mtime_s) AS end_s
            FROM (SELECT runtime_file, unnest(end_paths) AS path FROM unit_paths WHERE timed) e JOIN file_times f USING (path)
            GROUP BY e.runtime_file
        ),
        spans AS (
            SELECT p.stage, p.runtime_file, p.videoset, p.video, p.chain, v.rank,
                   CASE WHEN p.runtime_file LIKE '%/execution/%'
                        THEN regexp_replace(p.runtime_file, '/execution/[^/]+/', '/execution/*/') ELSE p.runtime_file END AS task,
                   s.start_s, e.end_s, coalesce(r.recorded_s, 0) AS recorded_s, coalesce(r.decode_s, 0) AS decode_s
            FROM unit_paths p LEFT JOIN recorded r USING (runtime_file) LEFT JOIN {videos} v USING (videoset, video)
            LEFT JOIN starts s USING (runtime_file) LEFT JOIN ends e USING (runtime_file)
            WHERE p.timed
        ),
        ordered AS (
            SELECT *, lag(rank) OVER w AS previous_rank, lag(end_s) OVER w AS previous_end_s, end_s - start_s AS own_s
            FROM spans WINDOW w AS (PARTITION BY stage, task, videoset ORDER BY rank)
        ),
        classified AS (
            SELECT *, coalesce(chain AND previous_rank = rank - 1 AND start_s >= previous_end_s
                               AND start_s - previous_end_s <= $gap
                               AND end_s - previous_end_s + $tolerance >= recorded_s, false) AS chained,
                      coalesce(own_s > $stale, false) AS stale_start
            FROM ordered
        )
        SELECT stage, runtime_file, task, start_s, end_s, recorded_s, decode_s, chained, stale_start,
               CASE WHEN chained THEN end_s - previous_end_s
                    WHEN stale_start THEN recorded_s
                    ELSE greatest(recorded_s, coalesce(own_s, 0)) END AS process_s
        FROM classified
        UNION ALL
        SELECT p.stage, p.runtime_file, p.runtime_file, NULL, NULL, coalesce(r.recorded_s, 0), coalesce(r.decode_s, 0),
               false, false, coalesce(r.recorded_s, 0)
        FROM unit_paths p LEFT JOIN recorded r USING (runtime_file) WHERE NOT p.timed
    """, dict(gap=CHAIN_GAP_S, tolerance=TIMESTAMP_TOLERANCE_S, stale=STALE_SPAN_S))


def stage_summary(con: duckdb.DuckDBPyConnection, manifest: str) -> list[dict]:
    """Per stage of a manifest(stage, runtime_file, ...) table: distinct work units and their seconds as recorded, at
    the instrumented boundary, and as process time, and the same if every manifest row (one per configuration and
    video) were charged its own file. Reads units; a file without units rows counts as zero seconds."""
    # Distinct files per stage, their seconds per boundary, and what charging every manifest row would cost.
    return query_dicts(con, f"""
        WITH distinct_files AS (SELECT DISTINCT stage, runtime_file FROM {manifest}),
        per_file AS (
            SELECT d.stage, d.runtime_file, coalesce(u.recorded_s, 0) AS recorded_s, coalesce(u.decode_s, 0) AS decode_s,
                   coalesce(u.process_s, 0) AS process_s, coalesce(u.chained, false) AS chained,
                   coalesce(u.stale_start, false) AS stale_start, u.start_s, u.end_s
            FROM distinct_files d LEFT JOIN units u USING (runtime_file)
        ),
        charged AS (
            SELECT m.stage, count(*) AS manifest_rows, sum(p.recorded_s - p.decode_s) AS without_sharing_instrumented_s,
                   sum(p.process_s) AS without_sharing_process_s
            FROM {manifest} m JOIN per_file p USING (runtime_file) GROUP BY m.stage
        )
        SELECT p.stage, count(*) AS work_units, sum(p.recorded_s) AS recorded_s, sum(p.decode_s) AS decode_s,
               sum(p.recorded_s - p.decode_s) AS instrumented_s, sum(p.process_s) AS process_s,
               count(*) FILTER (WHERE p.chained) AS chained_units, count(*) FILTER (WHERE p.stale_start) AS stale_starts,
               max(p.end_s) - min(p.start_s) AS elapsed_span_s,
               any_value(c.manifest_rows) AS manifest_rows,
               any_value(c.without_sharing_instrumented_s) AS without_sharing_instrumented_s,
               any_value(c.without_sharing_process_s) AS without_sharing_process_s
        FROM per_file p JOIN charged c USING (stage) GROUP BY p.stage ORDER BY p.stage
    """)


def dropped_ops(con: duckdb.DuckDBPyConnection, manifest: str) -> dict[str, dict[str, float]]:
    """{stage: {op: seconds}} for recorded ops outside p121's boundary, over a manifest's distinct files."""
    # Dropped-op seconds per stage and op over the manifest's distinct files.
    rows = query_dicts(con, f"""
        SELECT r.stage, r.op, round(sum(r.seconds), 1) AS seconds
        FROM runtime_ops r JOIN (SELECT DISTINCT runtime_file FROM {manifest}) USING (runtime_file)
        WHERE NOT r.counted GROUP BY r.stage, r.op ORDER BY r.stage, r.op
    """)
    # Nest as {stage: {op: seconds}} for the report (a handful of rows).
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(r['stage'], {})[r['op']] = r['seconds']
    return out


def training_runs(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> list[dict]:
    """Per classifier-training run: recorded fine-tuning time and the wall clock of both stages from file timestamps.

    `rows` are (model, runtime_file, start, frozen_end, end) in epoch seconds, where start is the
    result directory's creation, frozen_end the throughput file's creation, and end its last write
    (None when unknown). Reads runtime_ops for the recorded epochs.
    """
    # Timestamps per training run, as a table.
    con.execute('CREATE OR REPLACE TABLE train_times (model VARCHAR, runtime_file VARCHAR, start_s DOUBLE, '
                'frozen_end_s DOUBLE, end_s DOUBLE)')
    # Insert the timestamps, if any run was found.
    if rows:
        con.executemany('INSERT INTO train_times VALUES (?, ?, ?, ?, ?)', rows)
    # Recorded fine-tuning epochs, the two stage spans, and whether the spans agree with the epochs.
    return query_dicts(con, """
        WITH recorded AS (
            SELECT runtime_file, sum(seconds) FILTER (WHERE counted) AS finetune_recorded_s,
                   max(records) FILTER (WHERE op = 'train_step') AS finetune_epochs
            FROM runtime_ops GROUP BY runtime_file
        ),
        spans AS (
            SELECT t.*, coalesce(r.finetune_recorded_s, 0) AS finetune_recorded_s,
                   coalesce(r.finetune_epochs, 0) AS finetune_epochs,
                   t.end_s - t.start_s AS wall_s, t.frozen_end_s - t.start_s AS frozen_wall_s,
                   t.end_s - t.frozen_end_s AS finetune_wall_s
            FROM train_times t LEFT JOIN recorded r USING (runtime_file)
        ),
        checked AS (
            SELECT *, coalesce(finetune_epochs > 0 AND frozen_wall_s > 0
                               AND finetune_wall_s + $tolerance >= finetune_recorded_s
                               AND finetune_wall_s <= finetune_recorded_s * $ratio + $slack
                               AND frozen_wall_s <= $epochs / finetune_epochs * finetune_recorded_s * $ratio + $slack,
                               false) AS timestamps_consistent
            FROM spans
        )
        SELECT *, CASE WHEN timestamps_consistent THEN wall_s ELSE finetune_recorded_s END AS seconds
        FROM checked ORDER BY model
    """, dict(ratio=FINETUNE_WALL_MAX_RATIO, slack=FINETUNE_WALL_SLACK_S, tolerance=TIMESTAMP_TOLERANCE_S,
              epochs=float(P014_EPOCHS_PER_STAGE)))


def profile_estimates(con: duckdb.DuckDBPyConnection, rows: list[tuple], gap: float = CHAIN_GAP_S) -> list[dict]:
    """Estimated elapsed seconds per mistrack profile from (dataset, profile, videos, first_partial_s, last_output_s) rows.

    Profiles are ordered by last output (ties by dataset and profile). One that starts writing
    partial files no earlier than the previous profile's last output, and within `gap` seconds of
    it, is chained to it and costs the difference of the two last outputs; any other is the first
    of its run and costs its own span (a lower bound, since its first partial file is written
    when its first video finishes).
    """
    # Profile file times, as a table.
    con.execute('CREATE OR REPLACE TABLE profiles (dataset VARCHAR, profile VARCHAR, videos BIGINT, '
                'first_partial_s DOUBLE, last_output_s DOUBLE)')
    # Insert the profiles, if any were found.
    if rows:
        con.executemany('INSERT INTO profiles VALUES (?, ?, ?, ?, ?)', rows)
    # Chain each profile to the one before it in the same p016 run.
    con.execute("""
        CREATE OR REPLACE TABLE profile_estimates AS
        WITH ordered AS (
            SELECT *, lag(last_output_s) OVER (ORDER BY last_output_s, dataset, profile) AS previous_output_s
            FROM profiles
        ),
        chained AS (
            SELECT *, coalesce(first_partial_s >= previous_output_s AND first_partial_s - previous_output_s <= $gap,
                               false) AS chained
            FROM ordered
        )
        SELECT dataset, profile, videos, first_partial_s, last_output_s, chained,
               CASE WHEN chained THEN last_output_s - previous_output_s ELSE last_output_s - first_partial_s END AS seconds,
               NOT chained AS lower_bound
        FROM chained
    """, dict(gap=gap))
    # One row per profile.
    return query_dicts(con, 'SELECT * FROM profile_estimates ORDER BY dataset, profile')


def gather_profiles(datasets: list[str]) -> tuple[list[tuple], list[dict]]:
    """Complete p016 profiles as (dataset, profile, videos, first_partial_s, last_output_s) rows, and incomplete ones.

    A profile needs per-video partial files and aggregate outputs; one without both was not
    finished by p016 and is reported instead of estimated.
    """
    # (dataset, profile, partial files, output files) per profile directory.
    entries = []
    # Walk each dataset's track_rate directory (file metadata only, no data is read).
    for dataset in datasets:
        for profile_dir in sorted(Path(cache.index(dataset, 'track_rates')).glob('*')):
            # Skip stray files next to the profile directories.
            if not profile_dir.is_dir():
                continue
            # Per-video partial counts and the aggregate outputs written after them.
            partials = sorted(str(p) for p in profile_dir.glob('partial/*.npy'))
            outputs = sorted(str(p) for pattern in PROFILE_OUTPUT_GLOBS for p in profile_dir.glob(pattern))
            # Keep the profile's files for timestamping.
            entries.append((dataset, profile_dir.name, partials, outputs))
    # Modification times of every partial and output file at once.
    times = file_times([p for _, _, partials, outputs in entries for p in partials + outputs])
    # A profile with partial and output files was finished; its first partial and last output bound it.
    complete = [(d, name, len(partials), min(times[p][1] for p in partials), max(times[p][1] for p in outputs))
                for d, name, partials, outputs in entries if partials and outputs]
    # A profile missing either was not finished by p016; it is reported, not estimated.
    incomplete = [dict(dataset=d, profile=name, partials=len(partials), outputs=len(outputs))
                  for d, name, partials, outputs in entries if not (partials and outputs)]
    return complete, incomplete


def optimization_estimates(con: duckdb.DuckDBPyConnection, rows: list[tuple], gap: float = CHAIN_GAP_S) -> list[dict]:
    """Estimated seconds of p015's benchmark per (dataset, model) from (dataset, model, finished_s) rows.

    p015 benchmarks one resolution group after another in one process and writes every dataset
    of a group together. Files more than P015_GROUP_GAP_S apart start a new group; a group that
    finishes within `gap` of the previous group's finish took the difference, and the first group
    of a run (or one after a longer pause) has no estimate.
    """
    # The benchmark output files, as a table.
    con.execute('CREATE OR REPLACE TABLE compilations (dataset VARCHAR, model VARCHAR, finished_s DOUBLE)')
    # Insert the compilation files, if any were found.
    if rows:
        con.executemany('INSERT INTO compilations VALUES (?, ?, ?)', rows)
    # Group files written together, then time each group from the previous group's finish.
    return query_dicts(con, """
        WITH ordered AS (
            SELECT *, finished_s - lag(finished_s) OVER (ORDER BY finished_s, dataset, model) AS since_previous
            FROM compilations
        ),
        grouped AS (
            SELECT *, sum(CASE WHEN since_previous IS NULL OR since_previous > $group_gap THEN 1 ELSE 0 END)
                      OVER (ORDER BY finished_s, dataset, model) AS group_index
            FROM ordered
        ),
        groups AS (SELECT group_index, max(finished_s) AS finish_s FROM grouped GROUP BY group_index),
        timed AS (SELECT group_index, finish_s, finish_s - lag(finish_s) OVER (ORDER BY group_index) AS since_s FROM groups)
        SELECT g.dataset, g.model, g.group_index, t.finish_s, CASE WHEN t.since_s <= $gap THEN t.since_s END AS seconds
        FROM grouped g JOIN timed t USING (group_index) ORDER BY g.dataset, g.model
    """, dict(group_gap=P015_GROUP_GAP_S, gap=gap))


def gather_compilations(datasets: list[str]) -> list[tuple]:
    """(dataset, model, finished_s) for every p015 model_compilation.jsonl of the datasets."""
    # Every model_compilation.jsonl of the datasets.
    paths = [(dataset, p) for dataset in datasets
             for p in sorted(glob.glob(str(cache.index(dataset, 'training', 'results', '*', 'model_compilation.jsonl'))))]
    # Their last writes, at once.
    times = file_times([p for _, p in paths])
    # Each file's dataset, model, and last write.
    return [(dataset, Path(p).parent.name, times[p][1]) for dataset, p in paths]


def scoring_estimate(con: duckdb.DuckDBPyConnection, rows: list[tuple], configurations: int, videos: int) -> dict:
    """Scoring seconds per (configuration, video) from test-split p110 tasks, and the validation sweep's estimate.

    `rows` are (variant_id, videos, start_s, end_s): one p110 task (one process scoring one
    configuration on every test video), from its output directory's creation to its last result
    file. Spans that are unknown, negative, or over STALE_SPAN_S are left out.
    """
    # One row per test-split p110 task.
    con.execute('CREATE OR REPLACE TABLE scoring (variant_id VARCHAR, videos BIGINT, start_s DOUBLE, end_s DOUBLE)')
    # Insert the tasks, if any were found.
    if rows:
        con.executemany('INSERT INTO scoring VALUES (?, ?, ?, ?)', rows)
    # Seconds per (configuration, video) over the usable tasks.
    (row,) = query_dicts(con, """
        SELECT count(*) AS tasks, coalesce(sum(videos), 0) AS pairs, coalesce(sum(end_s - start_s), 0) AS seconds,
               sum(end_s - start_s) / sum(videos) AS seconds_per_pair,
               (SELECT count(*) FROM scoring) - count(*) AS unusable_tasks
        FROM scoring WHERE videos > 0 AND end_s - start_s BETWEEN 0 AND $stale
    """, dict(stale=STALE_SPAN_S))
    # Without one usable test task there is nothing to scale.
    estimate = None if row['seconds_per_pair'] is None else row['seconds_per_pair'] * configurations * videos
    return dict(row, valid_configurations=configurations, valid_videos=videos, estimate_s=estimate)


def gather_scoring(dataset: str) -> list[tuple]:
    """(variant_id, videos, start_s, end_s) for each Tetris task of the test split's p110 run."""
    # Test-split raw scores, one directory per scored configuration.
    root = Path(cache.eval(dataset, 'acc', 'raw', 'test'))
    # One directory per scored configuration (the reference's 'naive' task scores a different pipeline).
    tasks = [(d, sorted(str(p) for p in d.glob('*.json'))) for d in sorted(root.glob('*')) if d.is_dir() and d.name != 'naive']
    # Creation and modification times of every task directory and result file at once.
    times = file_times([str(d) for d, _ in tasks] + [p for _, files in tasks for p in files])
    # Per task: scored videos (not the combined DATASET.json), directory creation, and last result write.
    return [(d.name, sum(Path(p).name != 'DATASET.json' for p in files), times.get(str(d), (None, 0.0))[0],
             max(times[p][1] for p in files)) for d, files in tasks if files]


def video_split(dataset: str, split: str) -> dict:
    """Videos, frames, and minutes of video in one split, from the video files' metadata."""
    # Every video file of the split, with the pipeline's own extension list and order.
    names = list_split_videos(dataset, split)
    # An empty split would leave every rate undefined.
    assert names, f'{dataset}/{split}: no videos'
    # Running totals over the split's videos.
    frames, seconds = 0, 0.0
    # Read each container's frame count and frame rate (per-file metadata, not data analysis).
    for name in names:
        cap = cv2.VideoCapture(str(store.dataset(dataset, split, name)))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # An unreadable file would silently shorten the split.
        assert count > 0 and fps > 0, f'{dataset}/{split}/{name}: unreadable frame count or frame rate'
        # Add the video's frames and duration.
        frames += count
        seconds += count / fps
    return dict(videos=len(names), frames=frames, seconds=seconds, minutes=round(seconds / 60, 2), names=names)


def operating_point(con: duckdb.DuckDBPyConnection, dataset: str, loss: float) -> dict:
    """The fastest Tetris test configuration within `loss` of the reference pipeline's HOTA, with its parameters."""
    # The test split's Tetris configurations and the reference row, with HOTA as the accuracy.
    tetris, naive = load_test_tradeoff(dataset)
    load_tradeoff_points(con, 'points', {dataset: (tetris, naive)}, {'time': 'DOUBLE', 'frame_count': 'BIGINT'})
    # The fastest Tetris configuration whose HOTA is within the bound; without one there is nothing to break even at.
    chosen = fastest_tetris(con, 'points', loss)[dataset]
    # The chosen row's paper-boundary time and frame count, for the cross-checks.
    time_s, frames = con.execute("SELECT \"time\", frame_count FROM points WHERE system = 'tetris' AND config = ?",
                                 [chosen['config']]).fetchall()[0]
    # Decode the configuration's parameters from its name.
    params = parse_execution_param_str(chosen['config'])
    # The point with its parameters and the reference it is measured against.
    return dict(loss=loss, bound=chosen['bound'], reference_hota=chosen['reference_accuracy'],
                reference_fps=chosen['reference_fps'], reference_time_s=float(naive['time']),
                test_frames=int(naive['frame_count']), config=chosen['config'], fps=chosen['fps'], hota=chosen['accuracy'],
                time_s=time_s, frames=frames, classifier=params['classifier'], tilesize=params['tilesize'],
                tracker=params['tracker'], sample_rate=params['sample_rate'],
                provisional='pending R1.O1 re-scoring of B3D' if dataset in PROVISIONAL_DATASETS else None)


def break_even(preparation_s: float, reference_fps: float, tetris_fps: float, frames_per_minute: float) -> dict:
    """Extraction seconds per video minute for both pipelines and the video minutes that repay `preparation_s`."""
    # Throughputs and the video rate must be positive to define seconds per video minute.
    assert reference_fps > 0 and tetris_fps > 0 and frames_per_minute > 0, \
        f'non-positive rate: reference {reference_fps}, Tetris {tetris_fps}, frames per minute {frames_per_minute}'
    # A negative preparation time is a bug upstream.
    assert preparation_s >= 0, f'negative preparation time {preparation_s}'
    # Seconds each pipeline spends extracting one minute of video.
    reference = frames_per_minute / reference_fps
    tetris = frames_per_minute / tetris_fps
    # Seconds saved per video minute.
    saving = reference - tetris
    # Preparation is never repaid unless Tetris extracts faster.
    minutes = preparation_s / saving if saving > 0 else None
    # Rounded for the report.
    return dict(reference_fps=round(reference_fps, 2), tetris_fps=round(tetris_fps, 2),
                reference_extraction_s_per_video_minute=round(reference, 3),
                tetris_extraction_s_per_video_minute=round(tetris, 3),
                saving_s_per_video_minute=round(saving, 3),
                break_even_video_minutes=rounded(minutes, 1),
                break_even_video_hours=None if minutes is None else round(minutes / 60, 2))


def split_manifest(dataset: str, grid: pd.DataFrame | None = None, videoset: str = 'valid') -> pd.DataFrame:
    """p120's split manifest (reference pipeline and every configuration of `grid`) with an exists column.

    Built from the same p120 builders as build_query_execution_manifest, which asserts that every
    file exists; here a missing sweep file is recorded instead, so incomplete videos can be reported.
    `grid` defaults to every configured configuration.
    """
    # The split's videos, as p120 lists them.
    videos = build_split_video_manifest(datasets=[dataset], videosets=[videoset])
    # Every configuration of the grid on every video of the split.
    configs = videos.merge(build_polytris_variant_manifest() if grid is None else grid, how='cross')
    # The reference pipeline's files (asserted to exist by p120) and each sweep stage's files.
    df = pd.concat([build_naive_query_manifest(videos), *(build(configs) for build in STAGE_BUILDERS)], ignore_index=True)
    # Record which runtime files exist instead of asserting it.
    df['exists'] = df['runtime_file'].map(os.path.exists)
    return df


def load_manifest(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame):
    """Table `name`(stage, runtime_file, videoset, video, variant, variant_id, classifier, tilesize, exists) from a
    p120 manifest; a manifest without an exists column is taken as complete (p120 asserted it)."""
    # Keep only the columns used here, as plain strings so DuckDB sees no mixed or missing types.
    columns = ['stage', 'runtime_file', 'videoset', 'video', 'variant', 'variant_id', 'classifier', 'tilesize']
    table = df[columns].astype(str)
    # The existence flag as a boolean.
    table['exists'] = df['exists'].astype(bool) if 'exists' in df.columns else True
    # Materialize the frame as a DuckDB table.
    con.register(f'{name}_df', table)
    con.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}_df')
    con.unregister(f'{name}_df')


def sweep_coverage(con: duckdb.DuckDBPyConnection) -> dict:
    """Table complete_videos and the coverage of the validation sweep, from valid_manifest(variant, video, exists)."""
    # The one definition of a complete video: every sweep file present.
    con.execute("""
        CREATE OR REPLACE TABLE video_coverage AS
        SELECT video, bool_and("exists") AS complete, count(*) FILTER (WHERE NOT "exists") AS missing_files
        FROM valid_manifest WHERE variant = 'polytris' GROUP BY video
    """)
    # Videos with every sweep file.
    con.execute('CREATE OR REPLACE TABLE complete_videos AS SELECT video FROM video_coverage WHERE complete')
    # How many videos the split has and how many are complete.
    (row,) = query_dicts(con, 'SELECT count(*) AS videos, count(*) FILTER (WHERE complete) AS complete_videos FROM video_coverage')
    # Without one complete video nothing can be measured or scaled.
    assert row['complete_videos'] > 0, 'no validation video has every sweep file'
    # Missing files per incomplete video, for the report.
    incomplete = con.execute('SELECT video, missing_files FROM video_coverage WHERE NOT complete ORDER BY video').fetchall()
    # The scale factor turns the complete videos' sweep into the whole split's.
    return dict(row, scale=row['videos'] / row['complete_videos'], incomplete=dict(incomplete))


def component(basis: str, recorded_s: float, instrumented_s: float, process_s: float, unit: str = PROCESS_UNIT,
              **extra) -> dict:
    """One preparation component: its seconds as recorded, at the instrumented boundary, and as process time."""
    # Round once, here, so totals are sums of what the report shows.
    seconds = dict(recorded_s=round(recorded_s, 1), instrumented_s=round(instrumented_s, 1), process_s=round(process_s, 1))
    # Hours alongside seconds, plus the component's own fields.
    return dict(basis=basis, unit=unit, **seconds, instrumented_hours=round(seconds['instrumented_s'] / 3600, 3),
                process_hours=round(seconds['process_s'] / 3600, 3), **extra)


def stage_detail(rows: list[dict]) -> dict:
    """{stage: figures} from stage_summary rows, rounded for the report."""
    # Round floats to 0.1 s, keyed by stage.
    return {r['stage']: rounded({k: v for k, v in r.items() if k != 'stage'}, 1) for r in rows}


def total(rows: list[dict], key: str) -> float:
    """Sum of one figure over stage_summary rows."""
    # A plain sum over a handful of stages.
    return sum(r[key] for r in rows)


def manifests(con: duckdb.DuckDBPyConnection, dataset: str, point: dict) -> dict:
    """Load every manifest the dataset needs, the runtime ops and work units of their files, and the coverage."""
    # Index-stage (011, 012, 013), validation-split (reference and sweep), and test-split (reference and the
    # operating point's configuration) runtime files, as p120 lists them.
    load_manifest(con, 'index_manifest', build_index_construction_manifest([dataset]))
    load_manifest(con, 'valid_manifest', split_manifest(dataset))
    # The configured grid, to select the operating point's configuration on test.
    grid = build_polytris_variant_manifest()
    load_manifest(con, 'test_manifest', split_manifest(dataset, grid[grid['variant_id'] == point['config']], 'test'))
    # The operating point's test files were run (p120 keeps the Pareto set on test).
    missing = query_dicts(con, 'SELECT stage, runtime_file FROM test_manifest WHERE NOT "exists" LIMIT 5')
    assert not missing, f'{dataset}: operating-point test files missing: {missing}'
    # Which validation videos have every sweep file.
    coverage = sweep_coverage(con)
    # The tables each component reads.
    con.execute("CREATE OR REPLACE TABLE reference_manifest AS SELECT * FROM valid_manifest WHERE variant = 'naive'")
    con.execute("""CREATE OR REPLACE TABLE sweep_manifest AS SELECT * FROM valid_manifest
                   WHERE variant = 'polytris' AND video IN (SELECT video FROM complete_videos)""")
    con.execute("CREATE OR REPLACE TABLE reference_test AS SELECT * FROM test_manifest WHERE variant = 'naive'")
    con.execute("CREATE OR REPLACE TABLE point_test AS SELECT * FROM test_manifest WHERE variant = 'polytris'")
    con.execute("CREATE OR REPLACE TABLE train_det_manifest AS SELECT * FROM index_manifest WHERE stage = '011_tune_detect'")
    # p012 runs once per tile size; keep only the tile size the operating point uses ('<NA>' casts to NULL).
    con.execute("""CREATE OR REPLACE TABLE train_data_manifest AS SELECT * FROM index_manifest
                   WHERE stage = '012_tune_create_training_data' AND TRY_CAST(tilesize AS DOUBLE) = $tile""",
                dict(tile=point['tilesize']))
    # Every existing execution and index runtime file is a work unit, so each unit's predecessor is its true one,
    # even on videos left out of the recorded sweep.
    con.execute("""
        CREATE OR REPLACE TABLE work_units AS
        SELECT DISTINCT stage, runtime_file, videoset, video FROM (
            SELECT * FROM index_manifest WHERE stage <> '013_tune_train_classifier'
            UNION ALL SELECT * FROM valid_manifest WHERE "exists" UNION ALL SELECT * FROM test_manifest)
    """)
    # Each split's videos in the pipeline's sorted order, from the full manifests (a video whose files were all cleared
    # still sits between its neighbours, so they are not chained across it); 013's rows are per dataset, not per video.
    con.execute("""CREATE OR REPLACE TABLE split_videos AS SELECT videoset, video,
                   row_number() OVER (PARTITION BY videoset ORDER BY video) AS rank
                   FROM (SELECT videoset, video FROM index_manifest WHERE stage <> '013_tune_train_classifier'
                         UNION SELECT videoset, video FROM valid_manifest UNION SELECT videoset, video FROM test_manifest)""")
    # Every classifier ever trained on this dataset, configured or not, from its throughput file.
    trained = sorted(glob.glob(str(cache.index(dataset, 'training', 'results', '*', 'throughput_per_epoch.jsonl'))))
    # Op times of every work unit's runtime file and of every training run.
    files = [(r['stage'], r['runtime_file']) for r in query_dicts(con, 'SELECT stage, runtime_file FROM work_units')]
    load_runtime_ops(con, files + [('013_tune_train_classifier', p) for p in trained])
    # Recorded and process seconds per work unit.
    load_units(con, 'work_units', 'split_videos')
    # The coverage and the training runs, for the components.
    return dict(coverage=coverage, trained=trained)


def classifier_components(con: duckdb.DuckDBPyConnection, dataset: str, model: str, trained: list[str],
                          compilations: list[dict]) -> tuple[dict, dict, dict]:
    """Classifier training (p014) and optimization (p015) of the operating point's model, and the unneeded runs."""
    # Creation and modification times of each result directory and throughput file.
    dirs = [str(Path(p).parent) for p in trained]
    times = file_times(dirs + trained)
    # Each run's recorded epochs and timestamp wall clock.
    runs = training_runs(con, [(Path(p).parent.name, p, times.get(d, (None, 0))[0], times[p][0], times[p][1])
                               for d, p in zip(dirs, trained)])
    # The operating point's classifier must have exactly one run.
    relevant = [r for r in runs if r['model'] == model]
    assert len(relevant) == 1, f'{dataset}: no training run for the operating point classifier {model}'
    (run,) = relevant
    # Recorded is the fine-tuning stage only; both boundaries take the two-stage wall clock when consistent.
    training = component(
        'recorded: fine-tuning epochs; instrumented and process: wall clock of both stages from file timestamps'
        if run['timestamps_consistent'] else 'recorded fine-tuning epochs (timestamps inconsistent), a lower bound',
        run['finetune_recorded_s'], run['seconds'], run['seconds'], unit='process seconds of one training process',
        model=model, timestamps_consistent=run['timestamps_consistent'], finetune_epochs=run['finetune_epochs'],
        finetune_recorded_s=round(run['finetune_recorded_s'], 1),
        wall_s=rounded(run['wall_s'], 1), frozen_wall_s=rounded(run['frozen_wall_s'], 1),
        finetune_wall_s=rounded(run['finetune_wall_s'], 1),
        started=iso_utc(run['start_s']), frozen_stage_ended=iso_utc(run['frozen_end_s']), ended=iso_utc(run['end_s']))
    # p015's benchmark of this model (shared by the datasets of one resolution group; each camera alone would pay it).
    mine = [c for c in compilations if c['dataset'] == dataset and c['model'] == model]
    estimate = mine[0]['seconds'] if mine else None
    # Charged at its estimate when one exists, else zero and flagged missing.
    optimization = component(
        'estimate: gap since the previous resolution group finished in the same p015 run' if estimate is not None else
        'missing: first resolution group of its p015 run, or no model_compilation.jsonl', 0.0, estimate or 0.0,
        estimate or 0.0, unit='process seconds of one benchmark process', estimated=estimate is not None,
        finished=iso_utc(mine[0]['finish_s']) if mine else None)
    # Runs the reported configuration does not need (for example p017's baseline classifier).
    unneeded = {r['model']: dict(seconds=round(r['seconds'], 1), timestamps_consistent=r['timestamps_consistent'],
                                 started=iso_utc(r['start_s']), ended=iso_utc(r['end_s'])) for r in runs if r['model'] != model}
    return training, optimization, unneeded


def profiling_component(con: duckdb.DuckDBPyConnection, dataset: str, tilesize: int, train_videos: int) -> tuple[dict, dict]:
    """Mistrack profiling (p016) of the configured trackers at the operating point's tile size, and the unneeded ones."""
    # The profiles the configured sweep reads at this tile size.
    needed = [f'{t}_{tilesize}' for t in TRACKERS]
    # This dataset's profiles, with p016's elapsed time converted to an upper bound on process seconds.
    rows = query_dicts(con, """
        SELECT profile, videos, seconds, lower_bound, first_partial_s, last_output_s,
               seconds * least($workers, videos) AS process_upper_bound_s, list_contains($needed, profile) AS needed
        FROM profile_estimates WHERE dataset = $dataset ORDER BY profile
    """, dict(workers=P016_WORKERS, needed=needed, dataset=dataset))
    # Only the needed ones are charged.
    used = [r for r in rows if r['needed']]
    # Every configured tracker needs a finished profile.
    assert sorted(r['profile'] for r in used) == sorted(needed), \
        f'{dataset}: missing mistrack profiles {sorted(set(needed) - {r["profile"] for r in used})}'
    # A profile built from another video set (a stale partial file) would not be this dataset's cost.
    stale = [r['profile'] for r in used if r['videos'] != train_videos]
    assert not stale, f'{dataset}: profiles {stale} do not have one partial file per training video ({train_videos})'
    # Their process-second upper bounds, summed.
    seconds = sum(r['process_upper_bound_s'] for r in used)
    # The component, with each profile's provenance.
    profiling = component(
        'estimate: elapsed time from output modification times x min(40 workers, videos), an upper bound on process '
        'seconds', 0.0, seconds, seconds, unit=f'process seconds, upper bound (elapsed x min({P016_WORKERS} workers, videos))',
        elapsed_s=round(sum(r['seconds'] for r in used), 1), first_profile_of_run=any(r['lower_bound'] for r in used),
        profiles={r['profile']: dict(elapsed_s=round(r['seconds'], 1), videos=r['videos'], lower_bound=r['lower_bound'],
                                     first_partial=iso_utc(r['first_partial_s']), last_output=iso_utc(r['last_output_s']))
                  for r in used})
    # Profiles the reported configuration does not read.
    unneeded = {r['profile']: dict(elapsed_s=round(r['seconds'], 1), lower_bound=r['lower_bound']) for r in rows if not r['needed']}
    return profiling, unneeded


def extraction(con: duckdb.DuckDBPyConnection, point: dict, test: dict) -> dict:
    """Test-split seconds and throughput of the reference pipeline and of Tetris at its operating point, per boundary."""
    # Per-stage seconds of both pipelines on the test split.
    reference, tetris = stage_summary(con, 'reference_test'), stage_summary(con, 'point_test')
    frames = test['frames']
    # The tradeoff table and the video files must describe the same test split.
    assert frames == point['test_frames'] == point['frames'], \
        f"test frames: videos {frames}, reference row {point['test_frames']}, Tetris row {point['frames']}"
    # Seconds per boundary: the paper's (p121 as is), instrumented (without decoding), and process (timestamps).
    seconds = {system: dict(paper=total(rows, 'recorded_s'), instrumented=total(rows, 'instrumented_s'),
                            process=total(rows, 'process_s'))
               for system, rows in (('reference', reference), ('tetris', tetris))}
    # Every time must be positive to define a throughput.
    assert all(v > 0 for d in seconds.values() for v in d.values()), f'non-positive extraction time: {seconds}'
    # Seconds and throughput per system and boundary, with the per-stage detail and cross-checks.
    return dict(
        frames=frames,
        seconds=rounded(seconds, 3),
        fps={s: {b: frames / v for b, v in d.items()} for s, d in seconds.items()},
        tetris_stages=stage_detail(tetris), reference_stages=stage_detail(reference),
        # This recomputation must reproduce the tradeoff table's (paper) times; the ratios show it does.
        paper_time_ratio_to_tradeoff_table=dict(reference=round(seconds['reference']['paper'] / point['reference_time_s'], 4),
                                                tetris=round(seconds['tetris']['paper'] / point['time_s'], 4)),
        # How much the process boundary adds on each side.
        process_over_paper=dict(reference=round(seconds['reference']['process'] / seconds['reference']['paper'], 3),
                                tetris=round(seconds['tetris']['process'] / seconds['tetris']['paper'], 3)))


def measure_dataset(con: duckdb.DuckDBPyConnection, dataset: str, loss: float, compilations: list[dict]) -> dict:
    """Every preparation component, totals at each boundary, and the break-even durations for one dataset.

    Needs the profile_estimates table (profile_estimates) for every dataset.
    """
    # The operating point and the reference it is compared against.
    point = operating_point(con, dataset, loss)
    # The classifier and tile size the reported configuration uses (the configured grid has one of each).
    model = f"{point['classifier']}_{point['tilesize']}"
    # Manifests, runtime ops, and work units of every file this dataset's components read.
    loaded = manifests(con, dataset, point)
    coverage = loaded['coverage']
    # Videos, frames, and minutes of each split.
    train, valid, test = (video_split(dataset, s) for s in ('train', 'valid', 'test'))

    # Training-split reference detection; p011 must have timed every training frame.
    train_det = stage_summary(con, 'train_det_manifest')
    (frames,) = con.execute("""SELECT coalesce(sum(n), 0) FROM (SELECT max(records) AS n FROM runtime_ops
                               WHERE stage = '011_tune_detect' AND op = 'detect' GROUP BY runtime_file)""").fetchall()[0]
    assert frames == train['frames'], f'{dataset}: p011 timed {frames} of {train["frames"]} training frames'
    # Per-frame detection cost of the test-split reference pipeline (001's detect op), to show what estimating would give.
    (reference_detect_s,) = con.execute("""
        SELECT sum(r.seconds) FROM runtime_ops r JOIN (SELECT DISTINCT runtime_file FROM reference_test) USING (runtime_file)
        WHERE r.stage = '001_preprocess_groundtruth_detection' AND r.op = 'detect'""").fetchall()[0]
    # Recorded per frame; the estimate from the test reference's per-frame detection is kept for comparison.
    detection_train = component(
        'recorded: p011 times each frame\'s read and detect; process: recorded (file creation times are stale on some '
        'datasets); detector loading per video is not recorded', total(train_det, 'recorded_s'),
        total(train_det, 'instrumented_s'), total(train_det, 'recorded_s'), frames=frames,
        decode_s=round(total(train_det, 'decode_s'), 1),
        ms_per_frame=round(1000 * total(train_det, 'recorded_s') / frames, 2),
        estimate_from_test_reference_detect_s=round(frames * reference_detect_s / test['frames'], 1),
        note='p011 detects in batches of 16; the reference pipeline detects one frame at a time')

    # Training-data creation at the operating point's tile size.
    data_rows = stage_summary(con, 'train_data_manifest')
    data_files = [r['runtime_file'] for r in query_dicts(con, 'SELECT DISTINCT runtime_file FROM train_data_manifest')]
    # Recorded split and save; process time from the runtime files' timestamps.
    training_data = component(
        'recorded: split and save per frame; process: estimate from each video\'s runtime file creation to last write '
        '(adds the untimed seek, decode, and detection parsing)', total(data_rows, 'recorded_s'),
        total(data_rows, 'instrumented_s'), total(data_rows, 'process_s'), tilesize=point['tilesize'],
        stages=stage_detail(data_rows), first_frame_split_s=round(first_frame_seconds(con, data_files), 1))

    # Classifier training and optimization (p014, p015) of the operating point's model.
    training, optimization, unneeded_training = classifier_components(con, dataset, model, loaded['trained'], compilations)
    # Mistrack profiles (p016) of the configured trackers.
    profiling, unneeded_profiles = profiling_component(con, dataset, point['tilesize'], train['videos'])

    # Reference outputs on the validation split, the validation pseudo ground truth (p110 scores against them).
    reference_rows = stage_summary(con, 'reference_manifest')
    reference_valid = component(
        'recorded: read, detect, and track per frame; process: estimate, detection from its runtime file\'s creation to '
        'detection.jsonl\'s last write (not below the recorded ops), tracking as recorded',
        total(reference_rows, 'recorded_s'), total(reference_rows, 'instrumented_s'), total(reference_rows, 'process_s'),
        decode_s=round(total(reference_rows, 'decode_s'), 1), stages=stage_detail(reference_rows))

    # The validation sweep, recorded on complete videos and scaled to all of them.
    sweep_rows = stage_summary(con, 'sweep_manifest')
    # Configurations in the sweep, and the factor that scales the complete videos to the split.
    configurations = query_dicts(con, 'SELECT count(DISTINCT variant_id) AS n FROM sweep_manifest')[0]['n']
    scale = coverage['scale']
    instrumented_sweep, process_sweep = total(sweep_rows, 'instrumented_s'), total(sweep_rows, 'process_s')
    # Instrumented and process seconds are scaled to the split; the recorded figure is not.
    sweep = component(
        ('recorded' if scale == 1 else
         f"recorded on {coverage['complete_videos']} of {coverage['videos']} videos, scaled to all (estimate)")
        + '; process: estimate from file timestamps', total(sweep_rows, 'recorded_s'),
        scale * instrumented_sweep, scale * process_sweep, configurations=configurations, coverage=coverage,
        stages=stage_detail(sweep_rows),
        stage_shares={b: {r['stage']: round(r[f'{b}_s'] / total(sweep_rows, f'{b}_s'), 4) for r in sweep_rows}
                      for b in ('instrumented', 'process')},
        without_sharing_instrumented_s=round(scale * total(sweep_rows, 'without_sharing_instrumented_s'), 1),
        without_sharing_process_s=round(scale * total(sweep_rows, 'without_sharing_process_s'), 1),
        dropped_ops_s=dropped_ops(con, 'sweep_manifest'),
        lower_bound_units=sum(r['work_units'] - r['chained_units'] for r in sweep_rows),
        stale_starts=sum(r['stale_starts'] for r in sweep_rows))

    # Scoring every validation configuration on every validation video, estimated from the test split's scoring.
    scoring = scoring_estimate(con, gather_scoring(dataset), configurations, coverage['videos'])
    # Zero, flagged missing, when no test task could be used.
    scoring_s = scoring['estimate_s'] or 0.0
    validation_scoring = component(
        'estimate: test-split p110 seconds per (configuration, video) x validation configurations x videos'
        if scoring['estimate_s'] is not None else 'missing: no usable test-split scoring outputs', 0.0, scoring_s, scoring_s,
        unit='process seconds (one TrackEval process per configuration)', estimated=scoring['estimate_s'] is not None,
        **rounded({k: v for k, v in scoring.items() if k != 'estimate_s'}, 3))

    # Components in report order.
    components = dict(reference_detection_train=detection_train, training_data_creation=training_data,
                      classifier_training=training, classifier_optimization=optimization,
                      mistrack_profiling=profiling, reference_outputs_valid=reference_valid,
                      validation_sweep=sweep, validation_scoring=validation_scoring)
    assert tuple(components) == COMPONENTS
    # Totals per boundary, as sums of the rounded component figures the report shows.
    totals = {b: round(sum(c[f'{b}_s'] for c in components.values()), 1) for b in ('recorded', 'instrumented', 'process')}
    # Each component's share of each boundary's total.
    shares = {b: {k: round(c[f'{b}_s'] / totals[b], 4) for k, c in components.items()} for b in ('instrumented', 'process')}

    # Extraction at the operating point and the break-even durations, each boundary on both sides.
    extract = extraction(con, point, test)
    # Frames per minute of test video.
    frames_per_minute = 60 * test['frames'] / test['seconds']
    # The dataset's report.
    return dict(
        label=DATASET_LABELS[dataset],
        video={s: {k: v for k, v in d.items() if k != 'names'} for s, d in (('train', train), ('valid', valid), ('test', test))},
        operating_point=rounded(point),
        components=components,
        not_recorded=dict(
            training_data_pruning='p013 (deletes never-relevant tiles\' training images) writes no timings',
            tradeoff_and_pareto='p130 tradeoff and p135 Pareto selection write no timings (they read one score table)',
            detector_loading='detector loading and warm-up per p011 video, per reference video, and per p040 '
                             'configuration are not recorded; outside both boundaries on both sides',
            first_unit_of_each_task='process time of the first video of each sweep task is max(own span, recorded): '
                                    'it misses work done before its output directory was created'),
        not_needed_by_reported_configuration=dict(classifier_training=unneeded_training,
                                                  mistrack_profiling=unneeded_profiles),
        totals=dict(
            recorded_lower_bound_s=totals['recorded'], recorded_lower_bound_hours=round(totals['recorded'] / 3600, 3),
            recorded_lower_bound_omits=['classifier frozen stage', 'p015', 'p016', 'validation scoring', 'untimed work']
            + ([f"sweep on {coverage['videos'] - coverage['complete_videos']} incomplete videos"] if scale > 1 else []),
            instrumented_s=totals['instrumented'], instrumented_hours=round(totals['instrumented'] / 3600, 3),
            process_s=totals['process'], process_hours=round(totals['process'] / 3600, 3),
            shares=shares),
        extraction=extract,
        break_even=dict(frames_per_video_minute=round(frames_per_minute, 2),
                        **{b: break_even(totals[b], extract['fps']['reference'][b], extract['fps']['tetris'][b],
                                         frames_per_minute) for b in ('instrumented', 'process')}),
    )


def main():
    # Command-line options.
    args = parse_args()
    # A loss bound is a fraction of the reference HOTA.
    assert 0 <= args.loss < 1, f'--loss must be in [0, 1), got {args.loss}'
    # One in-memory DuckDB connection for every table.
    con = duckdb.connect()
    # Keep the log free of DuckDB progress bars.
    con.execute('SET enable_progress_bar = false')

    # Mistrack profiles and p015 benchmarks of every dataset, since consecutive ones of one run span datasets.
    profiles, incomplete_profiles = gather_profiles(list(DATASET_LABELS))
    profile_estimates(con, profiles)
    # p015 benchmark estimates of every dataset.
    compilations = optimization_estimates(con, gather_compilations(list(DATASET_LABELS)))
    # Measure each requested dataset.
    results = {}
    for dataset in args.datasets:
        results[dataset] = measure_dataset(con, dataset, args.loss, compilations)
        # One line per dataset and boundary for the run log.
        t, b = results[dataset]['totals'], results[dataset]['break_even']
        for boundary in ('instrumented', 'process'):
            print(f"{dataset:12s} {boundary:12s} preparation {t[f'{boundary}_hours']:.2f} h | Tetris "
                  f"{b[boundary]['tetris_fps']:.1f} fps vs reference {b[boundary]['reference_fps']:.1f} fps | saves "
                  f"{b[boundary]['saving_s_per_video_minute']:.1f} s per video minute | break-even "
                  f"{b[boundary]['break_even_video_minutes']} video minutes")

    # One JSON with every dataset, the method, and the baselines' re-timing needs.
    write_json(results_dir(__file__), 'preparation_cost.json', dict(
        datasets=results,
        method=dict(
            loss=args.loss, unit=PROCESS_UNIT,
            boundaries=dict(
                recorded='recorded op times only: p121\'s boundary as is, classifier fine-tuning epochs only, sweep '
                         'unscaled, no estimates; a lower bound',
                instrumented='p121\'s counted ops minus per-frame video decoding (VIDEO_DECODE_OPS), on preparation and '
                             'on both pipelines\' extraction; components without op records at their process time',
                process='whole-process time per work unit from file timestamps (sweep, p012, reference detection on '
                        'both pipelines\' sides); recorded time for p011 (stale creation times) and the reference\'s '
                        'tracking; components without op records at their process time; detector loading is outside on '
                        'both sides'),
            cache_dir=CONFIG['DATA']['CACHE_DIR'],
            inputs=['<ds>/indexing/segment/detection/*.detections.jsonl',
                    '<ds>/indexing/training/runtime/tilesize_*/*_creating_training_data.jsonl',
                    '<ds>/indexing/training/results/*/{throughput_per_epoch,model_compilation}.jsonl (+ timestamps)',
                    '<ds>/indexing/track_rate/*/ (modification times)',
                    '<ds>/execution/*/{002_naive,020,022,033,040,050,060}/** runtime files (+ timestamps)',
                    '<ds>/evaluation/070_accuracy/raw/test/*/ (timestamps)',
                    '<ds>/evaluation/090_tradeoff/tradeoff.csv'],
            ignored_ops=list(IGNORED_OPS), op_list_paths=OP_LIST_PATHS, video_decode_ops=VIDEO_DECODE_OPS,
            stage_spans={k: dict(start_level=level, end_siblings=ends, chained=chain)
                         for k, (level, ends, chain) in STAGE_SPANS.items()},
            chain_gap_s=CHAIN_GAP_S, stale_span_s=STALE_SPAN_S, timestamp_tolerance_s=TIMESTAMP_TOLERANCE_S,
            finetune_wall_max_ratio=FINETUNE_WALL_MAX_RATIO, finetune_wall_slack_s=FINETUNE_WALL_SLACK_S,
            p014_epochs_per_stage=P014_EPOCHS_PER_STAGE, p016_workers=P016_WORKERS, p015_group_gap_s=P015_GROUP_GAP_S,
            incomplete_profiles=incomplete_profiles,
            host=dict(host_info(), note='cache host, assumed for all runs; GPUs are those present when this script ran')),
        baselines=dict(BASELINE_PREPARATION, plan='measurements-gpu/r9o2_preparation_cost.py (dry run by default)'),
    ))


if __name__ == '__main__':
    main()
