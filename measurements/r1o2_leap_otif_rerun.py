#!/usr/local/bin/python
"""R1.O2 and R9.O2: LEAP rerun on our test clips and LEAP/OTIF preparation cost, from measurements-gpu/r1o2_leap_otif_rerun.py.

Depends on measurements-gpu/r1o2_leap_otif_rerun.py (run on bombe), whose outputs are in
measurements-results/r1o2_leap_otif_rerun/raw/; fails with a clear error when they are missing. The GPU script
reran LEAP's inference (LEAP's own main.py loop, unchanged, in the env of the container the scored runs used) on our
15 FPS test clips with the scored runs' configs, detectors (YOLOv5x for CalDoT and Amsterdam in place of LEAP's
distilled detector, RetinaNet for B3D), re-identification weights, and motion patterns, all copied from the scored
workspace (manchester:/data/alenachao/data/leap-dataset), and exported the state LEAP keeps per clip.

What this script reports per dataset:
  inputs       whether the videos in the scored workspace are our clips (MD5 of every train and test clip, from
               raw/manifest.json), and the image and env provenance.
  agreement    the rerun's parsed results (per object: interval, motion-pattern index, class id) against the scored
               run's (<clip>_results.json in the scored workspace), object by object; the motion patterns (hash per
               clip); the frames LEAP sampled (the scored speed.tsv's cumulative frames_sampled, differenced per run);
               and the track exports rebuilt with the scored runs' exporter (export-leap-outputs.py) from the rerun
               and from the scored results, each against the scored exports in /sota-results/<dataset>/leap/.
  frame_base   LEAP's sparse observations (the detector boxes it kept, [x1, y1, x2, y2, class, frame]) against the
               root dataset's ground truth on clip frame round(frame x scale) + lag: per scale (0.5, 1, 2) at lag 0
               and per lag (-3..3) at scale 1, the mean best IoU per observation and the share with IoU >= 0.5.
               LEAP's frame f is our frame f when scale 1 and lag 0 score best.
  hota         TrackEval (HOTA and Count, configured as evaluation/p141_otif_accuracy.py configures it, against the
               root dataset's 003_groundtruth, in a temporary directory) on three readings of the rerun:
                 raw      only the observations, each at its frame with LEAP's object id
                 linear   the observations linearly interpolated between consecutive observations of an object
                          (polyis.utilities.register_tracked_detections, as the pipeline interpolates tracks)
                 pattern  the rerun's export through p140's transform (transform_tracking_json with unwrap_leap),
                          which is how the reported LEAP tracks were built
               and the reported LEAP HOTA (p141's DATASET.json for parameter 000). pattern must reproduce it when
               the rerun's exports equal the scored ones (pattern_minus_reported_e6 is the difference in millionths).
  classes      class ids of LEAP's objects and observations (COCO 2 car, 5 bus, 7 truck); RetinaNet's B3D model
               has one class, which LEAP's adapter maps to 2. Also observations per object: an object LEAP saw on
               one sampled frame gives the linear reading nothing to interpolate (most of CalDoT2's and B3D3's).
  timing       the rerun's GPU, CPU set, contamination flag, GNU time, and LEAP's own timers against the scored
               speed.tsv (the mean of the scored runs of each clip; CalDoT's file holds two runs of every clip), and
               the other GPUs' activity during the run (see neighbors below).
  preparation  R9.O2: LEAP's and OTIF's preparation, re-timed on bombe by the GPU script's prep action (raw/prep/):
               per step GNU time's wall clock, CPU time, and peak memory, the GPU, its CPU set and the share of it the
               step kept busy (cpu_busy_share; each job had 10 cores, half of NUMA node 1, so a share near 1 marks a
               step more cores would speed up), GPU utilization and host load from the launcher's samples, the
               contamination flag, and the neighbors; per system and dataset the
               total, the total without evaluation-only steps (EVALUATION_ONLY_STEPS: LEAP's test-clip pseudo labels,
               which feed only LEAP's evaluation and the re-identification evaluation set), which steps are missing or
               failed, and whether every step was clean. OTIF's frame extraction reports how many frames it took from
               tracks (16_extract_frames.py's 'got N frames with track'). The reconstruction is checked against
               the scored run's outputs: LEAP's pseudo labels (identical lines, and lines within LABEL_PAIR_PX),
               re-identification crops (shared file names), re-identification training (fastreid's logged total, last
               Rank-1/mAP, which LEAP's empty evaluation table leaves null, and last training loss), motion-pattern
               cluster counts, and OTIF's validation curve. The scored run's own durations, where its files record
               them: LEAP's labeling (label files' modification times) and re-identification training (its log).
               Skipped, with a message, when raw/prep/steps.json is missing.
  neighbors    per timed run, from <record>_gpus.jsonl.gz (the GPU script's gpus action; null when absent): every other
               GPU's mean utilization, share of samples with utilization above 0, and peak memory, and the largest such
               share among the GPUs on the run's NUMA node (NODE_GPUS), since those share its CPUs' memory bus (and
               GPUs 5 and 6 a PCIe switch). The contamination flag covers only processes on the run's own GPU.

Writes measurements-results/r1o2_leap_otif_rerun/results.json, or results-partial.json when --datasets selects only
some datasets.
"""
import argparse
import json
import multiprocessing as mp
import tempfile
from pathlib import Path

import duckdb

from evaluation.manifests import list_split_videos
from evaluation.p140_otif_transform import dataset_name_in_sota, extract_video_id, transform_tracking_json
from measurements.common import (DATASET_LABELS, TRACK_COLUMNS, assert_files_exist, file_record, metric_summary,
                                 output_path, query_dicts, results_dir, rounded, trackeval_rerun, write_json)
from polyis.io import cache
from polyis.utilities import register_tracked_detections, save_tracking_results


# The GPU script's outputs, inside this script's result directory.
RAW_DIR = Path(__file__).resolve().parent.parent / 'measurements-results' / 'r1o2_leap_otif_rerun' / 'raw'
# LEAP has one parameter set, stored under parameter id 000 (p140, p141).
LEAP_PARAM = '000'
# Frame-base grid: time scales at lag 0 and lags at scale 1, and the IoU that counts an observation as matched.
SCALES = (0.5, 1.0, 2.0)
LAGS = tuple(range(-3, 4))
IOU_MATCH = 0.5
# Readings of the rerun scored with TrackEval.
HOTA_VARIANTS = ('raw', 'linear', 'pattern')
# Largest difference allowed between the pattern reading and the reported HOTA.
REPRODUCTION_TOLERANCE = 1e-9
# Schemas of the GPU script's JSON lines (fields not listed are not read).
OBJECT_TYPE = 'STRUCT(id BIGINT, "interval" BIGINT[], pattern BIGINT, class_id DOUBLE, observations DOUBLE[][])'
RERUN_COLUMNS = ("{'video_name': 'VARCHAR', 'frames_sampled': 'BIGINT[]', 'patterns_sha256': 'VARCHAR', "
                 "'timings': 'STRUCT(run_session_wall_s DOUBLE, detector_s DOUBLE, differencer_s DOUBLE, reid_s DOUBLE, "
                 f"match_s DOUBLE, decode_s DOUBLE, frames_filtered BIGINT)', 'objects': '{OBJECT_TYPE}[]'}}")
SCORED_OBJECT_TYPE = 'STRUCT(id BIGINT, "interval" BIGINT[], pattern BIGINT, class_id DOUBLE)'
SCORED_COLUMNS = f"{{'video_name': 'VARCHAR', 'patterns_sha256': 'VARCHAR', 'objects': '{SCORED_OBJECT_TYPE}[]'}}"
SPEED_TYPE = ('STRUCT(video_name VARCHAR, inference_total VARCHAR, detector VARCHAR, differencer VARCHAR, reid VARCHAR, '
              '"match" VARCHAR, decode VARCHAR, frames_sampled VARCHAR, frames_filtered VARCHAR, "row" BIGINT)[]')
EXPORT_COLUMNS = "{'video_name': 'VARCHAR', 'exporter_sha256': 'VARCHAR', 'frames': 'JSON'}"
# One exported frame list, as export-leap-outputs.py writes it (from_json's JSON structure form; a frame without
# boxes is null).
FRAMES_TYPE = ('[[{"left": "DOUBLE", "top": "DOUBLE", "right": "DOUBLE", "bottom": "DOUBLE", '
               '"class": "VARCHAR", "score": "DOUBLE", "track_id": "BIGINT"}]]')
# GNU time -v fields (regular expressions with one group) read from every timed record.
GNU_ELAPSED = r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'
GNU_FIELDS = {'user_s': r'User time \(seconds\): ([0-9.]+)', 'system_s': r'System time \(seconds\): ([0-9.]+)',
              'max_rss_kb': r'Maximum resident set size \(kbytes\): ([0-9]+)', 'exit_status': r'Exit status: ([0-9]+)'}
# The scored re-identification logs report the run's total as 'Total training time: h:mm:ss'.
REID_TOTAL = r'Total training time: ([0-9:]+)'
# One OTIF validation-curve line: detector, segmentation, tracker, runtime, and validation accuracy (tab-separated).
CURVE_FIELDS = ('detector', 'segmentation', 'tracker', 'runtime', 'valid_accuracy')
# Steps whose outputs only evaluate the system being prepared, per system: LEAP's test-clip pseudo labels stand in for
# ground truth in LEAP's evaluation and the re-identification evaluation set; LEAP's training never reads them.
EVALUATION_ONLY_STEPS = {'leap': ('labels_test',)}
# bombe's GPUs per NUMA node (measurements-gpu/BOMBE.md).
NODE_GPUS = {0: (0, 1, 2, 3), 1: (4, 5, 6)}
# Largest box-coordinate difference (pixels) at which a regenerated pseudo-label line still agrees with the scored one.
LABEL_PAIR_PX = 1.0
# 16_extract_frames.py's count of the frames it sampled from tracks (the rest are uniform random frames).
OTIF_TRACKED_FRAMES = r'got ([0-9]+) frames with track'
# The raw files of one dataset, by role.
RAW_FILES = {'rerun': 'leap_rerun.jsonl.gz', 'scored': 'leap_scored.jsonl.gz', 'scored_run': 'leap_scored_run.jsonl.gz',
             'rerun_export': 'leap_rerun_export.jsonl.gz', 'scored_export': 'leap_scored_export.jsonl.gz',
             'launch': 'leap_rerun_launch.json'}


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O2/R9.O2: LEAP rerun agreement, frame base, raw/linear HOTA, and preparation cost')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--sota-dir', type=Path, default=Path('/sota-results'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def create_macros(con: duckdb.DuckDBPyConnection):
    """SQL macros shared by the rerun's timing and the preparation records: GNU time's wall clock in seconds
    (h:mm:ss or m:ss), one numeric field, and the CPUs in a cpulist such as '20-29,60-69'."""
    # Each comma-separated part is one CPU or an inclusive range.
    con.execute("""
        CREATE OR REPLACE MACRO cpu_count(cpus) AS list_sum(list_transform(string_split(cpus, ','), lambda r:
            CASE WHEN contains(r, '-') THEN split_part(r, '-', 2)::BIGINT - split_part(r, '-', 1)::BIGINT + 1 ELSE 1 END))
    """)
    # The wall clock has two or three colon-separated parts.
    con.execute(f"""
        CREATE OR REPLACE MACRO gnu_elapsed_s(t) AS (
            WITH p AS (SELECT string_split(regexp_extract(t, '{GNU_ELAPSED}', 1), ':') AS l)
            SELECT CASE len(l) WHEN 3 THEN l[1]::DOUBLE * 3600 + l[2]::DOUBLE * 60 + l[3]::DOUBLE
                               WHEN 2 THEN l[1]::DOUBLE * 60 + l[2]::DOUBLE END FROM p)
    """)
    # A missing field is NULL.
    con.execute("CREATE OR REPLACE MACRO gnu_number(t, pattern) AS TRY_CAST(nullif(regexp_extract(t, pattern, 1), '') AS DOUBLE)")


def create_gpu_neighbors(con: duckdb.DuckDBPyConnection, runs: list[tuple[str, int, int, Path]]):
    """Table gpu_neighbors (key, other_gpus, node_neighbor_busy_share): for each timed run (key, its GPU, its NUMA node,
    its <record>_gpus.jsonl.gz) whose file exists, every other GPU's activity during the run (see the module
    docstring). Runs without the file get no row."""
    con.execute('CREATE OR REPLACE TABLE gpu_nodes (gpu BIGINT, node BIGINT)')
    con.executemany('INSERT INTO gpu_nodes VALUES (?, ?)', [(g, n) for n, gs in NODE_GPUS.items() for g in gs])
    con.execute('CREATE OR REPLACE TABLE gpu_runs (key VARCHAR, gpu BIGINT, node BIGINT, filename VARCHAR)')
    con.execute('CREATE OR REPLACE TABLE gpu_neighbors (key VARCHAR, other_gpus STRUCT(gpu BIGINT, util_mean DOUBLE, '
                'busy_share DOUBLE, mem_max_mib DOUBLE)[], node_neighbor_busy_share DOUBLE)')
    # Only runs whose GPU log exists; with none, both tables stay empty (DuckDB rejects an empty executemany).
    present = [(k, g, n, str(f)) for k, g, n, f in runs if Path(f).exists()]
    if not present:
        return
    con.executemany('INSERT INTO gpu_runs VALUES (?, ?, ?, ?)', present)
    # Per run and other GPU, the activity over the run's samples; then the run's same-node maximum.
    con.execute("""
        INSERT INTO gpu_neighbors
        WITH per_gpu AS (
            SELECT r.key, r.node, s.gpu, avg(s.util) AS util_mean, avg(CASE WHEN s.util > 0 THEN 1 ELSE 0 END) AS busy_share,
                   max(s.mem_mib) AS mem_max_mib
            FROM read_json(?, filename = true, format = 'newline_delimited',
                           columns = {'gpu': 'BIGINT', 'util': 'DOUBLE', 'mem_mib': 'DOUBLE'}) s
            JOIN gpu_runs r USING (filename)
            WHERE s.gpu <> r.gpu
            GROUP BY ALL
        )
        SELECT p.key, list({'gpu': p.gpu, 'util_mean': p.util_mean, 'busy_share': p.busy_share,
                            'mem_max_mib': p.mem_max_mib} ORDER BY p.gpu),
               coalesce(max(p.busy_share) FILTER (n.node = p.node), 0)
        FROM per_gpu p LEFT JOIN gpu_nodes n USING (gpu)
        GROUP BY p.key
    """, [[f for *_, f in present]])


def raw_paths(raw_dir: Path, dataset: str) -> dict[str, Path]:
    """The GPU script's files for one dataset, checked to exist."""
    # Every role must be present, or the GPU script has not finished this dataset.
    paths = {role: raw_dir / dataset / name for role, name in RAW_FILES.items()}
    assert_files_exist(paths.values(), f'{dataset} outputs of measurements-gpu/r1o2_leap_otif_rerun.py (run its '
                                       'setup, scored, leap, and export actions on bombe)')
    return paths


def load_dataset(con: duckdb.DuckDBPyConnection, dataset: str, paths: dict[str, Path], sota_dir: Path):
    """Tables clips, rerun_clips, scored_clips, rerun_objects, scored_objects, exports, observations, gt_boxes, and
    scored_speed for one dataset."""
    # Our test clips and LEAP's names for them (te07.mp4 is LEAP's 7).
    videos = list_split_videos(dataset, 'test')
    con.execute('CREATE OR REPLACE TABLE clips (video VARCHAR, leap_name VARCHAR)')
    con.executemany('INSERT INTO clips VALUES (?, ?)', [(v, str(extract_video_id(v))) for v in videos])
    # The rerun's and the scored run's clips.
    con.execute(f"CREATE OR REPLACE TABLE rerun_clips AS SELECT * FROM read_json(?, format = 'newline_delimited', "
                f"columns = {RERUN_COLUMNS})", [str(paths['rerun'])])
    con.execute(f"CREATE OR REPLACE TABLE scored_clips AS SELECT * FROM read_json(?, format = 'newline_delimited', "
                f"columns = {SCORED_COLUMNS})", [str(paths['scored'])])
    # Every clip on both sides, and no clip twice.
    bad = query_dicts(con, """
        SELECT c.leap_name, (SELECT count(*) FROM rerun_clips r WHERE r.video_name = c.leap_name) AS rerun,
               (SELECT count(*) FROM scored_clips s WHERE s.video_name = c.leap_name) AS scored
        FROM clips c WHERE rerun <> 1 OR scored <> 1
        UNION ALL SELECT video_name, 1, NULL FROM rerun_clips WHERE video_name NOT IN (SELECT leap_name FROM clips)
        UNION ALL SELECT video_name, NULL, 1 FROM scored_clips WHERE video_name NOT IN (SELECT leap_name FROM clips)
    """)
    assert not bad, f'{dataset}: clips missing, repeated, or unknown in the rerun or scored results: {bad[:5]}'
    # One row per object; an interval is [start, end).
    for side in ('rerun', 'scored'):
        observations = ', o.observations' if side == 'rerun' else ''
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_objects AS
            SELECT video_name, o.id, o."interval"[1] AS s, o."interval"[2] AS e, len(o."interval") AS interval_len,
                   o.pattern, o.class_id{observations}
            FROM (SELECT video_name, unnest(objects) AS o FROM {side}_clips)
        """)
        # LEAP's ids are unique within a clip, and every interval has two ends.
        bad = query_dicts(con, f"""
            SELECT video_name, id, count(*) AS n, any_value(interval_len) AS interval_len FROM {side}_objects
            GROUP BY ALL HAVING n > 1 OR any_value(interval_len) <> 2 LIMIT 5
        """)
        assert not bad, f'{dataset}: {side} objects repeated or without a two-ended interval: {bad}'
    # The observations, one row per kept detector box, on our clip names.
    con.execute("""
        CREATE OR REPLACE TABLE observations AS
        SELECT c.video, r.id, CAST(ob[6] AS BIGINT) AS frame, ob[1] AS x1, ob[2] AS y1, ob[3] AS x2, ob[4] AS y2,
               ob[5] AS class_id, len(ob) AS width
        FROM rerun_objects r JOIN clips c ON c.leap_name = r.video_name, unnest(r.observations) AS t(ob)
    """)
    # [x1, y1, x2, y2, class, frame], at most one per object and frame (LEAP excludes an object once matched).
    bad = query_dicts(con, """
        SELECT video, id, frame, count(*) AS n, any_value(width) AS width FROM observations
        GROUP BY ALL HAVING n > 1 OR any_value(width) <> 6 OR frame IS NULL OR frame < 0 LIMIT 5
    """)
    assert not bad, f'{dataset}: observations malformed or repeated within an object and frame: {bad}'
    # The rebuilt exports (rerun and scored) and the scored exports in /sota-results, as typed frame lists.
    sota = sota_dir / dataset_name_in_sota(dataset) / 'leap'
    sota_files = [(c, str(sota / f'{c}.json')) for c, in con.execute('SELECT leap_name FROM clips').fetchall()]
    assert_files_exist([p for _, p in sota_files], f'{dataset} scored LEAP exports')
    con.execute('CREATE OR REPLACE TABLE sota_files (leap_name VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO sota_files VALUES (?, ?)', sota_files)
    con.execute(f"""
        CREATE OR REPLACE TABLE exports AS
        SELECT 'sota' AS source, f.leap_name AS video_name, NULL AS exporter_sha256,
               coalesce(from_json(t.content, '{FRAMES_TYPE}'), []) AS frames
        FROM read_text(?) t JOIN sota_files f USING (filename)
        UNION ALL
        SELECT source, video_name, exporter_sha256, coalesce(from_json(frames, '{FRAMES_TYPE}'), [])
        FROM (SELECT 'rerun' AS source, * FROM read_json(?, format = 'newline_delimited', columns = {EXPORT_COLUMNS})
              UNION ALL
              SELECT 'scored', * FROM read_json(?, format = 'newline_delimited', columns = {EXPORT_COLUMNS}))
    """, [[p for _, p in sota_files], str(paths['rerun_export']), str(paths['scored_export'])])
    # One export per source and clip.
    bad = query_dicts(con, """
        SELECT c.leap_name, e.source, count(e.video_name) AS n FROM clips c CROSS JOIN (VALUES ('sota'), ('rerun'), ('scored')) s(source)
        LEFT JOIN exports e ON e.video_name = c.leap_name AND e.source = s.source GROUP BY ALL HAVING n <> 1 LIMIT 5
    """)
    assert not bad, f'{dataset}: exports missing or repeated: {bad}'
    # The scored run's speed.tsv rows, numbers parsed.
    con.execute(f"""
        CREATE OR REPLACE TABLE scored_speed AS
        SELECT r."row", r.video_name, CAST(r.inference_total AS DOUBLE) AS inference_total,
               CAST(r.detector AS DOUBLE) AS detector, CAST(r.differencer AS DOUBLE) AS differencer,
               CAST(r.reid AS DOUBLE) AS reid, CAST(r."match" AS DOUBLE) AS "match", CAST(r.decode AS DOUBLE) AS decode,
               CAST(r.frames_sampled AS BIGINT) AS frames_sampled_cumulative
        FROM (SELECT unnest(speed) AS r FROM read_json(?, format = 'newline_delimited', columns = {{'speed': '{SPEED_TYPE}'}}))
    """, [str(paths['scored_run'])])
    # The root dataset's ground truth, one row per box ([track_id, x1, y1, x2, y2], 1-based in DuckDB).
    gt_files = [(v, str(output_path(dataset, 'ground truth', '', v))) for v in videos]
    assert_files_exist([p for _, p in gt_files], f'{dataset} ground truth')
    con.execute('CREATE OR REPLACE TABLE gt_files (video VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO gt_files VALUES (?, ?)', gt_files)
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_boxes AS
        SELECT f.video, l.frame_idx AS frame, b[2] AS x1, b[3] AS y1, b[4] AS x2, b[5] AS y2
        FROM (SELECT filename, frame_idx, unnest(tracks) AS b
              FROM read_json(?, filename = true, format = 'newline_delimited', columns = {TRACK_COLUMNS})) l
        JOIN gt_files f USING (filename)
    """, [[p for _, p in gt_files]])
    return videos


def agreement(con: duckdb.DuckDBPyConnection) -> dict:
    """The rerun against the scored run: objects, clips, motion patterns, sampled frames, and exports."""
    # Object by object, matched on clip and LEAP's id.
    objects = query_dicts(con, """
        SELECT count(r.id) AS rerun_objects, count(s.id) AS scored_objects,
               count(*) FILTER (r.id IS NOT NULL AND s.id IS NOT NULL) AS matched_ids,
               count(*) FILTER (r.s = s.s AND r.e = s.e) AS same_interval,
               count(*) FILTER (r.pattern = s.pattern) AS same_pattern,
               count(*) FILTER (r.class_id = s.class_id) AS same_class,
               count(*) FILTER (r.s = s.s AND r.e = s.e AND r.pattern = s.pattern AND r.class_id = s.class_id) AS identical
        FROM rerun_objects r FULL JOIN scored_objects s USING (video_name, id)
    """)[0]
    # Clips whose every object agrees, with the clips that do not.
    clips = query_dicts(con, """
        WITH per_clip AS (
            SELECT video_name, bool_and(r.id IS NOT NULL AND s.id IS NOT NULL AND r.s = s.s AND r.e = s.e
                                        AND r.pattern = s.pattern AND r.class_id = s.class_id) AS identical
            FROM rerun_objects r FULL JOIN scored_objects s USING (video_name, id) GROUP BY video_name
        )
        SELECT (SELECT count(*) FROM clips) AS clips,
               (SELECT count(*) FROM clips) - count(*) FILTER (NOT identical) AS identical_clips,
               coalesce(list(video_name ORDER BY video_name) FILTER (NOT identical), []) AS differing_clips
        FROM per_clip
    """)[0]
    # The motion patterns LEAP loaded, per clip.
    patterns = query_dicts(con, """
        SELECT count(*) FILTER (r.patterns_sha256 = s.patterns_sha256) AS identical_clips,
               count(DISTINCT r.patterns_sha256) AS rerun_pattern_files, count(DISTINCT s.patterns_sha256) AS scored_pattern_files
        FROM rerun_clips r JOIN scored_clips s USING (video_name)
    """)[0]
    # Frames sampled per clip: the rerun's slices against the scored run's cumulative counts, differenced per run (the
    # scored file holds one block of rows per run, each in LEAP's clip order).
    sampled = query_dicts(con, """
        WITH runs AS (
            SELECT *, "row" // (SELECT count(*) FROM clips) AS run FROM scored_speed
        ),
        scored AS (
            SELECT run, video_name, frames_sampled_cumulative
                   - coalesce(lag(frames_sampled_cumulative) OVER (PARTITION BY run ORDER BY "row"), 0) AS sampled
            FROM runs
        )
        SELECT count(DISTINCT s.run) AS scored_runs, count(*) AS clip_runs,
               count(*) FILTER (s.sampled = len(r.frames_sampled)) AS same_sampled_count,
               sum(len(r.frames_sampled)) FILTER (s.run = 0) AS rerun_frames_sampled
        FROM scored s JOIN rerun_clips r USING (video_name)
    """)[0]
    # Each pair of exports, clip by clip, compared exactly (same exporter, so floats must match bit for bit).
    exports = {f'{a}_vs_{b}': query_dicts(con, """
                   SELECT count(*) AS clips, count(*) FILTER (x.frames IS NOT DISTINCT FROM y.frames) AS identical_clips,
                          coalesce(list(x.video_name ORDER BY x.video_name) FILTER (x.frames IS DISTINCT FROM y.frames), []) AS differing_clips
                   FROM exports x JOIN exports y USING (video_name) WHERE x.source = $a AND y.source = $b
               """, {'a': a, 'b': b})[0] for a, b in (('rerun', 'sota'), ('scored', 'sota'), ('rerun', 'scored'))}
    exporter = [r[0] for r in con.execute(
        "SELECT DISTINCT exporter_sha256 FROM exports WHERE source <> 'sota' ORDER BY ALL").fetchall()]
    return dict(objects=objects, clips=clips, patterns=patterns, frames_sampled=sampled, exports=exports,
                exporter_sha256=exporter)


def frame_base(con: duckdb.DuckDBPyConnection) -> dict:
    """Observations against the ground truth on clip frame round(frame x scale) + lag (see the module docstring)."""
    # The grid: every scale at lag 0, every lag at scale 1.
    shifts = sorted({(s, 0) for s in SCALES} | {(1.0, lag) for lag in LAGS})
    con.execute('CREATE OR REPLACE TABLE shifts (scale DOUBLE, lag BIGINT)')
    con.executemany('INSERT INTO shifts VALUES (?, ?)', shifts)
    # Best IoU of each observation with any ground-truth box on the shifted frame (0 when that frame has none).
    rows = query_dicts(con, """
        WITH best AS (
            SELECT o.video, o.id, o.frame, sh.scale, sh.lag,
                   coalesce(max(greatest(0, least(o.x2, g.x2) - greatest(o.x1, g.x1))
                                * greatest(0, least(o.y2, g.y2) - greatest(o.y1, g.y1))
                                / ((o.x2 - o.x1) * (o.y2 - o.y1) + (g.x2 - g.x1) * (g.y2 - g.y1)
                                   - greatest(0, least(o.x2, g.x2) - greatest(o.x1, g.x1))
                                     * greatest(0, least(o.y2, g.y2) - greatest(o.y1, g.y1)))), 0) AS iou
            FROM observations o CROSS JOIN shifts sh
            LEFT JOIN gt_boxes g ON g.video = o.video AND g.frame = CAST(round(o.frame * sh.scale) AS BIGINT) + sh.lag
            GROUP BY ALL
        )
        SELECT scale, lag, count(*) AS observations, avg(iou) AS mean_best_iou,
               avg(CASE WHEN iou >= $match THEN 1 ELSE 0 END) AS matched_share
        FROM best GROUP BY scale, lag ORDER BY scale, lag
    """, {'match': IOU_MATCH})
    # The best shift by mean IoU (ties to the identity, then the smaller shift).
    best = max(rows, key=lambda r: (r['mean_best_iou'], r['scale'] == 1.0 and r['lag'] == 0, -abs(r['lag'])), default=None)
    return dict(shifts=rows, best=best, aligned=best is not None and best['scale'] == 1.0 and best['lag'] == 0)


def classes(con: duckdb.DuckDBPyConnection) -> dict:
    """Objects per class id on both sides, observations per class id, and observations per rerun object (an object
    seen on one sampled frame gives the linear reading nothing to interpolate)."""
    # Class ids are COCO's as floats; show them as integers.
    def counts(sql: str) -> dict:
        return {str(int(r['class_id'])): r['n'] for r in query_dicts(con, sql)}
    # Every rerun object, with 0 observations when it has none.
    per_object = query_dicts(con, """
        WITH n AS (SELECT coalesce(len(observations), 0) AS n FROM rerun_objects)
        SELECT count(*) AS objects, sum(n) AS observations, count(*) FILTER (n = 0) AS objects_unobserved,
               count(*) FILTER (n = 1) AS objects_observed_once, count(*) FILTER (n >= 2) AS objects_observed_twice_or_more,
               avg(n) AS mean_observations, max(n) AS max_observations
        FROM n
    """)[0]
    return dict(rerun_objects=counts('SELECT class_id, count(*) AS n FROM rerun_objects GROUP BY 1 ORDER BY 1'),
                scored_objects=counts('SELECT class_id, count(*) AS n FROM scored_objects GROUP BY 1 ORDER BY 1'),
                observations=counts('SELECT class_id, count(*) AS n FROM observations GROUP BY 1 ORDER BY 1'),
                observations_per_object=per_object)


def timing(con: duckdb.DuckDBPyConnection, launch: dict, gpus_path: Path) -> dict:
    """The rerun's run record, the other GPUs' activity during it, and LEAP's timers against the scored run's (per
    clip mean over the scored runs)."""
    # GNU time's wall clock and peak memory, parsed as the preparation records are.
    seconds, rss = con.execute('SELECT gnu_elapsed_s($t), gnu_number($t, $rss)',
                               {'t': launch['gnu_time'], 'rss': GNU_FIELDS['max_rss_kb']}).fetchone()
    # LEAP's own timers summed over the clips, rerun against scored.
    timers = query_dicts(con, """
        WITH scored AS (
            SELECT video_name, avg(detector) AS detector, avg(reid) AS reid, avg("match") AS "match",
                   avg(decode) AS decode, avg(differencer) AS differencer, avg(inference_total) AS inference_total,
                   count(*) AS runs
            FROM scored_speed GROUP BY video_name
        )
        SELECT sum(r.timings.detector_s) AS rerun_detector_s, sum(s.detector) AS scored_detector_s,
               sum(r.timings.reid_s) AS rerun_reid_s, sum(s.reid) AS scored_reid_s,
               sum(r.timings.match_s) AS rerun_match_s, sum(s."match") AS scored_match_s,
               sum(r.timings.decode_s) AS rerun_decode_s, sum(s.decode) AS scored_decode_s,
               sum(r.timings.differencer_s) AS rerun_differencer_s, sum(s.differencer) AS scored_differencer_s,
               sum(s.inference_total) AS scored_inference_total_s, sum(r.timings.run_session_wall_s) AS rerun_run_session_s,
               min(s.runs) AS scored_runs_min, max(s.runs) AS scored_runs_max
        FROM rerun_clips r JOIN scored s USING (video_name)
    """)[0]
    # The other GPUs during the run (no row when the GPU script's gpus action has not written the file).
    create_gpu_neighbors(con, [('rerun', launch['gpu'], launch['numa_node'], gpus_path)])
    neighbors = query_dicts(con, 'SELECT other_gpus, node_neighbor_busy_share FROM gpu_neighbors')
    return dict(run=launch['run'], host=launch['host'], gpu=launch['gpu'], cpus=launch['cpus'],
                numa_node=launch['numa_node'], image=launch['image'], contaminated=launch['contaminated'],
                foreign_samples=launch['foreign_samples'], exit=launch['exit'], start=launch['start'], end=launch['end'],
                gnu_time_wall_s=seconds, max_rss_kb=None if rss is None else int(rss), leap_timers=timers,
                neighbors=neighbors[0] if neighbors else None)


def write_variants(con: duckdb.DuckDBPyConnection, dataset: str, videos: list[str], export_path: Path, base: Path) -> list[dict]:
    """Write the raw, linear, and pattern readings of the rerun as <base>/<variant>/<video>/tracking.jsonl and return
    one TrackEval task per reading."""
    # The observations of each clip in frame order, as (frame, x1, y1, x2, y2, id) rows.
    rows = con.execute('SELECT video, frame, x1, y1, x2, y2, id FROM observations ORDER BY video, frame, id').fetchall()
    by_video: dict[str, list] = {v: [] for v in videos}
    for video, *row in rows:
        by_video[video].append(row)
    # raw and linear go through the pipeline's own track writer, which interpolates gaps only when asked (a loop
    # over boxes, as the pipeline itself registers them; there is no vectorized form of it).
    for variant, interpolate in (('raw', False), ('linear', True)):
        for video, boxes in by_video.items():
            frame_tracks: dict = {}
            trajectories: dict = {}
            for frame, x1, y1, x2, y2, track_id in boxes:
                register_tracked_detections([(x1, y1, x2, y2, track_id)], frame, frame_tracks, trajectories, interpolate)
            path = base / variant / video / 'tracking.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tracking_results(frame_tracks, path)
    # pattern: the rerun's export of each clip through p140's transform, exactly as the reported tracks were made.
    names = dict(con.execute('SELECT leap_name, video FROM clips').fetchall())
    exports = con.execute("SELECT video_name, CAST(frames AS JSON) FROM read_json(?, format = 'newline_delimited', "
                          f"columns = {EXPORT_COLUMNS})", [str(export_path)]).fetchall()
    for leap_name, frames in exports:
        video = names[leap_name]
        source = base / 'pattern-exports' / f'{leap_name}.json'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(frames)
        transform_tracking_json(str(source), str(base / 'pattern' / video / 'tracking.jsonl'), dataset, video,
                                unwrap_leap=True)
    # One TrackEval task per reading, against the root dataset's ground truth as p141 scores.
    gt_dir = str(Path(output_path(dataset, 'ground truth', '', videos[0])).parents[2])
    return [dict(dataset=dataset, variant=v, videos=sorted(videos), gt_dir=gt_dir,
                 gt_file=str(Path('003_groundtruth', 'tracking.jsonl')), track_dir=str(base / v),
                 track_file='tracking.jsonl', output_dir=str(base / 'trackeval' / v)) for v in HOTA_VARIANTS]


def run_trackeval(task: dict) -> dict:
    """The task with metric_summary of its TrackEval rerun (top-level, so worker processes can run it)."""
    # HOTA (TrackEval adds Count), as p141 runs it.
    metrics = trackeval_rerun(task['gt_dir'], task['gt_file'], task['track_dir'], task['track_file'], task['videos'],
                              task['output_dir'], ('HOTA',))
    return dict(task, metrics=metric_summary(metrics))


def reported_hota(dataset: str) -> tuple[dict, dict]:
    """p141's combined LEAP result (DATASET.json) as metric_summary, and its file record."""
    # The cached combined result of LEAP's one parameter set.
    path = cache.sota('leap', dataset, 'accuracy', 'raw', 'test', LEAP_PARAM, 'DATASET.json')
    assert_files_exist([path], f'{dataset} LEAP DATASET.json')
    with open(path) as f:
        return metric_summary(json.load(f)['metrics']), file_record(path)


def hota_section(reruns: list[dict], reported: dict) -> dict:
    """HOTA, DetA, AssA, and Count of each reading and the reported result, with the pattern reading's difference."""
    # The scalar summary of one metric_summary.
    def scalars(m: dict) -> dict:
        return dict(HOTA=m['HOTA']['HOTA'], DetA=m['HOTA']['DetA'], AssA=m['HOTA']['AssA'], LocA=m['HOTA']['LocA'],
                    dets=m['Count']['Dets'], gt_dets=m['Count']['GT_Dets'], ids=m['Count']['IDs'])
    out = {r['variant']: scalars(r['metrics']) for r in reruns}
    out['reported'] = scalars(reported)
    # The pattern reading rebuilds the reported tracks from the rerun.
    difference = out['pattern']['HOTA'] - out['reported']['HOTA']
    # In millionths, so the 4-decimal rounding of the results file keeps it.
    out['pattern_minus_reported_e6'] = difference * 1e6
    out['pattern_reproduces_reported'] = abs(difference) <= REPRODUCTION_TOLERANCE
    return out


def load_prep(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> dict:
    """Tables prep_steps (one row per timed step, GNU time and GPU samples parsed) and prep_expected (every step the
    GPU script defines, per system, in order, from prep/steps.json), and the step lists."""
    # The step lists the GPU script runs, so a missing record shows as a missing step.
    steps_path = prep_dir / 'steps.json'
    assert_files_exist([steps_path], 'raw/prep/steps.json of measurements-gpu/r1o2_leap_otif_rerun.py (run its prep action)')
    steps = json.loads(steps_path.read_text())
    con.execute('CREATE OR REPLACE TABLE prep_expected (system VARCHAR, step VARCHAR, step_order BIGINT)')
    con.executemany('INSERT INTO prep_expected VALUES (?, ?, ?)', [(system, step, i) for system, names in steps.items() for i, step in enumerate(names)])
    # Evaluation-only steps, left out of the preparation total.
    con.execute('CREATE OR REPLACE TABLE prep_evaluation_only (system VARCHAR, step VARCHAR)')
    con.executemany('INSERT INTO prep_evaluation_only VALUES (?, ?)', [(system, step) for system, names in EVALUATION_ONLY_STEPS.items() for step in names])
    # The records: <system>/<dataset>/<step>.json (comparison files end in _compare.json).
    records = sorted(str(p) for p in prep_dir.glob('*/*/*.json') if not p.name.endswith('_compare.json'))
    assert records, f'no preparation records under {prep_dir}'
    con.execute(f"""
        CREATE OR REPLACE TABLE prep_records AS
        SELECT system, dataset, step, run, gpu, cpus, numa_node, image, start, "end", "exit", samples, foreign_samples,
               contaminated, gnu_time, output_size, compare_error, stdout_tail, filename
        FROM read_json(?, filename = true, union_by_name = true, columns = {{
            'system': 'VARCHAR', 'dataset': 'VARCHAR', 'step': 'VARCHAR', 'run': 'VARCHAR', 'gpu': 'BIGINT',
            'cpus': 'VARCHAR', 'numa_node': 'BIGINT', 'image': 'VARCHAR', 'start': 'VARCHAR', 'end': 'VARCHAR',
            'exit': 'BIGINT', 'samples': 'BIGINT', 'foreign_samples': 'BIGINT', 'contaminated': 'BOOLEAN',
            'gnu_time': 'VARCHAR', 'output_size': 'STRUCT(exists BOOLEAN, files BIGINT, bytes BIGINT)',
            'compare_error': 'VARCHAR', 'stdout_tail': 'VARCHAR'}})
    """, [records])
    # A record's place must match its content, and a step appears once.
    bad = query_dicts(con, """
        SELECT filename FROM prep_records
        WHERE NOT ends_with(filename, '/' || system || '/' || dataset || '/' || step || '.json')
        UNION ALL SELECT system || ' ' || dataset || ' ' || step FROM prep_records r
        WHERE NOT EXISTS (SELECT 1 FROM prep_expected e WHERE e.system = r.system AND e.step = r.step)
    """)
    assert not bad, f'preparation records misplaced or of unknown steps: {bad[:5]}'
    # Each step's GPU samples (the launcher's 1 s samples of its GPU and the host load).
    samples = sorted(str(p) for p in prep_dir.glob('*/*/*_samples.jsonl.gz'))
    con.execute('CREATE OR REPLACE TABLE prep_samples (k STRUCT(system VARCHAR, dataset VARCHAR, step VARCHAR), '
                'util DOUBLE, mem_mib DOUBLE, "foreign" BIGINT, load_1m DOUBLE)')
    if samples:
        con.execute("""
            INSERT INTO prep_samples
            SELECT regexp_extract(filename, '/([^/]+)/([^/]+)/([^/]+)_samples[.]jsonl[.]gz$', ['system', 'dataset', 'step']),
                   util, mem_mib, "foreign", load_1m
            FROM read_json(?, filename = true, format = 'newline_delimited',
                           columns = {'util': 'DOUBLE', 'mem_mib': 'DOUBLE', 'foreign': 'BIGINT', 'load_1m': 'DOUBLE'})
        """, [samples])
    # The other GPUs during each step, keyed system/dataset/step.
    runs = con.execute("SELECT system || '/' || dataset || '/' || step, gpu, numa_node, filename FROM prep_records").fetchall()
    create_gpu_neighbors(con, [(k, g, n, f.removesuffix('.json') + '_gpus.jsonl.gz') for k, g, n, f in runs])
    # One row per step: GNU time's numbers, the samples' summary, and the neighbors.
    con.execute(f"""
        CREATE OR REPLACE TABLE prep_steps AS
        SELECT r.system, r.dataset, r.step, e.step_order, r.run, r.gpu, r.cpus, r.image, r.start, r."end", r."exit",
               r.contaminated, r.foreign_samples, r.output_size, r.compare_error,
               gnu_elapsed_s(r.gnu_time) AS wall_s, gnu_number(r.gnu_time, '{GNU_FIELDS['user_s']}') AS user_s,
               gnu_number(r.gnu_time, '{GNU_FIELDS['system_s']}') AS system_s,
               gnu_number(r.gnu_time, '{GNU_FIELDS['max_rss_kb']}') AS max_rss_kb,
               gnu_number(r.gnu_time, '{GNU_FIELDS['exit_status']}') AS gnu_exit,
               cpu_count(r.cpus) AS cpu_threads,
               s.samples, s.gpu_util_mean, s.gpu_util_busy_share, s.gpu_mem_max_mib, s.load_1m_mean, s.load_1m_max,
               n.node_neighbor_busy_share, n.other_gpus,
               TRY_CAST(nullif(regexp_extract(r.stdout_tail, '{OTIF_TRACKED_FRAMES}', 1), '') AS BIGINT) AS tracked_frames,
               EXISTS (SELECT 1 FROM prep_evaluation_only v WHERE v.system = r.system AND v.step = r.step) AS evaluation_only
        FROM prep_records r JOIN prep_expected e USING (system, step)
        LEFT JOIN gpu_neighbors n ON n.key = r.system || '/' || r.dataset || '/' || r.step
        LEFT JOIN (SELECT k.system, k.dataset, k.step, count(*) AS samples, avg(util) AS gpu_util_mean,
                          avg(CASE WHEN util > 0 THEN 1 ELSE 0 END) AS gpu_util_busy_share, max(mem_mib) AS gpu_mem_max_mib,
                          avg(load_1m) AS load_1m_mean, max(load_1m) AS load_1m_max
                   FROM prep_samples GROUP BY ALL) s USING (system, dataset, step)
    """)
    # The share of the step's CPU set its CPU time kept busy (near 1 means the step was CPU-bound on that set).
    con.execute('CREATE OR REPLACE TABLE prep_steps AS SELECT *, (user_s + system_s) / nullif(wall_s, 0) / cpu_threads '
                'AS cpu_busy_share FROM prep_steps')
    return steps


def prep_totals(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per system and dataset: the steps in order, their total wall clock and CPU time (all succeeded steps, and the
    preparation totals without evaluation-only steps), the share of the CPU set the succeeded steps kept busy,
    whether the total is complete (every step recorded and succeeded) and clean (no foreign GPU process during any
    step), and the most any same-node GPU was busy."""
    return query_dicts(con, """
        WITH grid AS (
            SELECT d.system, d.dataset, e.step, e.step_order,
                   EXISTS (SELECT 1 FROM prep_evaluation_only v WHERE v.system = e.system AND v.step = e.step) AS evaluation_only
            FROM (SELECT DISTINCT system, dataset FROM prep_steps) d JOIN prep_expected e USING (system)
        )
        SELECT g.system, g.dataset,
               coalesce(list(g.step ORDER BY g.step_order) FILTER (s.step IS NULL), []) AS missing_steps,
               coalesce(list(g.step ORDER BY g.step_order) FILTER (s."exit" <> 0 OR s.gnu_exit <> 0), []) AS failed_steps,
               coalesce(list(g.step ORDER BY g.step_order) FILTER (s.contaminated), []) AS contaminated_steps,
               sum(s.wall_s) FILTER (s."exit" = 0) AS wall_s, sum(s.user_s + s.system_s) FILTER (s."exit" = 0) AS cpu_s,
               sum(s.wall_s) FILTER (s."exit" = 0 AND NOT g.evaluation_only) AS preparation_wall_s,
               sum(s.user_s + s.system_s) FILTER (s."exit" = 0 AND NOT g.evaluation_only) AS preparation_cpu_s,
               sum(s.user_s + s.system_s) FILTER (s."exit" = 0) / sum(s.wall_s * s.cpu_threads) FILTER (s."exit" = 0)
                   AS cpu_busy_share,
               coalesce(list(g.step ORDER BY g.step_order) FILTER (g.evaluation_only), []) AS evaluation_only_steps,
               max(s.node_neighbor_busy_share) AS node_neighbor_busy_share_max,
               count(s.step) FILTER (s."exit" = 0) = count(*) AS complete, NOT bool_or(coalesce(s.contaminated, false)) AS clean,
               list(DISTINCT s.cpus) FILTER (s.cpus IS NOT NULL) AS cpu_sets, list(DISTINCT s.gpu) FILTER (s.gpu IS NOT NULL) AS gpus
        FROM grid g LEFT JOIN prep_steps s USING (system, dataset, step)
        GROUP BY g.system, g.dataset ORDER BY g.system, g.dataset
    """)


def leap_label_agreement(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> list[dict]:
    """Per dataset and split: the regenerated pseudo labels against the scored ones (clips, lines, identical lines as
    a multiset per clip, identical files, and lines paired by clip, frame, and track id whose boxes differ by at most
    LABEL_PAIR_PX, since GPU nondeterminism moves boxes by fractions of a pixel) and the scored labeling's duration
    from its files' modification times (first to last file, scaled by n / (n - 1) for the first clip, whose start is
    not recorded)."""
    files = sorted(str(p) for p in prep_dir.glob('leap/*/labels_*_compare.jsonl.gz'))
    if not files:
        return []
    con.execute("""
        CREATE OR REPLACE TABLE label_files AS
        SELECT regexp_extract(filename, '/leap/([^/]+)/labels_([a-z]+)_compare', ['dataset', 'split']) AS k, source, clip,
               mtime, lines
        FROM read_json(?, filename = true, format = 'newline_delimited',
                       columns = {'source': 'VARCHAR', 'clip': 'BIGINT', 'mtime': 'DOUBLE', 'lines': 'VARCHAR[]'})
    """, [files])
    return query_dicts(con, """
        WITH lines AS (
            SELECT k.dataset, k.split, clip, source, line, count(*) AS n
            FROM (SELECT k, clip, source, unnest(lines) AS line FROM label_files) GROUP BY ALL
        ),
        per_line AS (
            SELECT dataset, split, clip, line, sum(n) FILTER (source = 'rerun') AS r, sum(n) FILTER (source = 'scored') AS s
            FROM lines GROUP BY ALL
        ),
        per_file AS (
            SELECT k.dataset, k.split, clip, count(*) FILTER (source = 'rerun') AS r, count(*) FILTER (source = 'scored') AS s,
                   count(DISTINCT lines) = 1 AND count(*) = 2 AS identical
            FROM label_files GROUP BY ALL
        ),
        spans AS (
            SELECT k.dataset, k.split, count(*) AS n, max(mtime) - min(mtime) AS span_s
            FROM label_files WHERE source = 'scored' GROUP BY ALL
        ),
        -- A label line is 'frame,track id,x,y,w,h,confidence,class,...'; key it by clip, frame, and track id.
        keyed AS (
            SELECT dataset, split, clip, source, f[1] AS frame, f[2] AS track, count(*) AS n,
                   any_value([f[3]::DOUBLE, f[4]::DOUBLE, f[5]::DOUBLE, f[6]::DOUBLE]) AS b
            FROM (SELECT k.dataset, k.split, clip, source, string_split(unnest(lines), ',') AS f FROM label_files)
            GROUP BY ALL
        ),
        -- Keys with one line on each side, and the largest coordinate difference of their boxes.
        paired AS (
            SELECT r.dataset, r.split, greatest(abs(r.b[1] - s.b[1]), abs(r.b[2] - s.b[2]), abs(r.b[3] - s.b[3]),
                                                abs(r.b[4] - s.b[4])) AS d
            FROM keyed r JOIN keyed s USING (dataset, split, clip, frame, track)
            WHERE r.source = 'rerun' AND s.source = 'scored' AND r.n = 1 AND s.n = 1
        ),
        pair_stats AS (
            SELECT dataset, split, count(*) AS paired_lines, count(*) FILTER (d <= $px) AS paired_within_px, max(d) AS paired_max_px
            FROM paired GROUP BY ALL
        )
        SELECT f.dataset, f.split, sum(f.r) AS rerun_clips, sum(f.s) AS scored_clips, count(*) FILTER (f.identical) AS identical_clips,
               (SELECT sum(r) FROM per_line l WHERE l.dataset = f.dataset AND l.split = f.split) AS rerun_lines,
               (SELECT sum(s) FROM per_line l WHERE l.dataset = f.dataset AND l.split = f.split) AS scored_lines,
               (SELECT sum(least(coalesce(r, 0), coalesce(s, 0))) FROM per_line l WHERE l.dataset = f.dataset AND l.split = f.split)
                   AS identical_lines,
               coalesce(any_value(ps.paired_lines), 0) AS paired_lines, coalesce(any_value(ps.paired_within_px), 0) AS paired_within_px,
               any_value(ps.paired_max_px) AS paired_max_px,
               any_value(sp.span_s) AS scored_span_s,
               CASE WHEN any_value(sp.n) > 1 THEN any_value(sp.span_s) * any_value(sp.n) / (any_value(sp.n) - 1) END
                   AS scored_duration_estimate_s
        FROM per_file f LEFT JOIN spans sp USING (dataset, split) LEFT JOIN pair_stats ps USING (dataset, split)
        GROUP BY f.dataset, f.split ORDER BY f.dataset, f.split
    """, {'px': LABEL_PAIR_PX})


def leap_crop_agreement(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> list[dict]:
    """Per dataset and crop directory: regenerated and scored re-identification crops and the file names they share
    (names hold the clip, object, class, box, and frame; LEAP picks each object's query crop at random)."""
    files = sorted(str(p) for p in prep_dir.glob('leap/*/reid_data_compare.jsonl.gz'))
    if not files:
        return []
    return query_dicts(con, """
        WITH f AS (
            SELECT regexp_extract(filename, '/leap/([^/]+)/reid_data_compare', 1) AS dataset, source, dir, names
            FROM read_json(?, filename = true, format = 'newline_delimited',
                           columns = {'source': 'VARCHAR', 'dir': 'VARCHAR', 'names': 'VARCHAR[]'})
        )
        SELECT r.dataset, r.dir, len(r.names) AS rerun_crops, len(s.names) AS scored_crops,
               len(list_intersect(r.names, s.names)) AS common_crops
        FROM f r JOIN f s ON r.dataset = s.dataset AND r.dir = s.dir AND r.source = 'rerun' AND s.source = 'scored'
        ORDER BY r.dataset, r.dir
    """, [files])


def leap_reid_training(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> list[dict]:
    """Per dataset: the re-identification run's total training time as fastreid logs it, rerun and scored (the last
    'Total training time' line), the last evaluation's Rank-1 and mAP of each (null when fastreid's evaluation table
    is empty, as in LEAP's logs), and the last logged training row's total loss and classification accuracy."""
    files = sorted(str(p) for p in prep_dir.glob('leap/*/reid_train_compare.json'))
    if not files:
        return []
    con.execute("""
        CREATE OR REPLACE TABLE reid_compare AS
        SELECT regexp_extract(filename, '/leap/([^/]+)/reid_train_compare', 1) AS dataset, unnest([
                   {'source': 'rerun', 'metrics': rerun.metrics, 'time_lines': rerun.time_lines},
                   {'source': 'scored', 'metrics': scored.metrics, 'time_lines': scored.time_lines}]) AS c
        FROM read_json(?, filename = true, columns = {
            'rerun': 'STRUCT(metrics JSON[], time_lines VARCHAR[])', 'scored': 'STRUCT(metrics JSON[], time_lines VARCHAR[])'})
    """, [files])
    return query_dicts(con, f"""
        WITH t AS (
            SELECT dataset, c.source AS source,
                   string_split(regexp_extract(list_filter(c.time_lines, lambda l: regexp_matches(l, '{REID_TOTAL}'))[-1],
                                               '{REID_TOTAL}', 1), ':') AS hms,
                   list_filter(c.metrics, lambda x: json_exists(x, '$."Rank-1"'))[-1] AS last_eval,
                   list_filter(c.metrics, lambda x: json_exists(x, '$.total_loss'))[-1] AS last_train
            FROM reid_compare
        )
        SELECT dataset, source, hms[1]::DOUBLE * 3600 + hms[2]::DOUBLE * 60 + hms[3]::DOUBLE AS total_training_s,
               (last_eval->>'$."Rank-1"')::DOUBLE AS rank1, (last_eval->>'$.mAP')::DOUBLE AS map,
               (last_train->>'$.total_loss')::DOUBLE AS last_total_loss,
               (last_train->>'$.cls_accuracy')::DOUBLE AS last_cls_accuracy
        FROM t ORDER BY dataset, source
    """)


def leap_motion_patterns(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> list[dict]:
    """Per dataset: clusters in the regenerated and the scored motion patterns (k-medoids starts at random, so the
    clusters themselves are not expected to match)."""
    files = sorted(str(p) for p in prep_dir.glob('leap/*/motion_patterns_compare.json'))
    if not files:
        return []
    return query_dicts(con, """
        SELECT regexp_extract(filename, '/leap/([^/]+)/motion_patterns_compare', 1) AS dataset, scored_dir,
               (SELECT sum(json_array_length(v)) FROM (SELECT unnest(json_extract(rerun, '$.*')) AS v)) AS rerun_clusters,
               (SELECT sum(json_array_length(v)) FROM (SELECT unnest(json_extract(scored, '$.*')) AS v)) AS scored_clusters
        FROM read_json(?, filename = true, columns = {'rerun': 'JSON', 'scored': 'JSON', 'scored_dir': 'VARCHAR'})
        ORDER BY dataset
    """, [files])


def otif_curves(con: duckdb.DuckDBPyConnection, prep_dir: Path) -> list[dict]:
    """Per dataset: OTIF's regenerated and scored validation curves (30_speed_accuracy.go's lines), their lengths, the
    best validation accuracy, and the configuration it chose at the end of each curve."""
    files = sorted(str(p) for p in prep_dir.glob('otif/*/tune_valid_compare.json'))
    if not files:
        return []
    fields = ', '.join(f"p[{i + 1}] AS {name}" for i, name in enumerate(CURVE_FIELDS))
    return query_dicts(con, f"""
        WITH lines AS (
            SELECT regexp_extract(filename, '/otif/([^/]+)/tune_valid_compare', 1) AS dataset, unnest([
                       {{'source': 'rerun', 'lines': rerun}}, {{'source': 'scored', 'lines': scored}}]) AS c
            FROM read_json(?, filename = true, columns = {{'rerun': 'VARCHAR[]', 'scored': 'VARCHAR[]'}})
        ),
        rows AS (
            SELECT dataset, c.source AS source, i, {fields}
            FROM (SELECT dataset, c, generate_subscripts(c.lines, 1) AS i, string_split(unnest(c.lines), chr(9)) AS p FROM lines)
        )
        SELECT dataset, source, count(*) AS points, max(valid_accuracy::DOUBLE) AS best_valid_accuracy,
               arg_max(detector || ' ' || segmentation || ' ' || tracker, i) AS last_config,
               arg_max(runtime::DOUBLE, i) AS last_runtime, arg_max(valid_accuracy::DOUBLE, i) AS last_valid_accuracy
        FROM rows GROUP BY dataset, source ORDER BY dataset, source
    """, [files])


def preparation(prep_dir: Path) -> dict:
    """R9.O2's LEAP and OTIF preparation as re-timed on bombe (see the module docstring)."""
    con = duckdb.connect()
    create_macros(con)
    steps = load_prep(con, prep_dir)
    out = dict(steps_defined=steps,
               steps=query_dicts(con, 'SELECT * EXCLUDE (step_order) FROM prep_steps ORDER BY system, dataset, step_order'),
               totals=prep_totals(con), leap_labels=leap_label_agreement(con, prep_dir),
               leap_crops=leap_crop_agreement(con, prep_dir), leap_reid_training=leap_reid_training(con, prep_dir),
               leap_motion_patterns=leap_motion_patterns(con, prep_dir), otif_curves=otif_curves(con, prep_dir))
    con.close()
    return out


def results_name(datasets: list[str]) -> str:
    """results.json for a run over every dataset, results-partial.json otherwise, so a partial run never replaces it."""
    # Every dataset selected (in any order) is a full run.
    return 'results.json' if set(datasets) == set(DATASET_LABELS) else 'results-partial.json'


def main():
    # Command-line options and the output directory.
    args = parse_args()
    out_dir = args.output_dir or results_dir(__file__)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The GPU script's manifest (inputs, images, env provenance).
    manifest_path = args.raw_dir / 'manifest.json'
    assert_files_exist([manifest_path], 'raw/manifest.json of measurements-gpu/r1o2_leap_otif_rerun.py (run setup)')
    manifest = json.loads(manifest_path.read_text())
    sections, tasks, records = {}, [], [file_record(manifest_path)]
    with tempfile.TemporaryDirectory(prefix='r1o2_leap_otif_rerun_') as tmp:
        con = duckdb.connect()
        create_macros(con)
        for dataset in args.datasets:
            # The dataset's files, tables, and sections.
            label = DATASET_LABELS[dataset]
            paths = raw_paths(args.raw_dir, dataset)
            records += [file_record(p) for p in paths.values()]
            videos = load_dataset(con, dataset, paths, args.sota_dir)
            launch = json.loads(paths['launch'].read_text())
            reported, reported_record = reported_hota(dataset)
            records.append(reported_record)
            sections[label] = dict(dataset=dataset, clips=len(videos),
                                   video_identity=[r for r in manifest['video_identity'] if r['dataset'] == dataset],
                                   agreement=agreement(con), frame_base=frame_base(con), classes=classes(con),
                                   timing=timing(con, launch, args.raw_dir / dataset / 'leap_rerun_gpus.jsonl.gz'),
                                   reported=reported)
            # The readings to score, written under the temporary directory.
            tasks += write_variants(con, dataset, videos, paths['rerun_export'], Path(tmp) / dataset)
            print(f"{label}: {sections[label]['agreement']['objects']['identical']} of "
                  f"{sections[label]['agreement']['objects']['scored_objects']} scored objects identical; frame base "
                  f"{'aligned' if sections[label]['frame_base']['aligned'] else 'NOT aligned'}", flush=True)
        con.close()
        # Every TrackEval rerun in parallel; each is independent.
        with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
            reruns = pool.map(run_trackeval, tasks)
    # Attach the readings to their datasets.
    for label, section in sections.items():
        section['hota'] = hota_section([r for r in reruns if r['dataset'] == section['dataset']], section.pop('reported'))
        h = section['hota']
        print(f"{label}: HOTA raw {h['raw']['HOTA']:.4f}, linear {h['linear']['HOTA']:.4f}, pattern "
              f"{h['pattern']['HOTA']:.4f}, reported {h['reported']['HOTA']:.4f}", flush=True)
    # R9.O2's preparation timing, when the GPU script has recorded any.
    prep_dir = args.raw_dir / 'prep'
    prep = preparation(prep_dir) if (prep_dir / 'steps.json').exists() else None
    if prep is None:
        print("no raw/prep/steps.json: preparation section skipped (run the GPU script's prep action)", flush=True)
    else:
        for t in prep['totals']:
            print(f"preparation {t['system']} {t['dataset']}: {t['wall_s'] or 0:.0f} s wall, complete {t['complete']}, "
                  f"clean {t['clean']}, missing {t['missing_steps']}, failed {t['failed_steps']}", flush=True)
    # Independent rerun queues share an output directory; keep each selected dataset set's summary separate.
    name = results_name(args.datasets) if args.output_dir is None else f"results-{'-'.join(sorted(args.datasets))}.json"
    write_json(out_dir, name, rounded(dict(
        method=dict(scales=SCALES, lags=LAGS, iou_match=IOU_MATCH, hota_variants=HOTA_VARIANTS, leap_param=LEAP_PARAM),
        environment={k: manifest[k] for k in ('images', 'dockerfile', 'scored_container', 'label_aliases')},
        inputs=records, datasets=sections, preparation=prep)))


if __name__ == '__main__':
    main()
