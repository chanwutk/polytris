#!/usr/local/bin/python
"""R1.O1: TrackEval's scoring window, and HOTA re-scored on full clips, for every system on every dataset.

Mechanism (read on the vendored TrackEval, reproduced by tests/test_measurements_r1o1_hota_scoring_window.py)
  polyis/trackeval/dataset.py (_load_raw_file) gives each file its own number of time steps, its largest
  frame_idx + 1 (lines 58-67), and builds that file's per-frame lists only that far (line 76).
  TrackEval's _base_dataset.get_raw_seq_data keeps the ground truth's num_timesteps (line 94) and computes
  similarities over zip(gt_dets, tracker_dets) (line 98); HOTA (hota.py lines 53 and 72), CLEAR (clear.py 68), and
  Identity (identity.py 53) iterate zip(gt_ids, tracker_ids). zip stops at the shorter list, so a clip is scored on
  its first min(T_gt, T_tracker) frames: ground-truth boxes after the tracker file's last row are not misses, and
  tracker boxes after the ground truth's last row are not false positives. Count, and the dataset's num_gt_dets and
  num_tracker_dets (dataset.py lines 158-159), count every box. When one side of a clip has no box at all, the
  metrics' early exits count the other side in full (all misses or all false positives), so that clip loses nothing.
  save_tracking_results (polyis/utilities.py) writes rows from the first to the last frame with a track, so every
  file (ground truth included) ends at its own last row. p110 (reference, Tetris) and p141 (OTIF, LEAP) both score
  with this dataset class and the HOTA metric only.

What this script measures, per dataset and test-split configuration of every system (common.system_configs), with
every tracking.jsonl and the ground truth (common.output_path(dataset, 'ground truth', ...), as p110 and p141 score
against) read at once by DuckDB:
  window      ground-truth boxes after the tracker file's last row in clips where both sides have boxes
              (gt_boxes_unscored), their share of all ground-truth boxes, the ground-truth rows and box frames after
              that row; tracker boxes after the ground truth's last row (tracker_boxes_unscored), their share of all
              tracker boxes, and the frames; clips whose tracker file has no box (scored in full as misses), and clips
              whose ground truth has no box (its tracker boxes scored in full as false positives). Checked to the box
              against every configuration's cached DATASET.json (Count GT_Dets - HOTA TP - FN and Dets - TP - FP at
              alpha 0.05, common.unscored_boxes), whose HOTA must also match the tradeoff table.
  padded      HOTA, DetA, AssA (means over TrackEval's 19 alphas, as p130 and p142 compute HOTA_HOTA) re-scored by
              common.trackeval_rerun (TrackEval with polyis.trackeval.dataset.Dataset, configured as p110 and p141, HOTA
              only) after padding both sides to the clip length n: a copy of each file with an empty row on frame
              n - 1 where the file ends earlier, written under a temporary directory and removed after the run. Every
              configuration is re-scored. Padded runs must score every box (TP + FN = GT_Dets, TP + FP = Dets), and a
              configuration without unscored boxes must keep its cached HOTA exactly.
  clip length n = get_video_frame_count (the frames every system decodes, as p140 uses it), or the ground truth's
              length where that is longer (clips_lengthened). Three B3D3 ground-truth files hold a row on frame
              frame_count, past every system's last decoded frame; clip_alignment compares those clips' ground truth
              with the reference's boxes one frame either way. Their boxes stay in the padded ground truth as misses
              for every system (window gt_boxes_past_frame_count; gt_unscored_share_within_frame_count leaves them out).
  reproduction  TrackEval on the unmodified cached files for the cached operating points (both rules) and the
              reference, which must reproduce DATASET.json, so the padded change is the padding's alone.
  operating_points  Both selection rules on the cached HOTA and on the padded HOTA (bounds from the padded reference),
              bounds named by common.bound_specs (each loss of the reference's HOTA, and 'leap_fit', LEAP's best HOTA):
              shared_bound, common.operating_points (Tetris, OTIF, and LEAP, each system's fastest configuration with
              HOTA at least the bound), and accuracy_matched, p203's rule (common.accuracy_matched, the rule the
              abstract's prior-system speedups quote): Tetris's shared-bound point against the fastest prior point
              meeting the bound if it is not more accurate, otherwise the most accurate Pareto prior point not above
              Tetris's HOTA. changes lists every choice, qualification, or speedup that differs under either rule, a
              shared-bound change with how far each changed choice is from the unrounded bound; speedup_ranges gives
              Tetris's speedup ranges over the datasets under both scorings.
  candidates  Tetris's candidates are the test runs of its validated Pareto set (common.system_configs), selected by
              p135 on unpadded validation HOTA, which the same window affects. This script re-ranks that fixed set; it
              does not re-derive the Pareto selection, which would need every validation configuration re-scored padded
              (and CalDoT2's validation cache cannot re-derive it, REVISION finding 5).
B3D (jnc*) is scored on automatic ground truth from the same detector family as Tetris and the reference, so its
numbers are provisional until R1.O1's human-verified labels replace it.

Writes measurements-results/r1o1_hota_scoring_window/results.json (method, summary, operating points, changes) and
configs.json (every configuration's window counts and cached and padded metrics), or results-partial.json and
configs-partial.json when --datasets selects only some datasets, so a partial run never replaces the full results.
"""
import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import duckdb

from evaluation.manifests import list_split_videos
from measurements.common import (DATASET_LABELS, LOOSEST_ALPHA, PRIOR_SYSTEMS, TRACK_COLUMNS, accuracy_matched,
                                 arrange_matched, assert_files_exist, bound_specs, choose, load_test_tradeoff,
                                 matched_stats, metric_summary, operating_points, output_path, query_dicts, results_dir,
                                 rounded, system_configs, trackeval_rerun, unscored_boxes, write_json)
from polyis.io import cache
from polyis.utilities import get_video_frame_count


# Systems compared at the operating points (the reference sets the bound).
SYSTEMS = ['tetris', *PRIOR_SYSTEMS]
# Schema of a tracking.jsonl line with each box kept as JSON text (only the box count is needed).
LINE_COLUMNS = "{'frame_idx': 'BIGINT', 'tracks': 'JSON[]'}"
# File name of a padded copy inside its clip's directory.
PADDED_FILE = 'tracking.jsonl'
# Largest difference allowed between a TrackEval rerun and DATASET.json, and between readings that must be equal.
TOLERANCE = 1e-9
# Ground truth behind each dataset label.
GROUND_TRUTH = {label: ('automatic: detector-generated 003_groundtruth, provisional until R1.O1'
                        if dataset.startswith('jnc') else '003_groundtruth of the root dataset')
                for dataset, label in DATASET_LABELS.items()}
# Fields of a configuration's padded and cached metrics whose change is reported.
DELTA_FIELDS = ('HOTA', 'DetA', 'AssA')


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O1: TrackEval's scoring window and HOTA re-scored on full clips")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--losses', nargs='+', type=float, default=[0.05, 0.10])
    parser.add_argument('--workers', type=int, default=24)
    return parser.parse_args()


def accuracy_path(dataset: str, system: str, config: str) -> Path:
    """The cached DATASET.json a configuration's HOTA comes from: p110's for the reference and Tetris, p141's for
    OTIF and LEAP (config 'param N')."""
    # p110 names its directory by variant id; the reference's is 'naive'.
    if system in ('reference', 'tetris'):
        return Path(cache.eval(dataset, 'acc', 'raw', 'test', config, 'DATASET.json'))
    # p141 names its directory by the three-digit parameter id.
    assert system in PRIOR_SYSTEMS, f'unknown system {system!r}'
    return Path(cache.sota(system, dataset, 'accuracy', 'raw', 'test', f"{int(config.removeprefix('param ')):03d}",
                           'DATASET.json'))


def split_path(path: Path, video: str) -> tuple[str, str]:
    """(directory, file relative to the clip directory) of a clip's file, as TrackEval's input_*_dir and input_*."""
    # Every cached file sits at <directory>/<video>/<relative path>.
    head, sep, tail = str(path).partition(f'/{video}/')
    assert sep and tail, f'{path} is not under a {video} directory'
    return head, tail


def register_inputs(con: duckdb.DuckDBPyConnection, datasets: list[str]):
    """Tables clips(dataset, label, video, frame_count), configs(dataset, label, system, config, fps, hota),
    gt_files(dataset, video, path), track_files(dataset, system, config, video, path), and accuracy_files(dataset,
    system, config, path).

    Every file must exist, and each dataset's frame counts must add up to the tradeoff table's test frame count.
    """
    # Replace any tables from a previous call.
    con.execute('CREATE OR REPLACE TABLE clips (dataset VARCHAR, label VARCHAR, video VARCHAR, frame_count BIGINT)')
    con.execute('CREATE OR REPLACE TABLE configs (dataset VARCHAR, label VARCHAR, system VARCHAR, config VARCHAR, '
                'fps DOUBLE, hota DOUBLE)')
    con.execute('CREATE OR REPLACE TABLE gt_files (dataset VARCHAR, video VARCHAR, path VARCHAR)')
    con.execute('CREATE OR REPLACE TABLE track_files (dataset VARCHAR, system VARCHAR, config VARCHAR, video VARCHAR, '
                'path VARCHAR)')
    con.execute('CREATE OR REPLACE TABLE accuracy_files (dataset VARCHAR, system VARCHAR, config VARCHAR, path VARCHAR)')
    # One dataset at a time (seven, not data rows): its clips, configurations, and their files.
    for dataset in datasets:
        # The test clips p110 and p141 score, and each clip's frame count as p140 reads it.
        videos = sorted(list_split_videos(dataset, 'test'))
        assert videos, f'{dataset}: no test videos'
        clips = [(dataset, DATASET_LABELS[dataset], v, int(get_video_frame_count(dataset, v))) for v in videos]
        # The clips must be the frames prior systems' throughput is divided by.
        frames = float(load_test_tradeoff(dataset)[1]['frame_count'])
        assert sum(n for *_, n in clips) == frames, f'{dataset}: clips add up to {sum(n for *_, n in clips)}, not {frames}'
        # Every system's test configurations, with their tradeoff-table throughput and HOTA.
        rows = system_configs(dataset)
        con.executemany('INSERT INTO clips VALUES (?, ?, ?, ?)', clips)
        con.executemany('INSERT INTO configs VALUES (?, ?, ?, ?, ?, ?)',
                        [(dataset, DATASET_LABELS[dataset], r['system'], r['config'], r['fps'], r['hota']) for r in rows])
        # The files each configuration is scored from, and the ground truth they are scored against.
        gt = [(dataset, v, str(output_path(dataset, 'ground truth', '003_groundtruth', v))) for v in videos]
        tracks = [(dataset, r['system'], r['config'], v, str(output_path(dataset, r['system'], r['config'], v)))
                  for r in rows for v in videos]
        accuracy = [(dataset, r['system'], r['config'], str(accuracy_path(dataset, r['system'], r['config'])))
                    for r in rows]
        # A missing file would silently drop a clip or a configuration.
        assert_files_exist([p for *_, p in gt + tracks + accuracy], f'{dataset} inputs')
        con.executemany('INSERT INTO gt_files VALUES (?, ?, ?)', gt)
        con.executemany('INSERT INTO track_files VALUES (?, ?, ?, ?, ?)', tracks)
        con.executemany('INSERT INTO accuracy_files VALUES (?, ?, ?, ?)', accuracy)
    # Configuration names identify rows of the report, so they must be unique per dataset and system.
    dup = query_dicts(con, 'SELECT dataset, system, config, count(*) AS n FROM configs GROUP BY ALL HAVING n > 1')
    assert not dup, f'duplicate configurations: {dup}'


def load_rows(con: duckdb.DuckDBPyConnection):
    """Tables rows(path, frame, boxes) of every ground-truth and tracking file, read at once; file_stats(path, lines,
    t, boxes) with t the file's length as TrackEval reads it (largest frame_idx + 1, 0 for an empty file); and
    clip_lengths(dataset, label, video, frame_count, t_gt, n, gt_boxes_past_frame_count).

    A clip's length n is its frame count (the frames every system decodes) or, where the ground truth holds a row past
    it, the ground truth's length, so no ground-truth box is dropped; such ground truth comes from a decode that is not
    the systems' (clip_alignment), and its boxes past the frame count are unreachable for every system. Scores do not
    depend on how far past both files a clip is padded, since those frames are empty on both sides. Repeated, negative,
    or missing frames and null track lists, which TrackEval's loader rejects or mis-reads, raise.
    """
    # Every distinct file once (the ground truth is shared by every configuration of its dataset).
    paths = [r[0] for r in con.execute('SELECT path FROM gt_files UNION SELECT path FROM track_files ORDER BY 1').fetchall()]
    # Each line's frame and box count; an empty file contributes no line.
    con.execute(f"""
        CREATE OR REPLACE TABLE rows AS
        SELECT filename AS path, frame_idx AS frame, len(tracks) AS boxes
        FROM read_json(?, filename = true, format = 'newline_delimited', columns = {LINE_COLUMNS})
    """, [paths])
    # TrackEval's loader asserts one line per frame and indexes its lists by frame.
    bad = query_dicts(con, """
        SELECT path, frame, count(*) AS n FROM rows GROUP BY ALL
        HAVING n > 1 OR frame IS NULL OR frame < 0 OR bool_or(boxes IS NULL) LIMIT 5
    """)
    assert not bad, f'tracking lines repeated, negative, or without a track list: {bad}'
    # Each file's lines, length, and boxes, also for a file without any line.
    con.execute("""
        CREATE OR REPLACE TABLE file_stats AS
        SELECT p.path, count(r.frame) AS lines, coalesce(max(r.frame) + 1, 0) AS t, coalesce(sum(r.boxes), 0) AS boxes
        FROM (SELECT unnest(?) AS path) p LEFT JOIN rows r USING (path) GROUP BY p.path
    """, [paths])
    # Each clip's length: its frame count, or the ground truth's length where that is longer.
    con.execute("""
        CREATE OR REPLACE TABLE clip_lengths AS
        WITH past AS (
            -- Ground-truth boxes at or past the frame count.
            SELECT g.dataset, g.video, sum(r.boxes) AS boxes
            FROM gt_files g JOIN clips c USING (dataset, video) JOIN rows r ON r.path = g.path
            WHERE r.frame >= c.frame_count GROUP BY ALL
        )
        SELECT c.dataset, c.label, c.video, c.frame_count, s.t AS t_gt, greatest(c.frame_count, s.t) AS n,
               coalesce(p.boxes, 0) AS gt_boxes_past_frame_count
        FROM clips c JOIN gt_files g USING (dataset, video) JOIN file_stats s ON s.path = g.path
        LEFT JOIN past p USING (dataset, video)
    """)


def track_overruns(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Tracking files longer than their clip (load_rows' n), which padding cannot bring to the clip's length."""
    # Every such file, with both lengths.
    return query_dicts(con, """
        SELECT t.dataset, t.system, t.config, t.video, l.n, s.t FROM track_files t
        JOIN clip_lengths l USING (dataset, video) JOIN file_stats s ON s.path = t.path
        WHERE s.t > l.n ORDER BY ALL
    """)


def long_clips(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Clips whose ground truth runs past their frame count, with the boxes there."""
    # Every such clip, in dataset and clip order.
    return query_dicts(con, """
        SELECT label AS dataset, video, frame_count, t_gt, gt_boxes_past_frame_count FROM clip_lengths
        WHERE n > frame_count ORDER BY label, video
    """)


def clip_alignment(con: duckdb.DuckDBPyConnection) -> dict:
    """{dataset label: {lengthened, other_clips}} for every dataset with a lengthened clip (long_clips): how the ground
    truth lines up with the frames the systems decode, measured against the reference's boxes.

    Per clip, iou_prev, iou_same, and iou_next are the mean, over the ground truth's boxes on frames
    1 <= f <= frame_count - 2, of each box's best IoU with the reference's boxes on frame f - 1, f, and f + 1 (0 when
    none overlaps). A ground truth aligned with the decoded frames peaks at f with f - 1 and f + 1 about equal.
    lengthened lists each lengthened clip with its lengths and the reference file's (reference_t); other_clips gives the
    medians of the three means over the dataset's other clips.
    """
    # The datasets with a lengthened clip, and each of their clips' ground-truth and reference files.
    con.execute("""
        CREATE OR REPLACE TABLE alignment_clips AS
        SELECT l.dataset, l.label, l.video, l.frame_count, l.t_gt, l.n > l.frame_count AS lengthened,
               g.path AS gt_path, t.path AS reference_path, s.t AS reference_t
        FROM clip_lengths l JOIN gt_files g USING (dataset, video)
        JOIN track_files t ON t.dataset = l.dataset AND t.video = l.video AND t.system = 'reference'
        JOIN file_stats s ON s.path = t.path
        WHERE l.dataset IN (SELECT dataset FROM clip_lengths WHERE n > frame_count)
    """)
    paths = [r[0] for r in con.execute('SELECT gt_path FROM alignment_clips UNION SELECT reference_path FROM '
                                       'alignment_clips ORDER BY 1').fetchall()]
    # Nothing to compare when no clip is lengthened.
    if not paths:
        return {}
    # Every box of those files, each ground-truth box's best IoU per offset, and the means per clip.
    con.execute(f"""
        CREATE OR REPLACE TABLE alignment AS
        WITH boxes AS (
            SELECT filename AS path, frame_idx AS frame, unnest(tracks) AS b
            FROM read_json($paths, filename = true, format = 'newline_delimited', columns = {TRACK_COLUMNS})
        ),
        gt AS (
            -- Ground-truth boxes on frames whose neighbours are both decoded frames.
            SELECT c.dataset, c.video, x.frame, x.b FROM boxes x JOIN alignment_clips c ON x.path = c.gt_path
            WHERE x.frame BETWEEN 1 AND c.frame_count - 2
        ),
        ref AS (SELECT c.dataset, c.video, x.frame, x.b FROM boxes x JOIN alignment_clips c ON x.path = c.reference_path),
        best AS (
            -- Boxes are [track_id, x1, y1, x2, y2]; IoU is the intersection over the union of the two areas.
            SELECT g.dataset, g.video, o.s, g.frame, g.b,
                   coalesce(max(greatest(0, least(g.b[4], r.b[4]) - greatest(g.b[2], r.b[2]))
                                * greatest(0, least(g.b[5], r.b[5]) - greatest(g.b[3], r.b[3]))
                                / ((g.b[4] - g.b[2]) * (g.b[5] - g.b[3]) + (r.b[4] - r.b[2]) * (r.b[5] - r.b[3])
                                   - greatest(0, least(g.b[4], r.b[4]) - greatest(g.b[2], r.b[2]))
                                     * greatest(0, least(g.b[5], r.b[5]) - greatest(g.b[3], r.b[3])))), 0) AS iou
            FROM gt g CROSS JOIN (SELECT unnest([-1, 0, 1]) AS s) o
            LEFT JOIN ref r ON r.dataset = g.dataset AND r.video = g.video AND r.frame = g.frame + o.s
            GROUP BY ALL
        ),
        means AS (
            SELECT dataset, video, avg(iou) FILTER (s = -1) AS iou_prev, avg(iou) FILTER (s = 0) AS iou_same,
                   avg(iou) FILTER (s = 1) AS iou_next
            FROM best GROUP BY ALL
        )
        SELECT c.label, c.video, c.lengthened, c.frame_count, c.t_gt, c.reference_t, m.iou_prev, m.iou_same, m.iou_next
        FROM alignment_clips c LEFT JOIN means m USING (dataset, video)
    """, {'paths': paths})
    # The lengthened clips, and the medians over each dataset's other clips for comparison.
    lengthened = query_dicts(con, 'SELECT * EXCLUDE (lengthened) FROM alignment WHERE lengthened ORDER BY label, video')
    others = query_dicts(con, """
        SELECT label, count(*) AS clips, median(iou_prev) AS iou_prev, median(iou_same) AS iou_same,
               median(iou_next) AS iou_next
        FROM alignment WHERE NOT lengthened GROUP BY label ORDER BY label
    """)
    # Arranged per dataset (one or a few, not data rows); a dataset of only lengthened clips has no comparison.
    out = {r['label']: dict(lengthened=[], other_clips=None) for r in lengthened}
    for r in others:
        out[r.pop('label')]['other_clips'] = r
    for r in lengthened:
        out[r.pop('label')]['lengthened'].append(r)
    return out


def scoring_window(con: duckdb.DuckDBPyConnection):
    """Tables clip_windows (one row per configuration and clip) and windows (per configuration): what TrackEval's
    zip leaves out.

    A clip is truncated when both its files have boxes; TrackEval then scores frames < min(t_gt, t_tracker), so
    ground-truth boxes at frames >= t_tracker and tracker boxes at frames >= t_gt are not scored. Rows unscored are
    the frames between the two lengths (gt side: t_gt - t_tracker, which may hold no box); box frames are those of
    them with at least one box. Ground-truth boxes on frames >= frame_count (past every system's decoded frames) are
    also counted apart, in total and among the unscored ones.
    """
    # Per configuration and clip: both lengths and box counts, and the boxes and frames past the other side's end.
    con.execute("""
        CREATE OR REPLACE TABLE clip_windows AS
        WITH pairs AS (
            -- Each configuration's clip with its file and the ground truth's.
            SELECT t.dataset, t.system, t.config, t.video, c.n, c.frame_count, c.gt_boxes_past_frame_count,
                   g.path AS gt_path, t.path AS tracker_path,
                   gs.t AS t_gt, gs.boxes AS gt_boxes, ts.t AS t_tracker, ts.boxes AS tracker_boxes,
                   gs.boxes > 0 AND ts.boxes > 0 AS truncated
            FROM track_files t JOIN clip_lengths c USING (dataset, video) JOIN gt_files g USING (dataset, video)
            JOIN file_stats gs ON gs.path = g.path JOIN file_stats ts ON ts.path = t.path
        ),
        gt_after AS (
            -- Ground-truth boxes on frames past the tracker file's last row, in truncated clips, and those of them
            -- past the clip's decoded frames.
            SELECT p.dataset, p.system, p.config, p.video, sum(r.boxes) AS gt_boxes_unscored,
                   count(*) FILTER (r.boxes > 0) AS gt_box_frames_unscored,
                   coalesce(sum(r.boxes) FILTER (r.frame >= p.frame_count), 0) AS gt_boxes_unscored_past_frame_count
            FROM pairs p JOIN rows r ON r.path = p.gt_path
            WHERE p.truncated AND r.frame >= p.t_tracker GROUP BY ALL
        ),
        tracker_after AS (
            -- Tracker boxes on frames past the ground truth's last row, in truncated clips.
            SELECT p.dataset, p.system, p.config, p.video, sum(r.boxes) AS tracker_boxes_unscored,
                   count(*) FILTER (r.boxes > 0) AS tracker_box_frames_unscored
            FROM pairs p JOIN rows r ON r.path = p.tracker_path
            WHERE p.truncated AND r.frame >= p.t_gt GROUP BY ALL
        )
        SELECT p.dataset, p.system, p.config, p.video, p.n, p.frame_count, p.t_gt, p.t_tracker, p.gt_boxes,
               p.tracker_boxes, p.truncated, p.gt_boxes_past_frame_count,
               coalesce(g.gt_boxes_unscored, 0) AS gt_boxes_unscored,
               coalesce(g.gt_box_frames_unscored, 0) AS gt_box_frames_unscored,
               coalesce(g.gt_boxes_unscored_past_frame_count, 0) AS gt_boxes_unscored_past_frame_count,
               CASE WHEN p.truncated THEN greatest(p.t_gt - p.t_tracker, 0) ELSE 0 END AS gt_rows_unscored,
               coalesce(a.tracker_boxes_unscored, 0) AS tracker_boxes_unscored,
               coalesce(a.tracker_box_frames_unscored, 0) AS tracker_box_frames_unscored,
               CASE WHEN p.truncated THEN greatest(p.t_tracker - p.t_gt, 0) ELSE 0 END AS tracker_rows_unscored
        FROM pairs p LEFT JOIN gt_after g USING (dataset, system, config, video)
        LEFT JOIN tracker_after a USING (dataset, system, config, video)
    """)
    # Per configuration: totals over the clips, the shares, and how the files end against the clips.
    con.execute("""
        CREATE OR REPLACE TABLE windows AS
        SELECT dataset, system, config, count(*) AS clips, sum(n) AS clip_frames,
               sum(gt_boxes) AS gt_boxes, sum(gt_boxes_unscored) AS gt_boxes_unscored,
               sum(gt_boxes_unscored) / nullif(sum(gt_boxes), 0) AS gt_unscored_share,
               sum(gt_boxes_past_frame_count) AS gt_boxes_past_frame_count,
               sum(gt_boxes_unscored_past_frame_count) AS gt_boxes_unscored_past_frame_count,
               (sum(gt_boxes_unscored) - sum(gt_boxes_unscored_past_frame_count)) / nullif(sum(gt_boxes), 0)
                   AS gt_unscored_share_within_frame_count,
               sum(gt_rows_unscored) AS gt_rows_unscored, sum(gt_box_frames_unscored) AS gt_box_frames_unscored,
               sum(tracker_boxes) AS tracker_boxes, sum(tracker_boxes_unscored) AS tracker_boxes_unscored,
               sum(tracker_boxes_unscored) / nullif(sum(tracker_boxes), 0) AS tracker_unscored_share,
               sum(tracker_rows_unscored) AS tracker_rows_unscored,
               sum(tracker_box_frames_unscored) AS tracker_box_frames_unscored,
               count(*) FILTER (gt_boxes_unscored > 0) AS clips_with_gt_unscored,
               count(*) FILTER (tracker_boxes_unscored > 0) AS clips_with_tracker_unscored,
               count(*) FILTER (tracker_boxes = 0) AS clips_without_tracker_boxes,
               count(*) FILTER (gt_boxes = 0) AS clips_without_gt_boxes,
               count(*) FILTER (t_tracker < n) AS tracker_files_ending_before_clip_end,
               count(*) FILTER (t_gt < n) AS gt_files_ending_before_clip_end,
               sum(n - least(t_tracker, n)) AS tracker_frames_short_of_clip_end
        FROM clip_windows GROUP BY ALL
    """)


def load_cached(con: duckdb.DuckDBPyConnection):
    """Table cached(dataset, system, config, HOTA, DetA, AssA, gt_dets, dets, tp, fn, fp) from every configuration's
    DATASET.json at once: HOTA fields as means over the alphas, counts at the loosest alpha (0.05)."""
    # Every cached combined result, attached to its configuration.
    paths = [r[0] for r in con.execute('SELECT path FROM accuracy_files ORDER BY path').fetchall()]
    means = ', '.join(f"list_avg(CAST(json_extract(o.json, '$.metrics.HOTA.{f}') AS DOUBLE[])) AS \"{f}\""
                      for f in DELTA_FIELDS)
    con.execute(f"""
        CREATE OR REPLACE TABLE cached AS
        SELECT a.dataset, a.system, a.config, {means},
               CAST(json_extract(o.json, '$.metrics.Count.GT_Dets') AS BIGINT) AS gt_dets,
               CAST(json_extract(o.json, '$.metrics.Count.Dets') AS BIGINT) AS dets,
               CAST(json_extract(o.json, '$.metrics.HOTA.HOTA_TP[{LOOSEST_ALPHA}]') AS DOUBLE) AS tp,
               CAST(json_extract(o.json, '$.metrics.HOTA.HOTA_FN[{LOOSEST_ALPHA}]') AS DOUBLE) AS fn,
               CAST(json_extract(o.json, '$.metrics.HOTA.HOTA_FP[{LOOSEST_ALPHA}]') AS DOUBLE) AS fp
        FROM read_json_objects(?, filename = true) o JOIN accuracy_files a ON a.path = o.filename
    """, [paths])


def cached_mismatches(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Configurations whose DATASET.json disagrees with the tradeoff table's HOTA or with the scoring window: Count
    must hold every box, and GT_Dets - TP - FN and Dets - TP - FP at alpha 0.05 must be exactly the unscored boxes."""
    # Every disagreement, also a configuration whose DATASET.json is missing or repeated.
    return query_dicts(con, """
        WITH joined AS (
            SELECT c.dataset, c.system, c.config, c.hota AS tradeoff_hota, k."HOTA" AS cached_hota,
                   w.gt_boxes, k.gt_dets, w.tracker_boxes, k.dets,
                   w.gt_boxes_unscored, k.gt_dets - round(k.tp + k.fn) AS cached_gt_unscored,
                   w.tracker_boxes_unscored, k.dets - round(k.tp + k.fp) AS cached_tracker_unscored,
                   count(k.config) OVER (PARTITION BY c.dataset, c.system, c.config) AS records
            FROM configs c LEFT JOIN cached k USING (dataset, system, config) LEFT JOIN windows w USING (dataset, system, config)
        )
        SELECT * FROM joined
        WHERE records <> 1 OR abs(tradeoff_hota - cached_hota) > $tol OR cached_hota IS NULL
           OR gt_boxes IS DISTINCT FROM gt_dets OR tracker_boxes IS DISTINCT FROM dets
           OR gt_boxes_unscored IS DISTINCT FROM cached_gt_unscored
           OR tracker_boxes_unscored IS DISTINCT FROM cached_tracker_unscored
        ORDER BY dataset, system, config
    """, {'tol': TOLERANCE})


def write_padded(source: str, target: Path, n: int, t: int):
    """Copy a tracking.jsonl to `target` with an empty row on frame n - 1 when the file ends earlier (t < n), so
    TrackEval reads it as n frames long. A file past its clip (t > n) cannot be padded to it and raises."""
    # Padding cannot shorten a file; such files are reported by the window and must be handled before scoring.
    assert t <= n, f'{source} runs to frame {t - 1}, past its clip of {n} frames'
    # The copy's clip directory.
    target.parent.mkdir(parents=True, exist_ok=True)
    data = Path(source).read_bytes()
    # A last line without its newline would merge with the padding row.
    if data and not data.endswith(b'\n'):
        data += b'\n'
    # The padding row, only where the file ends before the clip's last frame.
    if t < n:
        data += (json.dumps({'frame_idx': n - 1, 'tracks': []}) + '\n').encode()
    target.write_bytes(data)


def rescore(task: dict) -> dict:
    """The task's key with common.metric_summary of its TrackEval run (a top-level function, so pool workers can run it).

    A padded task writes padded copies of its files into its own temporary directory under work_dir, which is removed
    when the run ends; an unpadded one scores the cached files where they are, as p110 and p141 do.
    """
    # The task's own scratch directory, removed with everything in it when the run ends.
    with tempfile.TemporaryDirectory(dir=task['work_dir']) as scratch:
        if task['padded']:
            # Padded copies of the configuration's files, one per clip (a loop over files, not data rows).
            for video, source, n, t in task['files']:
                write_padded(source, Path(scratch) / video / PADDED_FILE, n, t)
            track_dir, track_file = scratch, PADDED_FILE
        else:
            # The cached files, addressed as the stage that wrote them is.
            track_dir, track_file = task['track_dir'], task['track_file']
        # Score against the padded or the cached ground truth, with the HOTA metric only, as p110 and p141 do.
        combined = trackeval_rerun(task['gt_dir'], task['gt_file'], track_dir, track_file,
                                   [v for v, *_ in task['files']], os.path.join(scratch, 'trackeval'), ('HOTA',))
    # The key and the summary only.
    return dict(dataset=task['dataset'], system=task['system'], config=task['config'], padded=task['padded'],
                metrics=metric_summary(combined))


def padded_ground_truth(con: duckdb.DuckDBPyConnection, work_dir: Path) -> dict[str, tuple[str, str]]:
    """Write every dataset's padded ground truth under work_dir once; return {dataset: (input_gt_dir, input_gt)}."""
    # Each clip's ground-truth file with its length and the clip's.
    rows = con.execute("""
        SELECT g.dataset, g.video, g.path, l.n, s.t FROM gt_files g JOIN clip_lengths l USING (dataset, video)
        JOIN file_stats s ON s.path = g.path ORDER BY ALL
    """).fetchall()
    # One padded copy per clip (a loop over files, not data rows).
    for dataset, video, path, n, t in rows:
        write_padded(path, work_dir / 'ground_truth' / dataset / video / PADDED_FILE, n, t)
    return {dataset: (str(work_dir / 'ground_truth' / dataset), PADDED_FILE) for dataset in {r[0] for r in rows}}


def rescore_tasks(con: duckdb.DuckDBPyConnection, work_dir: Path, gt_padded: dict, reproduce: set) -> list[dict]:
    """One padded task per configuration, and one unpadded task per configuration in `reproduce` ((dataset, system,
    config) keys), largest first so the pool's last runs are short."""
    # Each configuration's clips in clip order, each with its file, both lengths, and its ground-truth file.
    rows = query_dicts(con, """
        SELECT t.dataset, t.system, t.config,
               list([t.video, t.path, l.n::VARCHAR, s.t::VARCHAR, g.path] ORDER BY t.video) AS files, sum(l.n) AS frames
        FROM track_files t JOIN file_stats s USING (path) JOIN clip_lengths l USING (dataset, video)
        JOIN gt_files g USING (dataset, video)
        GROUP BY ALL ORDER BY frames DESC, t.dataset, t.system, t.config
    """)
    # One or two tasks per configuration (hundreds, not data rows).
    tasks = []
    for r in rows:
        # The clips' files with integer lengths.
        files = [(v, p, int(n), int(t)) for v, p, n, t, _ in r['files']]
        base = dict(dataset=r['dataset'], system=r['system'], config=r['config'], files=files, work_dir=str(work_dir))
        # Padded against the padded ground truth.
        tasks.append(dict(base, padded=True, gt_dir=gt_padded[r['dataset']][0], gt_file=gt_padded[r['dataset']][1]))
        # Unpadded against the cached ground truth, where p110 and p141 read both.
        if (r['dataset'], r['system'], r['config']) in reproduce:
            # Both locations from the same (first) clip, so the directory and the relative path pair up.
            video, track_path, *_, gt_path = r['files'][0]
            gt_dir, gt_file = split_path(Path(gt_path), video)
            track_dir, track_file = split_path(Path(track_path), video)
            tasks.append(dict(base, padded=False, gt_dir=gt_dir, gt_file=gt_file, track_dir=track_dir,
                              track_file=track_file))
    return tasks


def load_points(con: duckdb.DuckDBPyConnection, table: str, configs: list[dict], hota: dict):
    """Table <table>(dataset, system, config, fps, hota, accuracy), the shape common.operating_points reads, with each
    configuration's HOTA from `hota` ({(dataset, system, config): HOTA}) and the datasets keyed by their label."""
    # Replace any table from a previous call.
    con.execute(f'CREATE OR REPLACE TABLE {table} (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, '
                'hota DOUBLE, accuracy DOUBLE)')
    # One row per configuration; HOTA is both the reported metric and the selection accuracy.
    rows = [(c['label'], c['system'], c['config'], c['fps'], hota[(c['dataset'], c['system'], c['config'])],
             hota[(c['dataset'], c['system'], c['config'])]) for c in configs]
    con.executemany(f'INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)', rows)


def selections(con: duckdb.DuckDBPyConnection, table: str, losses: list[float]) -> dict:
    """Both selection rules on `table`, with bounds named by common.bound_specs (each loss as '0.05', ..., and leap_fit):
    {points: {bound: common.operating_points}, bounds: {bound: {dataset: fastest_within's unrounded bound}},
    matched: {bound: common.arrange_matched}}.

    operating_points rounds its bound to 4 decimals, too coarse to show how close a configuration came to it, so the
    bounds come from the same fastest_within rows (table <table>_chosen) unrounded.
    """
    # Relative bounds, and LEAP's best HOTA per dataset so LEAP just qualifies.
    specs = bound_specs(con, table, losses, 'leap')
    # The shared-bound rule as the JSON reports it, and the same choices as a table.
    points = {name: operating_points(con, table, SYSTEMS, **spec) for name, spec in specs.items()}
    choose(con, table, specs, SYSTEMS, f'{table}_chosen')
    # The unrounded bound of every bound and dataset (the same for every system of a dataset).
    bounds: dict = {}
    for r in query_dicts(con, f'SELECT DISTINCT bound, dataset, bound_accuracy FROM {table}_chosen ORDER BY ALL'):
        bounds.setdefault(r['bound'], {})[r['dataset']] = r['bound_accuracy']
    # p203's accuracy-matched rule on the same choices.
    accuracy_matched(con, table, f'{table}_chosen', f'{table}_matched')
    matched = arrange_matched(query_dicts(con, f'SELECT * FROM {table}_matched ORDER BY bound, dataset'),
                              matched_stats(con, f'{table}_chosen', f'{table}_matched'))
    return dict(points=points, bounds=bounds, matched=matched)


def speedup_ranges(selection: dict) -> dict:
    """{bound: {vs_reference, vs_otif, vs_leap, vs_accuracy_matched_prior: {min, max, datasets} or None}}: Tetris's
    speedups over the datasets where Tetris (and, for a prior system, that system) meets the bound, and over its
    accuracy-matched prior point, as the manuscript's ranges quote them."""
    out = {}
    # A few bounds and seven datasets, not data rows.
    for bound, by_dataset in selection['points'].items():
        # Tetris over the reference where Tetris qualifies, and over each prior system where both do.
        values = {'reference': [p['systems']['tetris']['speedup_vs_reference'] for p in by_dataset.values()
                                if p['systems']['tetris']['meets_bound']]}
        values.update({prior: [p[f'tetris_speedup_vs_{prior}'] for p in by_dataset.values()
                               if f'tetris_speedup_vs_{prior}' in p] for prior in PRIOR_SYSTEMS})
        # The range of each, or None where no dataset qualifies.
        out[bound] = {f'vs_{k}': dict(min=min(v), max=max(v), datasets=len(v)) if v else None for k, v in values.items()}
        # p203's accuracy-matched range, over the datasets with a matched prior point.
        m = selection['matched'][bound]
        out[bound]['vs_accuracy_matched_prior'] = dict(min=m['min'], max=m['max'], datasets=m['matched']) if m['matched'] else None
    return out


def chosen_keys(selection: dict, label_to_dataset: dict[str, str]) -> set[tuple[str, str, str]]:
    """(dataset, system, config) of every configuration chosen at any bound by either rule, plus each dataset's
    reference."""
    # Every qualifying system's shared-bound choice at every bound.
    keys = {(label_to_dataset[label], s, v['config']) for by_dataset in selection['points'].values()
            for label, p in by_dataset.items() for s, v in p['systems'].items() if v['meets_bound']}
    # Every accuracy-matched prior point (its Tetris point is the shared-bound choice).
    keys |= {(label_to_dataset[label], m['prior']['system'], m['prior']['config']) for by_bound in selection['matched'].values()
             for label, m in by_bound['per_dataset'].items() if m['prior'] is not None}
    # The reference pipeline sets every bound.
    return keys | {(d, 'reference', 'naive') for d in label_to_dataset.values()}


def point_changes(cached: dict, padded: dict, hota: dict) -> list[dict]:
    """Every difference between the cached and the padded shared-bound selections (selections' points): a system's
    chosen configuration or whether it meets the bound, its fps or speedup over the reference, and Tetris's speedup
    over each prior system. Each change carries rule 'shared_bound'.

    A system's change also carries both bounds (bound_hota, cached then padded, from selections' unrounded bounds) and,
    for each configuration chosen on either side, its HOTA minus the bound under each scoring (hota_minus_bound), from
    `hota` ({(dataset label, system, config): (cached HOTA, padded HOTA)}).
    """
    # The selections are nested {bound: {dataset: entry}} and share their keys.
    changes = []
    for bound, by_dataset in cached['points'].items():
        for label, before in by_dataset.items():
            after = padded['points'][bound][label]
            # Each system's selection (three, not data rows).
            for system in SYSTEMS:
                b, a = before['systems'].get(system), after['systems'].get(system)
                # A system without any configuration on the dataset has no selection on either side.
                if b is None and a is None:
                    continue
                fields = ('meets_bound', 'config', 'fps', 'speedup_vs_reference')
                diff = {f: [(b or {}).get(f), (a or {}).get(f)] for f in fields if (b or {}).get(f) != (a or {}).get(f)}
                if not diff:
                    continue
                # Both bounds, and how far each configuration chosen on either side is from them.
                margin = [cached['bounds'][bound][label], padded['bounds'][bound][label]]
                chosen = sorted({c for c in ((b or {}).get('config'), (a or {}).get('config')) if c is not None})
                changes.append(dict(rule='shared_bound', bound=bound, dataset=label, system=system, **diff,
                                    bound_hota=margin,
                                    hota_minus_bound={c: [hota[(label, system, c)][0] - margin[0],
                                                          hota[(label, system, c)][1] - margin[1]] for c in chosen}))
            # Tetris's speedups over the prior systems that meet the bound on either side.
            ratios = sorted(k for k in set(before) | set(after) if k.startswith('tetris_speedup_vs_'))
            for key in ratios:
                if before.get(key) != after.get(key):
                    changes.append(dict(rule='shared_bound', bound=bound, dataset=label, system='tetris',
                                        **{key: [before.get(key), after.get(key)]}))
    return changes


def matched_pairing(entry: dict | None) -> dict:
    """The fields of one dataset's common.arrange_matched entry that matched_changes compares, flat; {} for none."""
    # A dataset where Tetris misses the bound has no entry.
    if entry is None:
        return {}
    prior = entry['prior'] or {}
    return dict(tetris_config=entry['tetris']['config'], prior_system=prior.get('system'),
                prior_config=prior.get('config'), prior_meets_bound=prior.get('meets_bound'), prior_rule=prior.get('rule'),
                tetris_speedup_vs_prior=entry['tetris_speedup_vs_prior'])


def matched_changes(cached: dict, padded: dict) -> list[dict]:
    """Every difference between the cached and the padded accuracy-matched rule (selections' matched), with rule
    'accuracy_matched': per bound, the number of matched datasets, the speedup range, and how many matches miss the
    bound; per dataset, Tetris's point, the matched prior point, and the speedup, with both sides' HOTA of the points
    (tetris_hota, prior_hota) for context."""
    changes = []
    # A few bounds and seven datasets, not data rows.
    for bound, before in cached['matched'].items():
        after = padded['matched'][bound]
        # The bound's summary.
        fields = ('matched', 'min', 'max', 'matched_outside_bound', 'prior_systems')
        diff = {f: [before[f], after[f]] for f in fields if before[f] != after[f]}
        if diff:
            changes.append(dict(rule='accuracy_matched', bound=bound, **diff))
        # Each dataset's pairing, on either side.
        for label in sorted(set(before['per_dataset']) | set(after['per_dataset'])):
            b, a = before['per_dataset'].get(label), after['per_dataset'].get(label)
            fb, fa = matched_pairing(b), matched_pairing(a)
            diff = {f: [fb.get(f), fa.get(f)] for f in sorted(set(fb) | set(fa)) if fb.get(f) != fa.get(f)}
            if diff:
                context = dict(tetris_hota=[(x or {}).get('tetris', {}).get('hota') for x in (b, a)],
                               prior_hota=[((x or {}).get('prior') or {}).get('hota') for x in (b, a)])
                changes.append(dict(rule='accuracy_matched', bound=bound, dataset=label, **diff, **context))
    return changes


def point_metrics(cached: dict, metrics: dict, label_to_dataset: dict[str, str]) -> dict:
    """{bound: {dataset: {system: cached and padded HOTA/DetA/AssA and their change}}} at the cached operating points
    (selections' output), plus each dataset's reference and, under accuracy_matched_prior, its accuracy-matched prior
    point."""
    # The cached operating points (the reported ones), arranged like the selections.
    out: dict = {}
    for bound, by_dataset in cached['points'].items():
        for label, p in by_dataset.items():
            entry = out.setdefault(bound, {}).setdefault(label, {})
            dataset = label_to_dataset[label]
            # The reference and every system that meets the bound.
            chosen = {'reference': 'naive', **{s: v['config'] for s, v in p['systems'].items() if v['meets_bound']}}
            for system, config in chosen.items():
                entry[system] = dict(config=config, **metrics[(dataset, system, config)]['delta'])
            # The prior point p203 pairs with Tetris's, when there is one.
            matched = cached['matched'][bound]['per_dataset'].get(label)
            if matched is not None and matched['prior'] is not None:
                prior = matched['prior']
                entry['accuracy_matched_prior'] = dict(system=prior['system'], config=prior['config'],
                                                       **metrics[(dataset, prior['system'], prior['config'])]['delta'])
    return out


def config_metrics(con: duckdb.DuckDBPyConnection, padded: dict) -> dict:
    """{(dataset, system, config): window counts, cached and padded metrics (common.metric_summary), and their change},
    after the padded runs' checks: every box counted and scored, and an unchanged HOTA where nothing was unscored."""
    # The window and the cached result of every configuration.
    rows = query_dicts(con, """
        SELECT c.dataset, c.system, c.config, c.fps, w.* EXCLUDE (dataset, system, config),
               k."HOTA" AS cached_HOTA, k."DetA" AS cached_DetA, k."AssA" AS cached_AssA
        FROM configs c JOIN windows w USING (dataset, system, config) JOIN cached k USING (dataset, system, config)
    """)
    # Every configuration was re-scored.
    keys = {(r['dataset'], r['system'], r['config']) for r in rows}
    assert keys == set(padded), f'padded runs missing for {sorted(keys - set(padded))[:5]}'
    out = {}
    # One entry per configuration (hundreds, not data rows).
    for r in rows:
        key = (r['dataset'], r['system'], r['config'])
        p = padded[key]
        # Padding to the clip length counts and scores every box on both sides.
        count = (p['Count']['GT_Dets'], p['Count']['Dets'])
        assert count == (r['gt_boxes'], r['tracker_boxes']), f"{key}: padded Count {count} differs from the files' boxes"
        unscored = unscored_boxes(p)
        assert unscored['gt'] == 0, f'{key}: padded run left {unscored["gt"]} ground-truth boxes unscored'
        assert unscored['tracker'] == 0, f'{key}: padded run left {unscored["tracker"]} tracker boxes unscored'
        # Without unscored boxes, padding only adds frames empty on both sides.
        if r['gt_boxes_unscored'] == 0 and r['tracker_boxes_unscored'] == 0:
            assert abs(p['HOTA']['HOTA'] - r['cached_HOTA']) <= TOLERANCE, \
                f"{key}: HOTA moved {r['cached_HOTA']} -> {p['HOTA']['HOTA']}"
        # The change of each reported field.
        delta = {f'{f}_{kind}': v for f in DELTA_FIELDS
                 for kind, v in (('cached', r[f'cached_{f}']), ('padded', p['HOTA'][f]),
                                 ('change', p['HOTA'][f] - r[f'cached_{f}']))}
        out[key] = dict(window={k: v for k, v in r.items() if k not in ('dataset', 'system', 'config', 'fps')
                                and not k.startswith('cached_')},
                        fps=r['fps'], padded=p, delta=delta)
    return out


def check_reproduction(unpadded: dict, metrics: dict):
    """Unpadded reruns must reproduce DATASET.json's HOTA, DetA, AssA, and unscored counts."""
    # A few dozen reruns, not data rows.
    for key, m in unpadded.items():
        cached = metrics[key]
        # The same means as the cached result.
        for f in DELTA_FIELDS:
            assert abs(m['HOTA'][f] - cached['delta'][f'{f}_cached']) <= TOLERANCE, \
                f"{key}: rerun {f} {m['HOTA'][f]} differs from DATASET.json"
        # The same boxes left out.
        expected = dict(gt=cached['window']['gt_boxes_unscored'], tracker=cached['window']['tracker_boxes_unscored'])
        assert unscored_boxes(m) == expected, f'{key}: rerun leaves {unscored_boxes(m)} unscored, the window {expected}'


def summary(con: duckdb.DuckDBPyConnection, metrics: dict) -> dict:
    """Per dataset and system, over its configurations: how many leave boxes unscored, the unscored shares (also
    without the ground truth past the decoded frames) and the HOTA, DetA, and AssA changes (min, median, max); and the
    same over every dataset per system."""
    # One row per configuration with the readings summarized.
    con.execute('CREATE OR REPLACE TABLE config_results (dataset VARCHAR, system VARCHAR, config VARCHAR, '
                'gt_unscored_share DOUBLE, gt_unscored_share_within_frame_count DOUBLE, tracker_unscored_share DOUBLE, '
                'hota_change DOUBLE, deta_change DOUBLE, assa_change DOUBLE)')
    con.executemany('INSERT INTO config_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [(DATASET_LABELS[d], s, c, m['window']['gt_unscored_share'],
                      m['window']['gt_unscored_share_within_frame_count'], m['window']['tracker_unscored_share'],
                      m['delta']['HOTA_change'], m['delta']['DetA_change'], m['delta']['AssA_change'])
                     for (d, s, c), m in metrics.items()])
    # The statistics of each reading, per group.
    stats = ', '.join(f'min({c}) AS {c}_min, median({c}) AS {c}_median, max({c}) AS {c}_max'
                      for c in ('gt_unscored_share', 'gt_unscored_share_within_frame_count', 'tracker_unscored_share',
                                'hota_change', 'deta_change', 'assa_change'))
    counts = ('count(*) AS configs, count(*) FILTER (gt_unscored_share > 0) AS configs_with_gt_unscored, '
              'count(*) FILTER (tracker_unscored_share > 0) AS configs_with_tracker_unscored')
    per_dataset = query_dicts(con, f'SELECT dataset, system, {counts}, {stats} FROM config_results GROUP BY ALL ORDER BY ALL')
    overall = query_dicts(con, f'SELECT system, {counts}, {stats} FROM config_results GROUP BY ALL ORDER BY ALL')
    # Nest as {dataset: {system: statistics}} and {system: statistics}.
    nested: dict = {}
    for r in per_dataset:
        nested.setdefault(r.pop('dataset'), {})[r.pop('system')] = r
    return dict(per_dataset=nested, per_system={r.pop('system'): r for r in overall})


def nest_configs(metrics: dict) -> dict:
    """{dataset label: {system: {config: entry}}} for configs.json."""
    # Hundreds of entries, nested by key.
    out: dict = {}
    for (dataset, system, config), m in sorted(metrics.items()):
        out.setdefault(DATASET_LABELS[dataset], {}).setdefault(system, {})[config] = m
    return out


def method(args: argparse.Namespace) -> dict:
    """Inputs, rules, and checks, for results.json."""
    # Plain-text descriptions, keyed by the reading they describe.
    return dict(
        mechanism=('polyis/trackeval/dataset.py: num_timesteps = largest frame_idx + 1 per file (l.58-67), lists built '
                   'to that length (l.76); TrackEval _base_dataset.get_raw_seq_data keeps the ground truth\'s '
                   'num_timesteps (l.94) and zips similarities (l.98); HOTA (hota.py l.53, l.72), CLEAR (clear.py l.68), '
                   'Identity (identity.py l.53) iterate zip(gt_ids, tracker_ids); Count and num_*_dets (dataset.py '
                   'l.158-159) count every box; a side with no box takes the metrics\' early exit (all FN or all FP)'),
        scorers='p110 (reference, Tetris: 002_naive, 060_uncompressed_tracks) and p141 (OTIF, LEAP: SOTA '
                'tracking_results) against the root dataset\'s 003_groundtruth, HOTA metric only, skip 1',
        inputs='common.system_configs test-split configurations; common.output_path files; DATASET.json of p110/p141',
        clip_length=('n = max(polyis.utilities.get_video_frame_count (the frames the systems decode, as p140; checked '
                     'to add up to the tradeoff frame_count), ground-truth file length); clips_lengthened lists where '
                     'the ground truth is longer; scores do not depend on padding past both files, whose frames are '
                     'empty on both sides'),
        clip_alignment=('for datasets with a lengthened clip, per clip the mean over ground-truth boxes on frames '
                        '1..frame_count-2 of the best IoU with the reference\'s boxes on frame f-1, f, f+1 (0 without '
                        'overlap), and the medians over the other clips; reference_t is the reference file\'s length'),
        window=('a clip is truncated when both files have boxes; unscored = ground-truth boxes on frames >= t_tracker '
                'and tracker boxes on frames >= t_gt, t = largest frame_idx + 1; rows unscored = frames between the two '
                'lengths, box frames = those with a box; *_past_frame_count = ground-truth boxes on frames >= '
                'frame_count, past every system\'s decoded frames; checked to the box against DATASET.json Count - TP - '
                'FN/FP (common.unscored_boxes)'),
        padded=('both sides copied to a temporary directory with an empty row on frame n - 1 where they end earlier; '
                'common.trackeval_rerun (TrackEval HOTA with polyis.trackeval.dataset.Dataset as p110/p141); every '
                'configuration re-scored; checked: TP + FN = GT_Dets, TP + FP = Dets, unchanged HOTA where nothing '
                'was unscored'),
        reproduction=('unpadded TrackEval rerun of the cached operating points of both rules and the reference must '
                      'match DATASET.json'),
        selection=(f'bounds from common.bound_specs (losses {args.losses} of the reference HOTA, and leap_fit = '
                   'LEAP\'s best HOTA) on cached and on padded HOTA; shared_bound: common.operating_points (systems '
                   f'{SYSTEMS}); accuracy_matched: p203\'s rule (common.accuracy_matched, as r7o1_hota_matched_speedup '
                   'checks it against p203); fps from the tradeoff tables'),
        candidates=('Tetris candidates are the test runs of its validated Pareto set (p135, selected on unpadded '
                    'validation HOTA); they are re-ranked here, not re-selected'),
        hota='means over the 19 alphas (HOTA_HOTA of p130/p142); TP/FN/FP at alpha 0.05',
        changes=('rule shared_bound: bound_hota and hota_minus_bound use the unrounded bound (reference HOTA x '
                 '(1 - loss), or LEAP\'s best HOTA), cached then padded; rule accuracy_matched: per bound the matched '
                 'range, per dataset the pairing, with both sides\' HOTA'),
        speedup_ranges=('Tetris over the reference where it meets the bound, over a prior system where both meet it, '
                        'and over its accuracy-matched prior point where it has one'),
        rounding=('4 decimals when written, changes 8; fps and speedups as common.operating_points rounds them '
                  '(0.1 fps, 0.1x)'),
    )


def results_names(datasets: list[str]) -> tuple[str, str]:
    """(results, configs) file names: plain for a run over every dataset, *-partial otherwise."""
    # Every dataset selected (in any order) is a full run.
    suffix = '' if set(datasets) == set(DATASET_LABELS) else '-partial'
    return f'results{suffix}.json', f'configs{suffix}.json'


def main():
    # Command-line options and the result directory.
    args = parse_args()
    out_dir = results_dir(__file__)
    label_to_dataset = {DATASET_LABELS[d]: d for d in args.datasets}
    with tempfile.TemporaryDirectory(prefix='r1o1_hota_scoring_window_') as tmp:
        # Inputs, window, and cached checks in DuckDB.
        con = duckdb.connect()
        register_inputs(con, args.datasets)
        load_rows(con)
        scoring_window(con)
        load_cached(con)
        bad = cached_mismatches(con)
        assert not bad, f'{len(bad)} configurations disagree with their DATASET.json: {bad[:3]}'
        # Tracking files past their clip cannot be padded to it; ground truth past the frame count lengthens the clip.
        past = track_overruns(con)
        assert not past, f'{len(past)} tracking files run past their clip: {past[:5]}'
        lengthened, alignment = long_clips(con), clip_alignment(con)
        # The cached selection, whose points are also rerun unpadded.
        configs = query_dicts(con, 'SELECT * FROM configs ORDER BY dataset, system, config')
        load_points(con, 'cached_points', configs, {(c['dataset'], c['system'], c['config']): c['hota'] for c in configs})
        cached = selections(con, 'cached_points', args.losses)
        # Padded ground truth once per dataset, then one task per configuration (and per rerun).
        tasks = rescore_tasks(con, Path(tmp), padded_ground_truth(con, Path(tmp)), chosen_keys(cached, label_to_dataset))
        print(f'{len(configs)} configurations, {len(tasks)} TrackEval runs queued', flush=True)
        # Every TrackEval run in parallel; each is independent. Workers start from a fresh forkserver process, not
        # forked from this one, whose DuckDB connection and its threads stay open.
        runs = []
        with mp.get_context('forkserver').Pool(max(1, min(args.workers, len(tasks)))) as pool:
            for i, run in enumerate(pool.imap_unordered(rescore, tasks), 1):
                runs.append(run)
                # Progress every 50 runs.
                if i % 50 == 0 or i == len(tasks):
                    print(f'  {i}/{len(tasks)} runs done', flush=True)
        # Padded and unpadded results by configuration.
        padded = {(r['dataset'], r['system'], r['config']): r['metrics'] for r in runs if r['padded']}
        unpadded = {(r['dataset'], r['system'], r['config']): r['metrics'] for r in runs if not r['padded']}
        # Checks of the padded runs and the reruns.
        metrics = config_metrics(con, padded)
        check_reproduction(unpadded, metrics)
        # Both selections on padded HOTA, and what changes.
        load_points(con, 'padded_points', configs, {k: m['padded']['HOTA']['HOTA'] for k, m in metrics.items()})
        padded_sel = selections(con, 'padded_points', args.losses)
        hota = {(DATASET_LABELS[d], s, c): (m['delta']['HOTA_cached'], m['delta']['HOTA_padded'])
                for (d, s, c), m in metrics.items()}
        changes = point_changes(cached, padded_sel, hota) + matched_changes(cached, padded_sel)
        stats = summary(con, metrics)
        con.close()
    # One summary line per dataset and system.
    for label, by_system in rounded(stats['per_dataset']).items():
        for system, s in by_system.items():
            print(f"{label:9s} {system:9s} {s['configs']:3d} configs, ground truth unscored share "
                  f"{s['gt_unscored_share_min']}-{s['gt_unscored_share_max']}, HOTA change "
                  f"{s['hota_change_min']} to {s['hota_change_max']}")
    print(f'{len(changes)} operating-point changes')
    # Tetris's speedup ranges under both scorings, as the manuscript quotes them.
    for kind, sel in (('cached', cached), ('padded', padded_sel)):
        print(kind, {b: {k: (v['min'], v['max']) if v else None for k, v in r.items()}
                     for b, r in speedup_ranges(sel).items()})
    # The two result files, named by whether every dataset was run.
    results_name, configs_name = results_names(args.datasets)
    report = rounded(dict(
        method=method(args), ground_truth={DATASET_LABELS[d]: GROUND_TRUTH[DATASET_LABELS[d]] for d in args.datasets},
        summary=stats, clips_lengthened=lengthened, clip_alignment=alignment,
        reproduced=sorted(f'{DATASET_LABELS[d]} {s} {c}' for d, s, c in unpadded),
        operating_points=dict(cached=cached['points'], padded=padded_sel['points']),
        accuracy_matched=dict(cached=cached['matched'], padded=padded_sel['matched']),
        speedup_ranges=dict(cached=speedup_ranges(cached), padded=speedup_ranges(padded_sel)),
        operating_point_metrics=point_metrics(cached, metrics, label_to_dataset)))
    # The changes keep 8 decimals, since a bound margin can be far below 4.
    report['changes'] = rounded(changes, 8)
    write_json(out_dir, results_name, report)
    write_json(out_dir, configs_name, rounded(nest_configs(metrics)))


if __name__ == '__main__':
    main()
