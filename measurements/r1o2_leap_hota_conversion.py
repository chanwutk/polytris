#!/usr/local/bin/python
"""R1.O2: how LEAP's output becomes HOTA-scored tracks, and how much of LEAP's low HOTA the conversion explains.

LEAP answers with one predicted interval [start, end) and one motion-pattern id per object. The chain that turns
this into the tracks p141 scores:

  1. LEAP writes <video>_results.json ({object: [[start, end], pattern index, _]}) and <video>_tracks.json (the
     motion patterns, one box list each). These files are not on ace for the scored runs (see blockers).
  2. export-leap-outputs.py (ace:/work/alenachao/tunnel/work/scripts/) places each object's pattern over its
     interval: the pattern is linearly resampled when shorter than the interval, repeated when it has one box,
     and truncated when longer; box idx goes to index start + idx of a frame list as long as the largest end.
     A negative start therefore wraps to the end of the list (Python indexing), and an end past the clip
     lengthens the list past the clip. The exports are /sota-results/<dataset>/leap/<video id>.json.
  3. evaluation/p140_otif_transform.py turns an export into tracking.jsonl. Until 2026-10-26 it kept the wrapped
     tail and then interpolated the gap between the tail and the object's head, filling every frame between;
     frames past the clip were kept. Since then it drops the wrapped tail, keeps an object present on every
     exported frame whole (ambiguous: its wrap may overlap its real part, so both its true start and its true
     end are unknown), drops frames past the clip, and records what it removed in unwrap.json. The old cache is
     in SOTA/leap-wrapped-backup-20261026.
  4. evaluation/p141_otif_accuracy.py scores the tracks with TrackEval against the root dataset's 003_groundtruth
     through polyis/trackeval/dataset.py.

What this script measures per dataset (all from the cache and the raw exports, read-only):
  accuracy     LEAP's HOTA family (means over TrackEval's 19 alphas, and at the loosest alpha 0.05) and Count from
               p141's DATASET.json before and after the fix, checked against p142's tradeoff.csv.
  unwrap       totals of p140's unwrap.json: empty exports, wrapped and ambiguous objects, boxes and frames
               dropped past the clip.
  intervals    every exported object's interval recovered from the raw export (one run [a, b] is [a, b + 1);
               a head run from 0 plus a tail run to the export's end is a wrapped interval [tail - L, head + 1)),
               cross-checked object by object against unwrap.json; the share of objects whose interval was
               clipped at the start (wrapped), at the end (past the clip), or at either end, as lower and upper
               bounds over the ambiguous objects. An ambiguous object of an export longer than the clip is
               clipped at one end for certain (either it wraps, or it starts at 0 and ends at L > n), but which
               end is unknown; in a shorter export it is unclipped or start-clipped.
  box_frames   boxes of the ground truth, the old cache, the fixed cache, and the fixed cache without ambiguous
               objects; the old cache's surplus over the whole file and inside the clip, split into the wrapped
               tails, the boxes past the clip, and the gap fill of the old interpolation (checked to equal the
               frames between each wrapped object's head and tail).
  scored_window  boxes that p141's scoring leaves out. polyis/trackeval/dataset.py gives each file its own length
               (largest frame_idx + 1; save_tracking_results writes rows from the first to the last frame with a
               box), TrackEval keeps the ground truth's, and HOTA, CLEAR, and Identity iterate over
               zip(ground truth, tracker). When both sides have boxes, a clip is scored only on its first
               min(T_gt, T_tracker) frames, so ground-truth boxes after the tracker's last box are not misses and
               tracker boxes after the ground truth's last box are not false positives, while Count counts every
               box. Checked to the box against DATASET.json (GT_Dets - TP - FN and Dets - TP - FP at alpha 0.05).
  trackeval    TrackEval (HOTA, CLEAR, Identity, Count, as p141 runs it) rerun into a temporary directory on
               five variants: before and after (cached files; must reproduce DATASET.json), after without
               ambiguous objects (oracle-free lower reading of the ambiguous objects), and both of these on full
               clips (tracks and ground truth cut to the clip and padded with an empty row on its last frame, so
               every box inside the clip is scored).
  count_ceiling  an upper bound on DetA that holds for any box placement: on a frame with p predicted and g
               ground-truth boxes at most min(p, g) can match, so DetA <= sum min / (sum g + sum p - sum min) at
               every alpha, and HOTA <= sqrt(that) with perfect association. It depends only on LEAP's intervals
               (per-frame counts), not on where the exporter puts boxes; it is computed on full clips and compared
               with the full-clip TrackEval variants.
  time_base    whether LEAP's frame f is the clip's frame f, as every number above assumes (LEAP's configs set
               fps 15, as our clips). Export length over clip length, LEAP's interval length over the ground
               truth's track length, and the median per-clip correlation of LEAP's per-frame counts (its
               intervals on LEAP frames >= 0) with the ground truth's when clip frame t reads LEAP frame
               round(t / s), over a grid of time scales s. A best scale other than 1 that gains more than
               TIME_SCALE_WARN_GAIN flags the dataset. Shifts at scale 1 are reused from the native-queries lags.
  native_queries  per-frame count agreement with the ground truth, reused from
               measurements-results/r1o2_leap_native_queries/ (not recomputed): LEAP's aggregation error and
               selection F1 with and without ambiguous objects, against the reference pipeline; cross-checked
               against this script's unwrap totals and box counts, so both read the same cache.
  attribution  the HOTA changes of each measured conversion step, and LEAP's full-clip DetA split into the part
               the count ceiling removes (1 - ceiling) and the part lost to placement (ceiling - DetA), in
               absolute and in log terms. Placement covers the exporter's pattern placement, LEAP's own pattern
               choice, and any time-base mismatch, which cannot be split on ace. AssA is not attributed.
  summary      ranges over the datasets and the datasets each reading applies to.
  input_digests  per dataset and input source, the file count, bytes, and a combined SHA-256.

Writes measurements-results/r1o2_leap_hota_conversion/results.json, or results-partial.json when --datasets
selects only some datasets, so a partial run never replaces the full results.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import tempfile
from pathlib import Path

import duckdb

from evaluation.manifests import list_split_videos
from evaluation.p140_otif_transform import dataset_name_in_sota, extract_video_id
from measurements.common import (DATASET_LABELS, RESULTS_ROOT, assert_files_exist, file_record, leap_intervals,
                                 load_leap_exports, load_unwrap, metric_summary, output_path, query_dicts, results_dir,
                                 rounded, trackeval_rerun, unscored_boxes, unwrap_totals, write_json)
from polyis.io import cache
from polyis.utilities import dataset_root_name


# The LEAP cache p140 rewrote on 2026-10-26 and the copy of the cache it replaced, as directories under SOTA/.
CACHES = {'before': 'leap-wrapped-backup-20261026', 'after': 'leap'}
# LEAP has one parameter set, stored under parameter id 000.
LEAP_PARAM = '000'
# Where a video's LEAP tracks and the ground truth sit, relative to the video's directory (as in p141).
TRACK_FILE = Path('tracking_results', LEAP_PARAM, 'tracking.jsonl')
GT_FILE = Path('003_groundtruth', 'tracking.jsonl')
# Input sources of a dataset, as registered in table files.
INPUT_SOURCES = ('gt', 'before', 'after', 'unwrap', 'export')
# File name of rewritten tracks inside a variant's video directory.
REWRITTEN_FILE = Path('tracking.jsonl')
# TrackEval variants: which cache the tracks come from, whether ambiguous objects are dropped, and whether tracks
# and ground truth are cut and padded to the clip.
VARIANTS = {
    'before': dict(cache='before', drop_ambiguous=False, full_length=False),
    'after': dict(cache='after', drop_ambiguous=False, full_length=False),
    'after_ambiguous_dropped': dict(cache='after', drop_ambiguous=True, full_length=False),
    'after_full_length': dict(cache='after', drop_ambiguous=False, full_length=True),
    'after_ambiguous_dropped_full_length': dict(cache='after', drop_ambiguous=True, full_length=True),
}
# Box sources in table boxes: the ground truth, both caches, and the fixed cache without ambiguous objects.
BOX_SOURCES = ('gt', 'before', 'after', 'after_ambiguous_dropped')
# Largest difference allowed between a TrackEval rerun and the cached DATASET.json, and between equal readings.
REPRODUCTION_TOLERANCE = 1e-9
# The native-queries outputs round to 4 decimals, so their means can differ from exact ones by half a unit.
NATIVE_ROUNDING = 5e-5
# Systems read from the native-queries outputs, and the box source each LEAP reading corresponds to.
NATIVE_SYSTEMS = ('reference', 'leap', 'leap_ambiguous_dropped')
NATIVE_BOX_SOURCE = {'leap': 'after', 'leap_ambiguous_dropped': 'after_ambiguous_dropped'}
# Time scales s of the time-base diagnostic (clip frame t reads LEAP frame round(t / s)), the correlation gain over
# s = 1 that flags a dataset, and the decimals at which two scales' correlations count as tied (as the native-queries
# lag diagnostic).
TIME_SCALES = tuple(round(0.40 + 0.05 * i, 2) for i in range(23))
TIME_SCALE_WARN_GAIN = 0.05
TIME_SCALE_TIE_DECIMALS = 9
# Schema of a tracking.jsonl line with each box kept as JSON text, so rewritten files keep every digit
# (DuckDB's DOUBLE-to-JSON writer shortens 0.30000000000000004 to 0.3).
LINE_COLUMNS = "{'frame_idx': 'BIGINT', 'tracks': 'JSON[]'}"
# What the cached outputs cannot show, recorded in results.json.
BLOCKERS = {
    'raw_and_linear_interpolation_variants': (
        "LEAP's raw sparse observations (the boxes its detector found on the frames it sampled, with "
        're-identification ids) are not on ace: the exporter reads only intervals and motion patterns, the scored '
        'exports hold only pattern boxes, and /work/alenachao/tunnel/work/LEAP/outputs/reid/caldot1 and caldot2 '
        "are empty. The planned 'raw' (only observed boxes) and 'linear interpolation' (interpolation between "
        'observed boxes instead of pattern placement) HOTA variants cannot be scored.'),
    'native_results': (
        'The scored exports (/sota-results/*/leap/, written 2026-04-10) were built from '
        '/data/alenachao/data/leap-dataset/outputs/parsed_results/<dataset>, which is not on ace. ace holds an '
        'earlier run for CalDoT1/2 only (/work/alenachao/tunnel/work/LEAP/outputs/parsed_results/caldot1 and '
        'caldot2, 2026-03-12). Rebuilding every export from it with the logic of export-leap-outputs.py\'s '
        'reconstruct_tracks_json, in memory, and comparing the parsed JSON with the scored exports matches 3 of 49 '
        'non-empty CalDoT1 and 2 of 56 non-empty CalDoT2 exports, plus 7 of 11 and 4 of 4 empty ones. The check ran '
        'once on 2026-10-26 in a throwaway container with both directories mounted read-only; it is not part of '
        'this script, because the polyis container does not mount /work/alenachao/tunnel/work. The earlier run is '
        "therefore not the scored run. Without the scored run's results.json, both ends of the ambiguous objects "
        "and which objects' patterns were resampled, repeated, or truncated are unknown. The DetA lost between "
        "count_ceiling and LEAP's DetA (attribution.deta_decomposition.placement_loss_abs) mixes the exporter's "
        "pattern placement, LEAP's own pattern choice, and any time-base mismatch, and cannot be split."),
    'time_base': (
        "The frame base of LEAP's intervals cannot be confirmed on ace, since the scored run's inputs are not here. "
        "LEAP's configs (LEAP/configs/jnc*.yaml, caldot*.yaml) set fps 15, as our clips. ace's copies of LEAP's B3D "
        'test videos (/work/alenachao/tunnel/work/LEAP/data/videos/jnc0, jnc2, jnc6, jnc7 /test) are 29 fps with '
        '2150, 1876-1877, 2077, and 2090 frames, about twice our 15-fps clips (1075, 938-939, 1038-1039, and 1045 '
        'frames), while its CalDoT copies are 15 fps with 900 frames, as ours (ffprobe packet counts, 2026-10-26, '
        'throwaway container with the directory mounted read-only). Whether the scored B3D run read the 29-fps '
        'copies is unknown; datasets[*].time_base records what the counts suggest.'),
    'b3d_ground_truth': ('B3D (jnc*) is scored on automatic ground truth from the same detector family as Tetris '
                         'and the reference; its numbers are provisional until R1.O1.'),
}
# Findings for the authors, recorded in results.json.
FINDINGS = {
    'attribution': (
        "The conversion steps measured here (the p140 fix, dropping the ambiguous objects, and scoring full clips) "
        "move LEAP's HOTA by summary.measured_conversion_effect_range. They do not cover the core of the "
        "conversion, the exporter placing LEAP's motion pattern over each interval (resampled, repeated, or "
        'truncated). That step, LEAP\'s own pattern choice, and any time-base mismatch together make up the DetA '
        'lost between count_ceiling and LEAP\'s DetA (datasets[*].attribution.deta_decomposition), so the error the '
        'placement introduces is bounded above by that loss, not measured. Which of counts and placement loses '
        'more DetA depends on the scale (summary.count_loss_larger_absolute, summary.placement_loss_larger_log). '
        "AssA is not attributed, since no bound here constrains it. This does not support saying that the "
        'conversion introduces little error.'),
    'time_base': (
        "Every interval, clipping, ceiling, and attribution number assumes LEAP's frame f is the clip's frame f. "
        'On the datasets in summary.time_base_flagged, LEAP\'s counts correlate better with the ground truth '
        'under another time scale (datasets[*].time_base), and their exports run past the clip; there the '
        'boxes past the clip, the end-clipped shares, the count ceiling, and the attribution may reflect a '
        "frame-base mismatch rather than LEAP's intervals, until someone with the scored run's inputs confirms "
        'the frame base (blockers.time_base).'),
    'scored_window': (
        'polyis/trackeval/dataset.py with TrackEval scores a clip only on its first min(T_gt, T_tracker) frames '
        'when both sides have boxes (see datasets[*].scored_window). For LEAP it hides misses after its last '
        'exported box. evaluation/p110_accuracy_compute.py uses the same dataset class, so the reference, Tetris, '
        'and OTIF are scored the same way; they are not quantified here. The *_full_length TrackEval variants '
        'score full clips.'),
    'count_ids': ("TrackEval's Count IDs (and p142's Count_TracksMAPE) is max(track id) + 1 per clip, not the number "
                  'of tracks; box_frames.*.objects_in_clip counts distinct ids.'),
}


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: LEAP's HOTA conversion before and after the p140 fix")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--sota-dir', type=Path, default=Path('/sota-results'))
    parser.add_argument('--native-dir', type=Path, default=RESULTS_ROOT / 'r1o2_leap_native_queries')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def input_files(dataset: str, videos: list[str], sota_dir: Path) -> list[tuple[str, str, str]]:
    """(source, video, path) of every input of a dataset: ground truth, both caches' tracks, unwrap.json, raw exports."""
    # Ground truth of the root dataset, as p141 scores against it.
    rows = [('gt', v, str(output_path(dataset, 'ground truth', '003_groundtruth', v))) for v in videos]
    # LEAP's tracks in the old and the fixed cache.
    rows += [(source, v, str(cache.sota(directory, dataset, v, str(TRACK_FILE))))
             for source, directory in CACHES.items() for v in videos]
    # p140's statistics next to the fixed tracks.
    rows += [('unwrap', v, str(cache.sota(CACHES['after'], dataset, v, str(TRACK_FILE.parent), 'unwrap.json')))
             for v in videos]
    # The raw exports, named by the numeric video id.
    rows += [('export', v, str(sota_dir / dataset_name_in_sota(dataset) / 'leap' / f'{extract_video_id(v)}.json'))
             for v in videos]
    # Every clip needs every input; a missing one would silently shrink a total.
    assert_files_exist([p for *_, p in rows], f'{dataset} LEAP inputs')
    return rows


def register_files(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, str]]):
    """Table files(source, video, path)."""
    # One row per input file.
    con.execute('CREATE OR REPLACE TABLE files (source VARCHAR, video VARCHAR, path VARCHAR)')
    con.executemany('INSERT INTO files VALUES (?, ?, ?)', rows)


def input_digests(con: duckdb.DuckDBPyConnection) -> dict:
    """Per input source of table files: file count, bytes, latest modification time, and a combined SHA-256 over the
    sorted (video, file SHA-256) pairs, so a later change to any cached input shows in results.json."""
    out = {}
    # A handful of sources, each a loop over files (hashing is file I/O, not data analysis).
    for source in INPUT_SOURCES:
        # The registered files of the source, in clip order.
        records = [(video, file_record(path)) for video, path in source_files(con, source)]
        # One digest over every file's own digest, tied to its clip.
        combined = hashlib.sha256(''.join(f"{video}\t{r['sha256']}\n" for video, r in records).encode()).hexdigest()
        out[source] = dict(files=len(records), bytes=sum(r['bytes'] for _, r in records), sha256=combined,
                           latest_mtime=max((r['mtime'] for _, r in records), default=None))
    return out


def source_files(con: duckdb.DuckDBPyConnection, source: str) -> list[tuple[str, str]]:
    """(video, path) of the registered files of one source, in clip order."""
    # The pairs common.load_unwrap and common.load_leap_exports take.
    return con.execute('SELECT video, path FROM files WHERE source = ? ORDER BY video', [source]).fetchall()


def source_paths(con: duckdb.DuckDBPyConnection, sources: list[str]) -> list[str]:
    """Paths of the registered files of the given sources."""
    # The paths as DuckDB will report them in its filename column.
    return [r[0] for r in con.execute('SELECT path FROM files WHERE list_contains(?, source) ORDER BY path',
                                      [sources]).fetchall()]


def load_inputs(con: duckdb.DuckDBPyConnection):
    """Load the inputs registered in table files: unwrap.json (table unwrap), the tracks (tables lines and boxes), and
    the raw exports, whose recovered intervals become table objects."""
    # p140's statistics first, since the boxes without ambiguous objects read them.
    load_unwrap(con, source_files(con, 'unwrap'))
    # The ground truth's and both caches' tracks.
    load_lines(con)
    # The raw exports, and every exported object's interval and what p140 did with it.
    load_leap_exports(con, source_files(con, 'export'))
    classify_objects(con)


def load_lines(con: duckdb.DuckDBPyConnection):
    """Tables lines(source, video, frame, tracks JSON[]) of every tracking.jsonl and boxes(source, video, frame, track_id).

    boxes holds the sources of BOX_SOURCES; after_ambiguous_dropped is the fixed cache without unwrap.json's
    ambiguous objects. Duplicate rows, negative frames, null track lists, and repeated ids in a frame would break
    TrackEval's loader or make counts ambiguous, so they raise.
    """
    # Every line of the ground truth and both caches; an empty file contributes no line.
    con.execute(f"""
        CREATE OR REPLACE TABLE lines AS
        SELECT f.source, f.video, l.frame_idx AS frame, l.tracks
        FROM read_json(?, filename = true, format = 'newline_delimited', columns = {LINE_COLUMNS}) l
        JOIN files f ON f.path = l.filename AND f.source IN ('gt', 'before', 'after')
    """, [source_paths(con, ['gt', 'before', 'after'])])
    # TrackEval's loader asserts one line per frame and indexes lists by frame.
    bad = query_dicts(con, """
        SELECT source, video, frame, count(*) AS n FROM lines GROUP BY ALL
        HAVING n > 1 OR frame IS NULL OR frame < 0 OR bool_or(tracks IS NULL) LIMIT 5
    """)
    assert not bad, f'tracking lines repeated, negative, or without a track list: {bad}'
    # One row per box with its integer track id, then the fixed cache without ambiguous objects.
    con.execute("""
        CREATE OR REPLACE TABLE boxes AS
        SELECT source, video, frame, CAST(CAST(box->>0 AS DOUBLE) AS BIGINT) AS track_id
        FROM (SELECT source, video, frame, unnest(tracks) AS box FROM lines)
    """)
    con.execute("""
        INSERT INTO boxes SELECT 'after_ambiguous_dropped', b.video, b.frame, b.track_id
        FROM boxes b JOIN unwrap u USING (video)
        WHERE b.source = 'after' AND NOT list_contains(u.ambiguous_full_span_objects, b.track_id)
    """)
    # TrackEval requires unique ids within a frame.
    bad = query_dicts(con, """
        SELECT source, video, frame, track_id, count(*) AS n FROM boxes GROUP BY ALL HAVING n > 1 OR track_id IS NULL LIMIT 5
    """)
    assert not bad, f'boxes with a missing or repeated track id in one frame: {bad}'


def classify_objects(con: duckdb.DuckDBPyConnection):
    """Table objects: every exported object's recovered interval [interval_start, interval_end) and how p140 changed it.

    Columns: video, track_id, runs, L, n, head_end and tail_start (last frame of the first run, first frame of the last
    run), ambiguous (one run over the whole export; [interval_start, interval_end) is then [0, L), as kept, and both
    true ends are unknown), wrapped (start before the clip), end_clipped (a known end past the clip; never ambiguous),
    ambiguous_past_clip (ambiguous in an export longer than the clip: clipped at one end for certain, which one is
    unknown), entirely_past_clip (start at or past the clip end, removed whole), boxes, wrapped_boxes (the dropped
    tail), past_clip_boxes (boxes at or past the clip end outside the tail). The intervals come from
    common.leap_intervals, which raises on any other shape than the exporter's wrap.
    """
    # Every exported object's interval, recovered from the raw export.
    leap_intervals(con)
    # Attach each clip's length, then count each object's boxes by what p140 does with them.
    con.execute("""
        CREATE OR REPLACE TABLE objects AS
        WITH intervals AS (
            -- Each recovered interval with its clip's length n.
            SELECT i.*, u.n FROM leap_intervals i JOIN unwrap u USING (video)
        ),
        counts AS (
            -- Boxes per object, those in the wrapped tail, and those at or past the clip end outside the tail.
            SELECT o.video, o.track_id, count(*) AS boxes,
                   count(*) FILTER (i.wrapped AND o.frame >= i.tail_start) AS wrapped_boxes,
                   count(*) FILTER (o.frame >= i.n AND NOT (i.wrapped AND o.frame >= i.tail_start)) AS past_clip_boxes
            FROM export_obs o JOIN intervals i USING (video, track_id) GROUP BY ALL
        )
        SELECT i.video, i.track_id, i.runs, i.L, i.n, i.head_end, i.tail_start, i.ambiguous, i.wrapped,
               i.interval_start, i.interval_end, NOT i.ambiguous AND i.interval_end > i.n AS end_clipped,
               i.ambiguous AND i.L > i.n AS ambiguous_past_clip,
               NOT i.ambiguous AND i.interval_start >= i.n AS entirely_past_clip,
               c.boxes, c.wrapped_boxes, c.past_clip_boxes
        FROM intervals i JOIN counts c USING (video, track_id)
    """)


def unwrap_mismatches(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Clips where the recovery from the raw export disagrees with p140's unwrap.json (ids, counts, or lengths)."""
    # Every clip whose ids, box counts, or export length differ between the two.
    return query_dicts(con, """
        WITH per_video AS (
            -- The recovery's view of each clip, in unwrap.json's terms.
            SELECT video, count(*) AS objects,
                   coalesce(list(track_id ORDER BY track_id) FILTER (wrapped), []) AS wrapped_ids,
                   coalesce(list(track_id ORDER BY track_id) FILTER (ambiguous), []) AS ambiguous_ids,
                   sum(wrapped_boxes) AS wrapped_boxes, sum(past_clip_boxes) AS past_clip_boxes
            FROM objects GROUP BY video
        )
        SELECT u.video, u.objects, p.objects AS recovered_objects, u.export_frames, x.L,
               u.wrapped_boxes_dropped, p.wrapped_boxes, u.boxes_past_clip_dropped, p.past_clip_boxes
        FROM unwrap u JOIN export_len x USING (video) LEFT JOIN per_video p USING (video)
        WHERE u.export_frames <> x.L OR u.objects <> coalesce(p.objects, 0)
           OR list_sort(u.wrapped_objects) <> coalesce(p.wrapped_ids, []::BIGINT[])
           OR list_sort(u.ambiguous_full_span_objects) <> coalesce(p.ambiguous_ids, []::BIGINT[])
           OR u.wrapped_boxes_dropped <> coalesce(p.wrapped_boxes, 0)
           OR u.boxes_past_clip_dropped <> coalesce(p.past_clip_boxes, 0)
        ORDER BY u.video
    """)


def unwrap_lengths(con: duckdb.DuckDBPyConnection) -> dict:
    """Clip frames over the clips of table unwrap, and how the exports' lengths compare with their clips' (the rest of
    a section's unwrap totals is common.unwrap_totals, which r1o2_leap_native_queries' diagnostics also report)."""
    # One row over the dataset's clips.
    return query_dicts(con, """
        SELECT sum(n) AS clip_frames,
               -- Exported frames at or past the clip end, which p140 drops.
               sum(greatest(export_frames - n, 0)) AS export_frames_past_clip,
               count(*) FILTER (export_frames > n) AS clips_exported_past_clip,
               -- Non-empty exports that stop before the clip's last frame.
               count(*) FILTER (export_frames BETWEEN 1 AND n - 1) AS clips_exported_short_of_clip_end
        FROM unwrap
    """)[0]


def interval_summary(con: duckdb.DuckDBPyConnection) -> dict:
    """How many exported objects had their interval clipped at either end, and the boxes each clip removed.

    Known counts leave the ambiguous objects out; *_upper adds the ambiguous objects that may be clipped at that end
    (any at the start; at the end, those of exports longer than the clip). Either end: an ambiguous object of an
    export longer than the clip is clipped for certain (lower), any ambiguous object may be (upper).
    """
    # One row of counts, shares, and box totals over the dataset's exported objects.
    return query_dicts(con, """
        SELECT count(*) AS objects,
               count(*) FILTER (wrapped) AS start_clipped, count(*) FILTER (wrapped OR ambiguous) AS start_clipped_upper,
               count(*) FILTER (end_clipped) AS end_clipped,
               count(*) FILTER (end_clipped OR ambiguous_past_clip) AS end_clipped_upper,
               count(*) FILTER (wrapped AND end_clipped) AS both_ends_clipped,
               count(*) FILTER (wrapped OR end_clipped OR ambiguous_past_clip) AS clipped_either_end_lower,
               count(*) FILTER (wrapped OR end_clipped OR ambiguous) AS clipped_either_end_upper,
               count(*) FILTER (ambiguous) AS ambiguous, count(*) FILTER (ambiguous_past_clip) AS ambiguous_past_clip,
               count(*) FILTER (entirely_past_clip) AS entirely_past_clip,
               -- The same as shares of all exported objects.
               count(*) FILTER (wrapped) / nullif(count(*), 0) AS start_clipped_share,
               count(*) FILTER (wrapped OR ambiguous) / nullif(count(*), 0) AS start_clipped_share_upper,
               count(*) FILTER (end_clipped) / nullif(count(*), 0) AS end_clipped_share,
               count(*) FILTER (end_clipped OR ambiguous_past_clip) / nullif(count(*), 0) AS end_clipped_share_upper,
               count(*) FILTER (wrapped OR end_clipped OR ambiguous_past_clip) / nullif(count(*), 0)
                   AS clipped_either_end_share_lower,
               count(*) FILTER (wrapped OR end_clipped OR ambiguous) / nullif(count(*), 0) AS clipped_either_end_share_upper,
               -- Boxes of the raw export, and the share the fix removed.
               sum(boxes) AS export_boxes, sum(wrapped_boxes) AS wrapped_boxes, sum(past_clip_boxes) AS past_clip_boxes,
               (sum(wrapped_boxes) + sum(past_clip_boxes)) / nullif(sum(boxes), 0) AS removed_box_share,
               -- Frames between each wrapped object's head and tail, which the old interpolation filled.
               coalesce(sum(tail_start - head_end - 1) FILTER (wrapped), 0) AS wrapped_gap_frames,
               -- Interval lengths in frames (ambiguous ones are unknown and left out).
               median(interval_end - interval_start) FILTER (NOT ambiguous) AS median_interval_frames,
               median(interval_end - interval_start) FILTER (NOT ambiguous AND (wrapped OR end_clipped))
                   AS median_clipped_interval_frames
        FROM objects
    """)[0]


def box_frames(con: duckdb.DuckDBPyConnection) -> dict:
    """Per box source: boxes, boxes inside and past the clip, frames past the clip, and objects (all, inside the clip)."""
    # Box, frame, and object counts per source, split at the clip end n.
    rows = query_dicts(con, """
        SELECT b.source, count(*) AS boxes, count(*) FILTER (b.frame < u.n) AS boxes_in_clip,
               count(*) FILTER (b.frame >= u.n) AS boxes_past_clip,
               count(DISTINCT (b.video, b.frame)) FILTER (b.frame >= u.n) AS frames_past_clip,
               count(DISTINCT (b.video, b.track_id)) AS objects,
               count(DISTINCT (b.video, b.track_id)) FILTER (b.frame < u.n) AS objects_in_clip
        FROM boxes b JOIN unwrap u USING (video) GROUP BY b.source
    """)
    # Every source appears, also one without any box.
    empty = dict(boxes=0, boxes_in_clip=0, boxes_past_clip=0, frames_past_clip=0, objects=0, objects_in_clip=0)
    found = {r.pop('source'): r for r in rows}
    return {s: found.get(s, dict(empty)) for s in BOX_SOURCES}


def box_changes(frames: dict, intervals: dict) -> dict:
    """The old cache's surplus over the fixed one, split into wrapped tails, boxes past the clip, and gap fill.

    The raw export holds export_boxes; the fix removes the wrapped tails and the boxes past the clip, and nothing
    else, so the fixed cache must hold exactly the rest. The old cache held the whole export plus the boxes its
    interpolation filled between each wrapped tail and its head, which must be exactly the wrapped objects' gap
    frames. inflation counts every box in the file; inflation_in_clip only those HOTA can score (frames < n).
    """
    # Boxes the fixed cache must hold.
    kept = intervals['export_boxes'] - intervals['wrapped_boxes'] - intervals['past_clip_boxes']
    assert frames['after']['boxes'] == kept, \
        f"fixed cache holds {frames['after']['boxes']} boxes, the export minus p140's removals {kept}"
    # The old cache's gap fill: what it holds beyond the raw export, which must be the head-to-tail gaps.
    fill = frames['before']['boxes'] - intervals['export_boxes']
    assert fill >= 0, f'old cache holds {-fill} fewer boxes than the raw export'
    assert fill == intervals['wrapped_gap_frames'], \
        f"old cache's surplus over the export {fill} is not the wrapped objects' gap frames {intervals['wrapped_gap_frames']}"
    # Both caches' box counts.
    before, after = frames['before'], frames['after']
    # The decomposition, with the relative surplus over the whole file and inside the clip (None without fixed boxes).
    return dict(before=before['boxes'], after=after['boxes'], surplus=before['boxes'] - after['boxes'],
                before_in_clip=before['boxes_in_clip'], after_in_clip=after['boxes_in_clip'],
                surplus_in_clip=before['boxes_in_clip'] - after['boxes_in_clip'],
                wrapped_tails=intervals['wrapped_boxes'], past_clip=intervals['past_clip_boxes'], gap_fill=fill,
                inflation=before['boxes'] / after['boxes'] - 1 if after['boxes'] else None,
                inflation_in_clip=before['boxes_in_clip'] / after['boxes_in_clip'] - 1 if after['boxes_in_clip'] else None)


def scored_window(con: duckdb.DuckDBPyConnection, source: str) -> dict:
    """Boxes p141's TrackEval run leaves out for a cached box source ('before' or 'after'), and clips shorter than the clip.

    A file's length T is its largest frame_idx + 1 (0 when empty). When both the ground truth and the tracks of a clip
    have boxes, ground-truth boxes at frames >= T_tracker and tracker boxes at frames >= T_gt are not scored.
    """
    # One row of box and clip counts for the source.
    return query_dicts(con, """
        WITH lengths AS (SELECT source, video, max(frame) + 1 AS t FROM lines GROUP BY ALL),
        dets AS (SELECT source, video, count(*) AS dets FROM boxes GROUP BY ALL),
        seq AS (
            -- Each clip's two file lengths, and whether TrackEval truncates it (both sides have boxes).
            SELECT u.video, u.n, coalesce(lg.t, 0) AS t_gt, coalesce(lt.t, 0) AS t_tracker,
                   coalesce(dg.dets, 0) > 0 AND coalesce(dt.dets, 0) > 0 AS truncated
            FROM unwrap u
            LEFT JOIN lengths lg ON lg.source = 'gt' AND lg.video = u.video
            LEFT JOIN lengths lt ON lt.source = $source AND lt.video = u.video
            LEFT JOIN dets dg ON dg.source = 'gt' AND dg.video = u.video
            LEFT JOIN dets dt ON dt.source = $source AND dt.video = u.video
        ),
        unscored AS (
            -- Boxes past the other side's last row in a truncated clip.
            SELECT b.source = 'gt' AS is_gt, b.video
            FROM boxes b JOIN seq s USING (video)
            WHERE s.truncated AND ((b.source = 'gt' AND b.frame >= s.t_tracker) OR (b.source = $source AND b.frame >= s.t_gt))
        )
        SELECT (SELECT count(*) FROM boxes WHERE source = 'gt') AS gt_boxes,
               (SELECT count(*) FROM boxes WHERE source = $source) AS tracker_boxes,
               (SELECT count(*) FILTER (WHERE is_gt) FROM unscored) AS gt_boxes_not_scored,
               (SELECT count(*) FILTER (WHERE NOT is_gt) FROM unscored) AS tracker_boxes_not_scored,
               (SELECT count(DISTINCT video) FILTER (WHERE is_gt) FROM unscored) AS clips_with_gt_not_scored,
               (SELECT count(DISTINCT video) FILTER (WHERE NOT is_gt) FROM unscored) AS clips_with_tracker_not_scored,
               (SELECT count(*) FILTER (WHERE t_gt < n) FROM seq) AS gt_files_ending_before_clip_end,
               (SELECT count(*) FILTER (WHERE t_tracker < n) FROM seq) AS tracker_files_ending_before_clip_end
    """, {'source': source})[0]


def count_ceiling(con: duckdb.DuckDBPyConnection, source: str) -> dict:
    """The DetA bound any box placement of `source`'s per-frame counts obeys, on full clips (frames < n).

    On each frame at most min(p, g) boxes match, so DetA <= M / (G + P - M) with M = sum min(p, g); DetRe <= M / G,
    DetPr <= M / P, and HOTA <= sqrt(DetA bound) with perfect association.
    """
    # The three sums over the clip frames of the source and the ground truth.
    row = query_dicts(con, """
        WITH counts AS (
            -- Boxes per clip frame of the tracks and of the ground truth.
            SELECT b.video, b.frame, count(*) FILTER (b.source = $source) AS p, count(*) FILTER (b.source = 'gt') AS g
            FROM boxes b JOIN unwrap u USING (video)
            WHERE b.frame < u.n AND b.source IN ($source, 'gt') GROUP BY ALL
        )
        SELECT coalesce(sum(p), 0) AS tracker_boxes, coalesce(sum(g), 0) AS gt_boxes,
               coalesce(sum(least(p, g)), 0) AS matchable FROM counts
    """, {'source': source})[0]
    # The bounds from the three sums (None when a denominator is zero).
    m, g, p = row['matchable'], row['gt_boxes'], row['tracker_boxes']
    deta = m / (g + p - m) if g + p - m else None
    return dict(row, deta=deta, detre=m / g if g else None, detpr=m / p if p else None,
                hota_perfect_association=math.sqrt(deta) if deta is not None else None)


def time_base(con: duckdb.DuckDBPyConnection) -> dict:
    """Whether LEAP's frame f is the clip's frame f: export and interval lengths against the clip and the ground truth,
    and the median per-clip correlation of LEAP's and the ground truth's per-frame counts over TIME_SCALES.

    LEAP's count on its frame f >= 0 is the number of exported objects whose interval holds f (the raw export without
    the wrapped tails, past the clip included). Clip frame t reads LEAP frame round(t / s); a clip where either side is
    constant has no correlation and is left out of the median. The best scale has the highest median (equal to
    TIME_SCALE_TIE_DECIMALS), ties to the scale nearest 1, then the smaller one; the dataset is flagged when the best
    scale is not 1 and gains more than TIME_SCALE_WARN_GAIN over scale 1.
    """
    # Per-scale median correlations over the clips.
    con.execute("""
        CREATE OR REPLACE TABLE time_scales AS
        WITH leap AS (
            -- LEAP's count per frame of its own time line, wrapped tails left out.
            SELECT e.video, e.frame, count(*) AS p FROM export_obs e JOIN objects o USING (video, track_id)
            WHERE NOT (o.wrapped AND e.frame >= o.tail_start) GROUP BY ALL
        ),
        gt AS (
            -- The ground truth's count per clip frame (ids are unique per frame).
            SELECT b.video, b.frame, count(*) AS g FROM boxes b JOIN unwrap u USING (video)
            WHERE b.source = 'gt' AND b.frame < u.n GROUP BY ALL
        ),
        pairs AS (
            -- Every clip frame under every scale, with the LEAP frame it reads (0 where either side has no box).
            SELECT s.scale, f.video, coalesce(g.g, 0) AS g, coalesce(l.p, 0) AS p
            FROM (SELECT unnest($scales) AS scale) s
            CROSS JOIN (SELECT video, unnest(range(n)) AS frame FROM unwrap) f
            LEFT JOIN gt g ON g.video = f.video AND g.frame = f.frame
            LEFT JOIN leap l ON l.video = f.video AND l.frame = CAST(round(f.frame / s.scale) AS BIGINT)
        ),
        per_clip AS (
            SELECT scale, video, CASE WHEN var_pop(p) > 0 AND var_pop(g) > 0 THEN corr(p, g) END AS r
            FROM pairs GROUP BY ALL
        )
        SELECT scale, median(r) AS median_r, count(r) AS clips_with_r FROM per_clip GROUP BY scale
    """, {'scales': list(TIME_SCALES)})
    # The per-scale rows, in scale order, for results.json.
    scales = query_dicts(con, 'SELECT * FROM time_scales ORDER BY scale')
    # The best scale (none when no clip has a correlation) and the one-to-one reading.
    best = query_dicts(con, """
        SELECT scale, median_r FROM time_scales WHERE median_r IS NOT NULL
        ORDER BY round(median_r, $decimals) DESC, abs(scale - 1), scale LIMIT 1
    """, {'decimals': TIME_SCALE_TIE_DECIMALS})
    one = query_dicts(con, 'SELECT median_r FROM time_scales WHERE scale = 1.0')[0]['median_r']
    # The best scale and its correlation, or None for both.
    best_scale, best_r = (best[0]['scale'], best[0]['median_r']) if best else (None, None)
    # Lengths: each non-empty export over its clip, and LEAP's intervals against the ground truth's tracks.
    lengths = query_dicts(con, """
        SELECT (SELECT median(x.L / u.n) FROM export_len x JOIN unwrap u USING (video) WHERE x.L > 0) AS export_over_clip_median,
               (SELECT min(x.L / u.n) FROM export_len x JOIN unwrap u USING (video) WHERE x.L > 0) AS export_over_clip_min,
               (SELECT max(x.L / u.n) FROM export_len x JOIN unwrap u USING (video) WHERE x.L > 0) AS export_over_clip_max,
               (SELECT median(interval_end - interval_start) FROM objects WHERE NOT ambiguous) AS median_interval_frames,
               (SELECT median(t) FROM (SELECT max(frame) - min(frame) + 1 AS t FROM boxes b JOIN unwrap u USING (video)
                                       WHERE b.source = 'gt' AND b.frame < u.n GROUP BY b.video, b.track_id))
                   AS median_gt_track_frames
    """)[0]
    # The interval-to-track ratio, the scale readings, and the flag.
    ratio = (lengths['median_interval_frames'] / lengths['median_gt_track_frames']
             if lengths['median_interval_frames'] is not None and lengths['median_gt_track_frames'] else None)
    flagged = best_scale is not None and best_scale != 1.0 and (one is None or best_r - one > TIME_SCALE_WARN_GAIN)
    return dict(lengths, interval_over_gt_track=ratio, scales=scales, r_at_scale_1=one, best_scale=best_scale,
                best_r=best_r, flagged=flagged)


def cached_accuracy(con: duckdb.DuckDBPyConnection, dataset: str) -> tuple[dict, list[dict]]:
    """LEAP's p141 accuracy in each cache (metric_summary of DATASET.json) and file records of the files read.

    Each cache's p142 tradeoff.csv must carry the same HOTA, AssA, and DetA.
    """
    # Summaries by cache, and the provenance of every file read.
    out, records = {}, []
    for source, directory in CACHES.items():
        # p141's combined result and p142's table for this cache.
        accuracy = cache.sota(directory, dataset, 'accuracy', 'raw', 'test', LEAP_PARAM, 'DATASET.json')
        tradeoff = cache.sota(directory, dataset, 'tradeoff.csv')
        assert_files_exist([accuracy, tradeoff], f'{dataset} {source} LEAP accuracy files')
        records += [file_record(accuracy), file_record(tradeoff)]
        # The combined metrics, summarized like a rerun's.
        with open(accuracy) as f:
            out[source] = metric_summary(json.load(f)['metrics'])
        # The tradeoff row p200-p205 plot must match the JSON it was built from.
        table = query_dicts(con, 'SELECT HOTA_HOTA, HOTA_AssA, HOTA_DetA FROM read_csv(?)', [str(tradeoff)])
        assert len(table) == 1, f'{tradeoff}: expected one LEAP row, found {len(table)}'
        # Each of the three plotted columns against its DATASET.json mean.
        for column, field in (('HOTA_HOTA', 'HOTA'), ('HOTA_AssA', 'AssA'), ('HOTA_DetA', 'DetA')):
            assert abs(table[0][column] - out[source]['HOTA'][field]) <= REPRODUCTION_TOLERANCE, \
                f'{tradeoff}: {column} {table[0][column]} differs from DATASET.json {out[source]["HOTA"][field]}'
    return out, records


def check_scored_window(window: dict, accuracy: dict, source: str):
    """The scored_window counts must explain DATASET.json's Count and loosest-alpha HOTA totals to the box."""
    # Count counts every box; HOTA's TP + FN and TP + FP count only the scored ones.
    count, unscored = accuracy['Count'], unscored_boxes(accuracy)
    expected = dict(gt_boxes=count['GT_Dets'], tracker_boxes=count['Dets'], gt_boxes_not_scored=unscored['gt'],
                    tracker_boxes_not_scored=unscored['tracker'])
    wrong = {k: (window[k], v) for k, v in expected.items() if window[k] != v}
    assert not wrong, f'{source}: scored window (this script, DATASET.json) disagrees: {wrong}'


def build_variant_lines(con: duckdb.DuckDBPyConnection, source: str, drop_ambiguous: bool, full_length: bool):
    """Table variant_lines(video, frame_idx, tracks) of `source`'s lines, without ambiguous objects or cut to full clips.

    Full clips keep frames < n and add an empty row on frame n - 1 when the file has none, so TrackEval gives the file
    the clip's length. Boxes stay JSON text, so a rewritten file keeps every digit.
    """
    # The source's lines, filtered and cut as the variant asks.
    con.execute("""
        CREATE OR REPLACE TABLE variant_lines AS
        SELECT l.video, l.frame AS frame_idx,
               CASE WHEN $drop THEN list_filter(l.tracks, lambda b: NOT list_contains(
                        u.ambiguous_full_span_objects, CAST(CAST(b->>0 AS DOUBLE) AS BIGINT)))
                    ELSE l.tracks END AS tracks
        FROM lines l JOIN unwrap u USING (video)
        WHERE l.source = $source AND (NOT $full OR l.frame < u.n)
        UNION ALL
        -- The padding row on the clip's last frame.
        SELECT u.video, u.n - 1, []::JSON[] FROM unwrap u
        WHERE $full AND NOT EXISTS (SELECT 1 FROM lines l WHERE l.source = $source AND l.video = u.video AND l.frame = u.n - 1)
    """, {'source': source, 'drop': drop_ambiguous, 'full': full_length})


def copy_variant_lines(con: duckdb.DuckDBPyConnection, directory: Path, videos: list[str]):
    """Write table variant_lines as <directory>/<video>/tracking.jsonl, one file per clip (empty when it has no line)."""
    # One COPY per clip file (a loop over files, not data rows).
    for video in videos:
        # The clip's file and its directory.
        path = directory / video / REWRITTEN_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        # The names are embedded as SQL literals.
        assert "'" not in video and "'" not in str(path), f'unsupported path {path}'
        con.execute(f"COPY (SELECT frame_idx, tracks FROM variant_lines WHERE video = '{video}' ORDER BY frame_idx) "
                    f"TO '{path}' (FORMAT JSON)")


def variant_tasks(con: duckdb.DuckDBPyConnection, dataset: str, videos: list[str], work_dir: Path) -> list[dict]:
    """One TrackEval task per VARIANTS entry, writing the rewritten tracks and full-clip ground truth under work_dir."""
    # Cached files are scored where they are, as p141 does.
    base = work_dir / dataset
    gt_cached = (str(cache.execution(dataset_root_name(dataset))), str(GT_FILE))
    # The full-clip ground truth is written once for both full-clip variants.
    gt_full = (str(base / 'ground_truth_full_length'), str(REWRITTEN_FILE))
    build_variant_lines(con, 'gt', drop_ambiguous=False, full_length=True)
    copy_variant_lines(con, Path(gt_full[0]), videos)
    # One task per variant (five, not data rows).
    tasks = []
    for name, spec in VARIANTS.items():
        # Rewrite the tracks only when the variant changes them.
        if spec['drop_ambiguous'] or spec['full_length']:
            build_variant_lines(con, spec['cache'], spec['drop_ambiguous'], spec['full_length'])
            copy_variant_lines(con, base / name, videos)
            tracks = (str(base / name), str(REWRITTEN_FILE))
        else:
            tracks = (str(cache.sota(CACHES[spec['cache']], dataset)), str(TRACK_FILE))
        # Full-clip variants score against the full-clip ground truth, the others against the cached one.
        gt_dir, gt_file = gt_full if spec['full_length'] else gt_cached
        tasks.append(dict(dataset=dataset, variant=name, videos=sorted(videos), gt_dir=gt_dir, gt_file=gt_file,
                          track_dir=tracks[0], track_file=tracks[1], output_dir=str(base / 'trackeval' / name)))
    return tasks


def trackeval_combined(task: dict) -> dict:
    """TrackEval's combined HOTA, CLEAR, Identity, and Count results over a task's clips, configured as p141
    configures it (raw arrays and scalars). TrackEval writes nothing but an error log into output_dir."""
    # The shared rerun with the three families this script reports.
    return trackeval_rerun(task['gt_dir'], task['gt_file'], task['track_dir'], task['track_file'], task['videos'],
                           task['output_dir'], ('HOTA', 'CLEAR', 'Identity'), pad_scoring_window=False)


def run_trackeval(task: dict) -> dict:
    """The task with metric_summary of its TrackEval rerun (a top-level function, so worker processes can run it)."""
    # Rerun, then summarize like DATASET.json.
    return dict(task, metrics=metric_summary(trackeval_combined(task)))


def same_metrics(a: dict, b: dict, families: tuple[str, ...] = ('HOTA', 'Count')) -> list[str]:
    """Fields of the given families (loosest-alpha values included) that are missing on either side or differ by
    more than the tolerance, sorted."""
    # Names of the differing fields, family by family (a handful of fields, not data rows).
    differing = []
    for family in families:
        # Nested alpha_0.05 values are compared like the means.
        flat_a = {**a[family], **{f'alpha_0.05.{k}': v for k, v in a[family].get('alpha_0.05', {}).items()}}
        flat_b = {**b[family], **{f'alpha_0.05.{k}': v for k, v in b[family].get('alpha_0.05', {}).items()}}
        # Every field of either side; one missing from a side differs.
        differing += [f'{family}.{k}' for k in sorted((set(flat_a) | set(flat_b)) - {'alpha_0.05'})
                      if k not in flat_a or k not in flat_b or abs(flat_a[k] - flat_b[k]) > REPRODUCTION_TOLERANCE]
    return differing


def check_trackeval(section: dict, accuracy: dict):
    """Consistency of the reruns: cached variants reproduce DATASET.json, dropping no ambiguous object changes nothing,
    full-clip reruns score exactly count_ceiling's boxes, and no DetA exceeds its ceiling."""
    # Every rerun of this dataset by variant name.
    runs = section['trackeval']
    # The reruns of the cached files must reproduce p141's results.
    for source in CACHES:
        differing = same_metrics(runs[source], accuracy[source])
        assert not differing, f"{section['dataset']} {source}: TrackEval rerun differs from DATASET.json in {differing[:5]}"
    # Without ambiguous objects the dropped readings are the same files' boxes.
    if section['unwrap']['ambiguous_full_span_objects'] == 0:
        for full in ('', '_full_length'):
            differing = same_metrics(runs[f'after_ambiguous_dropped{full}'], runs[f'after{full}'], ('HOTA', 'CLEAR', 'Identity', 'Count'))
            assert not differing, f"{section['dataset']}: dropping no object changed {differing[:5]}"
    # Full-clip reruns count every box inside the clip, as count_ceiling does, and stay under its bound.
    for name, source in (('after_full_length', 'after'), ('after_ambiguous_dropped_full_length', 'after_ambiguous_dropped')):
        count, ceiling = runs[name]['Count'], section['count_ceiling'][source]
        # The same boxes on both sides of the comparison.
        assert (count['Dets'], count['GT_Dets']) == (ceiling['tracker_boxes'], ceiling['gt_boxes']), \
            f"{section['dataset']} {name}: Count {count} differs from the clip boxes {ceiling}"
        # No placement of these boxes can beat the count bound.
        if ceiling['deta'] is not None:
            assert runs[name]['HOTA']['DetA'] <= ceiling['deta'] + REPRODUCTION_TOLERANCE, \
                f"{section['dataset']} {name}: DetA {runs[name]['HOTA']['DetA']} exceeds its ceiling {ceiling['deta']}"


def load_native(native_dir: Path) -> tuple[dict, dict, list[dict]]:
    """r1o2_leap_native_queries' configs.json and diagnostics.json, and their file records."""
    # Both files are needed; they come from a separate measurement run.
    paths = [native_dir / 'configs.json', native_dir / 'diagnostics.json']
    assert_files_exist(paths, 'r1o2_leap_native_queries outputs (run measurements/r1o2_leap_native_queries.py first)')
    # Read both small JSON files whole.
    with open(paths[0]) as f:
        configs = json.load(f)
    with open(paths[1]) as f:
        diagnostics = json.load(f)
    # Both outputs and where they came from.
    return configs, diagnostics, [file_record(p) for p in paths]


def native_summary(configs: dict, diagnostics: dict, label: str, section: dict) -> dict:
    """Per-frame count agreement of LEAP (with and without ambiguous objects) and the reference, from the native-queries
    outputs, after checking they were computed on the same cache: same unwrap totals, frames, and mean counts."""
    # The dataset must have been scored by the native-queries run.
    dataset, diag = configs['datasets'].get(label), diagnostics.get(label)
    assert dataset is not None and diag is not None, f'{label} is missing from the r1o2_leap_native_queries outputs; rerun it'
    # Its LEAP statistics must be this cache's.
    unwrap = section['unwrap']
    differing = {k: (v, unwrap.get(k)) for k, v in diag['leap_unwrap'].items() if unwrap.get(k) != v}
    assert not differing, f'{label}: native-queries unwrap totals differ (native, here): {differing}'
    assert dataset['frames'] == unwrap['clip_frames'], f"{label}: native-queries frames {dataset['frames']} != {unwrap['clip_frames']}"
    # Per ground-truth variant, the reused per-frame count readings.
    counts = {}
    # Two ground-truth variants with about a hundred records each.
    for variant, records in dataset['records'].items():
        # This variant's records of the three systems, which must all be there.
        chosen = {r['system']: r for r in records if r['system'] in NATIVE_SYSTEMS}
        assert set(chosen) == set(NATIVE_SYSTEMS), f'{label} {variant}: native-queries systems {sorted(chosen)}'
        counts[variant] = dict(
            ground_truth_mean_count=dataset['ground_truth_summary'][variant]['mean_count'],
            systems={s: dict(hota=r['hota'], mean_predicted=r['aggregation']['mean_predicted'], mae=r['aggregation']['mae'],
                             acc_leap=r['aggregation']['acc_leap'], selection_precision=r['selection']['precision'],
                             selection_recall=r['selection']['recall'], selection_f1=r['selection']['f1'])
                     for s, r in chosen.items()})
    # The mean per-frame count of each LEAP reading must match this script's boxes inside the clip.
    for system, source in NATIVE_BOX_SOURCE.items():
        mean = section['box_frames'][source]['boxes_in_clip'] / unwrap['clip_frames']
        reported = counts['all']['systems'][system]['mean_predicted']
        assert abs(reported - mean) <= NATIVE_ROUNDING, f'{label} {system}: native mean count {reported}, here {mean}'
    # The reused counts, LEAP's empty-export rescoring and lag correlations, and the native run's own cache check.
    return dict(ground_truth=dataset['ground_truth'], per_frame_counts=counts,
                leap_excluding_empty_exports=diag['leap_excluding_empty_exports'],
                lag_correlations=[r for r in diag['lag_correlations'] if r['system'] in NATIVE_BOX_SOURCE],
                leap_cache_cross_check_equal=diag['leap_cache_cross_check']['equal'])


def deta_decomposition(deta: float | None, ceiling: float | None) -> dict:
    """LEAP's DetA loss (1 - DetA) split into what the count ceiling removes (1 - ceiling) and what placement loses
    (ceiling - DetA), in absolute DetA and as shares of the loss, and in log terms (-ln DetA = -ln ceiling +
    ln(ceiling / DetA)). Placement covers the exporter's pattern placement, LEAP's pattern choice, and any time-base
    mismatch together. Shares are None where undefined (no loss, or a DetA of 0 for the logs)."""
    # Without both numbers there is nothing to split.
    if deta is None or ceiling is None:
        return dict(deta=deta, ceiling=ceiling)
    # Absolute losses and their shares of the total.
    count_abs, placement_abs, total = 1 - ceiling, ceiling - deta, 1 - deta
    out = dict(deta=deta, ceiling=ceiling, count_loss_abs=count_abs, placement_loss_abs=placement_abs,
               count_share_abs=count_abs / total if total > 0 else None,
               placement_share_abs=placement_abs / total if total > 0 else None)
    # Log losses, defined for a positive DetA below 1.
    log_total = -math.log(deta) if 0 < deta < 1 else None
    out.update(count_loss_log=-math.log(ceiling) if ceiling > 0 else None,
               placement_loss_log=math.log(ceiling / deta) if deta > 0 else None,
               count_share_log=-math.log(ceiling) / log_total if log_total else None,
               placement_share_log=math.log(ceiling / deta) / log_total if log_total else None)
    return out


def attribution(section: dict) -> dict:
    """The HOTA change of each measured conversion step, LEAP's full-clip DetA split against the count ceiling, and
    the counts LEAP's intervals imply, side by side. AssA is reported, not attributed."""
    # The reruns, ceilings, and box counts of the section, HOTA by variant, the full-clip DetA, and the reused counts.
    runs, ceiling, frames = section['trackeval'], section['count_ceiling'], section['box_frames']
    hota = {name: r['HOTA']['HOTA'] for name, r in runs.items()}
    deta = runs['after_full_length']['HOTA']['DetA']
    native = section['native_queries']['per_frame_counts']['all']
    # Every reading, and whether a time-base mismatch may be part of the placement loss.
    return dict(
        reference_hota=native['systems']['reference']['hota'],
        leap_hota=hota,
        # The p140 fix, as cached.
        fix_effect=hota['after'] - hota['before'],
        # Dropping the ambiguous objects (both ends unknown), on the cached and on full clips.
        ambiguous_dropped_effect=hota['after_ambiguous_dropped'] - hota['after'],
        ambiguous_dropped_effect_full_length=hota['after_ambiguous_dropped_full_length'] - hota['after_full_length'],
        # Scoring every box inside the clip instead of p141's truncated window.
        full_length_effect=hota['after_full_length'] - hota['after'],
        # How far any box placement could lift detection with these intervals, and where the rest is lost.
        deta_full_length=deta, deta_ceiling=ceiling['after']['deta'],
        deta_share_of_ceiling=deta / ceiling['after']['deta'] if ceiling['after']['deta'] else None,
        deta_decomposition=deta_decomposition(deta, ceiling['after']['deta']),
        hota_ceiling_perfect_association=ceiling['after']['hota_perfect_association'],
        # Association is not bounded by counts, so it is not split.
        assa_full_length=runs['after_full_length']['HOTA']['AssA'],
        assa_attributed=False,
        time_base_flagged=section['time_base']['flagged'],
        # LEAP's output against the ground truth's, independent of box placement.
        objects_vs_ground_truth=frames['after']['objects_in_clip'] / frames['gt']['objects_in_clip'],
        box_frames_vs_ground_truth=frames['after']['boxes_in_clip'] / frames['gt']['boxes_in_clip'],
        per_frame_count_mae=native['systems']['leap']['mae'],
        per_frame_count_acc_leap=native['systems']['leap']['acc_leap'],
    )


def summary(sections: dict) -> dict:
    """Ranges over the datasets of the readings the findings quote, and which datasets each reading applies to."""
    # Each dataset's attribution (seven datasets, not data rows).
    a = {label: s['attribution'] for label, s in sections.items()}
    d = {label: x['deta_decomposition'] for label, x in a.items()}
    # Every measured conversion step's HOTA change, and LEAP's HOTA under every reading.
    effects = [x[k] for x in a.values() for k in ('fix_effect', 'ambiguous_dropped_effect',
                                                   'ambiguous_dropped_effect_full_length', 'full_length_effect')]
    hotas = [h for x in a.values() for h in x['leap_hota'].values()]
    shares = [x['deta_share_of_ceiling'] for x in a.values() if x['deta_share_of_ceiling'] is not None]
    # The ranges, and the datasets each reading applies to (shares of one loss sum to 1, so > 0.5 is the larger part).
    return dict(
        datasets=list(sections),
        measured_conversion_effect_range=value_range(effects), leap_hota_range=value_range(hotas),
        deta_share_of_ceiling_range=value_range(shares),
        assa_full_length_range=value_range([x['assa_full_length'] for x in a.values()]),
        # Where the count ceiling removes more DetA than placement in absolute DetA, and where placement does in logs.
        count_loss_larger_absolute=[k for k, x in d.items() if (x.get('count_share_abs') or 0) > 0.5],
        placement_loss_larger_log=[k for k, x in d.items() if (x.get('placement_share_log') or 0) > 0.5],
        time_base_flagged=[k for k, s in sections.items() if s['time_base']['flagged']],
        b3d_provisional=[k for k, s in sections.items() if s['dataset'].startswith('jnc')],
    )


def value_range(values: list) -> list | None:
    """[min, max] of a list, or None when it is empty."""
    # An empty list has no range.
    return [min(values), max(values)] if values else None


def analyze_dataset(con: duckdb.DuckDBPyConnection, dataset: str, sota_dir: Path, work_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Everything but the TrackEval reruns for one dataset; also returns the rerun tasks and the file records read."""
    # The test clips in teNN order, and every input of each.
    videos = list_split_videos(dataset, 'test')
    assert videos, f'{dataset}: no test videos'
    register_files(con, input_files(dataset, videos, sota_dir))
    # Load the statistics, tracks, and raw exports, and recover every exported interval.
    load_inputs(con)
    # The recovery and p140 must agree on every clip before either is reported.
    mismatches = unwrap_mismatches(con)
    assert not mismatches, f'{dataset}: raw-export recovery disagrees with unwrap.json on {len(mismatches)} clips: {mismatches[:3]}'
    # p141's cached accuracy, and the boxes its scoring window leaves out, checked against each other.
    accuracy, records = cached_accuracy(con, dataset)
    window = {source: scored_window(con, source) for source in CACHES}
    for source in CACHES:
        check_scored_window(window[source], accuracy[source], source)
    # The interval and box summaries, and everything else the section reports.
    intervals = interval_summary(con)
    frames = box_frames(con)
    section = dict(dataset=dataset, clips=len(videos), accuracy=accuracy,
                   unwrap=dict(unwrap_totals(con), **unwrap_lengths(con)), intervals=intervals, box_frames=frames,
                   box_changes=box_changes(frames, intervals), scored_window=window,
                   count_ceiling={s: count_ceiling(con, s) for s in ('after', 'after_ambiguous_dropped')},
                   time_base=time_base(con), input_digests=input_digests(con))
    # The rerun tasks write their files under work_dir.
    return section, variant_tasks(con, dataset, videos, work_dir), records


def method(args: argparse.Namespace) -> dict:
    """Inputs, rules, and variants, for results.json."""
    # Plain-text descriptions, keyed by the section they describe.
    return dict(
        chain=('LEAP results.json/tracks.json -> export-leap-outputs.py (pattern resampled, repeated, or truncated over '
               '[start, end); negative starts wrap to the list end) -> p140 (since 2026-10-26: wrapped tail dropped, '
               'full-span objects kept whole as ambiguous, frames >= clip length dropped, unwrap.json) -> p141 '
               '(TrackEval vs root 003_groundtruth) -> p142 tradeoff.csv'),
        caches={k: str(cache.sota(v, '<dataset>')) for k, v in CACHES.items()},
        raw_exports=f'{args.sota_dir}/<dataset in SOTA>/leap/<video id>.json',
        clip_length="unwrap.json clip_frames (p140's video frame count)",
        interval_recovery=('one run [a, b] -> [a, b + 1); head run from 0 plus tail run ending at L - 1 -> '
                           '[tail_start - L, head_end + 1); one run over the whole export -> ambiguous; checked per clip '
                           "against unwrap.json's ids and box counts"),
        clipped_share=('start-clipped = wrapped; end-clipped = a known interval end > clip length n (never ambiguous); '
                       '*_upper adds the ambiguous objects that may be clipped at that end (any at the start, those of '
                       'exports longer than n at the end); either end: lower adds the ambiguous objects of exports '
                       'longer than n (clipped at one end for certain), upper adds every ambiguous object'),
        box_changes=('surplus of the old cache over the fixed one over the whole file and inside the clip (frames < n); '
                     'gap_fill = old cache boxes - raw export boxes, asserted equal to the frames between each '
                     "wrapped object's head and tail"),
        time_base=(f"assumed: LEAP frame f = clip frame f (LEAP configs fps 15, clips 15 fps); checked by export "
                   f"length / clip length, LEAP's median interval / the ground truth's median track length, and the "
                   f"median per-clip Pearson correlation of LEAP's per-frame interval counts (LEAP frames >= 0) with "
                   f"the ground truth's, clip frame t reading LEAP frame round(t / s), s in {TIME_SCALES[0]}-"
                   f"{TIME_SCALES[-1]} by 0.05; flagged when the best s is not 1 and gains more than "
                   f"{TIME_SCALE_WARN_GAIN}; shifts at s = 1 are the native-queries lags (native_queries.lag_correlations)"),
        attribution=('measured steps: fix_effect, ambiguous_dropped_effect(_full_length), full_length_effect; '
                     'deta_decomposition: 1 - DetA = (1 - ceiling) + (ceiling - DetA) on full clips, and '
                     '-ln DetA = -ln ceiling + ln(ceiling / DetA); the placement part mixes the exporter, LEAP\'s '
                     'pattern choice, and any time-base mismatch; AssA not attributed'),
        input_digests='per source: file count, bytes, latest mtime, SHA-256 over sorted "video<TAB>file SHA-256" lines',
        scored_window=('a file is as long as its largest frame_idx + 1; with boxes on both sides a clip is scored on '
                       'min(T_gt, T_tracker) frames; checked against DATASET.json at alpha 0.05'),
        trackeval=dict(metrics='HOTA, CLEAR, Identity (threshold 0.5), Count, configured as p141', variants=VARIANTS,
                       full_length='tracks and ground truth cut to frames < n and padded with an empty row on frame n - 1',
                       output='temporary directory, removed after the run; only summaries are kept'),
        hota_summary=('means over the 19 alphas as p142 computes HOTA_HOTA; alpha_0.05 holds the loosest threshold; '
                      'HOTA_TP/FN/FP only at alpha 0.05'),
        count_ceiling=('M = sum over clip frames of min(p, g); DetA <= M / (G + P - M) for any box placement and any '
                       'alpha; HOTA <= sqrt(DetA bound)'),
        native_queries=f'{args.native_dir} (configs.json, diagnostics.json), cross-checked, not recomputed',
        rounding='4 decimals when written',
    )


def results_name(datasets: list[str]) -> str:
    """results.json for a run over every dataset, results-partial.json otherwise, so a partial run never replaces it."""
    # Every dataset selected (in any order) is a full run.
    return 'results.json' if set(datasets) == set(DATASET_LABELS) else 'results-partial.json'


def main():
    # Command-line options.
    args = parse_args()
    # Results go to measurements-results/r1o2_leap_hota_conversion/; the native-queries outputs are read first.
    out_dir = results_dir(__file__)
    configs, diagnostics, records = load_native(args.native_dir)
    # Per-dataset sections and the TrackEval tasks of every dataset.
    sections, tasks = {}, []
    with tempfile.TemporaryDirectory(prefix='r1o2_leap_hota_conversion_') as tmp:
        # DuckDB analysis per dataset; the connection is closed before TrackEval's workers fork.
        con = duckdb.connect()
        for dataset in args.datasets:
            # The dataset's section, its rerun tasks (files written under tmp), and the files it read.
            label = DATASET_LABELS[dataset]
            sections[label], dataset_tasks, dataset_records = analyze_dataset(con, dataset, args.sota_dir, Path(tmp))
            tasks += dataset_tasks
            records += dataset_records
            print(f"{label}: {sections[label]['clips']} clips, {sections[label]['intervals']['objects']} exported objects, "
                  f"{len(dataset_tasks)} TrackEval variants queued", flush=True)
        con.close()
        # Every TrackEval rerun in parallel; each is independent.
        with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
            reruns = pool.map(run_trackeval, tasks)
    # Attach each rerun to its dataset.
    labels = {dataset: DATASET_LABELS[dataset] for dataset in args.datasets}
    for run in reruns:
        sections[labels[run['dataset']]].setdefault('trackeval', {})[run['variant']] = run['metrics']
    # Checks, the native-queries reuse, and the attribution, per dataset.
    for label, section in sections.items():
        check_trackeval(section, section['accuracy'])
        section['native_queries'] = native_summary(configs, diagnostics, label, section)
        section['attribution'] = attribution(section)
        # One summary line per dataset.
        a = section['attribution']
        print(f"{label}: HOTA before {a['leap_hota']['before']:.4f} after {a['leap_hota']['after']:.4f} "
              f"(fix {a['fix_effect']:+.4f}), ambiguous dropped {a['ambiguous_dropped_effect']:+.4f}, "
              f"full clips {a['full_length_effect']:+.4f}; DetA {a['deta_full_length']:.4f} of ceiling "
              f"{a['deta_ceiling']:.4f}; objects {a['objects_vs_ground_truth']:.2f}x ground truth; "
              f"clipped objects {section['intervals']['clipped_either_end_share_lower']:.1%}-"
              f"{section['intervals']['clipped_either_end_share_upper']:.1%}; time scale best "
              f"{section['time_base']['best_scale']} ({'flagged' if section['time_base']['flagged'] else 'not flagged'})")
    # One rounded results file, named by whether every dataset was run.
    write_json(out_dir, results_name(args.datasets),
               rounded(dict(method=method(args), blockers=BLOCKERS, findings=FINDINGS, summary=summary(sections),
                            inputs=records, datasets=sections)))


if __name__ == '__main__':
    main()
