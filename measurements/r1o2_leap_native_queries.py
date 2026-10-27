#!/usr/local/bin/python
"""R1.O2: LEAP's native queries (selection, aggregation, top-k), scored the way LEAP's evaluator scores them.

LEAP's evaluator (LEAP/evaluate.py, class `evaluation`) turns ground truth and predictions into
one object count per frame, then scores three queries over the frames [start_frame, end_frame):

  selection_query_1 (Query-1)
    A frame is positive when its count is at least 1. F1 over frames. The code forces TP from 0
    to 1 and rounds recall and precision to 4 decimals before F1; that value is kept as
    `f1_leap_code`, while `f1` is the plain F1 (null when no frame is positive on either side).

  aggregation_query_1 (Query-2)
    Per frame, ACC_f = 1 - |p - g| / g when g > 0, 1 when g = p = 0, and 0 when g = 0 < p,
    averaged over frames; MAE is the mean |p - g|. With start 0 and end N (no padding),
    LEAP's ACC is exactly `acc_leap` = 1 - `mape_leap` here. `mape_pos` (frames with g > 0)
    and `zero_fp_rate` (share of g = 0 frames with p > 0) are extra columns.

  top_k_query_1 (Query-5)
    LEAP's paper asks for the top-10 frames with the most objects. The code clusters the 5000
    highest-count frames with a radius of int(0.25 * 300) = 75 frames, values the prediction's
    clusters with ground-truth counts, and scores the multiset overlap of the 10 largest cluster
    values. `topk_leap_code` transcribes that code with a stable argsort, and
    `topk_leap_code_default_sort` with numpy's default argsort, as the code is written.
    `topk_paper` scores the paper's query: a system greedily returns its 10 highest-count frames
    at least 75 frames apart, never a frame where it sees no object; tau is the smallest
    ground-truth count among the ground truth's 10 such frames; precision is the share of the
    10 returned frames whose ground-truth count reaches tau (the denominator stays 10). Counts
    are small integers, so many frames tie: `topk_paper` is the mean precision over
    TIE_DRAWS uniformly random tie orders (min and max kept), and `topk_paper_pooled_order` the
    earlier deterministic score (ties by pooled order, frames without objects returnable).

Evaluation unit: a dataset's test clips concatenated in teNN order (LEAP's paper evaluates one long
test video), with start 0 and end the pooled frame count, without padding to LEAP's YAML end_frame.
Queries are class-agnostic ("vehicle"), because no output or ground truth here carries a class
label (see blockers in configs.json); LEAP's per-class workload is therefore not reproduced.

Per-frame counts are the number of distinct track ids in each frame of a tracking.jsonl, frames
missing from the file count 0, and frames at or past the clip length are dropped and reported.
Systems: the reference pipeline (002_naive), every validated Tetris configuration, every OTIF
configuration, and LEAP, all read through common.output_path. LEAP's tracks are the corrected
cache (wrapped interval starts removed, ends clipped; see evaluation/p140_otif_transform.py).
Objects LEAP holds on every exported frame are ambiguous (their true end is unknown) and kept
whole as an upper bound; system `leap_ambiguous_dropped` removes them (the lower bound), with
LEAP's throughput and no HOTA of its own. The cache is cross-checked on one clip per dataset
against an interval recovery from LEAP's raw export in --sota-dir (diagnostics.json).

Constant answers (system `baseline`: 0 and 1 on every frame, and the ground truth's median count)
are scored like systems. An operating point whose bound a constant answer already meets is marked
`trivial`, since it cannot tell systems apart.

Ground truth: per dataset, <--gt-root>/<root dataset>/teNN.mp4/tracking.jsonl (for human-verified
tracks in the same format) when --gt-root holds every clip of that dataset, else the root dataset's
003_groundtruth tracks (common.output_path); a dataset only partly covered by --gt-root is an error.
Variant `all` (primary) counts every box, as LEAP's `pseudo_label`. Variant `moving` (sensitivity)
is LEAP's default `pseudo_tuple` built as LEAP/tools/data_prepare.py builds it: a track is kept
when last - first >= 15 frames (cfg fps 15 x 1 s) and its first and last box centers, computed
from int-truncated MOT left/top and width/height, are at least 50 pixels apart (traj_dist_min);
it counts on [first, last). B3D (jnc*) results on the automatic ground truth are provisional.

Throughput and HOTA come from common.system_configs. Operating points (common.operating_points)
use each query's paper-aligned accuracy (selection f1, aggregation acc_leap, top-k topk_paper) at
the --losses bounds relative to the reference pipeline and at `leap_fit`, LEAP's own accuracy per
dataset, for each ground-truth variant. A dataset whose reference accuracy is not positive has no
relative bound for that query; it is recorded under `skipped` and left out. Accuracies are rounded
only when written.

Writes measurements-results/r1o2_leap_native_queries/ (a run with --datasets rewrites all three
files with only those datasets):
  configs.json           method, inputs, blockers, and per dataset and ground-truth variant every
                         configuration's and constant answer's selection, aggregation, and top-k
                         scores with fps and HOTA
  operating_points.json  per ground-truth variant, query, and bound, each system's fastest
                         configuration within the bound, its speedup over the reference,
                         Tetris's speedup over OTIF and LEAP, and the trivial and
                         LEAP-ambiguity flags
  diagnostics.json       per dataset, frames dropped past the clip, LEAP's unwrap statistics, LEAP
                         rescored without its empty exports, lag correlations of counts against
                         ground truth, and the LEAP cache cross-check
"""
import argparse
import os
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

from evaluation.manifests import list_split_videos
from evaluation.p140_otif_transform import dataset_name_in_sota, extract_video_id
from measurements.common import (DATASET_LABELS, TRACK_COLUMNS, assert_files_exist, best_accuracy_bounds,
                                 operating_points, output_path, print_operating_points, query_dicts, results_dir,
                                 rounded, system_configs, write_json)
from polyis.io import cache
from polyis.utilities import dataset_root_name, get_video_frame_count


# LEAP's top-k query: k frames, gap 300, cluster radius int(0.25 * gap), 5000 candidate frames (evaluate.py).
TOPK = 10
TOPK_GAP = 300
TOPK_RADIUS = int(0.25 * TOPK_GAP)
TOPK_CANDIDATES = 5000
# Random tie orders averaged by topk_paper, and the seed of each configuration's generator.
TIE_DRAWS = 100
TIE_SEED = 0
# LEAP's pseudo_tuple filter: minimum track span in frames (cfg fps 15 x 1 s) and minimum travel in pixels.
MOVING_MIN_FRAMES = 15
MOVING_MIN_DISTANCE = 50
# Lag window, the correlation gain at a nonzero lag that triggers an alignment warning, and the
# decimals at which two lags' correlations count as tied.
MAX_LAG = 15
LAG_WARN_GAIN = 0.05
LAG_TIE_DECIMALS = 9
# Ground-truth variants: every box (primary) and LEAP's moving-track filter (sensitivity).
GT_VARIANTS = ('all', 'moving')
# Each query's paper-aligned accuracy, used for operating points.
QUERY_ACCURACY = {'selection': 'f1', 'aggregation': 'acc_leap', 'topk': 'topk_paper'}
# Systems held to each bound; Tetris is compared against the others.
POINT_SYSTEMS = ['tetris', 'otif', 'leap', 'leap_ambiguous_dropped']
# Schema of LEAP's unwrap.json, written next to each corrected LEAP tracking file by p140.
UNWRAP_COLUMNS = ("{'export_frames': 'BIGINT', 'clip_frames': 'BIGINT', 'objects': 'BIGINT', "
                  "'wrapped_objects': 'BIGINT[]', 'ambiguous_full_span_objects': 'BIGINT[]', "
                  "'wrapped_boxes_dropped': 'BIGINT', 'boxes_past_clip_dropped': 'BIGINT'}")
# One row (filename, frame_idx, box) per box of the tracking.jsonl files bound to the query's parameter.
TRACK_BOXES = ("(SELECT filename, frame_idx, unnest(tracks) AS box FROM read_json(?, filename = true, "
               f"format = 'newline_delimited', columns = {TRACK_COLUMNS}))")
# How each dataset's automatic ground truth was produced, for the report.
GT_SOURCES = {
    'caldot': "OTIF's single-class car YOLOv3 detections tracked with OC-SORT; trucks and buses have no separate labels",
    'ams': 'COCO YOLOv3 car detections tracked with OC-SORT, while the Tetris detector was trained on car, bus, and truck merged',
    'jnc': 'single-class RetinaNet on corner crops tracked with OC-SORT, the same detector family as Tetris and the reference',
}
# Work that cannot be done with the data available, and caveats on LEAP's runs, recorded in configs.json.
BLOCKERS = {
    'per_class_queries': (
        "LEAP's paper scores car, truck, and bus separately. No output or ground truth here has a class: the Tetris "
        "and reference detectors are single-class, tracking.jsonl rows are [track_id, x1, y1, x2, y2], OTIF labels "
        "every box car, LEAP's export labels every box car, and the ground truth has no truck or bus labels. Per-class "
        "queries need per-class ground truth, per-class Tetris and OTIF outputs (new GPU runs), and LEAP's native "
        "per-class results, which are not on ace. Only class-agnostic queries are scored."),
    'b3d_human_ground_truth': (
        'B3D (jnc*) is scored on automatic ground truth from the same detector family as Tetris and the reference, '
        'so its results are provisional until the human-verified tracks of R1.O1 are passed with --gt-root.'),
    'leap_run_caveats': [
        "LEAP's native results and pseudo-labels for these runs are not on ace (possibly on another host's local "
        "/data), so LEAP's own ground-truth variants, class ids, and the true ends of its ambiguous objects cannot "
        "be recovered.",
        'The detector, DETECT_CLASSES, and flags of the LEAP runs are unknown.',
        "B3D motion-pattern libraries may have been built from the 29-fps copies in LEAP/data/videos/jnc*, which "
        "would explain LEAP's long B3D intervals (about 3.5x the reference track length); unverified, and the "
        'libraries are not on ace.',
        "Whether LEAP's empty exports (diagnostics leap_unwrap.empty_export_videos) are genuine empty outputs or "
        'failed runs is unknown; diagnostics leap_excluding_empty_exports rescores LEAP without those clips.',
        "LEAP's runtime is inference_total - decode per video (evaluation/p140_otif_transform.py), so it excludes "
        'decoding and includes the JPEG writes of inference.py and the JSON writes of main.py.',
        "On B3D3 (jnc6), LEAP's reid stage takes 159.1 s of its 174.4 s inference total (/sota-results/jnc6/"
        "leap_jnc6.csv), against totals of 21.0-26.9 s on the other B3D cameras, so Tetris-over-LEAP ratios on "
        'B3D3 reflect that outlier.',
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: LEAP's native queries over every configuration")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--losses', nargs='+', type=float, default=[0.05, 0.10])
    parser.add_argument('--gt-root', type=Path, default=None)
    parser.add_argument('--sota-dir', type=Path, default=Path('/sota-results'))
    return parser.parse_args()


def selection_metrics(tp: int, fp: int, fn: int) -> dict:
    """Frame-presence precision, recall, and F1, plus LEAP's selection_query_1 F1 (`f1_leap_code`)."""
    # Plain metrics, null when their denominator is zero.
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # F1 = 2TP / (2TP + FP + FN): 0 when nothing matches, null when no frame is positive on either side.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    # LEAP's code: TP forced to at least 1, recall and precision rounded to 4 decimals, then F1 rounded.
    tp_code = max(tp, 1)
    recall_code = round(tp_code / (tp_code + fn), 4)
    precision_code = round(tp_code / (tp_code + fp), 4)
    f1_code = round(2 * (precision_code * recall_code) / (precision_code + recall_code), 4)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, f1_leap_code=f1_code)


def separated_top(values: np.ndarray, gid: np.ndarray, k: int = TOPK, separation: int = TOPK_RADIUS,
                  tiebreak: np.ndarray | None = None) -> np.ndarray:
    """Indices of up to k frames taken greedily by (value desc, tiebreak asc), each at least `separation` gids from the rest.

    tiebreak defaults to gid, so ties go to the earlier pooled frame. Repeatedly accepting the first
    remaining frame and removing its neighbors is the greedy scan, since a frame's eligibility only
    shrinks as frames are accepted.
    """
    # Rank frames by value (descending), then by the tie-break key (ascending).
    order = np.lexsort((gid if tiebreak is None else tiebreak, -values))
    # Global ids in rank order, and which ranked frames are still far enough from every accepted frame.
    ranked = gid[order]
    alive = np.ones(len(order), dtype=bool)
    chosen: list[int] = []
    # At most k vectorized passes over the frames.
    while len(chosen) < k and alive.any():
        # The best-ranked frame still eligible.
        first = int(np.argmax(alive))
        chosen.append(int(order[first]))
        # Remove every frame closer than the separation, including the accepted one.
        alive &= np.abs(ranked - ranked[first]) >= separation
    return np.array(chosen, dtype=np.int64)


def returnable(p: np.ndarray, k: int = TOPK, separation: int = TOPK_RADIUS) -> np.ndarray:
    """Indices of the frames a system can return under some tie order: count >= 1, cut to the highest counts.

    A frame where the system sees no object is not an answer to "the frames with the most objects",
    so it is never returned. Each accepted frame removes fewer than 2 * separation frames, so if the
    frames with count >= v number at least k * (2 * separation - 1), any scan accepts k of them before
    reaching a count below v; frames below the largest such v can never be returned.
    """
    # Frames with at least one object.
    positive = np.flatnonzero(p >= 1)
    # Frames that guarantee k separated answers in any order.
    need = k * (2 * separation - 1)
    # With fewer positive frames than that, every one of them may be returned.
    if len(positive) <= need:
        return positive
    # The need-th largest count: frames at or above it may be returned, frames below never.
    level = np.partition(p[positive], len(positive) - need)[len(positive) - need]
    return np.flatnonzero(p >= level)


def tie_draws(p: np.ndarray, gid: np.ndarray, draws: int = TIE_DRAWS, seed: int = TIE_SEED, k: int = TOPK) -> np.ndarray:
    """(draws, k) indices of the frames a system returns under `draws` uniformly random tie orders; -1 pads short answers.

    Each call seeds its own generator, so a configuration's draws depend only on its counts.
    """
    # Only returnable frames take part in the scan.
    candidates = returnable(p, k)
    # A generator of uniform tie-break keys, and the answers padded with -1.
    rng = np.random.default_rng(seed)
    out = np.full((draws, k), -1, dtype=np.int64)
    # One greedy scan per draw with fresh uniform tie-break keys (a loop over draws, each scan vectorized).
    for r in range(draws):
        chosen = candidates[separated_top(p[candidates], gid[candidates], k, tiebreak=rng.random(len(candidates)))]
        out[r, :len(chosen)] = chosen
    return out


def topk_tau(g: np.ndarray, gid: np.ndarray, k: int = TOPK) -> int:
    """tau: the smallest ground-truth count among the ground truth's k separated frames (ties by pooled order)."""
    # The ground truth's own separated frames in pooled order; score_topk records tau's range under random ties.
    return int(g[separated_top(g, gid, k)].min())


def topk_paper(draws: np.ndarray, pooled: np.ndarray, p: np.ndarray, g: np.ndarray, tau: int, k: int = TOPK) -> dict:
    """The paper's top-k precision against tau: mean, min, and max over the tie draws, and the pooled-order value.

    draws comes from tie_draws(p, gid); pooled is separated_top(p, gid), the earlier deterministic
    answer (ties by pooled order, frames without objects returnable).
    """
    # Per draw, the returned frames whose ground-truth count reaches tau, over k (padding never counts).
    hits = ((g[draws] >= tau) & (draws >= 0)).sum(axis=1) / k
    # Frames whose count equals the count of the last frame returned in pooled order (accepted and
    # suppressed frames included); a large number means the order among them decided the answer.
    ties = int((p == p[pooled[-1]]).sum()) if len(pooled) else 0
    return dict(topk_paper=float(hits.mean()), topk_paper_min=float(hits.min()), topk_paper_max=float(hits.max()),
                topk_paper_pooled_order=int((g[pooled] >= tau).sum()) / k, tau=tau, tau_zero=tau == 0,
                ties_at_cutoff=ties, returned_min=int((draws >= 0).sum(axis=1).min()))


def leap_clusters(values: np.ndarray, candidates: int = TOPK_CANDIDATES, radius: int = TOPK_RADIUS,
                  kind: str | None = 'stable') -> tuple[np.ndarray, np.ndarray]:
    """First and last frame of every cluster LEAP's _sort_max_intervals builds from the top `candidates` frames.

    The code visits frames by argsort()[::-1] (kind 'stable' puts the later frame first among ties;
    kind None is numpy's default sort, as LEAP calls it) and joins a frame to the first cluster, in
    creation order, that it falls strictly inside or strictly within `radius` of; otherwise it starts
    a cluster. Clusters never merge. The loop is sequential by construction; the per-frame test is
    vectorized over clusters.
    """
    # LEAP's candidate order under the requested sort.
    ids = values.argsort(kind=kind)[::-1][:candidates]
    # Cluster bounds, preallocated; m clusters exist so far.
    lo = np.empty(len(ids), dtype=np.int64)
    hi = np.empty(len(ids), dtype=np.int64)
    m = 0
    for frame in ids.tolist():
        # LEAP's three join conditions together: lo - radius < frame < hi + radius (frames are distinct).
        match = np.flatnonzero((lo[:m] - radius < frame) & (frame < hi[:m] + radius))
        if len(match):
            # Join the first matching cluster and widen it.
            c = match[0]
            lo[c], hi[c] = min(lo[c], frame), max(hi[c], frame)
        else:
            # Start a new single-frame cluster.
            lo[m] = hi[m] = frame
            m += 1
    return lo[:m], hi[:m]


def cluster_peaks(clusters: tuple[np.ndarray, np.ndarray], g: np.ndarray, k: int = TOPK) -> list[int]:
    """The k largest ground-truth maxima over the clusters' frame ranges (LEAP values every cluster with gt_result)."""
    # One range maximum per cluster (ranges may overlap, so no reduceat).
    peaks = [int(g[a:b + 1].max()) for a, b in zip(*clusters)]
    # Largest first, as max_set.sort() then [::-1][:k].
    return sorted(peaks, reverse=True)[:k]


def topk_leap_code(truth_clusters: tuple, pred_clusters: tuple, g: np.ndarray, k: int = TOPK) -> float:
    """LEAP's top_k_query_1: multiset overlap of the k largest cluster values of each side, over k."""
    # Both sides' clusters are valued with ground-truth counts, as in evaluate.py.
    overlap = Counter(cluster_peaks(truth_clusters, g, k)) & Counter(cluster_peaks(pred_clusters, g, k))
    return sum(overlap.values()) / k


def load_clips(con: duckdb.DuckDBPyConnection, dataset: str, videos: list[str]):
    """Tables clips(video, clip, n, pooled_offset) and frames(video, clip, frame, gidx, gid).

    n is the row count of the reference pipeline's detection.jsonl, which holds every decoded
    frame, checked against the video's frame count. gidx indexes the pooled sequence; gid adds a
    TOPK_RADIUS gap between clips so that frames of different clips are always separated.
    """
    # Reference detection files, one per clip.
    naive = [str(cache.exec(dataset, 'naive', v, 'detection.jsonl')) for v in videos]
    assert_files_exist(naive, f'{dataset} reference detection files')
    con.execute('CREATE OR REPLACE TABLE clip_files (video VARCHAR, clip BIGINT, filename VARCHAR, video_frames BIGINT)')
    # Clip order is teNN order; the video's own frame count comes from its container (per-file I/O).
    con.executemany('INSERT INTO clip_files VALUES (?, ?, ?, ?)',
                    [(v, c, f, get_video_frame_count(dataset, v)) for c, (v, f) in enumerate(zip(videos, naive))])
    # Count reference rows per clip and where the pooled sequence places each clip.
    con.execute("""
        CREATE OR REPLACE TABLE clips AS
        WITH rows AS (
            SELECT filename, count(*) AS n, count(DISTINCT frame_idx) AS distinct_frames,
                   min(frame_idx) AS first_frame, max(frame_idx) AS last_frame
            FROM read_json(?, filename = true, format = 'newline_delimited', columns = {'frame_idx': 'BIGINT'})
            GROUP BY filename
        )
        SELECT c.video, c.clip, r.n, c.video_frames, r.distinct_frames, r.first_frame, r.last_frame,
               coalesce(sum(r.n) OVER (ORDER BY c.clip ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS pooled_offset
        FROM clip_files c LEFT JOIN rows r USING (filename)
    """, [naive])
    # Each clip's reference rows must cover frames 0..n-1 once each, n being the video's frame count.
    bad = query_dicts(con, """
        SELECT video, n, video_frames FROM clips
        WHERE n IS NULL OR n <> video_frames OR distinct_frames <> n OR first_frame <> 0 OR last_frame <> n - 1
    """)
    assert not bad, f'{dataset}: reference frames do not match the videos: {bad}'
    create_frames(con)


def create_frames(con: duckdb.DuckDBPyConnection):
    """Table frames(video, clip, frame, gidx, gid) with one row per pooled frame of table clips(video, clip, n, pooled_offset)."""
    # Expand every clip into its frames, with pooled index and gap-separated global id.
    con.execute("""
        CREATE OR REPLACE TABLE frames AS
        SELECT video, clip, frame, pooled_offset + frame AS gidx, pooled_offset + $gap * clip + frame AS gid
        FROM (SELECT video, clip, pooled_offset, unnest(range(n)) AS frame FROM clips)
    """, {'gap': TOPK_RADIUS})


def drop_past_clip(con: duckdb.DuckDBPyConnection, table: str, keys: str) -> list[dict]:
    """Delete rows of `table` at or past their clip's length; return how many (video, frame) rows went, per `keys`."""
    # Negative frame indices would mean a broken file, not a clip-length mismatch.
    negative = query_dicts(con, f'SELECT count(*) AS n FROM {table} WHERE frame < 0')[0]['n']
    assert negative == 0, f'{table}: {negative} rows with negative frame indices'
    # Count the frames past each clip before deleting them (HAVING drops the empty group a constant key yields).
    dropped = query_dicts(con, f"""
        SELECT {keys}, count(DISTINCT (t.video, t.frame)) AS frames, count(DISTINCT t.video) AS clips
        FROM {table} t JOIN clips c USING (video) WHERE t.frame >= c.n GROUP BY ALL HAVING count(*) > 0 ORDER BY ALL
    """)
    # Remove them.
    con.execute(f'DELETE FROM {table} t USING clips c WHERE t.video = c.video AND t.frame >= c.n')
    return dropped


def load_counts(con: duckdb.DuckDBPyConnection, files: list[tuple], excluded: list[tuple]) -> list[dict]:
    """Table counts(system, config, video, frame, cnt): distinct track ids per frame of every system file.

    files holds (system, config, video, filename); excluded holds (system, video, track_id) ids to
    leave out (LEAP's ambiguous objects for leap_ambiguous_dropped). Returns frames dropped past the clip.
    """
    # Register which file belongs to which configuration and clip, and the ids to leave out.
    con.execute('CREATE OR REPLACE TABLE files (system VARCHAR, config VARCHAR, video VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO files VALUES (?, ?, ?, ?)', files)
    con.execute('CREATE OR REPLACE TABLE excluded (system VARCHAR, video VARCHAR, track_id BIGINT)')
    if excluded:
        con.executemany('INSERT INTO excluded VALUES (?, ?, ?)', excluded)
    # Read every file once, one row per box, and count distinct ids per configuration, clip, and frame.
    con.execute(f"""
        CREATE OR REPLACE TABLE counts AS
        WITH boxes AS (
            SELECT filename, frame_idx AS frame, CAST(box[1] AS BIGINT) AS track_id FROM {TRACK_BOXES}
        )
        SELECT f.system, f.config, f.video, b.frame, count(DISTINCT b.track_id) AS cnt
        FROM boxes b JOIN files f USING (filename)
        ANTI JOIN excluded e ON e.system = f.system AND e.video = f.video AND e.track_id = b.track_id
        GROUP BY ALL
    """, [sorted({f for *_, f in files})])
    return drop_past_clip(con, 'counts', 't.system, t.config')


def load_ground_truth(con: duckdb.DuckDBPyConnection, gt_files: list[tuple]) -> list[dict]:
    """Table gt_counts(variant, video, frame, g) for variants 'all' and 'moving'. Returns frames dropped past the clip.

    gt_files holds (video, filename). Box centers follow data_prepare.py on the MOT form of each box
    (left x1, width x2 - x1): xmin = int(left), xmax = xmin + int(width), center = int((xmin + xmax) / 2),
    likewise for y.
    """
    # Register which file belongs to which clip.
    con.execute('CREATE OR REPLACE TABLE gt_files (video VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO gt_files VALUES (?, ?)', gt_files)
    # Every ground-truth box with its track id and LEAP's integer center.
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_boxes AS
        SELECT g.video, b.frame_idx AS frame, CAST(b.box[1] AS BIGINT) AS track_id,
               trunc((2 * trunc(b.box[2]) + trunc(b.box[4] - b.box[2])) / 2) AS cx,
               trunc((2 * trunc(b.box[3]) + trunc(b.box[5] - b.box[3])) / 2) AS cy
        FROM {TRACK_BOXES} b JOIN gt_files g USING (filename)
    """, [[f for _, f in gt_files]])
    dropped = drop_past_clip(con, 'gt_boxes', "'ground truth' AS system")
    # Both variants' per-frame counts.
    con.execute("""
        CREATE OR REPLACE TABLE gt_counts AS
        -- Variant all: distinct track ids per frame (LEAP's pseudo_label).
        SELECT 'all' AS variant, video, frame, count(DISTINCT track_id) AS g FROM gt_boxes GROUP BY video, frame
        UNION ALL
        -- Variant moving: tracks spanning >= the minimum frames and travelling >= the minimum distance,
        -- each counted on [first, last) (LEAP's pseudo_tuple), gaps inside a track included.
        SELECT 'moving', f.video, f.frame, count(*) FROM (
            SELECT video, track_id, min(frame) AS first, max(frame) AS last,
                   arg_min(cx, frame) AS x0, arg_min(cy, frame) AS y0, arg_max(cx, frame) AS x1, arg_max(cy, frame) AS y1
            FROM gt_boxes GROUP BY video, track_id
        ) t JOIN frames f ON f.video = t.video AND f.frame >= t.first AND f.frame < t.last
        WHERE t.last - t.first >= $min_frames AND sqrt(power(t.x1 - t.x0, 2) + power(t.y1 - t.y0, 2)) >= $min_distance
        GROUP BY f.video, f.frame
    """, {'min_frames': MOVING_MIN_FRAMES, 'min_distance': MOVING_MIN_DISTANCE})
    return dropped


def densify(con: duckdb.DuckDBPyConnection):
    """Tables pred(system, config, clip, frame, gidx, p) and gt(variant, clip, frame, gidx, g) over every pooled frame."""
    # Every configuration on every pooled frame, 0 where its file has no row or no box.
    con.execute("""
        CREATE OR REPLACE TABLE pred AS
        SELECT c.system, c.config, f.clip, f.frame, f.gidx, coalesce(k.cnt, 0) AS p
        FROM configs c CROSS JOIN frames f
        LEFT JOIN counts k ON k.system = c.system AND k.config = c.config AND k.video = f.video AND k.frame = f.frame
    """)
    # Both ground-truth variants on every pooled frame.
    con.execute("""
        CREATE OR REPLACE TABLE gt AS
        SELECT v.variant, f.clip, f.frame, f.gidx, coalesce(k.g, 0) AS g
        FROM (SELECT unnest($variants) AS variant) v CROSS JOIN frames f
        LEFT JOIN gt_counts k ON k.variant = v.variant AND k.video = f.video AND k.frame = f.frame
    """, {'variants': list(GT_VARIANTS)})


def add_baselines(con: duckdb.DuckDBPyConnection):
    """Constant answers as configurations of system 'baseline' in tables configs and pred.

    'zero' is the empty output, 'one' answers every frame positive (and ranks frames at random for
    top-k), and 'median <variant>' is that ground-truth variant's median count on every frame.
    """
    # The constants: 0, 1, and each variant's lower median count over the pooled frames.
    con.execute("""
        CREATE OR REPLACE TABLE baseline_values AS
        SELECT 'zero' AS config, 0::BIGINT AS value UNION ALL SELECT 'one', 1
        UNION ALL SELECT 'median ' || variant, quantile_disc(g, 0.5) FROM gt GROUP BY variant
    """)
    # Register them as configurations without throughput or HOTA, then add their counts on every frame.
    con.execute("INSERT INTO configs SELECT 'baseline', config, NULL, NULL FROM baseline_values")
    con.execute("""
        INSERT INTO pred SELECT 'baseline', b.config, f.clip, f.frame, f.gidx, b.value
        FROM baseline_values b CROSS JOIN frames f
    """)


def score_frames(con: duckdb.DuckDBPyConnection, pred: str = 'pred') -> list[dict]:
    """Selection confusion counts and aggregation errors per ground-truth variant and configuration of table `pred`."""
    return query_dicts(con, f"""
        SELECT g.variant, p.system, p.config, count(*) AS frames, count(DISTINCT p.clip) AS clips,
               count(*) FILTER (WHERE g.g >= 1 AND p.p >= 1) AS tp,
               count(*) FILTER (WHERE g.g = 0 AND p.p >= 1) AS fp,
               count(*) FILTER (WHERE g.g >= 1 AND p.p = 0) AS fn,
               avg(abs(p.p - g.g)) AS mae,
               -- LEAP's per-frame error: relative when g > 0, else 0 for a correct zero and 1 for a false count.
               -- Relative errors are averaged as DECIMAL (exact sums), since a parallel DOUBLE average depends on
               -- summation order and identical answers (LEAP without ambiguous objects) could then differ in the last bit.
               avg(CAST(CASE WHEN g.g > 0 THEN abs(p.p - g.g) / g.g WHEN p.p = 0 THEN 0 ELSE 1 END AS DECIMAL(38, 15))) AS mape_leap,
               -- Frames with g > 0 only (avg skips the NULLs; a FILTER would still evaluate the division by zero).
               avg(CAST(CASE WHEN g.g > 0 THEN abs(p.p - g.g) / g.g END AS DECIMAL(38, 15))) AS mape_pos,
               count(*) FILTER (WHERE g.g = 0 AND p.p > 0) / nullif(count(*) FILTER (WHERE g.g = 0), 0) AS zero_fp_rate,
               avg(p.p) AS mean_predicted
        FROM {pred} p JOIN gt g USING (gidx)
        GROUP BY ALL ORDER BY ALL
    """)


def frame_metrics(r: dict) -> dict:
    """Selection and aggregation sections of a record from one score_frames row (unrounded)."""
    return dict(selection=selection_metrics(r['tp'], r['fp'], r['fn']),
                aggregation=dict(mae=r['mae'], mape_leap=r['mape_leap'], acc_leap=1 - r['mape_leap'],
                                 mape_pos=r['mape_pos'], zero_fp_rate=r['zero_fp_rate'],
                                 mean_predicted=r['mean_predicted']))


def score_topk(con: duckdb.DuckDBPyConnection) -> tuple[dict[tuple, dict], dict[str, dict]]:
    """Top-k scores per (variant, system, config) on pooled arrays fetched from DuckDB, and tau per variant."""
    # Configurations in DuckDB's order, so they line up with the rows of the count matrix.
    configs = con.execute('SELECT system, config FROM configs ORDER BY system, config').fetchall()
    # Pooled global ids, and every configuration's counts as one row of a matrix.
    gid = con.execute('SELECT gid FROM frames ORDER BY gidx').fetchnumpy()['gid']
    p_all = con.execute('SELECT p FROM pred ORDER BY system, config, gidx').fetchnumpy()['p']
    assert len(p_all) == len(configs) * len(gid), 'pred does not cover every configuration on every frame'
    p_all = p_all.reshape(len(configs), len(gid))
    # What each configuration returns, once: its tie draws, its pooled-order frames, and LEAP's clusters under both sorts.
    answers = [(tie_draws(p, gid), separated_top(p, gid), leap_clusters(p), leap_clusters(p, kind=None)) for p in p_all]
    out: dict[tuple, dict] = {}
    taus: dict[str, dict] = {}
    for variant in GT_VARIANTS:
        # This variant's pooled ground-truth counts and its tau.
        g = con.execute('SELECT g FROM gt WHERE variant = ? ORDER BY gidx', [variant]).fetchnumpy()['g']
        tau = topk_tau(g, gid)
        # tau under random ground-truth tie orders too (0 when fewer than k frames are returned), to show it is stable.
        gt_draws = tie_draws(g, gid)
        draw_taus = np.where((gt_draws >= 0).all(axis=1), g[gt_draws].min(axis=1), 0)
        taus[variant] = dict(tau=tau, tau_min_over_draws=int(draw_taus.min()), tau_max_over_draws=int(draw_taus.max()))
        # The ground truth's LEAP clusters under both sorts.
        truth_stable, truth_default = leap_clusters(g), leap_clusters(g, kind=None)
        # Score every configuration against this variant (a loop over configurations).
        for (system, config), p, (draws, pooled, stable, default) in zip(configs, p_all, answers):
            out[(variant, system, config)] = dict(topk_paper(draws, pooled, p, g, tau),
                                                   topk_leap_code=topk_leap_code(truth_stable, stable, g),
                                                   topk_leap_code_default_sort=topk_leap_code(truth_default, default, g))
    return out, taus


def lag_correlations(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per configuration, the median per-clip Pearson correlation of counts with ground truth 'all' at lag 0 and at the best lag.

    Lag L pairs system frame f with ground-truth frame f + L of the same clip. A clip where either
    side is constant has no correlation and is left out of the median. Constant answers are skipped.
    """
    return query_dicts(con, """
        WITH per_clip AS (
            SELECT p.system, p.config, p.clip, l.lag,
                   CASE WHEN var_pop(p.p) > 0 AND var_pop(g.g) > 0 THEN corr(p.p, g.g) END AS r
            FROM pred p CROSS JOIN (SELECT unnest(range(-$max_lag, $max_lag + 1)) AS lag) l
            JOIN gt g ON g.variant = 'all' AND g.clip = p.clip AND g.frame = p.frame + l.lag
            WHERE p.system <> 'baseline'
            GROUP BY ALL
        ),
        per_lag AS (
            SELECT system, config, lag, median(r) AS r, count(r) AS clips FROM per_clip GROUP BY ALL
        ),
        best AS (
            -- Highest median correlation (equal to LAG_TIE_DECIMALS); ties go to the smallest |lag|, then the negative lag.
            SELECT system, config, lag AS best_lag, r AS best_r FROM per_lag
            QUALIFY row_number() OVER (PARTITION BY system, config ORDER BY round(r, $decimals) DESC NULLS LAST, abs(lag), lag) = 1
        )
        SELECT z.system, z.config, z.r AS lag0_r, z.clips AS clips_with_r, b.best_lag, b.best_r,
               coalesce(b.best_lag <> 0 AND b.best_r - z.r > $gain, false) AS warn
        FROM per_lag z JOIN best b USING (system, config)
        WHERE z.lag = 0
        ORDER BY z.system, z.config
    """, {'max_lag': MAX_LAG, 'gain': LAG_WARN_GAIN, 'decimals': LAG_TIE_DECIMALS})


def load_unwrap(con: duckdb.DuckDBPyConnection, unwrap_files: list[tuple]) -> dict:
    """Table unwrap(video, ...) from p140's unwrap.json files ((video, filename) pairs); returns per-dataset totals."""
    # Missing statistics mean the LEAP cache predates the corrected p140.
    assert_files_exist([f for _, f in unwrap_files], 'LEAP unwrap.json files (rerun the fixed p140)')
    # Register which file belongs to which clip, then read them all.
    con.execute('CREATE OR REPLACE TABLE unwrap_files (video VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO unwrap_files VALUES (?, ?)', unwrap_files)
    con.execute(f"""
        CREATE OR REPLACE TABLE unwrap AS
        SELECT f.video, u.* EXCLUDE (filename)
        FROM read_json(?, filename = true, format = 'auto', columns = {UNWRAP_COLUMNS}) u JOIN unwrap_files f USING (filename)
    """, [[f for _, f in unwrap_files]])
    # Every clip needs one record, and p140 must have clipped to the same length as here.
    bad = query_dicts(con, """
        SELECT c.video, c.n, u.clip_frames FROM clips c LEFT JOIN unwrap u USING (video)
        WHERE u.clip_frames IS NULL OR u.clip_frames <> c.n
    """)
    assert not bad, f'LEAP unwrap.json clip lengths differ from the clips: {bad}'
    # Totals over the dataset's clips.
    return query_dicts(con, """
        SELECT count(*) AS clips, count(*) FILTER (WHERE export_frames = 0) AS empty_exports,
               sum(objects) AS objects, sum(len(wrapped_objects)) AS wrapped_objects,
               sum(len(ambiguous_full_span_objects)) AS ambiguous_full_span_objects,
               count(*) FILTER (WHERE len(ambiguous_full_span_objects) > 0) AS clips_with_ambiguous,
               sum(wrapped_boxes_dropped) AS wrapped_boxes_dropped, sum(boxes_past_clip_dropped) AS boxes_past_clip_dropped,
               coalesce(list(video ORDER BY video) FILTER (WHERE export_frames = 0), []) AS empty_export_videos
        FROM unwrap
    """)[0]


def leap_excluding_empty_exports(con: duckdb.DuckDBPyConnection) -> dict:
    """LEAP's selection F1 and acc_leap per ground-truth variant on the clips where its export is not empty."""
    # LEAP's counts on clips with a non-empty export.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW pred_leap_nonempty AS
        SELECT p.* FROM pred p JOIN clips c USING (clip) JOIN unwrap u USING (video)
        WHERE p.system = 'leap' AND u.export_frames > 0
    """)
    # The same scoring as every configuration, on those clips only (no row when every export is empty).
    return {r['variant']: dict(clips=r['clips'], frames=r['frames'], f1=selection_metrics(r['tp'], r['fp'], r['fn'])['f1'],
                               acc_leap=1 - r['mape_leap'])
            for r in score_frames(con, 'pred_leap_nonempty')}


def export_counts(con: duckdb.DuckDBPyConnection, path: str, n: int) -> np.ndarray:
    """Per-frame object counts of a raw LEAP export, recovering each object's interval [s, e) from its frames.

    The export lists boxes per frame; L is its length. One contiguous run [a, b] is [a, b + 1); two
    runs [0, a] and [b, L - 1] are a wrapped interval [b - L, a + 1); anything else raises. Each
    object counts on [max(s, 0), min(e, n)), as LEAP's preprocess does with start 0 and end n.
    """
    # Parse the export into (frame, track_id) pairs; a null file or frame holds no boxes.
    con.execute("""
        CREATE OR REPLACE TABLE export_obs AS
        WITH doc AS (SELECT from_json(content, '[[{"track_id": "BIGINT"}]]') AS frames FROM read_text(?)),
        per_frame AS (SELECT generate_subscripts(frames, 1) - 1 AS frame, unnest(frames) AS boxes FROM doc)
        SELECT DISTINCT frame, box.track_id AS track_id
        FROM (SELECT frame, unnest(boxes) AS box FROM per_frame)
    """, [path])
    # The export's length L (0 for a null file).
    length = con.execute("SELECT coalesce(len(from_json(content, '[\"JSON\"]')), 0) FROM read_text(?)", [path]).fetchone()[0]
    # Runs of consecutive frames per object, and each object's interval.
    rows = query_dicts(con, """
        WITH runs AS (
            SELECT track_id, min(frame) AS a, max(frame) AS b
            FROM (SELECT track_id, frame, frame - row_number() OVER (PARTITION BY track_id ORDER BY frame) AS run FROM export_obs)
            GROUP BY track_id, run
        ),
        shape AS (
            SELECT track_id, count(*) AS runs, min(a) AS head_start, min(b) AS head_end, max(a) AS tail_start, max(b) AS tail_end
            FROM runs GROUP BY track_id
        )
        SELECT track_id,
               CASE WHEN runs = 1 THEN head_start
                    WHEN runs = 2 AND head_start = 0 AND tail_end = $length - 1 THEN tail_start - $length END AS s,
               CASE WHEN runs = 1 THEN head_end + 1
                    WHEN runs = 2 AND head_start = 0 AND tail_end = $length - 1 THEN head_end + 1 END AS e
        FROM shape
    """, {'length': length})
    # Any other shape is not the exporter's wrap.
    invalid = [r['track_id'] for r in rows if r['s'] is None]
    assert not invalid, f'{path}: objects {invalid[:5]} are neither one interval nor one wrapped interval'
    # Count every interval on the clip's frames with a difference array.
    delta = np.zeros(n + 1, dtype=np.int64)
    starts = np.clip(np.array([r['s'] for r in rows], dtype=np.int64), 0, n)
    ends = np.clip(np.array([r['e'] for r in rows], dtype=np.int64), 0, n)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta)[:n]


def cross_check_leap(con: duckdb.DuckDBPyConnection, dataset: str, sota_dir: Path) -> dict:
    """Compare the corrected LEAP cache with the interval recovery from the raw export on the clip with the most corrections."""
    # The clip with the most wrapped and ambiguous objects (then the most objects) exercises both corrections.
    clip = query_dicts(con, """
        SELECT u.video, c.n, len(u.wrapped_objects) AS wrapped, len(u.ambiguous_full_span_objects) AS ambiguous,
               u.objects, u.export_frames
        FROM unwrap u JOIN clips c USING (video)
        ORDER BY len(u.wrapped_objects) + len(u.ambiguous_full_span_objects) DESC, u.objects DESC, u.video
        LIMIT 1
    """)[0]
    # LEAP's raw export for that clip, named by the numeric video id.
    path = sota_dir / dataset_name_in_sota(dataset) / 'leap' / f"{extract_video_id(clip['video'])}.json"
    assert_files_exist([str(path)], 'raw LEAP exports')
    recovered = export_counts(con, str(path), clip['n'])
    # The cache's counts for the same clip, from the pooled table (score_dataset checks there is one LEAP configuration).
    cached = con.execute("SELECT p FROM pred JOIN clips USING (clip) WHERE system = 'leap' AND video = ? ORDER BY frame",
                         [clip['video']]).fetchnumpy()['p']
    differing = int((recovered != cached).sum())
    return dict(clip, export=str(path), frames_differing=differing, equal=differing == 0,
                recovered_box_frames=int(recovered.sum()), cached_box_frames=int(cached.sum()))


def file_rows(dataset: str, videos: list[str], configs: list[tuple]) -> list[tuple]:
    """(system, config, video, filename) for every configuration and clip; leap_ambiguous_dropped reads LEAP's files."""
    # One file per configuration and clip.
    rows = [(s, c, v, str(output_path(dataset, 'leap' if s == 'leap_ambiguous_dropped' else s, c, v)))
            for s, c, _, _ in configs for v in videos]
    # A configuration missing a clip would be scored on fewer frames than the others.
    assert_files_exist([f for *_, f in rows], f'{dataset} tracking files')
    return rows


def ground_truth_files(dataset: str, videos: list[str], gt_root: Path | None) -> tuple[list[tuple], bool]:
    """(video, filename) of each clip's ground-truth tracks, and whether they come from gt_root.

    gt_root is used for a dataset when it holds every clip under <gt_root>/<root dataset>/; when it
    holds none, the root dataset's 003_groundtruth is used; when it holds only some, this raises.
    """
    # The clips gt_root holds for this dataset, if any.
    external = [] if gt_root is None else [(v, str(gt_root / dataset_root_name(dataset) / v / 'tracking.jsonl')) for v in videos]
    present = sum(os.path.exists(f) for _, f in external)
    assert present in (0, len(videos)), f'{dataset}: --gt-root holds {present} of {len(videos)} clips; supply all or none'
    if present:
        return external, True
    # Otherwise the automatic ground truth, which must exist for every clip.
    rows = [(v, str(output_path(dataset, 'ground truth', '003_groundtruth', v))) for v in videos]
    assert_files_exist([f for _, f in rows], f'{dataset} ground-truth files')
    return rows, False


def ground_truth_source(dataset: str, gt_root: Path | None, external: bool) -> dict:
    """Where a dataset's ground truth comes from and whether results on it are provisional."""
    # External (for example human-verified) ground truth is final.
    if external:
        return dict(kind='external', root=str(gt_root), dataset=dataset_root_name(dataset), provisional=False)
    # Automatic ground truth, described per dataset family; B3D's is provisional.
    family = next(k for k in GT_SOURCES if dataset.startswith(k))
    return dict(kind='automatic 003_groundtruth', dataset=dataset_root_name(dataset), source=GT_SOURCES[family],
                provisional=dataset.startswith('jnc'))


def score_dataset(con: duckdb.DuckDBPyConnection, dataset: str, gt_root: Path | None, sota_dir: Path) -> tuple[dict, dict]:
    """Every configuration's and constant answer's query scores on one dataset (unrounded), and that dataset's diagnostics."""
    # The test clips in teNN order, and the pooled frame tables.
    videos = list_split_videos(dataset, 'test')
    assert videos, f'{dataset}: no test videos'
    load_clips(con, dataset, videos)
    # Every configuration with throughput and HOTA; LEAP's ambiguous-dropped variant shares LEAP's runtime.
    configs = [(r['system'], r['config'], r['fps'], r['hota']) for r in system_configs(dataset)]
    leap = [c for s, c, _, _ in configs if s == 'leap']
    # unwrap.json, the ambiguous ids, and the cross-check are per LEAP configuration only when there is one.
    assert len(leap) == 1, f'{dataset}: expected exactly one LEAP configuration, found {leap}'
    configs += [('leap_ambiguous_dropped', c, f, None) for s, c, f, _ in configs if s == 'leap']
    con.execute('CREATE OR REPLACE TABLE configs (system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE)')
    con.executemany('INSERT INTO configs VALUES (?, ?, ?, ?)', configs)

    # LEAP's unwrap statistics and its ambiguous objects, excluded for the lower-bound variant.
    unwrap_summary = load_unwrap(con, [(v, str(output_path(dataset, 'leap', leap[0], v).parent / 'unwrap.json'))
                                       for v in videos])
    excluded = con.execute("""
        SELECT 'leap_ambiguous_dropped', video, unnest(ambiguous_full_span_objects) FROM unwrap
    """).fetchall()
    # Per-frame counts of every system and of both ground-truth variants, on every pooled frame, then the constant answers.
    gt_files, external = ground_truth_files(dataset, videos, gt_root)
    dropped = load_counts(con, file_rows(dataset, videos, configs), excluded)
    dropped += load_ground_truth(con, gt_files)
    densify(con)
    add_baselines(con)

    # Selection and aggregation in DuckDB, top-k on the fetched arrays.
    frame_scores = score_frames(con)
    topk, taus = score_topk(con)
    # Throughput and HOTA by configuration, to attach to each record.
    meta = {(s, c): (f, h) for s, c, f, h in configs}
    records: dict[str, list] = {v: [] for v in GT_VARIANTS}
    baselines: dict[str, dict] = {v: {} for v in GT_VARIANTS}
    # One record per ground-truth variant and configuration (a loop over result rows, not frames).
    for r in frame_scores:
        scores = dict(frame_metrics(r), topk=topk[(r['variant'], r['system'], r['config'])])
        if r['system'] != 'baseline':
            fps, hota = meta[(r['system'], r['config'])]
            records[r['variant']].append(dict(system=r['system'], config=r['config'], fps=fps, hota=hota, **scores))
        elif r['config'] in ('zero', 'one', f"median {r['variant']}"):
            # Constant answers, keyed zero, one, and median (this variant's own median only).
            baselines[r['variant']][r['config'].split(' ')[0]] = scores
    # Ground-truth summary per variant; with every frame positive, selection can only measure missed frames.
    gt_summary = {r['variant']: dict(positive_frames=r['positive'], all_frames_positive=r['positive'] == r['frames'],
                                     mean_count=r['mean'], max_count=r['max'], **taus[r['variant']])
                  for r in query_dicts(con, 'SELECT variant, count(*) FILTER (WHERE g > 0) AS positive, count(*) AS frames, '
                                            'avg(g) AS mean, max(g) AS max FROM gt GROUP BY variant')}
    frames = query_dicts(con, 'SELECT count(*) AS n FROM frames')[0]['n']
    scored = dict(ground_truth=ground_truth_source(dataset, gt_root, external), clips=len(videos), frames=frames,
                  ground_truth_summary=gt_summary, baselines=baselines, records=records)

    # Diagnostics: frames dropped, LEAP's unwrap statistics and empty-export check, lag correlations, and the cache cross-check.
    lags = lag_correlations(con)
    diagnostics = dict(frames_dropped_past_clip=dropped, leap_unwrap=unwrap_summary,
                       leap_excluding_empty_exports=leap_excluding_empty_exports(con), lag_correlations=lags,
                       lag_warnings=[r for r in lags if r['warn']], leap_cache_cross_check=cross_check_leap(con, dataset, sota_dir))
    return scored, diagnostics


def flag_cell(cell: dict, bound: float, trivial: dict | None):
    """Mark an operating-point cell trivial when a constant answer meets its bound, and LEAP-ambiguity-dependent
    when LEAP with and without its ambiguous objects disagree on meeting it."""
    # The best constant answer, and whether it already meets the (unrounded) bound.
    cell['trivial_baseline'] = None if trivial is None else dict(name=trivial['baseline'], accuracy=trivial['accuracy'])
    cell['trivial'] = trivial is not None and trivial['accuracy'] >= bound
    # Both LEAP readings, when both were scored.
    leap, dropped = cell['systems'].get('leap'), cell['systems'].get('leap_ambiguous_dropped')
    cell['leap_ambiguity_dependent'] = leap is not None and dropped is not None and leap['meets_bound'] != dropped['meets_bound']


def compute_operating_points(con: duckdb.DuckDBPyConnection, scored: dict, losses: list[float]) -> dict:
    """{gt variant: {query: {accuracy, skipped, non_informative, bounds: {bound: operating points}}}} over every scored dataset."""
    # Bound names for the relative losses.
    relative = {f'{loss:.2f}': loss for loss in losses}
    out: dict = {}
    for variant in GT_VARIANTS:
        for query, metric in QUERY_ACCURACY.items():
            # Each record's unrounded accuracy for this query sits in the section named after the query.
            con.execute('CREATE OR REPLACE TABLE points (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, '
                        'hota DOUBLE, accuracy DOUBLE)')
            con.executemany('INSERT INTO points VALUES (?, ?, ?, ?, ?, ?)',
                            [(label, r['system'], r['config'], r['fps'], r['hota'], r[query][metric])
                             for label, s in scored.items() for r in s['records'][variant]])
            # Each dataset's best constant answer: the accuracy a trivial answer reaches (ties to the first name).
            con.execute('CREATE OR REPLACE TABLE baseline_points (dataset VARCHAR, baseline VARCHAR, accuracy DOUBLE)')
            con.executemany('INSERT INTO baseline_points VALUES (?, ?, ?)',
                            [(label, name, b[query][metric]) for label, s in scored.items()
                             for name, b in s['baselines'][variant].items()])
            con.execute("""
                CREATE OR REPLACE TABLE trivial AS SELECT dataset, baseline, accuracy FROM baseline_points
                WHERE accuracy IS NOT NULL QUALIFY row_number() OVER (PARTITION BY dataset ORDER BY accuracy DESC, baseline) = 1
            """)
            # A dataset whose reference accuracy is missing or not positive has no relative bound: record and drop it.
            con.execute("""
                CREATE OR REPLACE TABLE skipped AS SELECT dataset, accuracy AS reference_accuracy FROM points
                WHERE system = 'reference' AND (accuracy IS NULL OR accuracy <= 0)
            """)
            con.execute('DELETE FROM points WHERE dataset IN (SELECT dataset FROM skipped)')
            # Configurations without an accuracy cannot meet any bound.
            con.execute('DELETE FROM points WHERE accuracy IS NULL')
            # Datasets where a constant answer is at least as accurate as the reference: every bound there is trivial.
            non_informative = query_dicts(con, """
                SELECT p.dataset, p.accuracy AS reference_accuracy, t.baseline, t.accuracy AS baseline_accuracy
                FROM points p JOIN trivial t USING (dataset) WHERE p.system = 'reference' AND t.accuracy >= p.accuracy
                ORDER BY p.dataset
            """)
            entry = dict(accuracy=metric, non_informative=non_informative, bounds={},
                         skipped=[dict(r, reason='reference accuracy is not positive')
                                  for r in query_dicts(con, 'SELECT * FROM skipped ORDER BY dataset')])
            out.setdefault(variant, {})[query] = entry
            # Nothing left to bound.
            if not query_dicts(con, 'SELECT 1 FROM points LIMIT 1'):
                continue
            # Relative bounds, then LEAP's own accuracy per dataset on the datasets where LEAP has an accuracy.
            entry['bounds'] = {name: operating_points(con, 'points', POINT_SYSTEMS, loss=loss) for name, loss in relative.items()}
            con.execute("CREATE OR REPLACE TEMP VIEW points_leap AS SELECT * FROM points "
                        "WHERE dataset IN (SELECT dataset FROM points WHERE system = 'leap')")
            fit = best_accuracy_bounds(con, 'points_leap', 'leap')
            entry['bounds']['leap_fit'] = operating_points(con, 'points_leap', POINT_SYSTEMS, bounds=fit)
            # Flag every cell against its unrounded bound (operating_points reports it rounded).
            trivial = {r['dataset']: r for r in query_dicts(con, 'SELECT * FROM trivial')}
            for name, by_dataset in entry['bounds'].items():
                for dataset, cell in by_dataset.items():
                    bound = fit[dataset] if name == 'leap_fit' else cell['reference']['accuracy'] * (1 - relative[name])
                    flag_cell(cell, bound, trivial.get(dataset))
    return out


def blockers(scored: dict) -> dict:
    """BLOCKERS, with the B3D ground-truth blocker naming the provisional datasets, or without it when there are none."""
    # Datasets whose results are provisional; the B3D blocker stays while there is one.
    provisional = [label for label, s in scored.items() if s['ground_truth']['provisional']]
    out = {k: v for k, v in BLOCKERS.items() if k != 'b3d_human_ground_truth'}
    if provisional:
        out['b3d_human_ground_truth'] = f"{BLOCKERS['b3d_human_ground_truth']} Provisional here: {', '.join(provisional)}."
    return out


def method(args: argparse.Namespace) -> dict:
    """Inputs, parameters, and every deviation from LEAP's evaluator, for configs.json."""
    return dict(
        evaluator='LEAP/evaluate.py (ace:/work/alenachao/tunnel/work/LEAP): selection_query_1, aggregation_query_1, top_k_query_1',
        unit='one pooled sequence per dataset: test clips concatenated in teNN order, start 0, end = pooled frames, no padding',
        classes='class-agnostic (vehicle); see blockers.per_class_queries',
        counts='distinct track ids per frame of tracking.jsonl; missing frames count 0; frames >= clip length dropped',
        clip_length="rows of the reference pipeline's 002_naive/detection.jsonl, checked against the video frame count",
        systems=dict(reference='002_naive tracks', tetris='every validated configuration (060 tracks)',
                     otif='every configuration (SOTA cache)',
                     leap='corrected SOTA cache (p140 unwrap); ambiguous full-span objects kept whole (upper bound)',
                     leap_ambiguous_dropped="LEAP without unwrap.json's ambiguous_full_span_objects (lower bound); "
                                            "LEAP's throughput, no HOTA"),
        baselines=dict(zero='0 on every frame (the empty output)',
                       one='1 on every frame (every frame positive; a random top-k ranking)',
                       median="the ground-truth variant's lower median count on every frame (uses the ground truth)",
                       use='datasets[*].baselines; a cell is trivial when the best of them meets its bound, and a '
                           "query is non-informative on a dataset when one is at least as accurate as the reference"),
        throughput_and_hota='measurements.common.system_configs',
        ground_truth=dict(
            path=('root dataset 003_groundtruth via common.output_path' if args.gt_root is None else
                  f'{args.gt_root}/<root dataset>/teNN.mp4/tracking.jsonl for datasets it fully covers, else the root '
                  "dataset's 003_groundtruth; see datasets[*].ground_truth"),
            variants=dict(all="every box, as LEAP's pseudo_label (primary)",
                          moving=(f"LEAP's pseudo_tuple: last - first >= {MOVING_MIN_FRAMES} frames and first/last box "
                                  f"centers >= {MOVING_MIN_DISTANCE} px apart, counted on [first, last); centers as "
                                  'data_prepare.py computes them from MOT left = x1 and width = x2 - x1: '
                                  'int((int(x1) + int(x1) + int(x2 - x1)) / 2), likewise for y '
                                  '(sensitivity; favours LEAP, which drops stopped vehicles itself)'))),
        selection=dict(f1='2TP / (2TP + FP + FN), presence = count >= 1; null when no frame is positive on either side',
                       degenerate=('when every ground-truth frame is positive (ground_truth_summary.all_frames_positive, '
                                   'as on B3D), the constant answer one scores F1 1.0 and every cell is trivial'),
                       f1_leap_code='TP forced to >= 1, recall and precision rounded to 4 decimals before F1 (evaluate.py)'),
        aggregation=dict(mape_leap='mean over frames of |p - g| / g (g > 0), 0 (g = p = 0), 1 (g = 0 < p); not clipped',
                         acc_leap="1 - mape_leap, LEAP's ACC on unpadded frames", mae='mean |p - g|',
                         mape_pos='mean |p - g| / g over frames with g > 0', zero_fp_rate='share of g = 0 frames with p > 0'),
        topk=dict(k=TOPK, separation_frames=TOPK_RADIUS, candidates=TOPK_CANDIDATES,
                  topk_paper=('mean over tie draws of the share of the system\'s returned frames with ground-truth count '
                              '>= tau, over k; the system greedily returns its k highest-count frames >= separation apart '
                              '(clips separated by a gap), never a frame where its count is 0; tau = min ground-truth '
                              'count of the ground truth\'s k such frames (ties by pooled order; '
                              'ground_truth_summary.tau_min/max_over_draws show it under random ties)'),
                  tie_draws=dict(draws=TIE_DRAWS, seed=TIE_SEED,
                                 rule='numpy default_rng(seed) per configuration; draw r breaks ties by the r-th vector '
                                      'of uniform keys over the frames it can return; topk_paper_min/max over draws'),
                  topk_paper_pooled_order='the earlier deterministic score: ties by pooled order, zero-count frames returnable',
                  ties_at_cutoff=('frames whose count equals that of the last frame returned in pooled order, accepted '
                                  'and suppressed frames included'),
                  topk_leap_code=('evaluate.py top_k_query_1 over pooled arrays with argsort(kind=stable); clusters may '
                                  'chain across clip boundaries; prediction clusters valued with ground-truth counts'),
                  topk_leap_code_default_sort="the same with numpy's default argsort, as evaluate.py calls it"),
        skipped_queries='Query-3 (distinct-id counts) and Query-4 (aggregation_query_2) are not queries in LEAP\'s paper',
        operating_points=dict(accuracy=QUERY_ACCURACY, losses=args.losses,
                              leap_fit=("bound = LEAP's best accuracy per dataset; a bound at or below a constant "
                                        "answer's accuracy (for example <= 0) is flagged trivial"),
                              trivial='cell.trivial: the best constant answer (cell.trivial_baseline) meets the bound',
                              leap_ambiguity_dependent=('cell.leap_ambiguity_dependent: leap and leap_ambiguous_dropped '
                                                        'disagree on meeting the bound; quote both'),
                              granularity='topk_paper is a mean of multiples of 1/k = 0.1 over draws',
                              undefined='datasets whose reference accuracy is not positive are skipped per query',
                              rounding='bounds and selection use unrounded accuracies; JSON values are rounded to 4 decimals'),
        lag_diagnostic=(f'median per-clip Pearson correlation of counts with ground truth all, system frame f vs ground-truth '
                        f'frame f + lag, lag in [-{MAX_LAG}, {MAX_LAG}]; warn when the best lag is nonzero and gains '
                        f'more than {LAG_WARN_GAIN}'),
        leap_moving_fps_note='LEAP configs set fps 15 for every dataset but caldot2.yaml, which has no fps key; 15 is used',
        sota_dir=str(args.sota_dir),
    )


def main():
    args = parse_args()
    # Results go to measurements-results/r1o2_leap_native_queries/; all analysis runs in one in-memory DuckDB.
    out_dir = results_dir(__file__)
    con = duckdb.connect()

    # Score every configuration of every system on each dataset.
    scored, diagnostics = {}, {}
    for dataset in args.datasets:
        label = DATASET_LABELS[dataset]
        scored[label], diagnostics[label] = score_dataset(con, dataset, args.gt_root, args.sota_dir)
        # One progress line per dataset with the cross-check outcome.
        check = diagnostics[label]['leap_cache_cross_check']
        print(f"{label}: {scored[label]['clips']} clips, {scored[label]['frames']} frames, "
              f"{len(scored[label]['records']['all'])} configurations, ground truth {scored[label]['ground_truth']['kind']}; "
              f"LEAP cache cross-check on {check['video']}: "
              f"{'equal' if check['equal'] else str(check['frames_differing']) + ' frames differ'}", flush=True)
        # Alignment warnings are printed so a misaligned system is not missed.
        for w in diagnostics[label]['lag_warnings']:
            print(f"  lag warning: {w['system']} {w['config']} lag0 {w['lag0_r']:.3f} best lag {w['best_lag']} {w['best_r']:.3f}")
    # Operating points from unrounded accuracies; every file is rounded only when written.
    points = compute_operating_points(con, scored, args.losses)
    write_json(out_dir, 'configs.json', rounded(dict(method=method(args), blockers=blockers(scored), datasets=scored)))
    write_json(out_dir, 'diagnostics.json', rounded(diagnostics))
    write_json(out_dir, 'operating_points.json', rounded(points))

    # Print the primary ground truth's operating points with their flags.
    for query, entry in points['all'].items():
        for s in entry['skipped']:
            print(f"{query}: {s['dataset']} skipped, reference accuracy {s['reference_accuracy']}")
        for s in entry['non_informative']:
            print(f"{query}: {s['dataset']} non-informative, baseline {s['baseline']} {s['baseline_accuracy']:.4f} "
                  f">= reference {s['reference_accuracy']:.4f}")
        print_operating_points(entry['bounds'], f"{query} ({entry['accuracy']}, ground truth all)")
        for name, by_dataset in entry['bounds'].items():
            print(f"  {name}: trivial {[d for d, c in by_dataset.items() if c['trivial']]}, "
                  f"LEAP-ambiguity-dependent {[d for d, c in by_dataset.items() if c['leap_ambiguity_dependent']]}")


if __name__ == '__main__':
    main()
