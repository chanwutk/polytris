#!/usr/local/bin/python
"""R1.O4: detection precision and recall before and after packing, on the same retained content.

The same detector, with the same weights and confidence threshold, runs twice in the cache:

  before packing  the reference pipeline on every original frame (002_naive/detection.jsonl,
                  preprocess/p002_preprocess_naive_detection.py, batch size 1);
  after packing   each Tetris configuration on its packed canvases (p040, batch size 4), with
                  every box mapped back to source-frame coordinates by p050
                  (050_uncompressed_detections/<detection config>/detections.jsonl).

p050 maps a canvas box through the group under its center or, when the center lies on an empty
canvas cell, under the first of its corners that lies on a group cell. It drops boxes that touch
no group cell before writing 050. Every box in 050 reaches the tracker (p060), so packed precision
is measured on what Tetris emits, after that drop.

Both are scored against the test ground truth the paper's HOTA uses (p110): the
003_groundtruth/tracking.jsonl of the detector-independent dataset (caldot1 for caldot1-y05,
ams for ams-y05, jncN for B3D). Restricting both to the same frames and the same retained
tiles separates what packing as implemented changes from what region selection drops. Packing
as implemented bundles tile rearrangement, black background, neighbouring polyominoes from other
frames, JPEG re-encoding of the canvas (cv2.imwrite, quality 95), batch-4 inference, and, on
Amsterdam, a 1260 px canvas for 1280 px frames (whole tiles only).

Frames      a configuration's frames are those with at least one tile on one of its canvases,
            i.e. the frames on which it actually ran detection.
Tiles       retained tiles are read from the canvases themselves (033_compressed_frames
            index_maps and offset_lookups), so they include tile padding. For padding 'none'
            they equal the p022 (pruned) or p020 (unpruned) grids.
Matching    per frame, detections in descending confidence each take the unmatched ground-truth
            box of highest IoU, if that IoU is at least 0.5 (COCO's greedy rule), one to one.
            Every detection on a frame is matched against every ground-truth box on it, retained
            or not. matched_nonretained_with_retained_candidate counts the detections for which
            COCO's preference for non-ignored ground truth could change that; 0 means it cannot.
Filters     a box is retained under
              'frame'           always (no region filter), which mixes region-selection loss in;
              'center'          when its center lies in a retained tile (at canvas scale 1, a
                                packed box passes exactly when p050 mapped it by its center);
              'center_cover50'  when its center lies in a retained tile and at least half of its
                                area lies in retained tiles;
              'inside'          when it lies entirely in retained tiles (area cover >= 1 - 1e-9),
                                so no counted box is cut at the region boundary.
Counting    retained ground truth is the recall denominator. A detection matched to a retained
            box is a true positive, and one matched to a non-retained box is ignored (so a box
            shifted across the region boundary is not counted twice). An unmatched detection that
            is itself retained is a false positive. Unmatched detections that are not retained
            are counted per side:
              reference              ignored, since they lie on content Tetris dropped or cross
                                     the region boundary;
              packed                 false positives under every filter when their center lies
                                     outside every retained tile, since p050 then mapped them
                                     through a corner: the detector drew them over black canvas
                                     cells or a neighbouring polyomino, and Tetris emits them to
                                     the tracker (fp_outside_filter counts them; many are
                                     tile-sized boxes on tile boundaries). Those with a retained
                                     center that fail a cover filter are ignored, like the
                                     reference's;
              packed_ignore_outside  ignored, i.e. the reference's rule applied to the packed
                                     detections (a sensitivity variant that hides those boxes).
            ignored_matched_nonretained and ignored_unmatched_outside count what each side ignored.
AP50        all-point interpolated average precision over the counted detections, ranked by
            confidence. Detections exist only above the detector's threshold (0.25 for the
            YOLO detectors, 0.45 for B3D's RetinaNet), so the curve is truncated there.

JPEG control  measurements-gpu/r1o4_packing_detection_accuracy.py reruns the reference detector
            on the operating point's frames as decoded ('rerun') and after a JPEG round trip at
            p030's quality ('jpeg'). Scored like the reference at the operating point, rerun vs
            reference is the environment's noise, jpeg vs rerun JPEG's effect alone, and packed vs
            jpeg the rest of packing. The script fails when those outputs are missing, unless
            --skip-jpeg-control is given.

Configurations are every validated Tetris test configuration in the tradeoff table (p130).
Configurations that differ only in the tracker share one detection directory (detection config),
so scores are stored once per detection config; sweep aggregates count each one once. The
operating point is the fastest configuration within --loss of the reference pipeline's HOTA
(common.fastest_within). For context, the reference detections are also scored on every test
frame without a region filter.

B3D (jnc*) still uses the automatic ground truth, produced with the same RetinaNet as the
detector under test, so its numbers are provisional until R1.O1's human-verified labels exist.

Writes measurements-results/r1o4_packing_detection_accuracy/:
  configs.json           per dataset: every detection config's frames, retained tiles, and scores
                         before and after packing under each filter (detection_configs); every
                         Tetris variant's fps, HOTA, and detection config (configs); and the
                         median, min, and max of packing's effect over detection configs, per
                         padding mode and overall (sweep)
  operating_points.json  per dataset, the operating point's scores, the reference pipeline's
                         scores on all frames, and the JPEG control (jpeg_control)
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd

from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements.common import DATASET_LABELS, fastest_within, load_test_tradeoff, query_dicts, results_dir, write_json
from polyis.io import cache, store
from polyis.utilities import build_param_str


# COCO's matching threshold: a detection matches a ground-truth box at IoU >= 0.5.
IOU_THRESHOLD = 0.5
# Region filters, from no filter to the strictest: the minimum fraction of a box's area in retained
# tiles, on top of a retained center (None: no region filter). 'inside' allows float error in the area sum.
FILTERS = {'frame': None, 'center': 0.0, 'center_cover50': 0.5, 'inside': 1 - 1e-9}
# Reported sides: the detections each scores (det_in.side), and whether its unmatched detections whose center
# lies outside every retained tile are false positives under every filter (True) or ignored when they fail it (False).
SIDES = {'reference': ('reference', False), 'packed': ('packed', True), 'packed_ignore_outside': ('packed', False),
         'rerun': ('rerun', False), 'jpeg': ('jpeg', False)}
# Sides of the sweep over every configuration, and of the JPEG control at the operating point.
MAIN_SIDES = ['reference', 'packed', 'packed_ignore_outside']
CONTROL_SIDES = MAIN_SIDES + ['rerun', 'jpeg']
# Detection sources on original frames: the reference (002_naive) and the JPEG control's two runs.
ORIGINAL_SOURCES = ['reference', 'rerun', 'jpeg']
# Compared pairs (side, baseline): each side against the reference, plus the JPEG control's decomposition
# (JPEG alone against the same-environment rerun, and the rest of packing against JPEG).
MAIN_PAIRS = [(s, 'reference') for s in MAIN_SIDES[1:]]
CONTROL_PAIRS = [(s, 'reference') for s in CONTROL_SIDES[1:]] + [('jpeg', 'rerun'), ('packed', 'jpeg')]
# The JPEG control's outputs inside this script's result directory (written by measurements-gpu/<same name>.py).
CONTROL_DIR = 'jpeg_control'
# Metrics compared between the sides.
METRICS = ('precision', 'recall', 'ap50')
# Schemas of the box files: frame index plus a list of boxes per frame (two lists in the JPEG control's files).
GT_COLUMNS = "{'frame_idx': 'BIGINT', 'tracks': 'DOUBLE[][]'}"
REFERENCE_COLUMNS = "{'frame_idx': 'BIGINT', 'detections': 'DOUBLE[][]'}"
PACKED_COLUMNS = "{'frame_idx': 'BIGINT', 'bboxes': 'DOUBLE[][]'}"
CONTROL_COLUMNS = "{'frame_idx': 'BIGINT', 'plain': 'DOUBLE[][]', 'jpeg': 'DOUBLE[][]'}"
# Pseudo-configuration under which the reference is scored on every test frame.
ALL_FRAMES = 'all frames'


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O4: detection precision/recall before and after packing')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    parser.add_argument('--skip-jpeg-control', action='store_true')
    return parser.parse_args()


def accuracy_threshold(row: dict) -> float | None:
    """The tracking-accuracy threshold p022 pruned a tradeoff row with, or None when it skipped pruning."""
    # The tradeoff table leaves the threshold empty (NaN) for unpruned configurations.
    threshold = row['tracking_accuracy_threshold']
    return None if threshold is None or pd.isna(threshold) else float(threshold)


def detection_config(row: dict) -> str:
    """The 030-050 stage directory of a tradeoff row (classifier_tile_sr[_thr]_rXXX_padding_sYYY[_tracker]).

    Stages 030-050 include the tracker only when p022 pruned with it; the 060 variant id always
    ends with the tracker. The name is rebuilt with the pipeline's own build_param_str and checked
    against the variant id, so a naming change fails here rather than reading the wrong directory.
    """
    # Unpruned configurations carry no threshold and no tracker before p060.
    threshold = accuracy_threshold(row)
    pruned = threshold is not None
    # Rebuild the directory name exactly as p030, p040, and p050 do.
    name = build_param_str(classifier=row['classifier'], tilesize=int(row['tilesize']),
                           sample_rate=int(row['sample_rate']), tilepadding=row['tilepadding'],
                           canvas_scale=float(row['canvas_scale']), tracker=row['tracker'] if pruned else None,
                           tracking_accuracy_threshold=threshold, relevance_threshold=float(row['relevance_threshold']))
    # p060 appends the tracker to unpruned names, which do not carry it yet.
    expected = name if pruned else f"{name}_{row['tracker']}"
    assert row['variant_id'] == expected, f"variant {row['variant_id']!r} does not match its parameters ({expected!r})"
    return name


def metadata_table(con: duckdb.DuckDBPyConnection, name: str, **columns: tuple[str, list]):
    """Temp table `name` from equal-length Python lists, one column per keyword given as (DuckDB type, values).

    For small metadata (configuration names, filters, tile grids); None becomes NULL.
    """
    # Equal lengths, or unnest would pad the shorter lists with NULL.
    assert len({len(values) for _, values in columns.values()}) <= 1, f'{name}: columns of different lengths'
    # unnest of several lists in one SELECT zips them row by row.
    select = ', '.join(f'unnest(${c}::{t}[]) AS {c}' for c, (t, _) in columns.items())
    con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS SELECT {select}', {c: values for c, (_, values) in columns.items()})


def canvas_tiles(directory: Path) -> np.ndarray:
    """Source tiles (frame, ty, tx) shown on the canvases of one 033_compressed_frames configuration directory.

    Canvas cell (r, c) holds group g = index_map[r, c] (0 is empty). Line g of the offset lookup
    is ((py, px), (oy, ox), frame), so the cell shows tile (r - py + oy, c - px + ox) of that
    frame: the inverse of p030's placement, with the offsets p050 uses to map boxes back.
    """
    # A configuration without its mapping files cannot define the retained region.
    assert (directory / 'index_maps').is_dir(), f'missing index maps in {directory}'
    parts = []
    # One index map and one offset lookup per canvas, sharing a stem (file I/O needs the loop).
    for path in sorted((directory / 'index_maps').glob('*.npy')):
        # Group id of every canvas cell.
        index_map = np.load(path).astype(np.int64)
        # Every canvas needs its offset lookup, or its groups cannot be placed.
        lookup_path = directory / 'offset_lookups' / f'{path.stem}.jsonl'
        assert lookup_path.is_file(), f'missing offset lookup {lookup_path}'
        # Offsets of every group, one JSON line per group in id order.
        with open(lookup_path) as f:
            lookup = np.array([[py, px, oy, ox, frame] for (py, px), (oy, ox), frame in map(json.loads, f)],
                              dtype=np.int64).reshape(-1, 5)
        # Occupied cells and their 0-based group index.
        rows, cols = np.nonzero(index_map)
        groups = index_map[rows, cols] - 1
        # Every occupied cell must refer to a group in the lookup.
        assert groups.size == 0 or groups.max() < len(lookup), f'{path}: group id beyond its offset lookup'
        # Each cell's group offsets, then the source tile it shows.
        g = lookup[groups]
        parts.append(np.column_stack([g[:, 4], rows - g[:, 0] + g[:, 2], cols - g[:, 1] + g[:, 3]]))
    # A configuration may retain nothing in a video.
    return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.int64)


def load_tiles(con: duckdb.DuckDBPyConnection, dataset: str, cfgs: list[str], videos: list[str]):
    """Table tiles(cfg, video, frame, ty, tx): distinct retained tiles of every detection configuration."""
    parts = []
    # Read every configuration's canvases in every test video (file I/O needs the loop).
    for ci, cfg in enumerate(cfgs):
        for vi, video in enumerate(videos):
            tiles = canvas_tiles(Path(cache.exec(dataset, 'comp-frames', video, cfg)))
            # Prefix the configuration and video indices so one array holds everything.
            parts.append(np.column_stack([np.full(len(tiles), ci), np.full(len(tiles), vi), tiles]))
    # Stack into one frame for DuckDB.
    raw = pd.DataFrame(np.concatenate(parts) if parts else np.empty((0, 5), dtype=np.int64),
                       columns=['ci', 'vi', 'frame', 'ty', 'tx'])
    # Expose the array to DuckDB, with the index-to-name lookups, for the joins.
    con.register('tiles_raw', raw)
    metadata_table(con, 'cfg_names', ci=('BIGINT', list(range(len(cfgs)))), cfg=('VARCHAR', cfgs))
    metadata_table(con, 'video_names', vi=('BIGINT', list(range(len(videos)))), video=('VARCHAR', videos))
    # A tile padded into two polyominoes is still one retained tile.
    con.execute("""
        CREATE OR REPLACE TABLE tiles AS
        SELECT DISTINCT c.cfg, v.video, CAST(r.frame AS BIGINT) AS frame, CAST(r.ty AS BIGINT) AS ty, CAST(r.tx AS BIGINT) AS tx
        FROM tiles_raw r JOIN cfg_names c USING (ci) JOIN video_names v USING (vi)
    """)
    # Release the registered array.
    con.unregister('tiles_raw')


def register_files(con: duckdb.DuckDBPyConnection, files: list[tuple[str, str, str]]) -> list[str]:
    """Temp table box_files(source, video, filename) for (source, video, path) rows; returns the paths."""
    # Reading zero files would fail inside DuckDB with a less clear message.
    assert files, 'no files to read'
    # Every file must exist, or a video would silently drop out.
    missing = [p for _, _, p in files if not os.path.exists(p)]
    assert not missing, f'{len(missing)} files missing, e.g. {missing[:3]}'
    # Map file names back to their source and video.
    sources, videos, paths = (list(column) for column in zip(*files))
    metadata_table(con, 'box_files', source=('VARCHAR', sources), video=('VARCHAR', videos), filename=('VARCHAR', paths))
    return paths


def read_boxes(con: duckdb.DuckDBPyConnection, table: str, files: list[tuple[str, str, str]], kind: str):
    """Tables `table` (one row per box) and `table`_lines(source, video, frame) (one row per file line) from box files.

    Files are given as (source, video, path). kind 'gt' reads tracking.jsonl ([track id, x1, y1,
    x2, y2]) into (video, frame, gid, x1, y1, x2, y2), with the track id as the box id; kind
    'reference' or 'packed' reads detection files ([x1, y1, x2, y2, score]) into (source, video,
    frame, did, x1, y1, x2, y2, score), with the box's 1-based position in its frame as the box id;
    kinds 'rerun' and 'jpeg' read the same columns from the JPEG control's 'plain' and 'jpeg' lists.
    Each file must hold at most one line per frame.
    """
    # Register the files, checking that each exists.
    paths = register_files(con, files)
    # Each kind's column holding the boxes and its schema.
    key, columns = {'gt': ('tracks', GT_COLUMNS), 'reference': ('detections', REFERENCE_COLUMNS),
                    'packed': ('bboxes', PACKED_COLUMNS), 'rerun': ('plain', CONTROL_COLUMNS),
                    'jpeg': ('jpeg', CONTROL_COLUMNS)}[kind]
    # Every line once, with its frame and box list.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE box_lines AS
        SELECT f.source, f.video, j.frame_idx AS frame, j.{key} AS boxes
        FROM read_json(?, filename = true, format = 'newline_delimited', columns = {columns}) j
        JOIN box_files f ON f.filename = j.filename
    """, [paths])
    # A repeated frame line would reuse box ids and double the frame's boxes.
    repeated = con.execute('SELECT count(*) FROM (SELECT source, video, frame FROM box_lines GROUP BY ALL HAVING count(*) > 1)').fetchone()[0]
    assert repeated == 0, f'{table}: {repeated} frames with more than one line in a file'
    # The lines without their boxes, for frame-level checks.
    con.execute(f'CREATE OR REPLACE TABLE {table}_lines AS SELECT source, video, frame FROM box_lines')
    # Ground truth: the track id identifies a box within its frame.
    if kind == 'gt':
        select = 'video, frame, CAST(b[1] AS BIGINT) AS gid, b[2] AS x1, b[3] AS y1, b[4] AS x2, b[5] AS y2'
    # Detections: position in the frame identifies a box; the fifth value is the confidence.
    else:
        select = 'source, video, frame, CAST(pos AS BIGINT) AS did, b[1] AS x1, b[2] AS y1, b[3] AS x2, b[4] AS y2, b[5] AS score'
    # One row per box, with its position in the frame's list.
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT {select} FROM (SELECT source, video, frame, unnest(boxes) AS b, generate_subscripts(boxes, 1) AS pos FROM box_lines)
    """)
    # Release the line lists.
    con.execute('DROP TABLE box_lines')
    # A short box list leaves trailing values NULL (DuckDB lists return NULL past their end).
    value_columns = ['x1', 'y1', 'x2', 'y2'] + ([] if kind == 'gt' else ['score'])
    short = con.execute(f"SELECT count(*) FROM {table} WHERE {' OR '.join(f'{c} IS NULL' for c in value_columns)}").fetchone()[0]
    assert short == 0, f'{table}: {short} boxes missing one of {value_columns}'
    # Ground-truth ids must be unique per frame (TrackEval requires it too).
    if kind == 'gt':
        dup = con.execute(f'SELECT count(*) FROM (SELECT video, frame, gid FROM {table} GROUP BY ALL HAVING count(*) > 1)').fetchone()[0]
        assert dup == 0, f'{table}: {dup} duplicate track ids within a frame'


def greedy_match(con: duckdb.DuckDBPyConnection, iou_threshold: float = IOU_THRESHOLD) -> int:
    """Table matches(source, video, frame, did, gid, iou): COCO's greedy one-to-one matching per frame.

    Sequentially, detections in descending confidence (ties to the lower did) each take the free
    ground-truth box of highest IoU (ties to the lower gid) if that IoU >= iou_threshold. This
    computes the same result in set-based rounds: a detection is final once no higher-priority
    unmatched detection competes for any of its free candidates, since nothing before it in the
    sequential order can then change its choice. Each round finalizes at least the top detection
    of every frame. Reads tables gt and dets, writes candidates (every pair at or above the
    threshold) and matches; returns the number of rounds.
    """
    # Candidate pairs on the same frame with enough overlap.
    con.execute("""
        CREATE OR REPLACE TABLE candidates AS
        SELECT * FROM (
            SELECT d.source, d.video, d.frame, d.did, d.score, g.gid,
                   greatest(0, least(d.x2, g.x2) - greatest(d.x1, g.x1)) * greatest(0, least(d.y2, g.y2) - greatest(d.y1, g.y1)) AS inter,
                   (d.x2 - d.x1) * (d.y2 - d.y1) + (g.x2 - g.x1) * (g.y2 - g.y1) - inter AS union_area,
                   CASE WHEN union_area > 0 THEN inter / union_area ELSE 0 END AS iou
            FROM dets d JOIN gt g USING (video, frame)
        )
        WHERE iou >= ?
    """, [iou_threshold])
    # Start with no matches and count the rounds.
    con.execute('CREATE OR REPLACE TABLE matches (source VARCHAR, video VARCHAR, frame BIGINT, did BIGINT, gid BIGINT, iou DOUBLE)')
    rounds = 0
    # Finalize safe detections until no free candidate pair is left.
    while True:
        inserted = con.execute("""
            INSERT INTO matches
            WITH free AS (
                -- Candidate pairs whose detection and ground-truth box are both still unmatched.
                SELECT c.* FROM candidates c
                ANTI JOIN matches m ON m.source = c.source AND m.video = c.video AND m.frame = c.frame AND m.did = c.did
                ANTI JOIN matches n ON n.source = c.source AND n.video = c.video AND n.frame = c.frame AND n.gid = c.gid
            ),
            blocked AS (
                -- Detections sharing a free candidate with a higher-priority detection.
                SELECT DISTINCT a.source, a.video, a.frame, a.did FROM free a JOIN free b
                  ON a.source = b.source AND a.video = b.video AND a.frame = b.frame AND a.gid = b.gid
                 AND (b.score > a.score OR (b.score = a.score AND b.did < a.did))
            )
            -- Every unblocked detection takes its best free candidate.
            SELECT source, video, frame, did, gid, iou FROM free
            ANTI JOIN blocked USING (source, video, frame, did)
            QUALIFY row_number() OVER (PARTITION BY source, video, frame, did ORDER BY iou DESC, gid) = 1
        """).fetchone()[0]
        # Stop when a round matches nothing (no free pairs remain).
        if inserted == 0:
            return rounds
        rounds += 1


def box_regions(con: duckdb.DuckDBPyConnection, tile_size: int):
    """Table regions(cfg, role, source, video, frame, bid, center_in, cover) for boxes touching retained tiles.

    center_in: the tile containing the box center (p050's floor(x / tile) rule) is retained.
    cover: the fraction of the box area inside retained tiles (tiles are disjoint, so the sum of
    per-tile overlaps). Ground-truth boxes and detections on original frames (ORIGINAL_SOURCES) are
    scored against every configuration's tiles, packed detections only against their own. Boxes with no retained tile, or with zero
    area, have no row (not retained). Reads gt, dets, and tiles.
    """
    # Every box's tile spans, joined with the retained tiles and summed per box and configuration.
    con.execute("""
        CREATE OR REPLACE TABLE regions AS
        WITH boxes AS (
            SELECT 'gt' AS role, 'gt' AS source, video, frame, gid AS bid, x1, y1, x2, y2 FROM gt
            UNION ALL
            SELECT 'det', source, video, frame, did, x1, y1, x2, y2 FROM dets
        ),
        box_rows AS (
            -- Every tile row a box spans (an edge on a tile boundary does not enter the next tile).
            SELECT *, unnest(generate_series(CAST(floor(y1 / $ts) AS BIGINT), CAST(ceil(y2 / $ts) AS BIGINT) - 1)) AS ty
            FROM boxes WHERE x2 > x1 AND y2 > y1
        ),
        cells AS (
            -- Every tile a box spans, with the overlap area and whether it holds the center.
            SELECT role, source, video, frame, bid, ty, tx, (x2 - x1) * (y2 - y1) AS area,
                   (least(x2, (tx + 1) * $ts) - greatest(x1, tx * $ts)) * (least(y2, (ty + 1) * $ts) - greatest(y1, ty * $ts)) AS overlap,
                   ty = CAST(floor((y1 + y2) / 2 / $ts) AS BIGINT) AND tx = CAST(floor((x1 + x2) / 2 / $ts) AS BIGINT) AS is_center
            FROM (SELECT *, unnest(generate_series(CAST(floor(x1 / $ts) AS BIGINT), CAST(ceil(x2 / $ts) AS BIGINT) - 1)) AS tx FROM box_rows)
        ),
        retained AS (
            -- Ground truth and detections on original frames against every configuration's tiles.
            SELECT t.cfg, c.* FROM cells c JOIN tiles t ON t.video = c.video AND t.frame = c.frame AND t.ty = c.ty AND t.tx = c.tx
            WHERE c.source = 'gt' OR list_contains($originals, c.source)
            UNION ALL
            -- Packed detections against their own configuration's tiles only.
            SELECT t.cfg, c.* FROM cells c JOIN tiles t ON t.cfg = c.source AND t.video = c.video AND t.frame = c.frame
                                                      AND t.ty = c.ty AND t.tx = c.tx
        )
        SELECT cfg, role, source, video, frame, bid, bool_or(is_center) AS center_in, sum(overlap) / any_value(area) AS cover
        FROM retained GROUP BY cfg, role, source, video, frame, bid
    """, {'ts': tile_size, 'originals': ORIGINAL_SOURCES})


def build_config_inputs(con: duckdb.DuckDBPyConnection):
    """Tables scopes, gt_in, det_in for score(): every configuration under every filter, both sides.

    Reads configs(cfg, ...), tiles, gt, dets (rows on original frames have a source in
    ORIGINAL_SOURCES, packed rows their detection configuration), matches, and regions. det_in.side is
    the source for original-frame rows and 'packed' for packed rows; center_in marks a box whose center
    lies in a retained tile.
    """
    # The filters as a table (min_cover NULL for 'frame'), the frames each configuration ran detection on, and the scopes.
    metadata_table(con, 'filters', filter=('VARCHAR', list(FILTERS)), min_cover=('DOUBLE', list(FILTERS.values())))
    con.execute('CREATE OR REPLACE TABLE detected_frames AS SELECT DISTINCT cfg, video, frame FROM tiles')
    con.execute('CREATE OR REPLACE TABLE scopes AS SELECT c.cfg, f.filter FROM configs c CROSS JOIN filters f')
    # Retention rule per filter, applied to a region row (NULL when the box touches no retained tile).
    rule = 'CASE WHEN fl.min_cover IS NULL THEN true ELSE coalesce(r.center_in AND r.cover >= fl.min_cover, false) END'
    # Ground truth on each configuration's frames, retained or not under each filter.
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_in AS
        SELECT u.cfg, fl.filter, g.video, g.frame, g.gid, {rule} AS retained
        FROM gt g JOIN detected_frames u USING (video, frame) CROSS JOIN filters fl
        LEFT JOIN regions r ON r.cfg = u.cfg AND r.role = 'gt' AND r.video = g.video AND r.frame = g.frame AND r.bid = g.gid
    """)
    # Original-frame detections on each configuration's frames, and each configuration's own packed detections.
    con.execute(f"""
        CREATE OR REPLACE TABLE det_in AS
        WITH placed AS (
            SELECT u.cfg, d.source AS side, d.* FROM dets d JOIN detected_frames u
              ON u.video = d.video AND u.frame = d.frame
            WHERE list_contains($originals, d.source)
            UNION ALL
            SELECT u.cfg, 'packed' AS side, d.* FROM dets d JOIN detected_frames u
              ON u.cfg = d.source AND u.video = d.video AND u.frame = d.frame
        )
        SELECT p.cfg, fl.filter, p.side, p.video, p.frame, p.did, p.score, {rule} AS passes,
               coalesce(r.center_in, false) AS center_in, m.gid
        FROM placed p CROSS JOIN filters fl
        LEFT JOIN regions r ON r.cfg = p.cfg AND r.role = 'det' AND r.source = p.source AND r.video = p.video
                           AND r.frame = p.frame AND r.bid = p.did
        LEFT JOIN matches m ON m.source = p.source AND m.video = p.video AND m.frame = p.frame AND m.did = p.did
    """, {'originals': ORIGINAL_SOURCES})


def build_all_frame_inputs(con: duckdb.DuckDBPyConnection):
    """Tables scopes, gt_in, det_in for score(): the reference on every frame in table frames, no filter."""
    con.execute('CREATE OR REPLACE TABLE scopes AS SELECT ? AS cfg, ? AS filter', [ALL_FRAMES, 'frame'])
    # Every ground-truth box on a clip frame is retained.
    con.execute("""
        CREATE OR REPLACE TABLE gt_in AS
        SELECT ? AS cfg, 'frame' AS filter, g.video, g.frame, g.gid, true AS retained
        FROM gt g JOIN frames USING (video, frame)
    """, [ALL_FRAMES])
    # Every reference detection on a clip frame counts.
    con.execute("""
        CREATE OR REPLACE TABLE det_in AS
        SELECT ? AS cfg, 'frame' AS filter, 'reference' AS side, d.video, d.frame, d.did, d.score, true AS passes,
               true AS center_in, m.gid
        FROM dets d JOIN frames USING (video, frame)
        LEFT JOIN matches m ON m.source = d.source AND m.video = d.video AND m.frame = d.frame AND m.did = d.did
        WHERE d.source = 'reference'
    """, [ALL_FRAMES])


def score(con: duckdb.DuckDBPyConnection, sides: list[str]) -> list[dict]:
    """Table scores and its rows: ground truth, TP, FP, precision, recall, AP50, and ignored counts per (cfg, filter, side).

    `sides` are keys of SIDES. Reads scopes(cfg, filter), gt_in(cfg, filter, video, frame, gid,
    retained), and det_in(cfg, filter, side, video, frame, did, score, passes, center_in, gid), where
    det_in.side is a source in ORIGINAL_SOURCES or 'packed' and det_in.gid is the matched ground-truth
    box (NULL when unmatched). Precision is NULL without counted detections, recall and AP50 without retained
    ground truth. AP50 is all-point interpolated: the sum, over true positives in confidence
    order, of the best precision at that rank or later, divided by the retained ground truth.
    """
    # Each requested side's input rows and its rule for unmatched detections centered outside the region.
    metadata_table(con, 'side_rules', side=('VARCHAR', sides), input=('VARCHAR', [SIDES[s][0] for s in sides]),
                   center_outside_fp=('BOOLEAN', [SIDES[s][1] for s in sides]))
    # Label, rank, and count every side's detections in one pass.
    con.execute("""
        CREATE OR REPLACE TABLE scores AS
        WITH joined AS (
            -- Each detection with the retention of the ground-truth box it matched (NULL when unmatched).
            SELECT d.*, g.retained AS gt_retained FROM det_in d LEFT JOIN gt_in g
              ON g.cfg = d.cfg AND g.filter = d.filter AND g.video = d.video AND g.frame = d.frame AND g.gid = d.gid
        ),
        labeled AS (
            -- 1 = true positive, 0 = false positive, NULL = ignored, under each side's rule.
            SELECT j.cfg, j.filter, s.side, j.video, j.frame, j.did, j.score, j.gid, j.passes,
                   CASE WHEN j.gid IS NOT NULL THEN CASE WHEN j.gt_retained THEN 1 END
                        WHEN j.passes OR (s.center_outside_fp AND NOT j.center_in) THEN 0 END AS tp
            FROM joined j JOIN side_rules s ON s.input = j.side
        ),
        ranked AS (
            -- Cumulative true positives in confidence order (ties broken by position for determinism).
            SELECT cfg, filter, side, tp, sum(tp) OVER w AS cum_tp, row_number() OVER w AS n
            FROM labeled WHERE tp IS NOT NULL
            WINDOW w AS (PARTITION BY cfg, filter, side ORDER BY score DESC, video, frame, did
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        ),
        enveloped AS (
            -- Interpolated precision: the best precision at this rank or any later one.
            SELECT *, max(cum_tp / n) OVER (PARTITION BY cfg, filter, side ORDER BY n
                                            ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS envelope
            FROM ranked
        ),
        per_side AS (
            -- Counted detections and the AP numerator.
            SELECT cfg, filter, side, sum(tp) AS tp, count(*) - sum(tp) AS fp,
                   coalesce(sum(envelope) FILTER (WHERE tp = 1), 0) AS ap_sum
            FROM enveloped GROUP BY cfg, filter, side
        ),
        tallies AS (
            -- What each side ignored, and the false positives counted although they fail the filter.
            SELECT cfg, filter, side,
                   count(*) FILTER (WHERE tp IS NULL AND gid IS NOT NULL) AS ignored_matched_nonretained,
                   count(*) FILTER (WHERE tp IS NULL AND gid IS NULL) AS ignored_unmatched_outside,
                   count(*) FILTER (WHERE tp = 0 AND NOT passes) AS fp_outside_filter
            FROM labeled GROUP BY cfg, filter, side
        ),
        positives AS (
            -- Retained ground truth per scope (zero when the scope has none).
            SELECT s.cfg, s.filter, count(g.gid) FILTER (WHERE g.retained) AS n_gt
            FROM scopes s LEFT JOIN gt_in g ON g.cfg = s.cfg AND g.filter = s.filter GROUP BY s.cfg, s.filter
        )
        -- Every scope and side, with zero counts when nothing was counted.
        SELECT p.cfg, p.filter, s.side, p.n_gt, CAST(coalesce(r.tp, 0) AS BIGINT) AS tp, CAST(coalesce(r.fp, 0) AS BIGINT) AS fp,
               CASE WHEN coalesce(r.tp, 0) + coalesce(r.fp, 0) > 0 THEN r.tp / (r.tp + r.fp) END AS precision,
               CASE WHEN p.n_gt > 0 THEN coalesce(r.tp, 0) / p.n_gt END AS recall,
               CASE WHEN p.n_gt > 0 THEN coalesce(r.ap_sum, 0) / p.n_gt END AS ap50,
               coalesce(t.ignored_matched_nonretained, 0) AS ignored_matched_nonretained,
               coalesce(t.ignored_unmatched_outside, 0) AS ignored_unmatched_outside,
               coalesce(t.fp_outside_filter, 0) AS fp_outside_filter
        FROM positives p CROSS JOIN side_rules s
        LEFT JOIN per_side r ON r.cfg = p.cfg AND r.filter = p.filter AND r.side = s.side
        LEFT JOIN tallies t ON t.cfg = p.cfg AND t.filter = p.filter AND t.side = s.side
    """)
    # The rows in a stable order.
    return query_dicts(con, 'SELECT * FROM scores ORDER BY cfg, filter, side')


def deltas(con: duckdb.DuckDBPyConnection, pairs: list[tuple[str, str]]) -> list[dict]:
    """Table deltas and its rows: per scope and (side, baseline) pair, the side's precision, recall, and AP50
    minus the baseline's.

    A pair is named after its side when the baseline is the reference, and '<side>_vs_<baseline>'
    otherwise. Computed from the unrounded values in table scores; a metric undefined on either side
    gives NULL.
    """
    # The pairs and their names.
    metadata_table(con, 'pairs', side=('VARCHAR', [a for a, _ in pairs]), base=('VARCHAR', [b for _, b in pairs]),
                   label=('VARCHAR', [a if b == 'reference' else f'{a}_vs_{b}' for a, b in pairs]))
    # Each pair's side and baseline on the same scope.
    con.execute("""
        CREATE OR REPLACE TABLE deltas AS
        SELECT p.cfg, p.filter, q.label AS side, p.precision - r.precision AS precision, p.recall - r.recall AS recall,
               p.ap50 - r.ap50 AS ap50
        FROM pairs q JOIN scores p ON p.side = q.side JOIN scores r ON r.cfg = p.cfg AND r.filter = p.filter AND r.side = q.base
    """)
    # The rows in a stable order.
    return query_dicts(con, 'SELECT * FROM deltas ORDER BY cfg, filter, side')


def retained_alternatives(con: duckdb.DuckDBPyConnection) -> dict:
    """{(cfg, filter, det_in side): n}: detections matched to a non-retained ground-truth box that also reach the
    IoU threshold with a retained one.

    Matching ignores retention, while COCO prefers non-ignored ground truth. Where n is 0 the two
    rules match identically: a detection that took a non-retained box had no retained candidate,
    and one that took the best free box overall took a retained one, so it was also the best free
    retained one. Reads det_in, gt_in, and candidates (from greedy_match).
    """
    # Detections whose matched box is not retained while a retained candidate reached the threshold.
    rows = query_dicts(con, """
        SELECT d.cfg, d.filter, d.side, count(DISTINCT (d.video, d.frame, d.did)) AS n
        FROM det_in d
        JOIN gt_in m ON m.cfg = d.cfg AND m.filter = d.filter AND m.video = d.video AND m.frame = d.frame AND m.gid = d.gid
        JOIN candidates c ON c.source = CASE WHEN d.side = 'packed' THEN d.cfg ELSE d.side END
                         AND c.video = d.video AND c.frame = d.frame AND c.did = d.did
        JOIN gt_in g ON g.cfg = d.cfg AND g.filter = d.filter AND g.video = c.video AND g.frame = c.frame AND g.gid = c.gid
        WHERE NOT m.retained AND g.retained
        GROUP BY d.cfg, d.filter, d.side
    """)
    # Keyed for lookup when arranging the report.
    return {(r['cfg'], r['filter'], r['side']): r['n'] for r in rows}


def side_entry(row: dict) -> dict:
    """One side's scores, with the metrics rounded for the report."""
    # Counts as they are, metrics to four decimals (None stays None).
    return dict(**{k: row[k] for k in ('n_gt', 'tp', 'fp', 'ignored_matched_nonretained', 'ignored_unmatched_outside',
                                       'fp_outside_filter')},
                **{k: None if row[k] is None else round(float(row[k]), 4) for k in METRICS})


def arrange(rows: list[dict], delta_rows: list[dict], alternatives: dict | None = None) -> dict:
    """{cfg: {filter: {side: scores, 'delta': {pair name: deltas}}}} from score() and deltas() rows.

    alternatives, from retained_alternatives(), adds matched_nonretained_with_retained_candidate to each side.
    """
    out: dict = {}
    # Place each side's scores under its configuration and filter (report arrangement, not analysis).
    for r in rows:
        entry = side_entry(r)
        # The count for the detections this side scores; absent keys mean none.
        if alternatives is not None:
            entry['matched_nonretained_with_retained_candidate'] = alternatives.get((r['cfg'], r['filter'], SIDES[r['side']][0]), 0)
        out.setdefault(r['cfg'], {}).setdefault(r['filter'], {})[r['side']] = entry
    # Packing's effect, rounded only after subtracting (+ 0.0 turns a float-noise -0.0 into 0.0).
    for d in delta_rows:
        out[d['cfg']][d['filter']].setdefault('delta', {})[d['side']] = {
            k: None if d[k] is None else round(float(d[k]), 4) + 0.0 for k in METRICS}
    return out


def sweep_summary(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Median, min, and max of each delta over a dataset's detection configs, per padding mode and overall ('all').

    Each detection config counts once (tracker variants share one). Padding modes also differ in
    sample rate, thresholds, and retained region, so a difference between modes is an association,
    not padding's effect. Reads deltas and configs(cfg, tilepadding).
    """
    # Deltas per padding mode, then over all modes, from one grouping-sets query (+ 0.0 turns -0.0 into 0.0).
    return query_dicts(con, """
        WITH padded AS (
            -- Each delta row with its detection config's padding mode.
            SELECT d.*, c.tilepadding FROM deltas d JOIN configs c USING (cfg)
        ),
        long AS (
            -- One row per detection config, filter, side, and metric (undefined deltas dropped).
            UNPIVOT padded ON precision, recall, ap50 INTO NAME metric VALUE delta
        )
        SELECT coalesce(tilepadding, 'all') AS tilepadding, filter, side, metric, count(*) AS detection_configs,
               round(median(delta), 4) + 0.0 AS median, round(min(delta), 4) + 0.0 AS min, round(max(delta), 4) + 0.0 AS max
        FROM long
        GROUP BY GROUPING SETS ((tilepadding, filter, side, metric), (filter, side, metric))
        ORDER BY 1, filter, side, metric
    """)


def operating_config(tetris: pd.DataFrame, naive: pd.Series, loss: float) -> dict:
    """The fastest Tetris configuration within `loss` of the reference pipeline's HOTA (inclusive)."""
    # A private connection, so the caller's tables are untouched.
    con = duckdb.connect()
    # Points table in the shape common.fastest_within expects: the reference row and every Tetris configuration.
    con.register('tetris_frame', tetris[['variant_id', 'throughput_fps', 'HOTA_HOTA']])
    con.execute("""
        CREATE TABLE pts AS
        SELECT 'd' AS dataset, 'reference' AS system, 'naive' AS config, CAST(? AS DOUBLE) AS fps, CAST(? AS DOUBLE) AS accuracy
        UNION ALL
        SELECT 'd', 'tetris', CAST(variant_id AS VARCHAR), CAST(throughput_fps AS DOUBLE), CAST(HOTA_HOTA AS DOUBLE) FROM tetris_frame
    """, [float(naive['throughput_fps']), float(naive['HOTA_HOTA'])])
    # The shared selection rule (inclusive bound, ties to the more accurate configuration).
    chosen = fastest_within(con, 'pts', loss, ['tetris'])[0]
    con.close()
    # Some configuration always meets a bound below the reference unless every one misses it.
    assert chosen['meets_bound'], f'no Tetris configuration within {loss:.0%} HOTA loss'
    return chosen


def video_grid(dataset: str, video: str, tile_size: int) -> tuple[int, int]:
    """(rows, cols) of the tile grid of a test video, as p030 computes it."""
    # Open the clip to read its size.
    cap = cv2.VideoCapture(str(store.dataset(dataset, 'test', video)))
    # Frame size from the container metadata.
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # An unreadable video has no size.
    assert width > 0 and height > 0, f'{dataset}/{video}: unreadable video'
    return height // tile_size, width // tile_size


def sample_frames(con: duckdb.DuckDBPyConnection):
    """Table sampled_frames(cfg, video, frame): the frames p020 classifies at each configuration's sample rate.

    p020 keeps frames whose index is a multiple of the sample rate and always adds the video's
    last frame. Reads configs(cfg, sample_rate) and frames(video, frame).
    """
    # Each video's last frame, then every configuration's sampled frames.
    con.execute("""
        CREATE OR REPLACE TABLE sampled_frames AS
        WITH last AS (SELECT video, max(frame) AS last_frame FROM frames GROUP BY video)
        SELECT c.cfg, f.video, f.frame
        FROM configs c CROSS JOIN frames f JOIN last l ON l.video = f.video
        WHERE f.frame % c.sample_rate = 0 OR f.frame = l.last_frame
    """)


def check_inputs(con: duckdb.DuckDBPyConnection, dataset: str, grids: dict[str, tuple[int, int]], naive_frames: int) -> dict:
    """Consistency checks across the loaded tables; returns counts recorded in the report.

    grids maps every test video to its (rows, cols) tile grid. Reads frames, packed_dets_lines,
    configs, tiles, sampled_frames, gt, and dets.
    """
    # Every test video's grid as a table.
    metadata_table(con, 'video_grids', video=('VARCHAR', list(grids)), grid_rows=('BIGINT', [r for r, _ in grids.values()]),
                   grid_cols=('BIGINT', [c for _, c in grids.values()]))
    # Reference frames per video must be distinct and contiguous from 0, for every test video and no other.
    bad = con.execute("""
        WITH per_video AS (SELECT video, count(*) AS n, count(DISTINCT frame) AS distinct_frames, min(frame) AS lo, max(frame) AS hi
                           FROM frames GROUP BY video)
        SELECT count(*) FROM video_grids g FULL JOIN per_video p USING (video)
        WHERE g.grid_rows IS NULL OR p.n IS NULL OR p.distinct_frames <> p.n OR p.lo <> 0 OR p.hi <> p.n - 1
    """).fetchone()[0]
    assert bad == 0, f'{dataset}: reference frames of {bad} videos are missing, repeated, or not contiguous from 0'
    # The clip frames add up to the tradeoff's frame count.
    total = con.execute('SELECT count(*) FROM frames').fetchone()[0]
    assert total == naive_frames, f'{dataset}: {total} reference frames, tradeoff counts {naive_frames}'
    # Every packed file has one line per clip frame (p050 writes every frame), so none is truncated.
    short = con.execute("""
        WITH expected AS (SELECT c.cfg, f.video, f.frame FROM configs c CROSS JOIN frames f),
             lines AS (SELECT source AS cfg, video, frame FROM packed_dets_lines)
        SELECT (SELECT count(*) FROM expected e ANTI JOIN lines l USING (cfg, video, frame))
             + (SELECT count(*) FROM lines l ANTI JOIN expected e USING (cfg, video, frame))
    """).fetchone()[0]
    assert short == 0, f'{dataset}: {short} packed frame lines missing or beyond the clip'
    # Retained tiles must lie on the configuration's sampled frames.
    off = con.execute("""
        SELECT count(*) FROM tiles t ANTI JOIN sampled_frames s ON s.cfg = t.cfg AND s.video = t.video AND s.frame = t.frame
    """).fetchone()[0]
    assert off == 0, f'{dataset}: {off} retained tiles off their configuration\'s sampled frames'
    # And inside their video's tile grid.
    outside = con.execute("""
        SELECT count(*) FROM tiles t LEFT JOIN video_grids g USING (video)
        WHERE g.grid_rows IS NULL OR t.ty < 0 OR t.tx < 0 OR t.ty >= g.grid_rows OR t.tx >= g.grid_cols
    """).fetchone()[0]
    assert outside == 0, f'{dataset}: {outside} retained tiles outside their video\'s tile grid'
    # Every packed detection must land on a frame its configuration packed (p050 maps it there).
    stray = con.execute("""
        SELECT count(*) FROM dets d ANTI JOIN tiles t ON t.cfg = d.source AND t.video = d.video AND t.frame = d.frame
        WHERE NOT list_contains(?, d.source)
    """, [ORIGINAL_SOURCES]).fetchone()[0]
    assert stray == 0, f'{dataset}: {stray} packed detections on frames without retained tiles'
    # Counts for the report: ground truth beyond the clip's frames, boxes with no area, and the grid sizes.
    return query_dicts(con, """
        SELECT (SELECT count(*) FROM gt ANTI JOIN frames USING (video, frame)) AS gt_boxes_outside_clip_frames,
               (SELECT count(*) FROM gt WHERE x2 <= x1 OR y2 <= y1) AS gt_boxes_zero_area,
               (SELECT count(*) FROM dets WHERE x2 <= x1 OR y2 <= y1) AS detections_zero_area,
               (SELECT count(*) FROM gt JOIN frames USING (video, frame)) AS gt_boxes,
               (SELECT string_agg(g, ', ' ORDER BY g) FROM (SELECT DISTINCT concat(grid_rows, 'x', grid_cols) AS g FROM video_grids)) AS tile_grids
    """)[0]


def load_control_manifest(directory: Path) -> dict:
    """The JPEG control's manifest (measurements-gpu/r1o4_packing_detection_accuracy.py), or a clear error."""
    # The GPU job writes its manifest after each dataset it finishes.
    path = directory / 'manifest.json'
    assert path.is_file(), (f'missing {path}: run measurements-gpu/r1o4_packing_detection_accuracy.py on a GPU first, '
                            'or pass --skip-jpeg-control')
    # The manifest's datasets, configuration, and environment.
    with open(path) as f:
        return json.load(f)


def score_controls(con: duckdb.DuckDBPyConnection, dataset: str, op_cfg: str, videos: list[str], tile_size: int,
                   directory: Path, entry: dict | None) -> dict:
    """The operating point's scores with the JPEG control's runs (rerun, jpeg) on the frames the control covered.

    entry is the manifest's record for this dataset and directory holds the control's files. Replaces
    tables tiles, configs, and dets with the operating point's rows on those frames (gt, frames, and
    sampled_frames are kept), then reruns matching and scoring with every side in CONTROL_SIDES.
    """
    # The control must have run on this dataset's operating point.
    assert entry is not None, f'{dataset}: no JPEG control in the manifest; rerun measurements-gpu/r1o4_packing_detection_accuracy.py'
    assert entry['detection_config'] == op_cfg, \
        f"{dataset}: the JPEG control ran on {entry['detection_config']}, the operating point is {op_cfg}"
    # Both runs of a video are in one file, read once per run.
    paths = {v: str(directory / dataset / f'{v}.jsonl.gz') for v in videos}
    read_boxes(con, 'rerun_dets', [('rerun', v, p) for v, p in paths.items()], 'rerun')
    read_boxes(con, 'jpeg_dets', [('jpeg', v, p) for v, p in paths.items()], 'jpeg')
    # The control covers exactly the frames on which the operating point ran detection.
    uncovered = con.execute("""
        WITH covered AS (SELECT video, frame FROM rerun_dets_lines),
             detected AS (SELECT DISTINCT video, frame FROM tiles WHERE cfg = ?)
        SELECT (SELECT count(*) FROM covered ANTI JOIN detected USING (video, frame))
             + (SELECT count(*) FROM detected ANTI JOIN covered USING (video, frame))
    """, [op_cfg]).fetchone()[0]
    assert uncovered == 0, f'{dataset}: the JPEG control covers {uncovered} frames more or fewer than {op_cfg} detected on'
    # Keep only the operating point's tiles and configuration.
    con.execute('CREATE OR REPLACE TABLE tiles AS SELECT * FROM tiles WHERE cfg = ?', [op_cfg])
    con.execute('CREATE OR REPLACE TABLE configs AS SELECT * FROM configs WHERE cfg = ?', [op_cfg])
    # Reference and packed detections on the covered frames, plus the control's two runs.
    con.execute("""
        CREATE OR REPLACE TABLE dets AS
        SELECT d.* FROM dets d SEMI JOIN rerun_dets_lines l ON l.video = d.video AND l.frame = d.frame
        WHERE d.source IN ('reference', ?)
        UNION ALL SELECT * FROM rerun_dets UNION ALL SELECT * FROM jpeg_dets
    """, [op_cfg])
    # Match, locate, and score every side on the operating point.
    greedy_match(con)
    box_regions(con, tile_size)
    build_config_inputs(con)
    scores = arrange(score(con, CONTROL_SIDES), deltas(con, CONTROL_PAIRS), retained_alternatives(con))[op_cfg]
    # How many frames and boxes each run covered.
    counts = query_dicts(con, """
        SELECT (SELECT count(*) FROM rerun_dets_lines) AS frames, (SELECT count(*) FROM rerun_dets) AS rerun_detections,
               (SELECT count(*) FROM jpeg_dets) AS jpeg_detections
    """)[0]
    return dict(detection_config=op_cfg, **counts, environment=entry.get('environment'), seconds=entry.get('seconds'),
                scores=scores)


def score_dataset(dataset: str, loss: float, control_dir: Path | None = None, control_entry: dict | None = None) -> dict:
    """Every detection config's before/after-packing scores on one dataset, plus the reference on all frames.

    With control_dir set, also scores the JPEG control (control_entry, its manifest record) at the operating point.
    """
    # One in-memory database per dataset.
    con = duckdb.connect()
    # Tetris's validated test configurations and the reference pipeline's row.
    tetris, naive = load_test_tradeoff(dataset)
    rows = tetris.to_dict('records')
    # Nothing to compare without a validated configuration.
    assert rows, f'{dataset}: no validated Tetris test configuration'
    # Each configuration's detection directory (configurations differing only in tracker share one).
    for r in rows:
        r['detection_config'] = detection_config(r)
    cfgs = sorted({r['detection_config'] for r in rows})
    # One tile size per dataset, since box-to-tile spans are computed once.
    sizes = {int(r['tilesize']) for r in rows}
    assert len(sizes) == 1, f'{dataset}: several tile sizes {sizes}'
    tile_size = sizes.pop()
    # Canvas scale 1 makes canvas cells source tiles, as p050's tile-unit offsets and the 'center' filter assume.
    assert all(float(r['canvas_scale']) == 1.0 for r in rows), f'{dataset}: a canvas scale other than 1'
    # Sample rate and padding of each detection configuration (both are part of its name, so one per name).
    params = sorted({(r['detection_config'], int(r['sample_rate']), r['tilepadding']) for r in rows})
    assert len(params) == len(cfgs), f'{dataset}: a detection config with several sample rates or paddings'
    metadata_table(con, 'configs', cfg=('VARCHAR', [p[0] for p in params]), sample_rate=('BIGINT', [p[1] for p in params]),
                   tilepadding=('VARCHAR', [p[2] for p in params]))
    # The test clips, as the pipeline stages list them.
    videos = sorted(f for f in os.listdir(store.dataset(dataset, 'test')) if f.endswith('.mp4'))
    # Ground truth from the detector-independent dataset, exactly as p110 resolves it.
    gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
    gt_files = [('gt', v, str(cache.exec(gt_dataset, 'groundtruth', v, 'tracking.jsonl'))) for v in videos]
    # Reference detections on original frames, and each detection configuration's unpacked detections.
    reference_files = [('reference', v, str(cache.exec(dataset, 'naive', v, 'detection.jsonl'))) for v in videos]
    packed_files = [(c, v, str(cache.exec(dataset, 'ucomp-dets', v, c, 'detections.jsonl'))) for c in cfgs for v in videos]

    # Load boxes; the reference files' lines are the clip frames (one line per decoded frame).
    read_boxes(con, 'gt', gt_files, 'gt')
    read_boxes(con, 'reference_dets', reference_files, 'reference')
    read_boxes(con, 'packed_dets', packed_files, 'packed')
    con.execute('CREATE OR REPLACE TABLE dets AS SELECT * FROM reference_dets UNION ALL SELECT * FROM packed_dets')
    con.execute('CREATE OR REPLACE TABLE frames AS SELECT video, frame FROM reference_dets_lines')
    # Retained tiles, sampled frames, and every video's tile grid (one metadata read per video).
    load_tiles(con, dataset, cfgs, videos)
    sample_frames(con)
    grids = {v: video_grid(dataset, v, tile_size) for v in videos}
    counts = check_inputs(con, dataset, grids, int(naive['frame_count']))

    # Match every detection set against the ground truth, then locate boxes against retained tiles.
    rounds = greedy_match(con)
    box_regions(con, tile_size)

    # Reference detections on every clip frame, for context.
    build_all_frame_inputs(con)
    all_frames = side_entry(score(con, ['reference'])[0])
    # Every configuration, before and after packing, under each filter.
    build_config_inputs(con)
    by_cfg = arrange(score(con, MAIN_SIDES), deltas(con, MAIN_PAIRS), retained_alternatives(con))
    sweep = sweep_summary(con)
    # Frames sampled, frames detected, and retained tiles per detection configuration.
    stats = {r['cfg']: r for r in query_dicts(con, """
        SELECT c.cfg,
               (SELECT count(*) FROM sampled_frames s WHERE s.cfg = c.cfg) AS frames_sampled,
               (SELECT count(*) FROM detected_frames d WHERE d.cfg = c.cfg) AS frames_detected,
               (SELECT count(*) FROM tiles t WHERE t.cfg = c.cfg) AS retained_tiles
        FROM configs c
    """)}
    # The dataset's operating point and its detection config.
    chosen = operating_config(tetris, naive, loss)
    op_cfg = next(r['detection_config'] for r in rows if r['variant_id'] == chosen['config'])
    # The JPEG control at the operating point (it replaces the loaded tables, so it runs last).
    control = None if control_dir is None else score_controls(con, dataset, op_cfg, videos, tile_size, control_dir, control_entry)
    con.close()

    # Variants by name, and the variants sharing each detection config (config metadata, not data).
    ordered = sorted(rows, key=lambda r: r['variant_id'])
    variants = {c: [r['variant_id'] for r in ordered if r['detection_config'] == c] for c in cfgs}
    # The first variant of each detection config carries its parameters; only pruned names fix the tracker.
    first = {c: next(r for r in ordered if r['detection_config'] == c) for c in cfgs}
    detection_configs = {c: dict(sample_rate=int(r['sample_rate']), tracking_accuracy_threshold=accuracy_threshold(r),
                                 relevance_threshold=float(r['relevance_threshold']), tilepadding=r['tilepadding'],
                                 tracker=r['tracker'] if accuracy_threshold(r) is not None else None,
                                 variants=variants[c], **{k: v for k, v in stats[c].items() if k != 'cfg'},
                                 scores=by_cfg[c])
                         for c, r in first.items()}
    # One record per Tetris variant, pointing at its detection config's scores.
    configs = [dict(config=r['variant_id'], detection_config=r['detection_config'], tracker=r['tracker'],
                    fps=round(float(r['throughput_fps']), 1), hota=round(float(r['HOTA_HOTA']), 4),
                    operating_point=r['variant_id'] == chosen['config'])
               for r in ordered]
    # B3D stays provisional until R1.O1's human-verified labels replace its automatic ground truth.
    return dict(provisional=dataset.startswith('jnc'), ground_truth_dataset=gt_dataset, videos=len(videos),
                frames=int(naive['frame_count']), tile_size=tile_size, matching_rounds=rounds, **counts,
                reference=dict(fps=round(float(naive['throughput_fps']), 1), hota=round(float(naive['HOTA_HOTA']), 4),
                               all_frames=all_frames),
                operating_point=dict(config=chosen['config'], detection_config=op_cfg, hota_bound=round(chosen['bound'], 4), loss=loss),
                detection_configs=detection_configs, configs=configs, sweep=sweep, jpeg_control=control)


def method(loss: float) -> dict:
    """What was measured, from which inputs, and how (recorded in every output file)."""
    # One description per input, rule, and caveat.
    return dict(
        item='R1.O4 detection precision/recall before and after packing',
        before_packing='002_naive/detection.jsonl: the reference pipeline detector on every original frame (batch size 1)',
        after_packing='050_uncompressed_detections/<detection config>/detections.jsonl: the same detector on packed canvases '
                      '(p040, batch size 4), mapped back by p050 (center, else first corner on a group cell; boxes touching '
                      'no group cell dropped before 050)',
        ground_truth='<p110 ground-truth dataset>/execution/<video>/003_groundtruth/tracking.jsonl (the boxes the paper scores HOTA against)',
        retained_tiles='033_compressed_frames/<detection config>/{index_maps,offset_lookups}: source tiles on the canvases, padding included',
        frames='per configuration, the frames with at least one retained tile (where it ran detection); '
               'frames_sampled counts p020\'s sampled frames (index % sample_rate == 0, plus each video\'s last frame)',
        matching=f'per frame, greedy by descending confidence, IoU >= {IOU_THRESHOLD}, one to one, all detections against all '
                 'ground truth; matched_nonretained_with_retained_candidate = 0 means COCO\'s prefer-non-ignored matching is identical',
        filters={'frame': 'every box on the frame (includes region-selection loss)',
                 'center': 'box center in a retained tile',
                 'center_cover50': f"box center in a retained tile and >= {FILTERS['center_cover50']:.0%} of its area in retained tiles",
                 'inside': 'box entirely inside retained tiles (area cover >= 1 - 1e-9); no counted box is cut at the region boundary'},
        counting='TP = detection matched to retained ground truth; detections matched to non-retained ground truth are ignored; '
                 'unmatched retained detections are FP. Unmatched detections that fail the filter: reference, rerun, jpeg '
                 'ignored (content Tetris dropped, or boxes crossing the region boundary); packed FP when the box center lies '
                 'outside every retained tile (p050 mapped it through a corner, the detector drew it over black canvas or a '
                 'neighbouring polyomino, and Tetris emits it to the tracker; fp_outside_filter), ignored otherwise; '
                 'packed_ignore_outside ignored (the reference rule on packed detections, a sensitivity variant)',
        ap50='all-point interpolated AP at IoU 0.5 over detections above the detector confidence threshold (truncated curve)',
        operating_point=f'fastest Tetris configuration with test HOTA >= reference HOTA x (1 - {loss})',
        detection_configs='variants differing only in tracker share one 030-050 directory and one score set; '
                          'aggregate over detection_configs, not configs, to count each once',
        sweep='median/min/max of each delta over detection configs, per padding mode and overall; padding modes also differ '
              'in sample rate, thresholds, and retained region, so differences between modes are associations',
        packing_includes='packing as implemented: tile rearrangement, black canvas background, neighbouring polyominoes from '
                         'other frames, JPEG re-encoding of canvases (cv2.imwrite default quality 95), batch-4 inference, '
                         'canvas size of whole tiles (Amsterdam 1260 px canvases for 1280 px frames, which also changes the '
                         'detector letterbox scale); jpeg_control separates the JPEG re-encoding',
        jpeg_control='measurements-gpu/r1o4_packing_detection_accuracy.py: the reference detector (as in p002, batch size 1) on '
                     'every frame the operating point detected on, as decoded (rerun) and after cv2.imencode/imdecode at the '
                     'default JPEG quality p030 uses (jpeg), on another GPU server with the same container image; scored like the '
                     'reference. delta.rerun = environment noise against 002_naive, delta.jpeg_vs_rerun = JPEG alone, '
                     'delta.packed_vs_jpeg = the rest of packing (plus the environment noise)',
        sampling='none: every validated test configuration, every test video, every detected frame',
        estimates='none',
        provisional='B3D (jnc*) scored against automatic ground truth from the same RetinaNet; rerun after R1.O1')


def main():
    # Datasets and the HOTA loss defining the operating point.
    args = parse_args()
    # A loss outside [0, 1) defines no meaningful bound.
    assert 0 <= args.loss < 1, f'--loss must be in [0, 1), got {args.loss}'
    # Results go to measurements-results/<script name>/, next to the JPEG control's outputs.
    out_dir = results_dir(__file__)
    control_dir = None if args.skip_jpeg_control else out_dir / CONTROL_DIR
    # The control's record per dataset (missing outputs fail here, before any work).
    manifest = {} if control_dir is None else load_control_manifest(control_dir)['datasets']

    # Score each dataset.
    scored = {}
    for dataset in args.datasets:
        scored[DATASET_LABELS[dataset]] = score_dataset(dataset, args.loss, control_dir, manifest.get(dataset))
        print(f"{DATASET_LABELS[dataset]}: {len(scored[DATASET_LABELS[dataset]]['detection_configs'])} detection configs scored")
    write_json(out_dir, 'configs.json', dict(method=method(args.loss), datasets=scored))

    # The operating point of each dataset, with the reference on all frames for context.
    points = {}
    for label, s in scored.items():
        # Exactly one variant is the operating point.
        ops = [c for c in s['configs'] if c['operating_point']]
        assert len(ops) == 1, f'{label}: {len(ops)} operating points'
        # Its detection config holds the scores.
        dc = s['detection_configs'][ops[0]['detection_config']]
        points[label] = dict(provisional=s['provisional'], reference_all_frames=s['reference']['all_frames'],
                             hota_bound=s['operating_point']['hota_bound'],
                             **{k: ops[0][k] for k in ('config', 'detection_config', 'fps', 'hota')},
                             **{k: dc[k] for k in ('tilepadding', 'frames_sampled', 'frames_detected', 'retained_tiles', 'scores')},
                             jpeg_control=s['jpeg_control'])
    write_json(out_dir, 'operating_points.json', dict(method=method(args.loss), datasets=points))

    # One line per dataset, filter, and side at the operating point.
    for label, p in points.items():
        ref = p['reference_all_frames']
        print(f"\n{label}{' (provisional)' if p['provisional'] else ''}: {p['config']} | reference on all frames "
              f"P {ref['precision']} R {ref['recall']} AP50 {ref['ap50']}")
        for name, e in p['scores'].items():
            print(f"  {name:15s} n_gt {e['reference']['n_gt']:7d} | before P {e['reference']['precision']} R {e['reference']['recall']} "
                  f"AP50 {e['reference']['ap50']} | after P {e['packed']['precision']} R {e['packed']['recall']} "
                  f"AP50 {e['packed']['ap50']} (fp outside filter {e['packed']['fp_outside_filter']}) | delta {e['delta']['packed']} "
                  f"| ignore-outside delta {e['delta']['packed_ignore_outside']}")
        # The JPEG control's decomposition, when it ran.
        if p['jpeg_control'] is not None:
            for name, e in p['jpeg_control']['scores'].items():
                print(f"  control {name:15s} rerun {e['delta']['rerun']} | jpeg vs rerun {e['delta']['jpeg_vs_rerun']} "
                      f"| packed vs jpeg {e['delta']['packed_vs_jpeg']}")


if __name__ == '__main__':
    main()
