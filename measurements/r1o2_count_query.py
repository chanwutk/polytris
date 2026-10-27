#!/usr/local/bin/python
"""R1.O2: OTIF's vehicle-counting query, scored with OTIF's own rule and metric.

The counting rule and accuracy are transcribed from OTIF's pipeline2/lib/eval.go:

  GetGoodTracks
    CalDoT1 and CalDoT2 define a top and a bottom polygon. A track counts in
    segment 0 when its first box center lies in the top polygon and its last in
    the bottom one, and in segment 1 for the reverse. Amsterdam counts every
    track lasting at least 10 frames whose first-to-last center moves at least
    50 pixels.

  Eval
    Per video and segment, accuracy is 1 when the counts match, 0 when either is
    zero, and 1 - |predicted - truth| / truth otherwise. Segment scores are
    averaged per video, then across the videos listed in the ground truth.

OTIF measures the Amsterdam track length in frames of its own video (30 FPS),
while every system here is tracked on the 15 FPS experiment clips. The
threshold is therefore rescaled per video by the frame-rate ratio, read from
the videos themselves (OTIF's test videos do not all share one frame rate).
Polygons and distances are in pixels, so clip and OTIF video resolutions must
match. OTIF rounds box centers to integers; centers here stay fractional.

Ground truth is the per-video count table shipped with the OTIF benchmark
(test/gt.txt). It is independent of every evaluated detector and tracker.

Each system is scored over all its test-split configurations: Tetris's
validated configurations, the reference pipeline, OTIF, and LEAP. Operating
points then hold every system to one bound, its fastest configuration whose
counting accuracy is within --losses of the reference pipeline's. A further
"leap_fit" bound is lowered, per dataset, to the best counting accuracy LEAP
reaches, so all three systems qualify and can be compared at LEAP's accuracy.

Writes measurements-results/r1o2_count_query/:
  configs.json           per dataset, the rule applied and every configuration's accuracy, fps, and HOTA
  operating_points.json  per bound ("0.05", "0.10", "leap_fit") and dataset, each system's fastest
                         configuration within the bound, its speedup over the reference, and
                         Tetris's speedup over OTIF and LEAP
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import duckdb

from measurements.common import (PRIOR_SYSTEMS, TRACK_COLUMNS, assert_files_exist, best_accuracy_bounds, operating_points,
                                 output_path, print_operating_points, query_dicts, results_dir, system_configs, write_json)
from polyis.io import store


# Polygons copied verbatim from OTIF's pipeline2/lib/eval.go, in OTIF video pixels.
CALDOT1_TOP = ((458, 344), (598, 138), (289, 136), (391, 298))
CALDOT1_BOTTOM = ((448, 369), (392, 480), (0, 480), (0, 37))
CALDOT2_TOP = ((334, 344), (588, 167), (100, 167), (104, 297))
CALDOT2_BOTTOM = ((322, 363), (0, 288), (0, 480), (322, 480))
# Amsterdam's travel rule from eval.go: minimum duration in OTIF video frames, minimum distance in pixels.
OTIF_MIN_DURATION = 10
OTIF_MIN_DISTANCE = 50


@dataclass(frozen=True)
class CountDataset:
    """One dataset of the counting query: our cache name, OTIF's name, and its rule."""
    key: str
    label: str
    otif_name: str
    kind: str
    top: tuple = ()
    bottom: tuple = ()


# The three datasets with OTIF counting ground truth (B3D has none).
DATASETS = (
    CountDataset('caldot1-y05', 'CalDoT1', 'caldot1', 'polygon', CALDOT1_TOP, CALDOT1_BOTTOM),
    CountDataset('caldot2-y05', 'CalDoT2', 'caldot2', 'polygon', CALDOT2_TOP, CALDOT2_BOTTOM),
    CountDataset('ams-y05', 'Amsterdam', 'amsterdam', 'travel'),
)


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O2: OTIF counting query over every configuration')
    parser.add_argument('--datasets', nargs='+', default=[d.key for d in DATASETS])
    parser.add_argument('--losses', nargs='+', type=float, default=[0.05, 0.10])
    return parser.parse_args()


def read_ground_truth(path: Path) -> dict[int, list[int]]:
    """OTIF's gt.txt as {video index: [count per segment]}, parsed like eval.go."""
    counts = {}
    # Read the small tab-separated table line by line, as eval.go does.
    for line in open(path):
        # Split on tabs after trimming whitespace.
        parts = line.strip().split('\t')
        # Skip blank lines and the '#' header.
        if len(parts) < 2 or parts[0] == '#':
            continue
        # First column is the video index; the rest are per-segment counts.
        counts[int(parts[0])] = [int(p) for p in parts[1:]]
    # A ground truth with no videos would make every accuracy undefined.
    assert counts, f'{path}: no ground-truth rows'
    # Every video must have the same number of segments.
    assert len({len(c) for c in counts.values()}) == 1, f'{path}: inconsistent segment counts'
    return counts


def create_edges(con: duckdb.DuckDBPyConnection, regions: dict[str, tuple]):
    """Table edges(region, ax, ay, bx, by): every polygon edge, each vertex to its successor."""
    # Build one row per edge, wrapping the last vertex back to the first.
    rows = [(name, *poly[i], *poly[(i + 1) % len(poly)]) for name, poly in regions.items() for i in range(len(poly))]
    # Replace any edges from a previous dataset.
    con.execute('CREATE OR REPLACE TABLE edges (region VARCHAR, ax DOUBLE, ay DOUBLE, bx DOUBLE, "by" DOUBLE)')
    # Insert the edges, if this dataset has polygons.
    if rows:
        con.executemany('INSERT INTO edges VALUES (?, ?, ?, ?, ?)', rows)


def load_endpoints(con: duckdb.DuckDBPyConnection, filenames: list[str]):
    """Table endpoints(filename, track_id, f0, x0, y0, f1, x1, y1): each track's first and last box center."""
    # Reading zero files would fail inside DuckDB with a less clear message.
    assert filenames, 'no tracking files to read'
    # Read every tracking file at once, one row per box, then keep each track's earliest and latest center.
    con.execute(f"""
        CREATE OR REPLACE TABLE endpoints AS
        WITH boxes AS (
            SELECT filename, frame_idx, unnest(tracks) AS box
            FROM read_json(?, filename = true, format = 'newline_delimited', columns = {TRACK_COLUMNS})
        )
        SELECT filename, CAST(box[1] AS BIGINT) AS track_id,
               min(frame_idx) AS f0, max(frame_idx) AS f1,
               arg_min((box[2] + box[4]) / 2, frame_idx) AS x0, arg_min((box[3] + box[5]) / 2, frame_idx) AS y0,
               arg_max((box[2] + box[4]) / 2, frame_idx) AS x1, arg_max((box[3] + box[5]) / 2, frame_idx) AS y1
        FROM boxes
        GROUP BY filename, track_id
    """, [filenames])


def locate_endpoints(con: duckdb.DuckDBPyConnection):
    """Table located(filename, track_id, endpoint, region): regions containing each track's start and end.

    Even-odd ray casting, as gomapinfer's Polygon.Contains: a point is inside a region when a ray
    to its right crosses an odd number of that region's edges.
    """
    con.execute("""
        CREATE OR REPLACE TABLE located AS
        WITH ends AS (
            SELECT filename, track_id, 'start' AS endpoint, x0 AS x, y0 AS y FROM endpoints
            UNION ALL
            SELECT filename, track_id, 'end' AS endpoint, x1 AS x, y1 AS y FROM endpoints
        )
        SELECT p.filename, p.track_id, p.endpoint, g.region
        FROM ends p JOIN edges g
          ON (g.ay > p.y) <> (g."by" > p.y)
         AND p.x < g.ax + (p.y - g.ay) / (g."by" - g.ay) * (g.bx - g.ax)
        GROUP BY p.filename, p.track_id, p.endpoint, g.region
        HAVING count(*) % 2 = 1
    """)


def count_tracks(con: duckdb.DuckDBPyConnection, kind: str):
    """Table counted(filename, track_id, segment): OTIF's GetGoodTracks over every loaded track.

    The travel rule reads each video's minimum duration from table video_rules(video, min_duration),
    joined to tracks through files(system, config, video, filename).
    """
    if kind == 'polygon':
        # Locate both endpoints, then apply eval.go's order: top->bottom is segment 0, else bottom->top is 1.
        locate_endpoints(con)
        con.execute("""
            CREATE OR REPLACE TABLE counted AS
            SELECT * FROM (
                SELECT s.filename, s.track_id,
                       min(CASE WHEN s.region = 'top' AND e.region = 'bottom' THEN 0
                                WHEN s.region = 'bottom' AND e.region = 'top' THEN 1 END) AS segment
                FROM located s JOIN located e
                  ON s.filename = e.filename AND s.track_id = e.track_id
                 AND s.endpoint = 'start' AND e.endpoint = 'end'
                GROUP BY s.filename, s.track_id
            )
            WHERE segment IS NOT NULL
        """)
    else:
        # Travel rule: keep tracks at least their video's minimum duration long that move at least the minimum distance.
        con.execute("""
            CREATE OR REPLACE TABLE counted AS
            SELECT DISTINCT p.filename, p.track_id, 0 AS segment
            FROM endpoints p
            JOIN files f ON f.filename = p.filename
            JOIN video_rules r ON r.video = f.video
            WHERE p.f1 - p.f0 >= r.min_duration AND sqrt(power(p.x1 - p.x0, 2) + power(p.y1 - p.y0, 2)) >= ?
        """, [OTIF_MIN_DISTANCE])


def score_configs(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """OTIF accuracy, predicted totals, throughput, and HOTA per (system, config).

    Reads tables files(system, config, video, filename), each configuration's tracking file per
    video; truth(video, segment, count), the ground-truth counts; counted, from count_tracks; and
    configs(system, config, fps, hota).
    """
    return query_dicts(con, """
        WITH per_segment AS (
            -- Predicted count per configuration, video, and segment (zero when nothing was counted).
            SELECT f.system, f.config, f.video, t.segment, t.count AS truth, count(c.track_id) AS predicted
            FROM files f
            JOIN truth t ON t.video = f.video
            LEFT JOIN counted c ON c.filename = f.filename AND c.segment = t.segment
            GROUP BY f.system, f.config, f.video, t.segment, t.count
        ),
        scored AS (
            -- eval.go's per-segment accuracy.
            SELECT *, CASE WHEN predicted = truth THEN 1.0
                           WHEN predicted = 0 OR truth = 0 THEN 0.0
                           ELSE 1 - abs(predicted - truth) / CAST(truth AS DOUBLE) END AS accuracy
            FROM per_segment
        ),
        per_video AS (
            -- Mean over segments within each video.
            SELECT system, config, video, avg(accuracy) AS accuracy FROM scored GROUP BY system, config, video
        ),
        totals AS (
            -- Predicted totals per segment, in segment order.
            SELECT system, config, list(predicted ORDER BY segment) AS predicted_per_segment
            FROM (SELECT system, config, segment, sum(predicted)::BIGINT AS predicted FROM scored
                  GROUP BY system, config, segment)
            GROUP BY system, config
        )
        -- Mean over videos, with each configuration's throughput and HOTA attached.
        SELECT v.system, v.config, any_value(c.fps) AS fps, any_value(c.hota) AS hota,
               round(avg(v.accuracy), 4) AS accuracy, count(*) AS videos, any_value(t.predicted_per_segment) AS predicted_per_segment
        FROM per_video v JOIN totals t USING (system, config) JOIN configs c USING (system, config)
        GROUP BY v.system, v.config
        ORDER BY v.system, v.config
    """)


def frame_rate_and_size(path: Path) -> tuple[float, int, int]:
    """(fps, width, height) of a video file."""
    # Fail clearly on a missing video rather than reading zeros.
    assert path.exists(), f'missing video {path}'
    cap = cv2.VideoCapture(str(path))
    # Read frame rate and resolution from the container metadata.
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # A zero frame rate means the file could not be decoded.
    assert fps > 0, f'{path}: unreadable frame rate'
    return fps, width, height


def scaled_min_duration(otif_fps: float, clip_fps: float) -> int:
    """OTIF's minimum duration converted from OTIF video frames to clip frames.

    The clip must be OTIF's video subsampled by a whole factor, so the threshold
    converts exactly; anything else raises.
    """
    # Subsampling factor between OTIF's video and the experiment clip.
    factor = otif_fps / clip_fps
    # Only a whole factor gives an exact conversion (allowing 29.97 vs 14.985 rounding).
    assert abs(factor - round(factor)) < 1e-3 and round(factor) >= 1, \
        f'clip fps {clip_fps} is not OTIF fps {otif_fps} divided by a whole number'
    # The threshold in OTIF frames must also divide evenly.
    assert OTIF_MIN_DURATION % round(factor) == 0, \
        f'{OTIF_MIN_DURATION} OTIF frames is not a whole number of clip frames at factor {round(factor)}'
    return OTIF_MIN_DURATION // round(factor)


def check_videos(dataset: CountDataset, videos: list[int]) -> tuple[dict, list[tuple[int, int]]]:
    """Check clip/OTIF resolutions for every ground-truth video and convert OTIF's duration per video.

    Returns the rule applied (with the frame-rate pairs found) and (video, min_duration) rows.
    OTIF's test videos need not share one frame rate, so the threshold is converted per video.
    """
    # Read OTIF's original test video and our experiment clip of the same index (per-file I/O).
    meta = [(frame_rate_and_size(store.otif(dataset.otif_name, 'test', 'video', f'{idx}.mp4')),
             frame_rate_and_size(store.dataset(dataset.key, 'test', f'te{idx:02d}.mp4'))) for idx in videos]
    # Pixel rules need identical resolutions.
    mismatched = [idx for idx, ((_, ow, oh), (_, cw, ch)) in zip(videos, meta) if (ow, oh) != (cw, ch)]
    assert not mismatched, f'{dataset.label}: OTIF and clip resolutions differ for videos {mismatched}'
    # Convert OTIF's duration threshold for each video's frame-rate pair.
    rows = [(idx, scaled_min_duration(o[0], c[0])) for idx, (o, c) in zip(videos, meta)]
    # Summarize the frame-rate pairs and which videos use each, for the report.
    groups: dict[tuple, list[int]] = {}
    for idx, (o, c) in zip(videos, meta):
        groups.setdefault((round(o[0], 3), round(c[0], 3)), []).append(idx)
    rates = [dict(otif_fps=o, clip_fps=c, min_duration_clip_frames=scaled_min_duration(o, c), videos=v)
             for (o, c), v in sorted(groups.items())]
    sizes = sorted({(c[1], c[2]) for _, c in meta})
    rule = dict(kind=dataset.kind, frame_rates=rates, resolutions=sizes, min_distance_pixels=OTIF_MIN_DISTANCE,
                min_duration_otif_frames=OTIF_MIN_DURATION)
    return rule, rows


def config_files(dataset: CountDataset, videos: list[int]) -> tuple[list[tuple], list[tuple]]:
    """Rows for tables files(system, config, video, filename) and configs(system, config, fps, hota)."""
    # Our automatic ground truth (scored for reference, no throughput) and every system's configurations.
    configs = [('ground truth', '003_groundtruth', None, None)]
    configs += [(r['system'], r['config'], r['fps'], r['hota']) for r in system_configs(dataset.key)]
    # One file per configuration and ground-truth video.
    files = [(s, c, idx, str(output_path(dataset.key, s, c, f'te{idx:02d}.mp4'))) for s, c, _, _ in configs for idx in videos]
    return files, configs


def score_dataset(con: duckdb.DuckDBPyConnection, dataset: CountDataset) -> dict:
    """Rule applied and every configuration's counting accuracy for one dataset."""
    # Ground-truth counts and the videos they cover.
    truth = read_ground_truth(store.otif(dataset.otif_name, 'test', 'gt.txt'))
    videos = sorted(truth)
    rule, video_rules = check_videos(dataset, videos)
    files, configs = config_files(dataset, videos)
    # Every ground-truth video must have tracks, or the mean would cover fewer videos for some systems.
    assert_files_exist([f for *_, f in files], f'{dataset.label} tracking files')

    # Load inputs into DuckDB tables.
    con.execute('CREATE OR REPLACE TABLE files (system VARCHAR, config VARCHAR, video INTEGER, filename VARCHAR)')
    con.executemany('INSERT INTO files VALUES (?, ?, ?, ?)', files)
    con.execute('CREATE OR REPLACE TABLE truth (video INTEGER, segment INTEGER, count INTEGER)')
    con.executemany('INSERT INTO truth VALUES (?, ?, ?)',
                    [(v, s, n) for v, counts in truth.items() for s, n in enumerate(counts)])
    con.execute('CREATE OR REPLACE TABLE configs (system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE)')
    con.executemany('INSERT INTO configs VALUES (?, ?, ?, ?)', configs)
    create_edges(con, {'top': dataset.top, 'bottom': dataset.bottom} if dataset.kind == 'polygon' else {})
    con.execute('CREATE OR REPLACE TABLE video_rules (video INTEGER, min_duration INTEGER)')
    con.executemany('INSERT INTO video_rules VALUES (?, ?)', video_rules)

    # Endpoints of every track in every file, then OTIF's counting rule, then OTIF's accuracy.
    load_endpoints(con, sorted({f for *_, f in files}))
    count_tracks(con, dataset.kind)
    records = score_configs(con)
    # Every configuration must have been scored on every ground-truth video.
    assert len(records) == len(configs) and all(r['videos'] == len(videos) for r in records), \
        f'{dataset.label}: some configurations were not scored on every video'
    truth_totals = [sum(c[i] for c in truth.values()) for i in range(len(truth[videos[0]]))]
    return dict(rule=rule, videos=len(videos), truth_per_segment=truth_totals, records=records)


def load_points(con: duckdb.DuckDBPyConnection, scored: dict):
    """Table count_points(dataset, system, config, fps, hota, accuracy) from every dataset's scored configurations."""
    con.execute('CREATE OR REPLACE TABLE count_points (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE, accuracy DOUBLE)')
    # Our automatic ground truth has no throughput, so it is not a candidate.
    con.executemany('INSERT INTO count_points VALUES (?, ?, ?, ?, ?, ?)',
                    [(label, r['system'], r['config'], r['fps'], r['hota'], r['accuracy'])
                     for label, s in scored.items() for r in s['records'] if r['system'] != 'ground truth'])


def main():
    args = parse_args()
    # Restrict to the requested datasets, failing on unknown names.
    by_key = {d.key: d for d in DATASETS}
    unknown = set(args.datasets) - set(by_key)
    assert not unknown, f'unknown datasets {sorted(unknown)}; choose from {sorted(by_key)}'
    out_dir = results_dir(__file__)
    con = duckdb.connect()

    # Score every configuration of every system on each dataset.
    scored = {by_key[k].label: score_dataset(con, by_key[k]) for k in args.datasets}
    for label, s in scored.items():
        print(f"{label}: {len(s['records'])} configurations scored, rule {s['rule']}")
    write_json(out_dir, 'configs.json', scored)

    # Operating points at each requested relative bound, and at the bound LEAP just meets on each dataset.
    load_points(con, scored)
    systems = ['tetris', *PRIOR_SYSTEMS]
    points = {f'{loss:.2f}': operating_points(con, 'count_points', systems, loss=loss) for loss in args.losses}
    points['leap_fit'] = operating_points(con, 'count_points', systems,
                                          bounds=best_accuracy_bounds(con, 'count_points', 'leap'))
    write_json(out_dir, 'operating_points.json', points)
    print_operating_points(points, 'counting accuracy')

if __name__ == '__main__':
    main()
