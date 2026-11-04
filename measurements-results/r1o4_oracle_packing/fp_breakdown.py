"""Break down retained-content false positives of the oracle-packing runs by type, original vs. packed.

Run in a bombe container whose /polyis-cache is one dataset's oracle-packing scratch cache:
  python measurements-results/r1o4_oracle_packing/fp_breakdown.py --dataset caldot2-y05 --stride 16 --padding none tr square
A false positive is an unmatched detection that passes the center filter (Table 3's scoring). Its type comes from
its best IoU with any ground-truth box on the same frame:
  duplicate      IoU >= 0.5: the object was already matched by a higher-scoring detection;
  localization   0.1 <= IoU < 0.5: a box on an object but misplaced, cut, or merged;
  background     IoU < 0.1: no annotated object there.
For packed false positives, 'also_original' counts those overlapping (IoU >= 0.5) an original-frame detection.
It also characterizes packed false positives whose mapped center lies outside retained content ('center_outside').
The saved sr16/fp_breakdown/*.json came from the runs before r1o4_oracle_packing.py enabled p050's
drop_center_outside, and show that those boxes (CalDoT2 only) explained its packed precision drop.
"""
import argparse
import json
from pathlib import Path

import duckdb

from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from evaluation.manifests import list_split_videos
from measurements import r1o4_packing_detection_accuracy as pa
from measurements.r1o4_oracle_packing_stats import TILE_SIZE, config_name
from polyis.io import cache


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--padding', nargs='+', default=['none', 'square'])
    return parser.parse_args()


def breakdown(dataset: str, stride: int, padding: str) -> list[dict]:
    """False positives by side and type for one padding, under Table 3's center filter."""
    videos = sorted(list_split_videos(dataset, 'test'))
    cfg = config_name(stride, padding)
    gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
    con = duckdb.connect(config={'threads': 4})
    # The same inputs and matching as r1o4_oracle_packing.score_dataset.
    pa.read_boxes(con, 'gt', [('gt', v, str(cache.exec(gt_dataset, 'groundtruth', v, 'tracking.jsonl'))) for v in videos], 'gt')
    pa.read_boxes(con, 'reference_dets', [('reference', v, str(cache.exec(dataset, 'naive', v,
                  f'oracle_detection_sr{stride}.jsonl'))) for v in videos], 'reference')
    pa.read_boxes(con, 'packed_dets', [(cfg, v, str(cache.exec(dataset, 'ucomp-dets', v, cfg, 'detections.jsonl')))
                                      for v in videos], 'packed')
    con.execute('CREATE TABLE frames AS SELECT video, frame FROM reference_dets_lines')
    con.execute('DELETE FROM gt WHERE frame % ? != 0', [stride])
    con.execute('CREATE TABLE dets AS SELECT * FROM reference_dets UNION ALL SELECT * FROM packed_dets')
    pa.greedy_match(con)
    pa.load_tiles(con, dataset, [cfg], videos)
    pa.metadata_table(con, 'configs', cfg=('VARCHAR', [cfg]), sample_rate=('BIGINT', [stride]),
                      tilepadding=('VARCHAR', [padding]))
    pa.box_regions(con, TILE_SIZE)
    pa.build_config_inputs(con)
    # Unmatched detections that count against precision under the center filter, with their boxes.
    con.execute("""
        CREATE TABLE fp AS
        SELECT i.side, d.source, d.video, d.frame, d.did, d.score, d.x1, d.y1, d.x2, d.y2
        FROM det_in i JOIN dets d ON d.video = i.video AND d.frame = i.frame AND d.did = i.did
             AND d.source = CASE WHEN i.side = 'packed' THEN i.cfg ELSE i.side END
        WHERE i.filter = 'center' AND i.passes AND i.gid IS NULL
    """)
    # Best IoU with ground truth on the frame, and (for packed) with any original-frame detection.
    iou = """greatest(0, least(a.x2, b.x2) - greatest(a.x1, b.x1)) * greatest(0, least(a.y2, b.y2) - greatest(a.y1, b.y1))
             / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1)
                - greatest(0, least(a.x2, b.x2) - greatest(a.x1, b.x1)) * greatest(0, least(a.y2, b.y2) - greatest(a.y1, b.y1)))"""
    rows = con.execute(f"""
        WITH best AS (
            SELECT a.side, a.did, a.video, a.frame, a.score,
                   coalesce((SELECT max({iou}) FROM gt b WHERE b.video = a.video AND b.frame = a.frame), 0) AS gt_iou,
                   coalesce((SELECT max({iou}) FROM dets b WHERE b.source = 'reference' AND b.video = a.video
                             AND b.frame = a.frame), 0) AS ref_iou
            FROM fp a
        )
        SELECT side,
               CASE WHEN gt_iou >= 0.5 THEN 'duplicate' WHEN gt_iou >= 0.1 THEN 'localization' ELSE 'background' END AS kind,
               count(*) AS n, round(median(score), 3) AS median_score,
               sum(CASE WHEN side = 'packed' AND ref_iou >= 0.5 THEN 1 ELSE 0 END) AS also_original
        FROM best GROUP BY ALL ORDER BY side, kind
    """).fetchall()
    totals = dict(con.execute("""SELECT side, count(*) FROM det_in WHERE filter = 'center' AND passes GROUP BY side""").fetchall())
    out = [dict(padding=padding, side=s, kind=k, fp=n, median_score=m, also_original=o, detections=totals[s])
           for s, k, n, m, o in rows]
    # Packed false positives whose mapped center lies outside retained content (counted by Table 3's scoring).
    outside = con.execute(f"""
        WITH o AS (
            SELECT d.*, coalesce(r.cover, 0) AS cover
            FROM det_in i JOIN dets d ON d.source = i.cfg AND d.video = i.video AND d.frame = i.frame AND d.did = i.did
            LEFT JOIN regions r ON r.cfg = i.cfg AND r.role = 'det' AND r.source = d.source AND r.video = d.video
                               AND r.frame = d.frame AND r.bid = d.did
            WHERE i.filter = 'center' AND i.side = 'packed' AND i.gid IS NULL AND NOT i.center_in
        ), tp AS (
            SELECT d.* FROM det_in i JOIN dets d ON d.source = i.cfg AND d.video = i.video AND d.frame = i.frame AND d.did = i.did
            WHERE i.filter = 'center' AND i.side = 'packed' AND i.gid IS NOT NULL AND i.passes
        )
        SELECT count(*), round(median(score), 3),
               round(median(x2 - x1), 1), round(median(y2 - y1), 1),
               (SELECT round(median(x2 - x1), 1) FROM tp), (SELECT round(median(y2 - y1), 1) FROM tp),
               round(avg(CASE WHEN cover > 0 THEN 1 ELSE 0 END), 3),
               round(avg(CASE WHEN coalesce((SELECT max({iou}) FROM gt b WHERE b.video = a.video AND b.frame = a.frame), 0) >= 0.1
                              THEN 1 ELSE 0 END), 3),
               round(avg(CASE WHEN x1 < 0 OR y1 < 0 THEN 1 ELSE 0 END), 3),
               count(DISTINCT video)
        FROM o a
    """).fetchone()
    keys = ['n', 'median_score', 'median_w', 'median_h', 'tp_median_w', 'tp_median_h', 'touches_retained',
            'overlaps_gt', 'negative_coords', 'videos']
    out.append(dict(padding=padding, side='packed', kind='center_outside', **dict(zip(keys, outside))))
    return out


def main():
    args = parse_args()
    out = [row for padding in args.padding for row in breakdown(args.dataset, args.stride, padding)]
    path = Path(f'/polyis/measurements-results/r1o4_oracle_packing/sr{args.stride}/fp_breakdown/{args.dataset}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    for row in out:
        print(row)


if __name__ == '__main__':
    main()
