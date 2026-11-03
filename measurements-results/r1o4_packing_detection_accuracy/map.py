"""Rescore the saved Table 2 operating points over IoU 0.50:0.05:0.95 without inference."""
import hashlib
import json
from pathlib import Path
import sys

import duckdb

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))
from evaluation.manifests import list_split_videos
from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements import r1o4_packing_detection_accuracy as packing
from measurements.common import DATASET_LABELS, load_test_tradeoff, write_json
from polyis.io import cache


def main():
    source = DIRECTORY / 'operating_points.json'
    saved = json.loads(source.read_text())
    report = dict(method=dict(
        metric='Single-class mean AP over IoU 0.50:0.05:0.95, using COCO\'s 101 recall levels.',
        scope='Same saved 5%-loss operating points, retained tiles, confidence thresholds and matching/ignore rules as packing.json.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), datasets={})
    for dataset, label in DATASET_LABELS.items():
        point = saved['datasets'][label]
        cfg = point['detection_config']
        tetris, naive = load_test_tradeoff(dataset)
        row = tetris.loc[tetris.variant_id == point['config']].iloc[0]
        assert packing.detection_config(row) == cfg
        videos = list_split_videos(dataset, 'test')
        gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
        with duckdb.connect() as con:
            con.execute('SET threads = 4')
            packing.metadata_table(con, 'configs', cfg=('VARCHAR', [cfg]),
                sample_rate=('BIGINT', [int(row['sample_rate'])]), tilepadding=('VARCHAR', [row['tilepadding']]))
            packing.read_boxes(con, 'gt', [('gt', v, str(cache.exec(gt_dataset, 'groundtruth', v, 'tracking.jsonl')))
                                           for v in videos], 'gt')
            packing.read_boxes(con, 'reference_dets', [('reference', v, str(cache.exec(dataset, 'naive', v, 'detection.jsonl')))
                                                       for v in videos], 'reference')
            packing.read_boxes(con, 'packed_dets', [(cfg, v, str(cache.exec(dataset, 'ucomp-dets', v, cfg, 'detections.jsonl')))
                                                    for v in videos], 'packed')
            con.execute('CREATE TABLE dets AS SELECT * FROM reference_dets UNION ALL SELECT * FROM packed_dets')
            con.execute('CREATE TABLE frames AS SELECT video, frame FROM reference_dets_lines')
            packing.load_tiles(con, dataset, [cfg], videos)
            packing.sample_frames(con)
            tile_size = int(row['tilesize'])
            counts = packing.check_inputs(con, dataset,
                {v: packing.video_grid(dataset, v, tile_size) for v in videos}, int(naive['frame_count']))
            assert con.execute('SELECT count(*) FROM tiles').fetchone()[0] == point['retained_tiles']
            packing.box_regions(con, tile_size)
            packing.greedy_match(con)
            packing.build_config_inputs(con)
            for result in packing.score(con, packing.MAIN_SIDES):
                expected = point['scores'][result['filter']][result['side']]
                for metric in packing.METRICS:
                    assert abs(result[metric] - expected[metric]) <= .000051, (label, result)
            results = packing.mean_ap(con)
            report['datasets'][label] = dict(config=point['config'], detection_config=cfg,
                provisional=point['provisional'], counts=counts, scores=results)
        write_json(DIRECTORY, 'map.json', report)
        print(label, [(r['side'], r['map']) for r in results if r['filter'] == 'center'], flush=True)


if __name__ == '__main__':
    main()
