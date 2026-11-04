"""Detection-only packing comparison with annotation-derived tiles on sampled test frames.

Run inside an isolated bombe_container.sh scratch cache. Reuses p030/p040/p050,
without learned classification, ILP pruning, tracking, or track interpolation.
One run packs every requested tile padding against one shared set of original-frame detections.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import duckdb
import numpy as np
import torch

from evaluation.manifests import list_split_videos
from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements import r1o4_packing_detection_accuracy as pa
from measurements.common import DATASET_LABELS
from polyis.io import cache, store
from polyis.models import detector as detectors
from polyis.utilities import build_param_str, mark_detections
from scripts import p030_exec_compress as packing
from scripts import p040_exec_detect as detection
from scripts import p050_exec_uncompress as unpacking


TILE_SIZE = 60
CLASSIFIER = 'OracleGT'
# Padding choices: none (P=0) and square (P=1); both reuse one set of original-frame detections.
PADDINGS = ('none', 'square')
RESULTS = Path('/polyis/measurements-results/r1o4_oracle_packing')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(DATASET_LABELS), required=True)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--padding', nargs='+', choices=PADDINGS, default=list(PADDINGS))
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def config_name(stride, padding):
    # The stage-output configuration name p030/p040/p050 use for one stride and padding.
    return build_param_str(classifier=CLASSIFIER, tilesize=TILE_SIZE, sample_rate=stride,
                           tilepadding=padding, canvas_scale=1.0, tracker=None,
                           tracking_accuracy_threshold=None, relevance_threshold=0.5)


def output_path(stride, padding, dataset):
    # One result file per stride, padding and dataset.
    return RESULTS / f'sr{stride}' / padding / f'{dataset}.json'


def oracle_bitmap(tracks, width, height):
    # Use source pixel coordinates, matching p030's fixed-size tiles, without p021's resizing.
    return mark_detections(tracks, width, height, TILE_SIZE, detection_slice=slice(1, 5)) * 255


def all_frame_scores(con, cfg):
    # Include every sampled frame, including frames with no ground-truth objects or retained tiles.
    pa.build_all_frame_inputs(con)
    con.execute("""
        INSERT INTO det_in
        SELECT ?, 'frame', 'packed', d.video, d.frame, d.did, d.score,
               true, true, m.gid
        FROM dets d JOIN frames USING (video, frame)
        LEFT JOIN matches m ON m.source = d.source AND m.video = d.video
                           AND m.frame = d.frame AND m.did = d.did
        WHERE d.source = ?
    """, [pa.ALL_FRAMES, cfg])
    rows = pa.score(con, ['reference', 'packed'])
    return {row['side']: pa.side_entry(row) for row in rows}


def run_video(dataset, video, stride, paddings, model, methods):
    gt_path = cache.exec(resolve_groundtruth_tracking_dataset(dataset), 'groundtruth', video, 'tracking.jsonl')
    reference = cache.exec(dataset, 'naive', video, f'oracle_detection_sr{stride}.jsonl')
    gt_sha256 = sha256(gt_path)
    cap = cv2.VideoCapture(str(store.dataset(dataset, 'test', video)))
    assert cap.isOpened(), f'{dataset}/{video}: unreadable video'
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = range(0, frame_count, stride)
    assert frame_count > 0 and width >= TILE_SIZE and height >= TILE_SIZE

    # Annotation files contain track IDs, but only their per-frame boxes select tiles.
    tracks = {}
    with gt_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['frame_idx'] % stride == 0:
                tracks[row['frame_idx']] = row['tracks']
    assert all(0 <= frame < frame_count for frame in tracks)
    # The relevance bitmaps do not depend on padding; p030 adds padding when it groups tiles.
    score_path = cache.exec(dataset, 'relevancy', video, f'{CLASSIFIER}_{TILE_SIZE}_{stride}',
                            'score', 'score.jsonl')
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open('w') as stream:
        for frame in samples:
            bitmap = oracle_bitmap(tracks.get(frame, []), width, height)
            stream.write(json.dumps(dict(idx=frame, classification_size=list(bitmap.shape),
                                         classification_hex=bitmap.tobytes().hex())) + '\n')

    # Original-frame detections, shared by every padding; written to a temporary file so a partial file is never reused.
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        temporary = reference.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream, torch.inference_mode():
            for frame in range(frame_count):
                ok, image = cap.read()
                assert ok, f'{dataset}/{video}: could not decode frame {frame}'
                if frame % stride == 0:
                    boxes = detectors.detect(image, model)
                    stream.write(json.dumps(dict(frame_idx=frame, detections=boxes.tolist())) + '\n')
        assert not cap.read()[0], f'{dataset}/{video}: video frame count disagrees with metadata'
        temporary.replace(reference)
    cap.release()
    reference_sha256 = sha256(reference)

    records = {}
    for padding in paddings:
        cfg = config_name(stride, padding)
        done = cache.exec(dataset, 'ucomp-dets', video, cfg, 'oracle_done.json')
        inputs = dict(method=methods[padding], video=video, ground_truth_sha256=gt_sha256,
                      reference_sha256=reference_sha256)
        if done.exists():
            previous = json.loads(done.read_text())
            assert previous['inputs'] == inputs, f'{dataset}/{video}/{padding}: completed inputs changed'
            records[padding] = previous
            continue
        # Calling individual stages avoids their sweep entry points, which clear sibling configurations.
        packing.compress(dataset, 'test', video, CLASSIFIER, TILE_SIZE, stride, padding, 1.0,
                         None, None, 0.5, packing.PackMode.Best_Fit)
        detection.detect_objects(dataset, video, CLASSIFIER, TILE_SIZE, stride, padding, 1.0,
                                 None, None, 0.5, 4, model)
        unpacking.unpack(dataset, 'test', video, CLASSIFIER, TILE_SIZE, stride, padding, 1.0,
                         None, None, 0.5)
        records[padding] = dict(inputs=inputs, frames=frame_count, sampled_frames=len(samples),
                                width=width, height=height,
                                grid_width=width // TILE_SIZE * TILE_SIZE,
                                grid_height=height // TILE_SIZE * TILE_SIZE,
                                canvases=len(list(cache.exec(dataset, 'comp-frames', video, cfg, 'images').glob('*.jpg'))))
        save_json(done, records[padding])
    return records


def score_dataset(dataset, videos, cfg, stride, padding):
    gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
    with duckdb.connect(config={'threads': 4}) as con:
        pa.read_boxes(con, 'gt', [('gt', v, str(cache.exec(gt_dataset, 'groundtruth', v, 'tracking.jsonl')))
                                  for v in videos], 'gt')
        pa.read_boxes(con, 'reference_dets', [('reference', v, str(cache.exec(dataset, 'naive', v,
                                    f'oracle_detection_sr{stride}.jsonl'))) for v in videos], 'reference')
        pa.read_boxes(con, 'packed_dets', [(cfg, v, str(cache.exec(dataset, 'ucomp-dets', v, cfg,
                                    'detections.jsonl'))) for v in videos], 'packed')
        con.execute('CREATE TABLE frames AS SELECT video, frame FROM reference_dets_lines')
        assert con.execute('SELECT count(*) FROM frames WHERE frame % ? != 0', [stride]).fetchone()[0] == 0
        assert con.execute('SELECT count(*) FROM packed_dets WHERE frame % ? != 0', [stride]).fetchone()[0] == 0
        con.execute('DELETE FROM gt WHERE frame % ? != 0', [stride])
        con.execute('CREATE TABLE dets AS SELECT * FROM reference_dets UNION ALL SELECT * FROM packed_dets')
        pa.greedy_match(con)
        full = all_frame_scores(con, cfg)

        # Retain the existing Table 3 scoring filters as a directly comparable sensitivity analysis.
        pa.load_tiles(con, dataset, [cfg], videos)
        pa.metadata_table(con, 'configs', cfg=('VARCHAR', [cfg]), sample_rate=('BIGINT', [stride]),
                          tilepadding=('VARCHAR', [padding]))
        pa.box_regions(con, TILE_SIZE)
        pa.build_config_inputs(con)
        rows = pa.score(con, pa.MAIN_SIDES)
        retained = pa.arrange(rows, pa.deltas(con, pa.MAIN_PAIRS), pa.retained_alternatives(con))[cfg]
        frame_count = con.execute('SELECT count(*) FROM frames').fetchone()[0]
        assert full['reference']['n_gt'] == full['packed']['n_gt']
        assert full['reference']['n_gt'] >= retained['center']['reference']['n_gt']
        return dict(sampled_frames=frame_count, all_sampled_frames=full, retained_content=retained)


def main():
    args = parse_args()
    assert args.stride > 0
    # A stock ace cache must never receive this experiment's stage outputs.
    assert Path('/scratch/polyis-cache').samefile(Path(cache.CACHE_DIR)), 'Use bombe_container.sh with a new scratch cache'
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    paddings = list(dict.fromkeys(args.padding))
    videos = list_split_videos(args.dataset, 'test')
    assert videos
    model = detectors.get_detector(args.dataset, args.gpu, batch_size=4)
    weights = Path(detectors.get_detector_info(args.dataset).get('model_path') or model.cfg.MODEL.WEIGHTS)
    methods = {padding: dict(split='test', stride=args.stride, first_frame=0, tile_size=TILE_SIZE,
                             classifier='test annotation box overlap', tile_padding=padding,
                             ilp_pruning=False, tracking=False, track_interpolation=False,
                             original_batch_size=1, packed_batch_size=4, packed_jpeg_quality=95,
                             weights=str(weights), weights_sha256=sha256(weights),
                             pipeline_sha256={stage: sha256(f'/polyis/scripts/{stage}.py') for stage in
                                              ('p030_exec_compress', 'p040_exec_detect', 'p050_exec_uncompress')})
               for padding in paddings}
    records = {}
    for index, video in enumerate(videos, 1):
        records[video] = run_video(args.dataset, video, args.stride, paddings, model, methods)
        print(f'{DATASET_LABELS[args.dataset]} {index}/{len(videos)} {video}: '
              + ', '.join(f'{p} {r["sampled_frames"]} frames {r["canvases"]} canvases' for p, r in records[video].items()),
              flush=True)
    detectors.delete(model)
    for padding in paddings:
        scores = score_dataset(args.dataset, videos, config_name(args.stride, padding), args.stride, padding)
        result = dict(dataset=args.dataset, label=DATASET_LABELS[args.dataset], method=methods[padding],
                      videos={video: r[padding] for video, r in records.items()}, **scores,
                      environment=dict(gpu=os.environ.get('BOMBE_GPU'), run=os.environ.get('BOMBE_RUN'),
                                       torch=torch.__version__, opencv=cv2.__version__),
                      source_sha256=sha256(__file__))
        save_json(output_path(args.stride, padding, args.dataset), result)
        print(json.dumps(dict(label=result['label'], padding=padding, sampled_frames=result['sampled_frames'],
                              center=result['retained_content']['center']), indent=2), flush=True)


if __name__ == '__main__':
    main()
