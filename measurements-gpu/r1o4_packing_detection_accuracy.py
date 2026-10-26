#!/usr/local/bin/python
"""R1.O4 JPEG control (GPU): the reference detector on the operating point's frames, as decoded and after JPEG.

Packing as implemented re-encodes every canvas as JPEG (p030 writes it with cv2.imwrite at the
default quality, 95) before p040 detects on it. This job separates that step from the rest of
packing. On every frame where the operating point's detection configuration ran detection (the
frames on its 033 canvases), it runs the reference pipeline's detector as p002 does (batch size 1,
frames decoded with cv2, 32 warm-up frames) twice:

  plain  the decoded frame, which should reproduce 002_naive (the environment's noise floor);
  jpeg   the decoded frame after cv2.imencode('.jpg') and cv2.imdecode, i.e. the JPEG round trip
         a canvas goes through, at the same default quality.

measurements/r1o4_packing_detection_accuracy.py scores both like the reference, on the same
retained tiles and ground truth. The operating point uses that script's rule (fastest Tetris
configuration within --loss of the reference pipeline's HOTA).

Writes measurements-results/r1o4_packing_detection_accuracy/jpeg_control/ (never the cache):
  <dataset>/<video>.jsonl.gz  one line per covered frame: frame_idx, plain, jpeg ([x1, y1, x2, y2, score] lists)
  manifest.json               per finished dataset: configuration, frames per video, runtime; and the environment
Run on a GPU server in the polyis container, e.g. ./run measurements-gpu/r1o4_packing_detection_accuracy.py.
"""
import argparse
import gzip
import json
import os
import platform
import socket
import time
from pathlib import Path

import cv2
import torch
import ultralytics

import polyis.models.detector
from measurements import r1o4_packing_detection_accuracy as pa
from measurements.common import DATASET_LABELS, load_test_tradeoff, results_dir, write_json
from polyis.io import cache, store


# Warm-up frames before timing, as p002 uses.
WARMUP_FRAMES = 32


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O4 JPEG control: reference detector on decoded and JPEG round-tripped frames')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def jpeg_round_trip(frame):
    """The frame after encoding as JPEG at cv2's default quality and decoding, as a canvas is written and read."""
    # p030 writes canvases with cv2.imwrite (default parameters); p040 reads them with cv2.imread (color).
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok, 'JPEG encoding failed'
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def operating_frames(dataset: str, loss: float, videos: list[str]) -> tuple[dict, str, dict[str, list[int]]]:
    """The operating point's tradeoff row, its detection config, and the frames it detected on in each video."""
    # The operating point, chosen as the measurement script chooses it.
    tetris, naive = load_test_tradeoff(dataset)
    chosen = pa.operating_config(tetris, naive, loss)
    row = tetris[tetris['variant_id'] == chosen['config']].iloc[0].to_dict()
    cfg = pa.detection_config(row)
    # Frames with at least one tile on the configuration's canvases.
    frames = {v: sorted({int(f) for f in pa.canvas_tiles(Path(cache.exec(dataset, 'comp-frames', v, cfg)))[:, 0]}) for v in videos}
    return row, cfg, frames


def detect_video(dataset: str, video: str, frames: list[int], detector, path: Path) -> int:
    """Detect on the listed frames of one test video, plain and after JPEG, and write one line per frame; returns frames read."""
    # Decode sequentially, as p002 does.
    cap = cv2.VideoCapture(str(store.dataset(dataset, 'test', video)))
    assert cap.isOpened(), f'{dataset}/{video}: cannot open'
    # Frames to detect on, and how many were written.
    wanted = set(frames)
    written = 0
    # Write compressed lines as they come.
    with gzip.open(path, 'wt') as f:
        # Index of the next decoded frame.
        index = 0
        # Read up to the last wanted frame.
        while wanted and index <= frames[-1]:
            ok, frame = cap.read()
            # The clip ended before a wanted frame: the frame list and the video disagree.
            assert ok, f'{dataset}/{video}: frame {index} unreadable'
            if index in wanted:
                # Both runs on the same decoded frame.
                plain = polyis.models.detector.detect(frame, detector)
                jpeg = polyis.models.detector.detect(jpeg_round_trip(frame), detector)
                f.write(json.dumps(dict(frame_idx=index, plain=[] if plain is None else plain.tolist(),
                                        jpeg=[] if jpeg is None else jpeg.tolist())) + '\n')
                written += 1
            index += 1
    cap.release()
    # Every wanted frame was detected on.
    assert written == len(wanted), f'{dataset}/{video}: {written} of {len(wanted)} frames written'
    return index


def warm_up(dataset: str, video: str, detector):
    """Run the detector on the first frames of a video so timing and cuDNN algorithm choice settle, as p002 does."""
    # Decode the first frames.
    cap = cv2.VideoCapture(str(store.dataset(dataset, 'test', video)))
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        # A short clip ends the warm-up early.
        if not ok:
            break
        polyis.models.detector.detect(frame, detector)
    cap.release()
    # Wait for the GPU before timing.
    torch.cuda.synchronize()


def environment(gpu: int) -> dict:
    """Host, GPU, and library versions, recorded so the noise floor against 002_naive can be explained."""
    # Everything that could change the detector output between two runs.
    return dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(gpu), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), ultralytics=ultralytics.__version__,
                opencv=cv2.__version__, python=platform.python_version(),
                container_image=os.environ.get('POLYIS_IMAGE', 'polyis (unrecorded tag)'))


def main():
    # Datasets, the operating point's loss, and the GPU.
    args = parse_args()
    # The control's directory inside the measurement's result directory (same script name).
    out_dir = results_dir(__file__) / pa.CONTROL_DIR
    out_dir.mkdir(exist_ok=True)
    # Keep the records of datasets finished by an earlier run.
    manifest_path = out_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(datasets={})
    manifest.update(environment=environment(args.gpu), method=dict(
        frames='every frame with a tile on the operating point configuration\'s 033 canvases',
        plain='reference detector (polyis.models.detector, batch size 1) on the cv2-decoded frame',
        jpeg='the same after cv2.imencode(".jpg") with default parameters (quality 95, as p030) and cv2.imdecode',
        operating_point=f'fastest Tetris configuration with test HOTA >= reference HOTA x (1 - {args.loss})',
        sampling='none', estimates='none'))

    for dataset in args.datasets:
        # The test clips and the operating point's detected frames.
        videos = sorted(f for f in os.listdir(store.dataset(dataset, 'test')) if f.endswith('.mp4'))
        row, cfg, frames = operating_frames(dataset, args.loss, videos)
        (out_dir / dataset).mkdir(exist_ok=True)
        # The reference pipeline's detector for this dataset, as p002 loads it.
        detector = polyis.models.detector.get_detector(dataset, args.gpu, batch_size=1)
        warm_up(dataset, videos[0], detector)
        start = time.time()
        # Every test video, including those where the configuration detected on no frame (an empty file).
        for video in videos:
            detect_video(dataset, video, frames[video], detector, out_dir / dataset / f'{video}.jsonl.gz')
            print(f'{dataset} {video}: {len(frames[video])} frames', flush=True)
        # Record the finished dataset and save the manifest, so a later failure keeps it.
        manifest['datasets'][dataset] = dict(config=row['variant_id'], detection_config=cfg, frames=sum(map(len, frames.values())),
                                             videos={v: len(f) for v, f in frames.items()}, seconds=round(time.time() - start, 1),
                                             environment=environment(args.gpu))
        write_json(out_dir, 'manifest.json', manifest)
        # Free the detector before the next dataset's.
        polyis.models.detector.delete(detector)


if __name__ == '__main__':
    main()
