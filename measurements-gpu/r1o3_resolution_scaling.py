#!/usr/local/bin/python
"""R1.O3 resolution scaling (GPU): reference-detector and relevance-classifier time per frame as the frame size grows.

--classifier --max-batch times the relevance classifier instead: R9.O1's classifier for the dataset (ShuffleNet05 at its
tile size, from measurements-results/r9o1_matched_timing/), loaded and optimized as p020 does, run through p020's
classify_batch on each frame paired with its previous frame, over every tile of the grid (no never-relevant mask, so
the tile count follows the frame size), at the largest batch of frames the GPU holds.

Per detector (one dataset per distinct detector checkpoint), the first --frames frames of the dataset's first test
video are decoded once and resized with cv2 (bilinear) to each frame size in SIZES and to the dataset's own size. Each
size is timed at batch size 1 (p002's reference batching) after an untimed warm-up of WARMUP_FRAMES frames.

Input resolution: each detector is forced to run at the resized frame's own resolution, never at a native size of
its own. RetinaNet's ResizeShortestEdge is replaced with one whose short and long edge equal the frame's, so the
transform is the identity (asserted). Ultralytics is called with imgsz=(height, width); it pads each side up to the
model stride, and the effective input shape is recorded. A 'default' row times the dataset's own frame size with
the loader's untouched settings (detectron2 MIN_SIZE_TEST / ultralytics' checkpoint imgsz), as a cross-check
against r9o1_matched_timing.

Timing: torch.cuda.synchronize() before and after every call, time.perf_counter_ns between.

Writes measurements-results/r1o3_resolution_scaling/detector_timing.jsonl (one line per detector and size) and
manifest.json. Run on bombe with
  measurements-gpu/bombe_container.sh <GPU> /data/cwkt/data/bombe-revision/scratch/r1o3_resolution \
    'python measurements-gpu/r1o3_resolution_scaling.py'
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from detectron2.data import transforms as T

import polyis.models.detector
import polyis.models.retinanet_b3d
from measurements.common import DATASET_LABELS, RESULTS_ROOT, host_info, iso_utc, results_dir, write_json
from polyis.io import cache, store
from polyis.train.select_model_optimization import select_model_optimization
from scripts.p020_exec_classify import classify_batch, load_model


# One dataset per distinct detector checkpoint (the four B3D datasets share one RetinaNet).
DATASETS = ('jnc0', 'ams-y05', 'caldot1-y05', 'caldot2-y05')
# Frame sizes (width, height), 16:9 from 360p to 4K, for the batch-size-1 sweep.
SIZES = ((640, 360), (960, 540), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))
# Batched sweep: pixel-count multiples of the dataset's own frame size, each side scaled by the square root.
SCALES = (1, 2, 4)
# Batch size at the dataset's own size (p020's classifier batch); the 'scaled' schedule divides it by the scale.
BATCH = 16
# Max-batch sweep: largest batch tried, timed calls at the found batch, and the step down when timing runs out of memory.
MAX_BATCH_CAP = 1024
MAX_BATCH_CALLS = 8
MAX_BATCH_BACKOFF = 0.9
# Untimed frames per size before timing (also lets cuDNN pick kernels for the new shape).
WARMUP_FRAMES = 32
# The device the models use; the launcher exposes one GPU as cuda:0.
DEVICE = 'cuda:0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--batched', action='store_true')
    parser.add_argument('--max-batch', action='store_true')
    parser.add_argument('--classifier', action='store_true')
    return parser.parse_args()


def decode(path, n: int) -> list[np.ndarray]:
    """The first n frames of a video, decoded with cv2 (BGR)."""
    # Open the video for sequential decoding.
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f'{path}: cannot open'
    frames = []
    # Read until n frames or the end of the stream.
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    # A short video would time fewer frames than asked for.
    assert len(frames) == n, f'{path}: decoded {len(frames)} of {n} frames'
    return frames


def timed(fn, *args) -> tuple[object, float]:
    """fn's result and its time in ms, with the GPU synchronized before and after."""
    # Nothing queued earlier may run inside the interval.
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    out = fn(*args)
    # All of the call's GPU work finishes inside the interval.
    torch.cuda.synchronize()
    return out, (time.perf_counter_ns() - t0) / 1e6


def native_detect(detector, width: int, height: int, native: bool):
    """detect(frame) running at the frame's own resolution (native=True) or the loader's defaults, and its input shape."""
    # RetinaNet (detectron2 DefaultPredictor): resizing comes from detector.aug.
    if isinstance(detector, polyis.models.retinanet_b3d.DefaultPredictor):
        if native:
            # Short and long edges equal the frame's, so the resize is the identity.
            detector.aug = T.ResizeShortestEdge([min(width, height)] * 2, max(width, height))
        else:
            # The loader's test-time resize, from its config.
            detector.aug = T.ResizeShortestEdge([detector.cfg.INPUT.MIN_SIZE_TEST] * 2, detector.cfg.INPUT.MAX_SIZE_TEST)
        # The transform the wrapper applies to a frame of this size.
        tf = detector.aug.get_transform(np.zeros((height, width, 3), np.uint8))
        shape = [int(tf.new_h), int(tf.new_w)]
        # Native mode must not resize.
        assert not native or shape == [height, width], f'RetinaNet resizes {height}x{width} to {shape}'
        return (lambda frames: polyis.models.retinanet_b3d.detect_batch(frames, detector)), shape
    # Ultralytics YOLO: the input size comes from imgsz (the checkpoint's training imgsz when not given).
    kwargs = dict(imgsz=(height, width)) if native else {}

    def detect(frames):
        # The same call and box extraction as polyis.models.ultralytics.detect_batch, with imgsz set.
        results = detector(frames, verbose=False, **kwargs)
        return [np.empty((0, 5)) if r.boxes is None else
                np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1) for r in results]
    # One call fixes the predictor's letterboxed input shape, which is read back.
    detect([np.zeros((height, width, 3), np.uint8)])
    return detect, [int(s) for s in detector.predictor.imgsz]


def is_oom(e: BaseException) -> bool:
    """Whether an exception is CUDA running out of memory (raised directly or through the TorchScript interpreter)."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(e)


def cycled_calls(frames: list[np.ndarray], batch: int, n_calls: int) -> list[list[np.ndarray]]:
    """n_calls calls of `batch` consecutive frames, wrapping around the frame list when it runs out."""
    return [[frames[(i * batch + j) % len(frames)] for j in range(batch)] for i in range(n_calls)]


def fits(run_one, batch: int) -> bool:
    """Whether one call of run_one at this batch runs without running out of GPU memory."""
    try:
        run_one(batch)
        torch.cuda.synchronize()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # TorchScript re-raises CUDA out-of-memory as a plain RuntimeError; anything else propagates.
        if not is_oom(e):
            raise
        return False
    finally:
        # Release what the probe left cached before the next one.
        torch.cuda.empty_cache()


def find_max_batch(run_one) -> int:
    """Largest batch run_one(batch) fits: doubling from BATCH, then bisection between the last fit and first failure."""
    # Below BATCH, start from 1.
    lo = BATCH if fits(run_one, BATCH) else 0
    if lo == 0:
        hi = BATCH
    else:
        # Double until a batch fails or the cap is reached.
        hi = lo * 2
        while hi <= MAX_BATCH_CAP and fits(run_one, hi):
            lo, hi = hi, hi * 2
        if hi > MAX_BATCH_CAP:
            return lo
    # Bisect: lo fits (or is 0), hi does not.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(run_one, mid) else (lo, mid)
    assert lo > 0, 'not even batch 1 fits'
    return lo


def detector_max_batch(detector, frames: list[np.ndarray], width: int, height: int) -> int:
    """Largest detector batch at this size, the detector running at the frame's own resolution."""
    # Resize once; probe calls wrap around the frames.
    resized = [cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR) for f in frames]
    detect, _ = native_detect(detector, width, height, True)
    return find_max_batch(lambda b: detect(cycled_calls(resized, b, 1)[0]))


def matched_classifier(dataset: str) -> dict:
    """R9.O1's classifier name and tile size for the dataset (its operating point's classifier)."""
    # The matched timing keys datasets by their paper label.
    matched = json.loads((RESULTS_ROOT / 'r9o1_matched_timing' / 'matched_timing.json').read_text())
    return matched['datasets'][DATASET_LABELS[dataset]]['classifier']


def load_classifier(dataset: str, name: str, tilesize: int):
    """The classifier loaded and optimized as p020's classify_all does (load_model, then the benchmark's fastest method)."""
    # Trained weights in half precision, on the GPU.
    model = load_model(dataset, tilesize, name, DEVICE).to(DEVICE)
    # p020's choice of optimization from the compilation benchmark (trace-based methods are shape-generic).
    with open(cache.index(dataset, 'training', 'results', f'{name}_{tilesize}', 'model_compilation.jsonl')) as f:
        benchmark = [json.loads(line) for line in f]
    return select_model_optimization(model, benchmark, DEVICE, tilesize, BATCH)


def classifier_call(model, tilesize: int, width: int, height: int):
    """classify(frames, prev_frames) on every tile of a width x height frame, with p020's classify_batch and constants."""
    # The tile grid; p020 drops pixels beyond the last full tile.
    grid_w, grid_h = width // tilesize, height // tilesize
    # p020's ImageNet normalization of the frame and difference channels, in half precision.
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1)
    # Tile positions (row, column) in row-major order, as p020 builds them.
    y = torch.arange(grid_h, device=DEVICE).repeat_interleave(grid_w)
    x = torch.arange(grid_w, device=DEVICE).repeat(grid_h)
    positions = torch.stack([y, x], dim=1).float()
    # Every tile is inferred, so the tile count follows the frame size.
    mask = torch.ones(grid_h * grid_w, device=DEVICE, dtype=torch.uint8)

    def classify(frames, prev):
        return classify_batch(grid_w, grid_h, positions, frames, prev, model, tilesize, DEVICE, mean, std, mask)
    return classify, grid_w * grid_h


def time_classifier(dataset: str, model, tilesize: int, frames: list[np.ndarray], width: int, height: int, batch: int,
                    n_calls: int) -> dict:
    """Classifier ms per frame at one size and batch of frames (n_calls timed calls after one warm-up call), with peak memory."""
    # RGB copies resized once (p020 classifies RGB frames); frame i is paired with frame i - 1, the first with itself.
    resized = [np.ascontiguousarray(cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]) for f in frames]
    pairs = [(resized[i], resized[max(i - 1, 0)]) for i in range(len(resized))]
    calls = [list(zip(*c)) for c in cycled_calls(pairs, batch, n_calls + 1)]
    classify, tiles = classifier_call(model, tilesize, width, height)
    row = dict(dataset=dataset, model='classifier', width=width, height=height, pixels=width * height, batch=batch,
               tiles=tiles)
    # Peak memory counts this size only.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # One untimed warm-up call, then the timed calls; a call's time split evenly over its frames.
        timed(classify, *calls[0])
        ms = np.asarray([timed(classify, *c)[1] / batch for c in calls[1:]])
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # TorchScript re-raises CUDA out-of-memory as a plain RuntimeError; anything else propagates.
        if not is_oom(e):
            raise
        print(f'{dataset} classifier {width}x{height} batch {batch}: out of memory', flush=True)
        torch.cuda.empty_cache()
        return dict(row, oom=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f'{dataset} classifier {width}x{height} batch {batch} ({tiles} tiles/frame): {ms.mean():.3f} ms/frame, '
          f'peak {peak:.0f} MiB', flush=True)
    return dict(row, oom=False, frames=batch * n_calls, mean_ms=float(ms.mean()), median_ms=float(np.median(ms)),
                peak_allocated_mib=peak, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def time_size(dataset: str, detector, frames: list[np.ndarray], width: int, height: int, native: bool, batch: int = 1,
              n_calls: int | None = None) -> dict:
    """Mean, median, and p95 ms per frame at one size and batch size, after warm-up, with peak GPU memory.

    With n_calls, that many calls are timed, wrapping around the frames; otherwise the frames split into calls exactly.
    """
    # Resize every frame once, outside the timed region.
    resized = [cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR) for f in frames]
    if n_calls is None:
        # Consecutive frames grouped into calls of `batch` (frames is a multiple of every batch size used).
        assert len(resized) % batch == 0, f'{len(resized)} frames do not split into batches of {batch}'
        calls = [resized[i:i + batch] for i in range(0, len(resized), batch)]
    else:
        calls = cycled_calls(resized, batch, n_calls)
    row = dict(dataset=dataset, mode='native' if native else 'default', width=width, height=height, pixels=width * height,
               batch=batch)
    # Peak memory counts this size only: drop cached blocks and reset the peak first.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        detect, shape = native_detect(detector, width, height, native)
        # Untimed warm-up at this shape and batch.
        for c in calls[:max(1, WARMUP_FRAMES // batch)]:
            timed(detect, c)
        # Timed calls, each call's time split evenly over its frames; detections as a sanity check.
        ms, boxes = [], []
        for c in calls:
            out, t = timed(detect, c)
            ms += [t / batch] * batch
            boxes += [len(o) for o in out]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # TorchScript re-raises CUDA out-of-memory as a plain RuntimeError; anything else propagates.
        if not is_oom(e):
            raise
        # A size and batch that does not fit is recorded, not shrunk.
        print(f'{dataset} {width}x{height} batch {batch}: out of memory', flush=True)
        torch.cuda.empty_cache()
        return dict(row, oom=True)
    ms = np.asarray(ms)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f'{dataset} {width}x{height} batch {batch} {row["mode"]} input {shape}: {ms.mean():.2f} ms/frame, '
          f'peak {peak:.0f} MiB', flush=True)
    return dict(row, oom=False, input_shape=shape, frames=len(ms), mean_ms=float(ms.mean()), median_ms=float(np.median(ms)),
                p95_ms=float(np.percentile(ms, 95)), mean_detections=float(np.mean(boxes)), peak_allocated_mib=peak,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def main():
    args = parse_args()
    out = results_dir(__file__)
    started = time.time()
    rows = []
    for dataset in args.datasets:
        # The dataset's first test video and its own frame size.
        video = sorted(v for v in os.listdir(store.dataset(dataset, 'test')) if v.endswith('.mp4'))[0]
        frames = decode(store.dataset(dataset, 'test', video), args.frames)
        own = (frames[0].shape[1], frames[0].shape[0])
        if args.classifier:
            # R9.O1's classifier, at own size times each pixel scale, at the largest batch of frames the GPU holds.
            assert args.max_batch, '--classifier runs only with --max-batch'
            info = matched_classifier(dataset)
            model, method = load_classifier(dataset, info['name'], info['tilesize'])
            for scale in SCALES:
                w, h = (round(d * scale ** 0.5 / 2) * 2 for d in own)
                # Probe with one RGB frame pair per batch slot, wrapping around the frames.
                classify, _ = classifier_call(model, info['tilesize'], w, h)
                rgb = [np.ascontiguousarray(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]) for f in frames]
                found = batch = find_max_batch(lambda b: classify(*zip(*cycled_calls(list(zip(rgb, [rgb[0]] + rgb[:-1])), b, 1)[0])))
                del rgb
                while True:
                    row = time_classifier(dataset, model, info['tilesize'], frames, w, h, batch, MAX_BATCH_CALLS)
                    if not row['oom']:
                        break
                    # A batch of 1 that runs out of memory cannot be stepped down further.
                    assert batch > 1, f'{dataset} classifier {w}x{h}: batch 1 runs out of memory'
                    batch = max(1, int(batch * MAX_BATCH_BACKOFF))
                rows.append(dict(row, video=video, scale=scale, schedule='max', probed_max_batch=found,
                                 classifier=info['name'], tilesize=info['tilesize'], optimization=method))
            del model
            torch.cuda.empty_cache()
            continue
        # The reference detector, as p002 loads it.
        detector = polyis.models.detector.get_detector(dataset, 0, batch_size=1)
        if args.max_batch:
            # Own size times each pixel scale, at the largest batch the GPU holds (stepping down if timing runs out).
            for scale in SCALES:
                w, h = (round(d * scale ** 0.5 / 2) * 2 for d in own)
                batch = detector_max_batch(detector, frames, w, h)
                found = batch
                while True:
                    row = time_size(dataset, detector, frames, w, h, native=True, batch=batch, n_calls=MAX_BATCH_CALLS)
                    if not row['oom']:
                        break
                    # A batch of 1 that runs out of memory cannot be stepped down further.
                    assert batch > 1, f'{dataset} {w}x{h}: batch 1 runs out of memory'
                    batch = max(1, int(batch * MAX_BATCH_BACKOFF))
                rows.append(dict(row, video=video, scale=scale, schedule='max', probed_max_batch=found))
        elif args.batched:
            # Own size times each pixel scale (sides by its square root, rounded to even), at an equal batch and
            # at a batch divided by the scale.
            for scale in SCALES:
                w, h = (round(d * scale ** 0.5 / 2) * 2 for d in own)
                for schedule, batch in (('equal', BATCH), ('scaled', BATCH // scale)):
                    rows.append(dict(time_size(dataset, detector, frames, w, h, native=True, batch=batch), video=video,
                                     scale=scale, schedule=schedule))
        else:
            # The loader's defaults at the dataset's own size, then every size at its own resolution.
            rows.append(dict(time_size(dataset, detector, frames, *own, native=False), video=video))
            for w, h in sorted(set(SIZES) | {own}, key=lambda s: s[0] * s[1]):
                rows.append(dict(time_size(dataset, detector, frames, w, h, native=True), video=video))
        # Free the model before the next dataset's.
        polyis.models.detector.delete(detector)
        torch.cuda.empty_cache()
    # One line per detector, mode, and size.
    suffix = ('_classifier' if args.classifier else '') + ('_maxbatch' if args.max_batch else '_batched' if args.batched else '')
    with open(out / f'detector_timing{suffix}.jsonl', 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    write_json(out, f'manifest{suffix}.json', dict(
        datasets=args.datasets, sizes=SIZES, scales=SCALES, batch=BATCH, batched=args.batched, max_batch=args.max_batch,
        max_batch_cap=MAX_BATCH_CAP, max_batch_calls=MAX_BATCH_CALLS, frames=args.frames, warmup_frames=WARMUP_FRAMES,
        gpu=torch.cuda.get_device_name(0), bombe_gpu=os.environ.get('BOMBE_GPU'), bombe_run=os.environ.get('BOMBE_RUN'),
        host=host_info(), started=iso_utc(started), finished=iso_utc(time.time())))


if __name__ == '__main__':
    main()
