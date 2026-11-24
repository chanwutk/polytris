"""
Compare decord against OpenCV before execution switches decoders (PLAN.md step 1).

Two modes, one JSONL row per video:

  pixels   Frame counts (OpenCV reported, OpenCV read, decord len), pixel agreement over
           every frame (OpenCV BGR flipped to RGB), and whether decord random access
           (get_batch on the needed indices) returns the same pixels as decord sequential
           reads, for each sample rate.  Run datasets in parallel; no timing.

  timings  Wall time to obtain the needed frames for each sample rate and decoder
           thread count, with three strategies: decord get_batch, decord
           sequential-and-keep, OpenCV sequential-and-keep.  Each measurement is
           repeated and the median kept.  Run alone on an idle machine.

"Needed frames" follow p020: sampled idx % s == 0, plus idx-1 (idx+1 for frame 0) for the
diff channel.  A DuckDB summary is printed at the end.
"""

import argparse
import hashlib
import json
import os
import statistics
import time

import cv2
import decord
import duckdb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pixels', 'timings'])
    parser.add_argument('--datasets-root', default='/polyis-data/datasets')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--videoset', default='test')
    parser.add_argument('--videos-per-dataset', type=int, default=None)
    parser.add_argument('--sample-rates', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=64)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def needed_indices(frame_count: int, sample_rate: int) -> list[int]:
    """Sorted frame indices p020 reads: sampled frames plus each one's diff partner."""
    # Sampled frames, exactly as p020 selects them.
    sampled = range(0, frame_count, sample_rate)
    # Diff partner: previous frame, or the next frame for frame 0.
    partners = (idx - 1 if idx > 0 else idx + 1 for idx in sampled)
    # Union, dropping partners past the end (only possible for a 1-frame video).
    return sorted({*sampled, *(p for p in partners if p < frame_count)})


def open_opencv(path: str, threads: int) -> cv2.VideoCapture:
    """Open a video with OpenCV's FFmpeg backend using a fixed decoder thread count (0 = auto)."""
    # Pin the FFmpeg backend and its thread count so timings are comparable.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    assert cap.isOpened(), f'OpenCV could not open {path}'
    return cap


def open_decord(path: str, threads: int) -> decord.VideoReader:
    """Open a video with decord on CPU using a fixed decoder thread count (0 = auto)."""
    # Fresh reader so no decoder state carries over between measurements.
    return decord.VideoReader(path, ctx=decord.cpu(0), num_threads=threads)


def opencv_frames(path: str, threads: int = 0):
    """Yield every frame OpenCV decodes, converted to RGB."""
    cap = open_opencv(path, threads)
    try:
        while True:
            # Read the next frame; stop at the first failed read.
            ok, frame = cap.read()
            if not ok:
                return
            # Flip BGR to RGB to match decord's channel order.
            yield np.ascontiguousarray(frame[:, :, ::-1])
    finally:
        # Release the capture even if the consumer stops early.
        cap.release()


def decord_sequential(path: str, chunk: int, threads: int = 0):
    """Yield every frame decord decodes, reading contiguous chunks in order."""
    vr = open_decord(path, threads)
    # Walk the video in contiguous chunks to bound memory.
    for start in range(0, len(vr), chunk):
        batch = vr.get_batch(list(range(start, min(start + chunk, len(vr))))).asnumpy()
        # Hand out frames one at a time.
        yield from batch


def decord_random(path: str, needed: list[int], chunk: int, threads: int = 0) -> dict[int, np.ndarray]:
    """Fetch only the needed frames with decord random access, in chunks."""
    vr = open_decord(path, threads)
    frames: dict[int, np.ndarray] = {}
    # Request needed indices chunk by chunk to bound memory.
    for start in range(0, len(needed), chunk):
        idxs = needed[start:start + chunk]
        for idx, frame in zip(idxs, vr.get_batch(idxs).asnumpy()):
            frames[idx] = frame
    return frames


def video_info(path: str) -> dict:
    """Frame counts from both decoders plus the frame size."""
    # Reported count and size from container metadata.
    cap = open_opencv(path, 0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        'width': width, 'height': height,
        'opencv_reported_frames': reported,
        # Frames actually decodable, counted by reading to the end.
        'opencv_read_frames': sum(1 for _ in opencv_frames(path)),
        'decord_frames': len(open_decord(path, 0)),
    }


def pixel_row(path: str, frame_count: int, sample_rates: list[int], chunk: int) -> dict:
    """Pixel agreement between decoders, and random-access exactness per sample rate."""
    # Counters accumulated over all frames both decoders produce.
    compared = identical = differing = 0
    max_abs_diff = 0
    first_mismatch = None
    # Keep a digest of each decord sequential frame for the random-access check below
    # (digests, not frames, so long videos do not exhaust memory).
    sequential: list[bytes] = []
    # Walk both decoders in lockstep; zip stops at the shorter one.
    for idx, (a, b) in enumerate(zip(opencv_frames(path), decord_sequential(path, chunk))):
        # A shape mismatch is a hard disagreement; record it and stop.
        if a.shape != b.shape:
            return {'shape_mismatch': [list(a.shape), list(b.shape)], 'frames_compared': idx}
        sequential.append(hashlib.blake2b(b.tobytes(), digest_size=16).digest())
        # Absolute per-pixel difference in a wider dtype to avoid uint8 wraparound.
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        frame_max = int(diff.max())
        compared += 1
        # Track identical frames and the worst difference seen.
        if frame_max == 0:
            identical += 1
        else:
            differing += int(np.count_nonzero(diff))
            first_mismatch = idx if first_mismatch is None else first_mismatch
        max_abs_diff = max(max_abs_diff, frame_max)

    # Random access must return exactly the frames sequential decoding returns.
    random_access = []
    for s in sample_rates:
        needed = needed_indices(frame_count, s)
        fetched = decord_random(path, needed, chunk)
        mismatched = [i for i in needed
                      if hashlib.blake2b(fetched[i].tobytes(), digest_size=16).digest() != sequential[i]]
        random_access.append({'sample_rate': s, 'needed_frames': len(needed),
                              'mismatches': len(mismatched),
                              'first_mismatch': mismatched[0] if mismatched else None})
    return {
        'frames_compared': compared, 'frames_identical': identical,
        'max_abs_diff': max_abs_diff, 'differing_channel_values': differing,
        'first_mismatch_frame': first_mismatch, 'random_access': random_access,
    }


def timed(fn) -> float:
    """Wall-clock seconds to run fn once."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def timing_rows(path: str, frame_count: int, sample_rates: list[int], threads_list: list[int],
                repeats: int, chunk: int) -> list[dict]:
    """Median wall time of each strategy for every (sample rate, thread count)."""
    rows = []
    for threads in threads_list:
        for s in sample_rates:
            # Frames this sample rate needs, as a set for O(1) membership tests.
            needed = needed_indices(frame_count, s)
            needed_set = set(needed)
            # Each strategy obtains exactly the needed frames and discards the rest.
            strategies = {
                'decord_get_batch': lambda: decord_random(path, needed, chunk, threads),
                'decord_sequential': lambda: [f for i, f in enumerate(decord_sequential(path, chunk, threads))
                                              if i in needed_set],
                'opencv_sequential': lambda: [f for i, f in enumerate(opencv_frames(path, threads))
                                              if i in needed_set],
            }
            for name, fn in strategies.items():
                # Repeat and keep the median to damp noise from other tenants.
                samples = [timed(fn) for _ in range(repeats)]
                rows.append({'threads': threads, 'sample_rate': s, 'strategy': name,
                             'needed_frames': len(needed), 'median_s': statistics.median(samples),
                             'samples_s': samples})
    return rows


def main():
    args = parse_args()
    # Write rows as we go so a crash keeps finished videos.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as out:
        for dataset in args.datasets:
            # Sorted video files of this dataset's videoset, optionally truncated.
            video_dir = os.path.join(args.datasets_root, dataset, args.videoset)
            videos = sorted(f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv')))
            assert videos, f'No videos in {video_dir}'
            videos = videos[:args.videos_per_dataset]
            for video in videos:
                path = os.path.join(video_dir, video)
                info = video_info(path)
                # Use the smaller count so every needed index exists in both decoders.
                frame_count = min(info['opencv_read_frames'], info['decord_frames'])
                row = {'dataset': dataset, 'video': video, **info}
                if args.mode == 'pixels':
                    row |= pixel_row(path, frame_count, args.sample_rates, args.chunk)
                else:
                    row['timings'] = timing_rows(path, frame_count, args.sample_rates, args.threads,
                                                 args.repeats, args.chunk)
                out.write(json.dumps(row) + '\n')
                out.flush()
                print(f"{dataset}/{video} done", flush=True)

    if args.mode == 'pixels':
        # Per-dataset agreement: counts, identical frames, worst diff, random-access misses.
        print(duckdb.sql(f"""
            SELECT dataset, count(*) AS videos,
                   sum((opencv_read_frames != decord_frames)::INT) AS count_mismatch_videos,
                   sum((opencv_reported_frames != opencv_read_frames)::INT) AS cv_metadata_off_videos,
                   sum(frames_compared) AS frames, sum(frames_identical) AS identical_frames,
                   max(max_abs_diff) AS max_abs_diff,
                   sum(list_sum(list_transform(random_access, lambda r: r.mismatches))) AS random_access_mismatches
            FROM read_json_auto('{args.out}') GROUP BY dataset ORDER BY dataset
        """))
    else:
        # Per-dataset median time per (threads, sample rate, strategy), summed over videos.
        print(duckdb.sql(f"""
            WITH t AS (SELECT dataset, unnest(timings, recursive := true) FROM read_json_auto('{args.out}'))
            PIVOT (SELECT dataset, threads, sample_rate, strategy, median_s FROM t)
            ON strategy USING round(sum(median_s), 2)
            ORDER BY dataset, threads, sample_rate
        """))


if __name__ == '__main__':
    main()
