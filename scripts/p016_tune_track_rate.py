#!/usr/local/bin/python

import argparse
import json
import os
import multiprocessing as mp
from functools import partial
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from polyis.b3d.sort import iou_batch
from polyis.utilities import create_tracker, register_tracked_detections, ProgressBar, get_config, get_video_resolution, get_overlapping_tiles
from polyis.io import cache, store


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
TRACKERS = CONFIG['EXEC']['TRACKERS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']

SAMPLE_RATES = [1, 2, 4, 8, 16]
ACCURACY_THRESHOLDS = [60, 70, 80, 90, 95, 100]


def parse_args():
    parser = argparse.ArgumentParser(description='Compute per-tile tracking accuracy at different sample rates')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for Hungarian matching (default: 0.3)')
    parser.add_argument('--num_workers', type=int, default=40,
                        help='Number of parallel workers (default: 40)')
    return parser.parse_args()


def run_tracker_on_detections(
    all_dets: dict[int, np.ndarray],
    total_frames: int,
    tracker_name: str,
    img_size: tuple[int, int],
    sample_rate: int
) -> dict[int, list[list[float]]]:
    """
    Run tracker over all frames, providing detections only at sampled frames.

    Returns frame_tracks: {frame_idx: [[track_id, x1, y1, x2, y2], ...]}
    """
    # Create a fresh tracker instance
    tracker = create_tracker(tracker_name, img_size)
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    for frame_idx in range(total_frames):
        # Provide detections only at sampled frames (every sample_rate-th frame)
        if frame_idx % sample_rate == 0 and frame_idx in all_dets:
            dets = all_dets[frame_idx]
        else:
            dets = np.empty((0, 5))

        # Update tracker (must be called every frame for Kalman filter state)
        tracked_dets = tracker.update(dets)

        # Register tracks only at sampled frames
        if frame_idx % sample_rate == 0:
            register_tracked_detections(tracked_dets, frame_idx, frame_tracks, trajectories, interpolate=False)

    return frame_tracks


def _is_correct_association(
    frame_idx: int,
    sample_rate: int,
    step_direction: int,
    end_bound: int,
    gt_track_id: int,
    sampled_track_id: int,
    gt_presence: set[tuple[int, int]],
    sampled_presence: set[tuple[int, int]],
    per_frame_matched_pairs: dict[int, set[tuple[int, int]]],
) -> bool:
    """
    Walk forward (step_direction=1) or backward (step_direction=-1) from frame_idx;
    return True if association is maintained or both tracks end together, False if broken.
    """
    check_frame = frame_idx + step_direction * sample_rate
    while (check_frame - end_bound) * step_direction <= 0:
        gt_has = (check_frame, gt_track_id) in gt_presence
        sampled_has = (check_frame, sampled_track_id) in sampled_presence

        if gt_has != sampled_has:
            # Exactly one has detection (not both) → association broken
            return False
        if gt_has and sampled_has:
            # Both have detections: check if they match on this frame
            frame_matches = per_frame_matched_pairs.get(check_frame, set())
            if (gt_track_id, sampled_track_id) in frame_matches:
                # Match → association maintained (correct), stop
                return True
            # Don't match → association broken (ID switch)
            return False
        # Neither has detection → continue walking by sample_rate

        check_frame += step_direction * sample_rate
    # Exited without finding any detection until bound → tracks ended together (correct)
    return True


def _build_track_presence(
    tracks: dict[int, list[list[float]]]
) -> tuple[set[tuple[int, int]], dict[int, int], dict[int, int]]:
    """
    Build track presence set and per-track min/max frame indices.

    Returns (presence, track_min_frame, track_max_frame) where:
    - presence: set of (frame_idx, track_id) pairs indicating detection exists
    - track_min_frame: earliest frame each track_id appears in
    - track_max_frame: latest frame each track_id appears in
    """
    presence: set[tuple[int, int]] = set()
    track_min_frame: dict[int, int] = {}
    track_max_frame: dict[int, int] = {}

    # Iterate all frames and detections to record presence and frame bounds
    for frame_idx, dets in tracks.items():
        for det in dets:
            tid = int(det[0])
            presence.add((frame_idx, tid))
            track_min_frame[tid] = min(track_min_frame.get(tid, frame_idx), frame_idx)
            track_max_frame[tid] = max(track_max_frame.get(tid, frame_idx), frame_idx)

    return presence, track_min_frame, track_max_frame


def _match_detections_per_frame(
    gt_tracks: dict[int, list[list[float]]],
    sampled_tracks: dict[int, list[list[float]]],
    sample_rate: int,
    iou_threshold: float,
) -> tuple[
    list[int],
    dict[int, set[tuple[int, int]]],
    dict[int, list[tuple[int, int, int, int]]],
    dict[int, set[int]],
]:
    """
    Precompute Hungarian matching for all sampled frames with GT detections.

    Returns (all_gt_frames, per_frame_matched_pairs, per_frame_match_details, per_frame_matched_gt):
    - all_gt_frames: sorted list of GT frame indices
    - per_frame_matched_pairs: {frame: set of (gt_track_id, sampled_track_id)}
    - per_frame_match_details: {frame: [(gt_det_idx, sampled_det_idx, gt_track_id, sampled_track_id)]}
    - per_frame_matched_gt: {frame: set of matched GT detection indices}
    """
    per_frame_matched_pairs: dict[int, set[tuple[int, int]]] = {}
    per_frame_match_details: dict[int, list[tuple[int, int, int, int]]] = {}
    per_frame_matched_gt: dict[int, set[int]] = {}

    # Collect and sort all GT frame indices
    all_gt_frames = sorted(gt_tracks.keys())

    for frame_idx in all_gt_frames:
        # Only evaluate at sampled frames
        if frame_idx % sample_rate != 0:
            continue

        gt_dets = gt_tracks[frame_idx]
        sampled_dets = sampled_tracks.get(frame_idx, [])

        matched_pairs: set[tuple[int, int]] = set()
        match_details: list[tuple[int, int, int, int]] = []
        matched_gt: set[int] = set()

        if len(gt_dets) > 0 and len(sampled_dets) > 0:
            # Extract bboxes and track IDs
            gt_boxes = np.array([[d[1], d[2], d[3], d[4]] for d in gt_dets])
            gt_ids = [int(d[0]) for d in gt_dets]
            sampled_boxes = np.array([[d[1], d[2], d[3], d[4]] for d in sampled_dets])
            sampled_ids = [int(d[0]) for d in sampled_dets]

            # Hungarian matching using IoU cost matrix
            iou_matrix = iou_batch(gt_boxes, sampled_boxes)
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Collect matches that exceed the IoU threshold
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_threshold:
                    g_id = gt_ids[r]
                    s_id = sampled_ids[c]
                    matched_pairs.add((g_id, s_id))
                    match_details.append((r, c, g_id, s_id))
                    matched_gt.add(r)

        # Store per-frame matching results
        per_frame_matched_pairs[frame_idx] = matched_pairs
        per_frame_match_details[frame_idx] = match_details
        per_frame_matched_gt[frame_idx] = matched_gt

    return all_gt_frames, per_frame_matched_pairs, per_frame_match_details, per_frame_matched_gt


def _calculate_misstrack_stats_per_tile(
    gt_tracks: dict[int, list[list[float]]],
    all_gt_frames: list[int],
    sample_rate: int,
    tile_size: int,
    grid_h: int,
    grid_w: int,
    gt_presence: set[tuple[int, int]],
    gt_track_min_frame: dict[int, int],
    gt_track_max_frame: dict[int, int],
    sampled_presence: set[tuple[int, int]],
    sampled_track_min_frame: dict[int, int],
    sampled_track_max_frame: dict[int, int],
    per_frame_matched_pairs: dict[int, set[tuple[int, int]]],
    per_frame_match_details: dict[int, list[tuple[int, int, int, int]]],
    per_frame_matched_gt: dict[int, set[int]],
) -> np.ndarray:
    """
    Evaluate temporal consistency of matched pairs and accumulate per-tile counts.

    For each matched pair, walks forward and backward to verify temporal association.
    Unmatched GT detections are counted as incorrect.

    Returns array of shape (grid_h, grid_w, 2) with [correct, incorrect] counts.
    """
    counts = np.zeros((grid_h, grid_w, 2), dtype=np.int64)

    # Compute the global max frame for default backward-walk bound
    max_frame = max(all_gt_frames) if all_gt_frames else 0

    # Evaluate each sampled frame
    for frame_idx in all_gt_frames:
        # Only evaluate at sampled frames
        if frame_idx % sample_rate != 0:
            continue

        gt_dets = gt_tracks[frame_idx]
        if len(gt_dets) == 0:
            continue

        match_details = per_frame_match_details.get(frame_idx, [])
        matched_gt = per_frame_matched_gt.get(frame_idx, set())

        # Evaluate each matched pair for temporal consistency (README Step 3.2)
        for gt_idx, _sampled_idx, gt_track_id, sampled_track_id in match_details:
            # Determine forward walk boundary from both tracks' max frames
            forward_end = max(
                gt_track_max_frame.get(gt_track_id, -1),
                sampled_track_max_frame.get(sampled_track_id, -1),
            )
            correct = _is_correct_association(
                frame_idx, sample_rate, 1, forward_end,
                gt_track_id, sampled_track_id,
                gt_presence, sampled_presence, per_frame_matched_pairs,
            )
            # Determine backward walk boundary from both tracks' min frames
            backward_start = min(
                gt_track_min_frame.get(gt_track_id, max_frame + 1),
                sampled_track_min_frame.get(sampled_track_id, max_frame + 1),
            )
            correct = correct and _is_correct_association(
                frame_idx, sample_rate, -1, backward_start,
                gt_track_id, sampled_track_id,
                gt_presence, sampled_presence, per_frame_matched_pairs,
            )

            # Assign result to all overlapping tiles of the GT detection
            g_det = gt_dets[gt_idx]
            r0, r1, c0, c1 = get_overlapping_tiles(g_det[1], g_det[2], g_det[3], g_det[4],
                                                    tile_size, grid_h, grid_w)
            if correct:
                counts[r0:r1+1, c0:c1+1, 0] += 1  # correct
            else:
                counts[r0:r1+1, c0:c1+1, 1] += 1  # incorrect

        # Unmatched GT detections → incorrect
        for i in range(len(gt_dets)):
            if i not in matched_gt:
                g_det = gt_dets[i]
                r0, r1, c0, c1 = get_overlapping_tiles(g_det[1], g_det[2], g_det[3], g_det[4],
                                                        tile_size, grid_h, grid_w)
                counts[r0:r1+1, c0:c1+1, 1] += 1  # incorrect

    return counts


def calculate_misstrack_stats_per_tile(
    gt_tracks: dict[int, list[list[float]]],
    sampled_tracks: dict[int, list[list[float]]],
    sample_rate: int,
    tile_size: int,
    grid_h: int,
    grid_w: int,
    iou_threshold: float = 0.3
) -> np.ndarray:
    """
    Compute per-tile (correct, incorrect) counts using temporal consistency.

    For each matched GT-sampled detection pair at a sampled frame:
    - Walk forward/backward by sample_rate until finding a frame with detection(s)
    - When found: if only one track has detection → incorrect
    - When found: if both have detections and match → correct (stop)
    - When found: if both have detections but don't match → incorrect
    - If never find any detection → correct (tracks ended together)
    - Count as correct ONLY if BOTH forward and backward checks pass
    - Unmatched GT detections → incorrect

    Returns array of shape (grid_h, grid_w, 2) with [correct, incorrect] counts.
    """
    # Build presence sets and frame bounds for GT and sampled tracks
    gt_presence, gt_track_min_frame, gt_track_max_frame = _build_track_presence(gt_tracks)
    sampled_presence, sampled_track_min_frame, sampled_track_max_frame = _build_track_presence(sampled_tracks)

    # Precompute per-frame Hungarian matching between GT and sampled detections
    all_gt_frames, per_frame_matched_pairs, per_frame_match_details, per_frame_matched_gt = (
        _match_detections_per_frame(gt_tracks, sampled_tracks, sample_rate, iou_threshold)
    )

    # Evaluate temporal consistency and accumulate per-tile correct/incorrect counts
    return _calculate_misstrack_stats_per_tile(
        gt_tracks, all_gt_frames, sample_rate, tile_size, grid_h, grid_w,
        gt_presence, gt_track_min_frame, gt_track_max_frame,
        sampled_presence, sampled_track_min_frame, sampled_track_max_frame,
        per_frame_matched_pairs, per_frame_match_details, per_frame_matched_gt,
    )


def process_video_tracker(
    dataset: str,
    video: str,
    tracker_name: str,
    tile_size: int,
    iou_threshold: float,
    partial_dir: str,
    gpu_id: int,
    command_queue: mp.Queue
):
    """Worker: load detections, run GT + sampled trackers, save per-video partial counts."""
    device = f'cuda:{gpu_id}'

    # Send initial progress update
    command_queue.put((device, {
        'description': f'{dataset}/{video} {tracker_name} tile={tile_size}',
        'completed': 0,
        'total': len(SAMPLE_RATES)
    }))

    # Load detections from p011 output
    detections_path = cache.index(dataset, 'det', f'{video}.detections.jsonl')
    assert os.path.exists(detections_path), f"Detections not found: {detections_path}"

    all_dets: dict[int, np.ndarray] = {}
    with open(detections_path, 'r') as f:
        for line in f:
            frame_idx, dets, _ = json.loads(line)
            if len(dets) > 0:
                all_dets[frame_idx] = np.array(dets, dtype=np.float64)
            else:
                all_dets[frame_idx] = np.empty((0, 5))

    # Get video resolution and compute tile-aligned dimensions
    width, height = get_video_resolution(dataset, video)
    target_h = (height // tile_size) * tile_size
    target_w = (width // tile_size) * tile_size
    grid_h = target_h // tile_size
    grid_w = target_w // tile_size

    # Scale detection coordinates to tile-aligned resolution
    scale_x = target_w / width
    scale_y = target_h / height
    scaled_dets: dict[int, np.ndarray] = {}
    for frame_idx, dets in all_dets.items():
        if dets.shape[0] > 0:
            scaled = dets.copy()
            scaled[:, 0] *= scale_x  # x1
            scaled[:, 1] *= scale_y  # y1
            scaled[:, 2] *= scale_x  # x2
            scaled[:, 3] *= scale_y  # y2
            scaled_dets[frame_idx] = scaled
        else:
            scaled_dets[frame_idx] = dets

    # Get total frame count from video file
    video_path = store.dataset(dataset, 'train', video)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Run GT tracker (sample_rate=1)
    gt_tracks = run_tracker_on_detections(scaled_dets, total_frames, tracker_name,
                                          (target_h, target_w), sample_rate=1)

    # Compute per-tile counts for each sample rate
    # Shape: (num_rates, grid_h, grid_w, 2) where last dim is [correct, incorrect]
    partial_counts = np.zeros((len(SAMPLE_RATES), grid_h, grid_w, 2), dtype=np.int64)

    for rate_idx, rate in enumerate(SAMPLE_RATES):
        # Run sampled tracker
        sampled_tracks = run_tracker_on_detections(scaled_dets, total_frames, tracker_name,
                                                   (target_h, target_w), sample_rate=rate)

        # Compute per-tile correct/incorrect counts using temporal consistency
        counts = calculate_misstrack_stats_per_tile(gt_tracks, sampled_tracks, rate, tile_size, grid_h, grid_w, iou_threshold)
        partial_counts[rate_idx] = counts

        # Send progress update
        command_queue.put((device, {'completed': rate_idx + 1}))

    # Save partial per-video counts
    os.makedirs(partial_dir, exist_ok=True)
    np.save(os.path.join(partial_dir, f'{video}.npy'), partial_counts)


def main(args):
    """
    Orchestrate per-tile tracking accuracy computation across datasets, trackers, and tile sizes.

    For each combination, runs parallel workers per video, then aggregates partial counts
    and computes Laplace-smoothed accuracy.
    """
    mp.set_start_method('spawn', force=True)

    for dataset in DATASETS:
        dataset_dir = store.dataset(dataset, 'train')
        assert dataset_dir.exists(), f"Dataset directory {dataset_dir} does not exist"

        # Get list of training videos with detections
        det_dir = cache.index(dataset, 'det')
        assert det_dir.exists(), f"Detection directory {det_dir} does not exist"

        videos = [
            f.stem.replace('.detections', '')
            for f in det_dir.iterdir()
            if f.name.endswith('.detections.jsonl')
        ]
        assert len(videos) > 0, f"No detection files found in {det_dir}"

        print(f"Dataset {dataset}: {len(videos)} videos")

        for tracker_name in TRACKERS:
            for tile_size in TILE_SIZES:
                print(f"  Processing tracker={tracker_name}, tile_size={tile_size}")

                # Output directory for this (dataset, tracker, tile_size) combination
                output_dir = cache.index(dataset, 'track_rates', f'{tracker_name}_{tile_size}')
                partial_dir = output_dir / 'partial'

                # Create task functions for each video
                funcs = [
                    partial(process_video_tracker, dataset, video, tracker_name, tile_size,
                            args.iou_threshold, str(partial_dir))
                    for video in videos
                ]

                # Run in parallel (CPU-only tracking)
                ProgressBar(num_workers=args.num_workers, num_tasks=len(funcs)).run_all(funcs)

                # Aggregate per-video counts by summing across all videos
                total_counts = None
                for video in videos:
                    counts_path = partial_dir / f'{video}.npy'
                    assert counts_path.exists(), f"Partial counts not found: {counts_path}"
                    video_counts = np.load(str(counts_path))

                    if total_counts is None:
                        total_counts = video_counts.copy()
                    else:
                        total_counts += video_counts

                assert total_counts is not None, "No partial results to aggregate"

                # Compute Laplace-smoothed accuracy from aggregated counts
                correct = total_counts[:, :, :, 0].astype(np.float32)
                incorrect = total_counts[:, :, :, 1].astype(np.float32)
                accuracy = (correct + 1) / (correct + incorrect + 2)

                # Build lookup table: for each threshold, find the lowest (least frequent)
                # sampling rate that still achieves the required accuracy per tile.
                # Shape: (grid_h, grid_w, num_threshold_levels)
                grid_h = accuracy.shape[1]
                grid_w = accuracy.shape[2]
                num_thresholds = len(ACCURACY_THRESHOLDS)
                max_rate_table = np.full((grid_h, grid_w, num_thresholds), SAMPLE_RATES[0], dtype=np.int32)

                for t_idx, threshold_pct in enumerate(ACCURACY_THRESHOLDS):
                    threshold = threshold_pct / 100.0
                    # Iterate rates from lowest (most frequent) to highest (least frequent);
                    # each qualifying rate overwrites the previous, so the final value is
                    # the highest (least frequent) rate that still meets the threshold.
                    for rate_idx in range(len(SAMPLE_RATES)):
                        # Overwrite with this rate wherever accuracy meets the threshold
                        meets = accuracy[rate_idx] >= threshold
                        max_rate_table[:, :, t_idx] = np.where(meets, SAMPLE_RATES[rate_idx],
                                                               max_rate_table[:, :, t_idx])

                # Save results
                os.makedirs(str(output_dir), exist_ok=True)
                np.save(str(output_dir / 'accuracy.npy'), accuracy)
                np.save(str(output_dir / 'counts.npy'), total_counts)
                np.save(str(output_dir / 'max_rate_table.npy'), max_rate_table)
                with open(str(output_dir / 'sample_rates.json'), 'w') as f:
                    json.dump(SAMPLE_RATES, f)

                # Save max rate table as text files, one per threshold level
                for t_idx, threshold_pct in enumerate(ACCURACY_THRESHOLDS):
                    txt_path = output_dir / f'max_rate_{threshold_pct:03d}.txt'
                    np.savetxt(str(txt_path), max_rate_table[:, :, t_idx], fmt='%2d')

                print(f"    Saved accuracy.npy {accuracy.shape}, counts.npy {total_counts.shape}, "
                      f"max_rate_table.npy {max_rate_table.shape}")

    print("All tracking rate evaluations completed!")


if __name__ == '__main__':
    main(parse_args())
