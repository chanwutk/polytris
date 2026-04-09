from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from scripts.p016_tune_track_rate import (
    _build_track_presence,
    _is_correct_association,
    _match_detections_per_frame,
    run_tracker_on_detections,
)

from .common import (
    GRID_COLS,
    GRID_ROWS,
    HEURISTIC_THRESHOLDS,
    RATE_CHOICES,
    ensure_dir,
    frame_metadata,
    get_overlapping_rect_cells,
    heuristic_accuracy_path,
    heuristic_counts_path,
    heuristic_dir,
    heuristic_metadata_path,
    heuristic_rate_table_path,
    list_split_videos,
    load_naive_tracking_source,
    subsample_videos,
    tracks_to_detection_arrays,
)


def calculate_rectangular_misstrack_counts(
    gt_tracks: dict[int, list[list[float]]],
    sampled_tracks: dict[int, list[list[float]]],
    sample_rate: int,
    width: int,
    height: int,
    iou_threshold: float = 0.3,
    num_rows: int = GRID_ROWS,
    num_cols: int = GRID_COLS,
) -> np.ndarray:
    gt_presence, gt_track_min_frame, gt_track_max_frame = _build_track_presence(gt_tracks)
    sampled_presence, sampled_track_min_frame, sampled_track_max_frame = _build_track_presence(sampled_tracks)
    all_gt_frames, per_frame_matched_pairs, per_frame_match_details, per_frame_matched_gt = (
        _match_detections_per_frame(gt_tracks, sampled_tracks, sample_rate, iou_threshold)
    )

    counts = np.zeros((num_rows, num_cols, 2), dtype=np.int64)
    max_frame = max(all_gt_frames) if all_gt_frames else 0

    for frame_idx in all_gt_frames:
        if frame_idx % sample_rate != 0:
            continue

        gt_dets = gt_tracks[frame_idx]
        if len(gt_dets) == 0:
            continue

        match_details = per_frame_match_details.get(frame_idx, [])
        matched_gt = per_frame_matched_gt.get(frame_idx, set())

        for gt_idx, _sampled_idx, gt_track_id, sampled_track_id in match_details:
            forward_end = max(
                gt_track_max_frame.get(gt_track_id, -1),
                sampled_track_max_frame.get(sampled_track_id, -1),
            )
            correct = _is_correct_association(
                frame_idx,
                sample_rate,
                1,
                forward_end,
                gt_track_id,
                sampled_track_id,
                gt_presence,
                sampled_presence,
                per_frame_matched_pairs,
            )

            backward_start = min(
                gt_track_min_frame.get(gt_track_id, max_frame + 1),
                sampled_track_min_frame.get(sampled_track_id, max_frame + 1),
            )
            correct = correct and _is_correct_association(
                frame_idx,
                sample_rate,
                -1,
                backward_start,
                gt_track_id,
                sampled_track_id,
                gt_presence,
                sampled_presence,
                per_frame_matched_pairs,
            )

            gt_det = gt_dets[gt_idx]
            overlap = get_overlapping_rect_cells(
                float(gt_det[1]),
                float(gt_det[2]),
                float(gt_det[3]),
                float(gt_det[4]),
                width,
                height,
                num_rows,
                num_cols,
            )
            if overlap is None:
                continue

            row_start, row_end, col_start, col_end = overlap
            counts[row_start:row_end + 1, col_start:col_end + 1, 0 if correct else 1] += 1

        for det_idx, gt_det in enumerate(gt_dets):
            if det_idx in matched_gt:
                continue

            overlap = get_overlapping_rect_cells(
                float(gt_det[1]),
                float(gt_det[2]),
                float(gt_det[3]),
                float(gt_det[4]),
                width,
                height,
                num_rows,
                num_cols,
            )
            if overlap is None:
                continue

            row_start, row_end, col_start, col_end = overlap
            counts[row_start:row_end + 1, col_start:col_end + 1, 1] += 1

    return counts


def build_rate_tables_from_counts(
    counts: np.ndarray,
    thresholds: tuple[float, ...] = HEURISTIC_THRESHOLDS,
    rate_choices: tuple[int, ...] = RATE_CHOICES,
) -> tuple[np.ndarray, np.ndarray]:
    correct = counts[..., 0].astype(np.float32)
    incorrect = counts[..., 1].astype(np.float32)
    accuracy = (correct + 1.0) / (correct + incorrect + 2.0)

    num_rows = counts.shape[1]
    num_cols = counts.shape[2]
    rate_table = np.full((num_rows, num_cols, len(thresholds)), rate_choices[0], dtype=np.int32)

    for threshold_idx, threshold_pct in enumerate(thresholds):
        threshold = float(threshold_pct) / 100.0
        for rate_idx, rate in enumerate(rate_choices):
            meets_threshold = accuracy[rate_idx] >= threshold
            rate_table[:, :, threshold_idx] = np.where(
                meets_threshold,
                rate,
                rate_table[:, :, threshold_idx],
            )

    return accuracy, rate_table


def load_heuristic_rate_grids(dataset: str, tracker_name: str) -> dict[float, np.ndarray]:
    rate_table = np.load(heuristic_rate_table_path(dataset, tracker_name))
    metadata_path = heuristic_metadata_path(dataset, tracker_name)
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    thresholds = metadata['thresholds']
    return {
        float(threshold): rate_table[:, :, idx].astype(np.int32)
        for idx, threshold in enumerate(thresholds)
    }


def _process_single_video(
    dataset: str,
    video: str,
    tracker_name: str,
    iou_threshold: float,
) -> np.ndarray:
    # Load source data and run ground-truth tracker.
    width, height, total_frames = frame_metadata(dataset, video)
    source_tracks = load_naive_tracking_source(dataset, video)
    source_detections = tracks_to_detection_arrays(source_tracks)
    gt_tracks = run_tracker_on_detections(
        source_detections,
        total_frames,
        tracker_name,
        (height, width),
        sample_rate=1,
    )

    # Run tracker at each sample rate and count mistrack events per cell.
    video_counts = np.zeros((len(RATE_CHOICES), GRID_ROWS, GRID_COLS, 2), dtype=np.int64)
    for rate_idx, rate in enumerate(RATE_CHOICES):
        sampled_tracks = run_tracker_on_detections(
            source_detections,
            total_frames,
            tracker_name,
            (height, width),
            sample_rate=rate,
        )
        video_counts[rate_idx] = calculate_rectangular_misstrack_counts(
            gt_tracks=gt_tracks,
            sampled_tracks=sampled_tracks,
            sample_rate=rate,
            width=width,
            height=height,
            iou_threshold=iou_threshold,
        )

    return video_counts


def run_heuristic_stage(
    dataset: str,
    tracker_name: str,
    iou_threshold: float = 0.3,
    video_fraction_divisor: int = 1,
    num_workers: int = 1,
    force: bool = False,
) -> dict[float, np.ndarray]:
    output_dir = ensure_dir(heuristic_dir(dataset, tracker_name))
    counts_path = heuristic_counts_path(dataset, tracker_name)
    accuracy_path = heuristic_accuracy_path(dataset, tracker_name)
    rate_table_path = heuristic_rate_table_path(dataset, tracker_name)
    metadata_path = heuristic_metadata_path(dataset, tracker_name)

    if (
        not force
        and counts_path.exists()
        and accuracy_path.exists()
        and rate_table_path.exists()
        and metadata_path.exists()
    ):
        return load_heuristic_rate_grids(dataset, tracker_name)

    train_videos = subsample_videos(
        list_split_videos(dataset, 'train'),
        divisor=video_fraction_divisor,
    )
    partial_dir = ensure_dir(output_dir / 'partial')

    total_counts = np.zeros((len(RATE_CHOICES), GRID_ROWS, GRID_COLS, 2), dtype=np.int64)

    # Parallelize per-video heuristic computation.
    fork_context = mp.get_context('fork')
    worker_count = min(max(1, num_workers), len(train_videos)) if train_videos else 1
    with fork_context.Pool(processes=worker_count) as pool:
        video_results = pool.starmap(
            _process_single_video,
            [(dataset, video, tracker_name, iou_threshold) for video in train_videos],
        )

    # Aggregate per-video counts and save partial results.
    for video, video_counts in zip(train_videos, video_results):
        np.save(partial_dir / f'{Path(video).stem}.npy', video_counts)
        total_counts += video_counts

    accuracy, rate_table = build_rate_tables_from_counts(total_counts)
    np.save(counts_path, total_counts)
    np.save(accuracy_path, accuracy)
    np.save(rate_table_path, rate_table)

    metadata = {
        'dataset': dataset,
        'tracker': tracker_name,
        'thresholds': list(HEURISTIC_THRESHOLDS),
        'rate_choices': list(RATE_CHOICES),
        'grid_rows': GRID_ROWS,
        'grid_cols': GRID_COLS,
        'train_videos': train_videos,
        'video_fraction_divisor': video_fraction_divisor,
        'iou_threshold': iou_threshold,
    }
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

    return {
        float(threshold): rate_table[:, :, idx].astype(np.int32)
        for idx, threshold in enumerate(HEURISTIC_THRESHOLDS)
    }
