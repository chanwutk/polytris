from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import uuid
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from polyis.pack.adapters import group_tiles_all
from polyis.sample.ilp.gurobi import solve_ilp
from scripts.p016_tune_track_rate import (
    _build_track_presence,
    _match_detections_per_frame,
    run_tracker_on_detections,
)

from .analysis import annotate_pareto_flags, save_results
from .common import (
    GRID_COLS,
    GRID_ROWS,
    PreparedVideoData,
    ablation_root,
    build_relevance_bitmaps,
    count_active_cells,
    decode_rate_grid,
    encode_rate_grid,
    ensure_dir,
    evaluation_dir,
    filter_tracks_to_detection_arrays,
    frame_metadata,
    heuristic_metadata_path,
    iter_rate_grids,
    list_split_videos,
    load_naive_tracking_source,
    make_variant_id,
    rate_grid_json,
    retention_rate,
    save_tracking_results_with_total_frames,
    subsample_videos,
    tracks_to_detection_arrays,
)
from .heuristic import load_heuristic_rate_grids, run_heuristic_stage


@dataclass(frozen=True)
class EvaluationTask:
    variant_id: str
    method: str
    encoded_grid: str
    heuristic_threshold: int | None


_WORKER_EVALUATION_STATE: dict[str, object] | None = None


def _prepare_video_data(
    dataset: str,
    video: str,
    tracker_name: str,
) -> PreparedVideoData:
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
    relevance_bitmaps = build_relevance_bitmaps(source_tracks, total_frames, width, height)
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(relevance_bitmaps.astype(np.uint8), 0)

    return PreparedVideoData(
        video=video,
        width=width,
        height=height,
        total_frames=total_frames,
        source_tracks=source_tracks,
        source_detections=source_detections,
        gt_tracks=gt_tracks,
        relevance_bitmaps=relevance_bitmaps,
        tile_to_polyomino_id=np.asarray(tile_to_polyomino_id),
        polyomino_lengths=polyomino_lengths,
    )


def _prune_video_bitmaps(video_data: PreparedVideoData, rate_grid: np.ndarray, time_limit_seconds: float) -> np.ndarray:
    ilp_result = solve_ilp(
        tile_to_polyomino_id=video_data.tile_to_polyomino_id,
        polyomino_lengths=video_data.polyomino_lengths,
        max_sampling_distance=rate_grid.astype(np.int32),
        grid_height=GRID_ROWS,
        grid_width=GRID_COLS,
        time_limit_seconds=time_limit_seconds,
    )

    # Pre-group selected polyomino IDs by frame to avoid O(F * |selected|) scan.
    selected_by_frame: dict[int, set[int]] = defaultdict(set)
    for selected_frame, poly_id in ilp_result.selected:
        selected_by_frame[selected_frame].add(poly_id)

    pruned_bitmaps = np.zeros_like(video_data.relevance_bitmaps, dtype=np.uint8)

    for frame_idx in range(len(video_data.polyomino_lengths)):
        selected_ids = selected_by_frame.get(frame_idx, set())
        if not selected_ids:
            continue
        tile_ids = video_data.tile_to_polyomino_id[frame_idx]
        mask = np.isin(tile_ids, list(selected_ids)) & (tile_ids >= 0)
        pruned_bitmaps[frame_idx] = mask.astype(np.uint8)

    return pruned_bitmaps


def _build_anchor_frames_by_track(
    gt_tracks: dict[int, list[list[float]]],
    retained_bitmaps: np.ndarray,
    width: int,
    height: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    from .common import track_center_is_active

    anchors_by_track: dict[int, list[int]] = {}
    anchors_by_frame: dict[int, list[int]] = {}

    for frame_idx, gt_dets in gt_tracks.items():
        if frame_idx >= len(retained_bitmaps):
            continue

        active_bitmap = retained_bitmaps[frame_idx]
        anchor_indices: list[int] = []
        for det_idx, gt_det in enumerate(gt_dets):
            if not track_center_is_active(gt_det, active_bitmap, width, height, bbox_slice=slice(1, 5)):
                continue

            track_id = int(gt_det[0])
            anchors_by_track.setdefault(track_id, []).append(frame_idx)
            anchor_indices.append(det_idx)

        if anchor_indices:
            anchors_by_frame[frame_idx] = anchor_indices

    return anchors_by_track, anchors_by_frame


def _association_holds_at_anchor(
    gt_track_id: int,
    pred_track_id: int,
    frame_idx: int,
    anchors_by_track: dict[int, list[int]],
    pred_presence: set[tuple[int, int]],
    matched_pairs: dict[int, set[tuple[int, int]]],
    direction: int,
) -> bool:
    anchor_frames = anchors_by_track.get(gt_track_id, [])
    if not anchor_frames:
        return True

    anchor_pos = bisect_left(anchor_frames, frame_idx)
    if anchor_pos >= len(anchor_frames) or anchor_frames[anchor_pos] != frame_idx:
        return True

    next_pos = anchor_pos + direction
    if next_pos < 0 or next_pos >= len(anchor_frames):
        return True

    next_frame = anchor_frames[next_pos]
    if (next_frame, pred_track_id) not in pred_presence:
        return False

    return (gt_track_id, pred_track_id) in matched_pairs.get(next_frame, set())


def calculate_schedule_aware_mistrack_totals(
    gt_tracks: dict[int, list[list[float]]],
    predicted_tracks: dict[int, list[list[float]]],
    retained_bitmaps: np.ndarray,
    width: int,
    height: int,
    iou_threshold: float = 0.3,
) -> tuple[int, int]:
    pred_presence, _, _ = _build_track_presence(predicted_tracks)
    _, matched_pairs, match_details, matched_gt = _match_detections_per_frame(
        gt_tracks,
        predicted_tracks,
        sample_rate=1,
        iou_threshold=iou_threshold,
    )
    anchors_by_track, anchors_by_frame = _build_anchor_frames_by_track(
        gt_tracks,
        retained_bitmaps,
        width,
        height,
    )

    correct = 0
    incorrect = 0

    for frame_idx, anchor_indices in anchors_by_frame.items():
        gt_dets = gt_tracks.get(frame_idx, [])
        frame_match_map = {
            gt_idx: predicted_track_id
            for gt_idx, _pred_idx, _gt_track_id, predicted_track_id in match_details.get(frame_idx, [])
        }
        frame_matched_gt = matched_gt.get(frame_idx, set())

        for gt_idx in anchor_indices:
            gt_det = gt_dets[gt_idx]
            gt_track_id = int(gt_det[0])

            if gt_idx not in frame_matched_gt:
                incorrect += 1
                continue

            pred_track_id = int(frame_match_map[gt_idx])
            forward_ok = _association_holds_at_anchor(
                gt_track_id,
                pred_track_id,
                frame_idx,
                anchors_by_track,
                pred_presence,
                matched_pairs,
                direction=1,
            )
            backward_ok = _association_holds_at_anchor(
                gt_track_id,
                pred_track_id,
                frame_idx,
                anchors_by_track,
                pred_presence,
                matched_pairs,
                direction=-1,
            )

            if forward_ok and backward_ok:
                correct += 1
            else:
                incorrect += 1

    return correct, incorrect


def _resolve_groundtruth_dataset(dataset: str) -> str:
    if dataset.startswith('caldot1-y'):
        return 'caldot1'
    if dataset.startswith('caldot2-y'):
        return 'caldot2'
    if dataset.startswith('ams-y'):
        return 'ams'
    return dataset


def _compute_hota(
    dataset: str,
    videos: list[str],
    prediction_tracks: dict[str, dict[int, list[list[float]]]],
    total_frames_by_video: dict[str, int],
    tracker_name: str,
    keep_temp_tracks: bool,
) -> float:
    if '/polyis/modules/TrackEval' not in sys.path:
        sys.path.append('/polyis/modules/TrackEval')

    import trackeval
    from trackeval.metrics import HOTA

    from polyis.trackeval.dataset import Dataset

    gt_dataset = _resolve_groundtruth_dataset(dataset)
    gt_root = Path('/polyis-cache') / gt_dataset / 'execution'
    temp_parent = ensure_dir(ablation_root(dataset, tracker_name, 'tmp_tracks'))

    with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for video in videos:
            output_path = temp_dir / video / 'tracking.jsonl'
            save_tracking_results_with_total_frames(
                frame_tracks=prediction_tracks[video],
                total_frames=total_frames_by_video[video],
                output_path=output_path,
            )

        dataset_config = {
            'output_fol': str(temp_dir),
            'output_sub_fol': 'mistrack_rate_ablation',
            'input_gt': str(Path('003_groundtruth') / 'tracking.jsonl'),
            'input_track': 'tracking.jsonl',
            'skip': 1,
            'tracker': 'mistrack-rate-ablation',
            'seq_list': sorted(videos),
            'input_dir': str(temp_dir),
            'input_gt_dir': str(gt_root),
            'input_track_dir': str(temp_dir),
        }
        eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'BREAK_ON_ERROR': True,
            'LOG_ON_ERROR': str(temp_dir / 'LOG.txt'),
            'PRINT_RESULTS': False,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'OUTPUT_SUMMARY': False,
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
            'OUTPUT_EMPTY_CLASSES': False,
        }

        evaluator = trackeval.Evaluator(eval_config)
        eval_dataset = Dataset(dataset_config)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = evaluator.evaluate([eval_dataset], [HOTA()])

        tracker_results = results[0].get('Dataset', {}).get('sort', {})
        combined = tracker_results['COMBINED_SEQ']['vehicle']['HOTA']
        hota_value = float(sum(combined['HOTA']) / len(combined['HOTA']))

        if keep_temp_tracks:
            preserved_dir = ensure_dir(evaluation_dir(dataset, tracker_name) / 'preserved_tracks')
            target_dir = preserved_dir / uuid.uuid4().hex
            shutil.copytree(temp_dir, target_dir)

        return hota_value


def _evaluate_rate_grid(
    dataset: str,
    tracker_name: str,
    prepared_videos: list[PreparedVideoData],
    rate_grid: np.ndarray,
    iou_threshold: float,
    time_limit_seconds: float,
    compute_hota: bool,
    keep_temp_tracks: bool,
    method: str,
    heuristic_threshold: int | None,
) -> dict[str, object]:
    total_correct = 0
    total_incorrect = 0
    retention_numerator = 0
    retention_denominator = 0
    prediction_tracks: dict[str, dict[int, list[list[float]]]] = {}

    for video_data in prepared_videos:
        pruned_bitmaps = _prune_video_bitmaps(video_data, rate_grid, time_limit_seconds)
        filtered_detections = filter_tracks_to_detection_arrays(
            frame_tracks=video_data.source_tracks,
            active_bitmaps=pruned_bitmaps,
            width=video_data.width,
            height=video_data.height,
        )
        predicted_tracks = run_tracker_on_detections(
            filtered_detections,
            video_data.total_frames,
            tracker_name,
            (video_data.height, video_data.width),
            sample_rate=1,
        )
        correct, incorrect = calculate_schedule_aware_mistrack_totals(
            gt_tracks=video_data.gt_tracks,
            predicted_tracks=predicted_tracks,
            retained_bitmaps=pruned_bitmaps,
            width=video_data.width,
            height=video_data.height,
            iou_threshold=iou_threshold,
        )
        total_correct += correct
        total_incorrect += incorrect
        retention_numerator += count_active_cells(pruned_bitmaps)
        retention_denominator += count_active_cells(video_data.relevance_bitmaps)
        prediction_tracks[video_data.video] = predicted_tracks

    hota_value = float('nan')
    if compute_hota:
        hota_value = _compute_hota(
            dataset=dataset,
            videos=[video_data.video for video_data in prepared_videos],
            prediction_tracks=prediction_tracks,
            total_frames_by_video={
                video_data.video: video_data.total_frames
                for video_data in prepared_videos
            },
            tracker_name=tracker_name,
            keep_temp_tracks=keep_temp_tracks,
        )

    total_anchors = total_correct + total_incorrect
    mistrack_rate = (
        float(total_incorrect) / float(total_anchors)
        if total_anchors > 0
        else 0.0
    )
    retention_value = (
        float(retention_numerator) / float(retention_denominator)
        if retention_denominator > 0
        else 0.0
    )
    encoded_grid = encode_rate_grid(rate_grid)

    return {
        'dataset': dataset,
        'split': 'test',
        'tracker': tracker_name,
        'method': method,
        'heuristic_threshold': heuristic_threshold,
        'variant_id': make_variant_id(method, rate_grid, heuristic_threshold),
        'grid_key': encoded_grid,
        'grid_rates_json': rate_grid_json(rate_grid),
        'mistrack_rate': mistrack_rate,
        'HOTA_HOTA': hota_value,
        'retention_rate': retention_value,
        'anchor_correct': total_correct,
        'anchor_incorrect': total_incorrect,
        'anchor_total': total_anchors,
    }


def _set_worker_evaluation_state(
    dataset: str,
    tracker_name: str,
    prepared_videos: list[PreparedVideoData],
    iou_threshold: float,
    time_limit_seconds: float,
    compute_hota: bool,
    keep_temp_tracks: bool,
) -> None:
    global _WORKER_EVALUATION_STATE

    _WORKER_EVALUATION_STATE = {
        'dataset': dataset,
        'tracker_name': tracker_name,
        'prepared_videos': prepared_videos,
        'iou_threshold': iou_threshold,
        'time_limit_seconds': time_limit_seconds,
        'compute_hota': compute_hota,
        'keep_temp_tracks': keep_temp_tracks,
    }


def _clear_worker_evaluation_state() -> None:
    global _WORKER_EVALUATION_STATE

    _WORKER_EVALUATION_STATE = None


def _evaluate_task_worker(task: EvaluationTask) -> dict[str, object]:
    if _WORKER_EVALUATION_STATE is None:
        raise RuntimeError('Worker evaluation state has not been initialized')

    return _evaluate_rate_grid(
        dataset=str(_WORKER_EVALUATION_STATE['dataset']),
        tracker_name=str(_WORKER_EVALUATION_STATE['tracker_name']),
        prepared_videos=list(_WORKER_EVALUATION_STATE['prepared_videos']),
        rate_grid=decode_rate_grid(task.encoded_grid),
        iou_threshold=float(_WORKER_EVALUATION_STATE['iou_threshold']),
        time_limit_seconds=float(_WORKER_EVALUATION_STATE['time_limit_seconds']),
        compute_hota=bool(_WORKER_EVALUATION_STATE['compute_hota']),
        keep_temp_tracks=bool(_WORKER_EVALUATION_STATE['keep_temp_tracks']),
        method=task.method,
        heuristic_threshold=task.heuristic_threshold,
    )


def _evaluate_candidate_tasks(
    candidate_tasks: list[EvaluationTask],
    dataset: str,
    tracker_name: str,
    prepared_videos: list[PreparedVideoData],
    iou_threshold: float,
    time_limit_seconds: float,
    compute_hota: bool,
    keep_temp_tracks: bool,
    num_workers: int,
) -> list[dict[str, object]]:
    if not candidate_tasks:
        return []

    worker_count = max(1, int(num_workers))
    task_order = {
        task.variant_id: index
        for index, task in enumerate(candidate_tasks)
    }
    _set_worker_evaluation_state(
        dataset=dataset,
        tracker_name=tracker_name,
        prepared_videos=prepared_videos,
        iou_threshold=iou_threshold,
        time_limit_seconds=time_limit_seconds,
        compute_hota=compute_hota,
        keep_temp_tracks=keep_temp_tracks,
    )

    try:
        progress = Progress(
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not sys.stderr.isatty(),
        )

        with progress:
            progress_task_id = progress.add_task(
                description=f'{dataset} {tracker_name} configs',
                total=len(candidate_tasks),
            )

            fork_context = mp.get_context('fork')
            rows: list[dict[str, object]] = []
            with fork_context.Pool(processes=worker_count) as pool:
                for row in pool.imap_unordered(_evaluate_task_worker, candidate_tasks):
                    rows.append(row)
                    progress.advance(progress_task_id)

            rows.sort(key=lambda row: task_order[str(row['variant_id'])])
            return rows
    finally:
        _clear_worker_evaluation_state()


def _build_prepared_videos(
    dataset: str,
    tracker_name: str,
    video_fraction_divisor: int,
    num_workers: int = 1,
) -> list[PreparedVideoData]:
    test_videos = subsample_videos(
        list_split_videos(dataset, 'test'),
        divisor=video_fraction_divisor,
    )

    fork_context = mp.get_context('fork')
    worker_count = min(max(1, num_workers), len(test_videos)) if test_videos else 1
    with fork_context.Pool(processes=worker_count) as pool:
        return pool.starmap(
            _prepare_video_data,
            [(dataset, video, tracker_name) for video in test_videos],
        )


def run_evaluation_stage(
    dataset: str,
    tracker_name: str,
    iou_threshold: float = 0.3,
    time_limit_seconds: float = 0.1,
    limit_configs: int | None = None,
    compute_hota: bool = True,
    keep_temp_tracks: bool = False,
    num_workers: int = 75,
    video_fraction_divisor: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    # Ensure heuristic grids are available before evaluation.
    run_heuristic_stage(
        dataset,
        tracker_name,
        iou_threshold=iou_threshold,
        video_fraction_divisor=video_fraction_divisor,
        num_workers=num_workers,
        force=force,
    )

    output_dir = ensure_dir(evaluation_dir(dataset, tracker_name))
    existing_path = output_dir / 'results.csv'
    existing_df = pd.DataFrame()
    seen_variants: set[str] = set()
    if existing_path.exists() and not force:
        existing_df = pd.read_csv(existing_path)
        seen_variants = set(existing_df['variant_id'])

    if force and existing_path.exists():
        existing_path.unlink()

    prepared_videos = _build_prepared_videos(
        dataset,
        tracker_name,
        video_fraction_divisor=video_fraction_divisor,
        num_workers=num_workers,
    )
    heuristic_grids = load_heuristic_rate_grids(dataset, tracker_name)
    with open(heuristic_metadata_path(dataset, tracker_name), 'r', encoding='utf-8') as file:
        heuristic_metadata = json.load(file)
    heuristic_thresholds = [int(value) for value in heuristic_metadata['thresholds']]

    candidate_tasks: list[EvaluationTask] = []

    # Exhaustive configs are subject to limit_configs budget.
    exhaustive_count = 0
    for rate_grid in iter_rate_grids():
        variant_id = make_variant_id('exhaustive', rate_grid)
        if variant_id in seen_variants:
            continue
        if limit_configs is not None and exhaustive_count >= limit_configs:
            break

        candidate_tasks.append(EvaluationTask(
            variant_id=variant_id,
            method='exhaustive',
            encoded_grid=encode_rate_grid(rate_grid),
            heuristic_threshold=None,
        ))
        exhaustive_count += 1

    # Heuristic configs are always included regardless of limit_configs.
    for threshold in heuristic_thresholds:
        rate_grid = heuristic_grids[threshold]
        variant_id = make_variant_id('heuristic', rate_grid, threshold)
        if variant_id in seen_variants:
            continue

        candidate_tasks.append(EvaluationTask(
            variant_id=variant_id,
            method='heuristic',
            encoded_grid=encode_rate_grid(rate_grid),
            heuristic_threshold=threshold,
        ))

    candidate_rows = _evaluate_candidate_tasks(
        candidate_tasks=candidate_tasks,
        dataset=dataset,
        tracker_name=tracker_name,
        prepared_videos=prepared_videos,
        iou_threshold=iou_threshold,
        time_limit_seconds=time_limit_seconds,
        compute_hota=compute_hota,
        keep_temp_tracks=keep_temp_tracks,
        num_workers=num_workers,
    )

    new_df = pd.DataFrame.from_records(candidate_rows)
    if existing_df.empty:
        combined_df = new_df
    elif new_df.empty:
        combined_df = existing_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    if combined_df.empty:
        return combined_df

    combined_df = combined_df.drop_duplicates(subset=['variant_id'], keep='last').reset_index(drop=True)
    combined_df = annotate_pareto_flags(combined_df)
    save_results(combined_df, dataset, tracker_name)
    return combined_df
