import sys
from pathlib import Path

import numpy as np
import pandas as pd


ABLATION_DIR = Path(__file__).resolve().parents[1] / 'ablation' / 'mistrack-rate'
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

from mistrack_rate.analysis import annotate_pareto_flags
from mistrack_rate.common import (
    GRID_COLS,
    GRID_ROWS,
    PreparedVideoData,
    center_cell_for_box,
    decode_rate_grid,
    encode_rate_grid,
    get_overlapping_rect_cells,
    iter_rate_grids,
    subsample_videos,
)
import mistrack_rate.evaluate as evaluate_module
from mistrack_rate.evaluate import (
    EvaluationTask,
    _evaluate_candidate_tasks,
    calculate_schedule_aware_mistrack_totals,
)
from mistrack_rate.heuristic import build_rate_tables_from_counts


def _fake_evaluate_rate_grid(
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
    _ = prepared_videos
    _ = iou_threshold
    _ = time_limit_seconds
    _ = compute_hota
    _ = keep_temp_tracks

    return {
        'dataset': dataset,
        'tracker': tracker_name,
        'method': method,
        'heuristic_threshold': heuristic_threshold,
        'variant_id': (
            f'{method}_{encode_rate_grid(rate_grid)}'
            if heuristic_threshold is None
            else f'{method}_t{heuristic_threshold:03d}_{encode_rate_grid(rate_grid)}'
        ),
        'grid_key': encode_rate_grid(rate_grid),
        'grid_rates_json': rate_grid.astype(int).tolist(),
        'score': int(rate_grid.sum()),
    }


def test_rectangular_overlap_maps_box_to_expected_3x3_cells():
    overlap = get_overlapping_rect_cells(
        x1=120.0,
        y1=80.0,
        x2=480.0,
        y2=260.0,
        width=1080,
        height=720,
    )

    assert overlap == (0, 1, 0, 1)


def test_center_cell_for_box_uses_rectangular_cells():
    center_cell = center_cell_for_box(
        x1=730.0,
        y1=500.0,
        x2=770.0,
        y2=620.0,
        width=1080,
        height=720,
    )

    assert center_cell == (2, 2)


def test_iter_rate_grids_enumerates_all_3_power_9_configs():
    encoded = {encode_rate_grid(rate_grid) for rate_grid in iter_rate_grids()}

    assert len(encoded) == 3 ** (GRID_ROWS * GRID_COLS)


def test_encode_rate_grid_round_trips():
    rate_grid = np.asarray([
        [1, 2, 4],
        [4, 2, 1],
        [2, 1, 4],
    ], dtype=np.int32)

    assert np.array_equal(decode_rate_grid(encode_rate_grid(rate_grid)), rate_grid)


def test_subsample_videos_keeps_every_third_video_deterministically():
    videos = [f'v{index:02d}.mp4' for index in range(10)]

    assert subsample_videos(videos, divisor=3) == [
        'v00.mp4',
        'v03.mp4',
        'v06.mp4',
        'v09.mp4',
    ]


def test_build_rate_tables_from_counts_selects_highest_rate_meeting_threshold():
    counts = np.zeros((3, GRID_ROWS, GRID_COLS, 2), dtype=np.int64)

    counts[0, 0, 0] = [14, 6]
    counts[1, 0, 0] = [16, 4]
    counts[2, 0, 0] = [18, 0]

    counts[0, 1, 1] = [18, 0]
    counts[1, 1, 1] = [18, 0]
    counts[2, 1, 1] = [18, 0]

    accuracy, rate_table = build_rate_tables_from_counts(
        counts,
        thresholds=(70, 85),
        rate_choices=(1, 2, 4),
    )

    assert accuracy.shape == (3, GRID_ROWS, GRID_COLS)
    assert rate_table[0, 0, 0] == 4
    assert rate_table[0, 0, 1] == 4
    assert rate_table[1, 0, 1] == 1
    assert rate_table[1, 1, 1] == 4


def test_schedule_aware_mistrack_totals_counts_anchor_mismatch():
    gt_tracks = {
        0: [[1, 0.0, 0.0, 20.0, 20.0]],
        1: [[1, 0.0, 0.0, 20.0, 20.0]],
    }
    predicted_tracks = {
        0: [[10, 0.0, 0.0, 20.0, 20.0]],
    }
    retained_bitmaps = np.ones((2, GRID_ROWS, GRID_COLS), dtype=np.uint8)

    correct, incorrect = calculate_schedule_aware_mistrack_totals(
        gt_tracks=gt_tracks,
        predicted_tracks=predicted_tracks,
        retained_bitmaps=retained_bitmaps,
        width=1080,
        height=720,
        iou_threshold=0.3,
    )

    assert correct == 0
    assert incorrect == 2


def test_annotate_pareto_flags_marks_exhaustive_frontier_and_matching_heuristic():
    results_df = pd.DataFrame([
        {
            'method': 'exhaustive',
            'variant_id': 'exhaustive_a',
            'grid_key': 'a',
            'mistrack_rate': 0.10,
            'retention_rate': 0.60,
            'HOTA_HOTA': 0.80,
        },
        {
            'method': 'exhaustive',
            'variant_id': 'exhaustive_b',
            'grid_key': 'b',
            'mistrack_rate': 0.20,
            'retention_rate': 0.65,
            'HOTA_HOTA': 0.75,
        },
        {
            'method': 'exhaustive',
            'variant_id': 'exhaustive_c',
            'grid_key': 'c',
            'mistrack_rate': 0.30,
            'retention_rate': 0.70,
            'HOTA_HOTA': 0.78,
        },
        {
            'method': 'heuristic',
            'variant_id': 'heuristic_a',
            'grid_key': 'a',
            'mistrack_rate': 0.10,
            'retention_rate': 0.60,
            'HOTA_HOTA': 0.80,
        },
    ])

    flagged_df = annotate_pareto_flags(results_df)

    assert bool(flagged_df.loc[flagged_df['grid_key'] == 'a', 'is_pareto_mistrack_vs_hota'].all())
    assert not bool(flagged_df.loc[flagged_df['grid_key'] == 'b', 'is_pareto_mistrack_vs_hota'].iloc[0])


def test_evaluate_candidate_tasks_parallel_matches_sequential(monkeypatch):
    monkeypatch.setattr(evaluate_module, '_evaluate_rate_grid', _fake_evaluate_rate_grid)

    candidate_tasks = [
        EvaluationTask(
            variant_id='exhaustive_1-1-1-1-1-1-1-1-1',
            method='exhaustive',
            encoded_grid='1-1-1-1-1-1-1-1-1',
            heuristic_threshold=None,
        ),
        EvaluationTask(
            variant_id='exhaustive_2-2-2-2-2-2-2-2-2',
            method='exhaustive',
            encoded_grid='2-2-2-2-2-2-2-2-2',
            heuristic_threshold=None,
        ),
        EvaluationTask(
            variant_id='heuristic_t090_4-4-4-4-4-4-4-4-4',
            method='heuristic',
            encoded_grid='4-4-4-4-4-4-4-4-4',
            heuristic_threshold=90,
        ),
    ]

    sequential_rows = _evaluate_candidate_tasks(
        candidate_tasks=candidate_tasks,
        dataset='jnc0',
        tracker_name='sortcython',
        prepared_videos=[],
        iou_threshold=0.3,
        time_limit_seconds=0.1,
        compute_hota=False,
        keep_temp_tracks=False,
        num_workers=1,
    )
    parallel_rows = _evaluate_candidate_tasks(
        candidate_tasks=candidate_tasks,
        dataset='jnc0',
        tracker_name='sortcython',
        prepared_videos=[],
        iou_threshold=0.3,
        time_limit_seconds=0.1,
        compute_hota=False,
        keep_temp_tracks=False,
        num_workers=2,
    )

    assert sequential_rows == parallel_rows
