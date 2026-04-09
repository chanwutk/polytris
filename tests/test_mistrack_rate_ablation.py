import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
    retention_rate,
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


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

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


def test_overlap_returns_none_for_box_entirely_outside_frame():
    # Box entirely to the right of the frame clips to zero width.
    assert get_overlapping_rect_cells(
        x1=1200.0, y1=100.0, x2=1400.0, y2=200.0,
        width=1080, height=720,
    ) is None

    # Box entirely above the frame clips to zero height.
    assert get_overlapping_rect_cells(
        x1=100.0, y1=-200.0, x2=200.0, y2=-100.0,
        width=1080, height=720,
    ) is None


def test_center_cell_returns_none_for_degenerate_box():
    result = center_cell_for_box(
        x1=-100.0,
        y1=-100.0,
        x2=-50.0,
        y2=-50.0,
        width=1080,
        height=720,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Grid enumeration and encoding tests
# ---------------------------------------------------------------------------

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


def test_decode_rate_grid_rejects_malformed_input():
    with pytest.raises(AssertionError, match='Expected 9 values'):
        decode_rate_grid('1-2-3')

    with pytest.raises(AssertionError, match='Expected 9 values'):
        decode_rate_grid('1-2-3-4-5-6-7-8-9-10')


# ---------------------------------------------------------------------------
# Video subsampling tests
# ---------------------------------------------------------------------------

def test_subsample_videos_keeps_every_third_video_deterministically():
    videos = [f'v{index:02d}.mp4' for index in range(10)]

    assert subsample_videos(videos, divisor=3) == [
        'v00.mp4',
        'v03.mp4',
        'v06.mp4',
        'v09.mp4',
    ]


def test_subsample_videos_rejects_invalid_divisor():
    videos = ['a.mp4', 'b.mp4']

    with pytest.raises(ValueError, match='positive divisor'):
        subsample_videos(videos, divisor=0)

    with pytest.raises(ValueError, match='positive divisor'):
        subsample_videos(videos, divisor=-1)


def test_subsample_videos_rejects_invalid_remainder():
    videos = ['a.mp4', 'b.mp4']

    with pytest.raises(ValueError, match='remainder'):
        subsample_videos(videos, divisor=3, remainder=-1)

    with pytest.raises(ValueError, match='remainder'):
        subsample_videos(videos, divisor=3, remainder=3)


def test_subsample_videos_empty_list():
    assert subsample_videos([], divisor=2) == []


# ---------------------------------------------------------------------------
# Heuristic rate table tests
# ---------------------------------------------------------------------------

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


def test_build_rate_tables_invariant_highest_rate_meets_threshold():
    # For each cell and threshold, the selected rate must be the highest
    # rate whose Laplace-smoothed accuracy meets the threshold.
    counts = np.zeros((3, GRID_ROWS, GRID_COLS, 2), dtype=np.int64)

    # Set up varied counts per cell so different rates have different accuracies.
    counts[0, 0, 0] = [10, 10]  # rate=1: accuracy ~0.52
    counts[1, 0, 0] = [15, 5]   # rate=2: accuracy ~0.73
    counts[2, 0, 0] = [19, 1]   # rate=4: accuracy ~0.91

    counts[0, 2, 2] = [19, 1]   # rate=1: accuracy ~0.91
    counts[1, 2, 2] = [10, 10]  # rate=2: accuracy ~0.52
    counts[2, 2, 2] = [5, 15]   # rate=4: accuracy ~0.30

    thresholds = (50, 70, 90)
    rate_choices = (1, 2, 4)
    accuracy, rate_table = build_rate_tables_from_counts(
        counts,
        thresholds=thresholds,
        rate_choices=rate_choices,
    )

    # Verify invariant: for each cell and threshold, the selected rate is the
    # highest rate whose accuracy >= threshold.
    for threshold_idx, threshold_pct in enumerate(thresholds):
        threshold_value = float(threshold_pct) / 100.0
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                selected_rate = rate_table[row, col, threshold_idx]
                # The selected rate must meet the threshold (or be the fallback rate=1).
                if selected_rate > rate_choices[0]:
                    rate_idx = rate_choices.index(selected_rate)
                    assert accuracy[rate_idx, row, col] >= threshold_value, (
                        f'Selected rate {selected_rate} at ({row},{col}) threshold={threshold_pct}% '
                        f'has accuracy {accuracy[rate_idx, row, col]:.3f} < {threshold_value}'
                    )
                # No higher rate should also meet the threshold.
                selected_rate_idx = rate_choices.index(selected_rate)
                for higher_rate_idx in range(selected_rate_idx + 1, len(rate_choices)):
                    assert accuracy[higher_rate_idx, row, col] < threshold_value, (
                        f'Higher rate {rate_choices[higher_rate_idx]} at ({row},{col}) '
                        f'threshold={threshold_pct}% has accuracy '
                        f'{accuracy[higher_rate_idx, row, col]:.3f} >= {threshold_value} '
                        f'but was not selected'
                    )


# ---------------------------------------------------------------------------
# Mistrack counting tests
# ---------------------------------------------------------------------------

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


def test_schedule_aware_mistrack_totals_empty_inputs():
    # No GT tracks means no anchors, so both counts should be zero.
    correct, incorrect = calculate_schedule_aware_mistrack_totals(
        gt_tracks={},
        predicted_tracks={},
        retained_bitmaps=np.ones((5, GRID_ROWS, GRID_COLS), dtype=np.uint8),
        width=1080,
        height=720,
    )

    assert correct == 0
    assert incorrect == 0


# ---------------------------------------------------------------------------
# Retention rate tests
# ---------------------------------------------------------------------------

def test_retention_rate_zero_original_returns_zero():
    original = np.zeros((3, GRID_ROWS, GRID_COLS), dtype=np.uint8)
    pruned = np.zeros((3, GRID_ROWS, GRID_COLS), dtype=np.uint8)

    assert retention_rate(original, pruned) == 0.0


def test_retention_rate_full_retention():
    original = np.ones((3, GRID_ROWS, GRID_COLS), dtype=np.uint8)
    pruned = np.ones((3, GRID_ROWS, GRID_COLS), dtype=np.uint8)

    assert retention_rate(original, pruned) == 1.0


# ---------------------------------------------------------------------------
# Pareto annotation tests
# ---------------------------------------------------------------------------

def _make_pareto_test_data() -> pd.DataFrame:
    return pd.DataFrame([
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


def test_annotate_pareto_flags_marks_exhaustive_frontier_only():
    results_df = _make_pareto_test_data()
    flagged_df = annotate_pareto_flags(results_df)

    # Point 'a' is Pareto-optimal for mistrack_vs_hota (lowest mistrack, highest HOTA).
    assert bool(
        flagged_df.loc[
            (flagged_df['grid_key'] == 'a') & (flagged_df['method'] == 'exhaustive'),
            'is_pareto_mistrack_vs_hota',
        ].iloc[0]
    )
    # Point 'b' is dominated.
    assert not bool(
        flagged_df.loc[flagged_df['grid_key'] == 'b', 'is_pareto_mistrack_vs_hota'].iloc[0]
    )
    # Heuristic row with same grid_key 'a' should NOT be flagged as Pareto.
    assert not bool(
        flagged_df.loc[
            (flagged_df['grid_key'] == 'a') & (flagged_df['method'] == 'heuristic'),
            'is_pareto_mistrack_vs_hota',
        ].iloc[0]
    )


def test_annotate_pareto_flags_mistrack_vs_retention():
    # mistrack_vs_retention: minimize mistrack (x), minimize retention (y).
    # Point 'a' has (0.10, 0.60) which dominates 'b' (0.20, 0.65) and 'c' (0.30, 0.70).
    results_df = _make_pareto_test_data()
    flagged_df = annotate_pareto_flags(results_df)

    assert bool(
        flagged_df.loc[
            (flagged_df['grid_key'] == 'a') & (flagged_df['method'] == 'exhaustive'),
            'is_pareto_mistrack_vs_retention',
        ].iloc[0]
    )


def test_annotate_pareto_flags_retention_vs_hota():
    # retention_vs_hota: minimize retention (x), maximize HOTA (y).
    # Point 'a' has (0.60, 0.80) — best retention AND best HOTA.
    results_df = _make_pareto_test_data()
    flagged_df = annotate_pareto_flags(results_df)

    assert bool(
        flagged_df.loc[
            (flagged_df['grid_key'] == 'a') & (flagged_df['method'] == 'exhaustive'),
            'is_pareto_retention_vs_hota',
        ].iloc[0]
    )


# ---------------------------------------------------------------------------
# Parallelism tests
# ---------------------------------------------------------------------------

# This test relies on fork-based multiprocessing inheriting the monkeypatched
# _evaluate_rate_grid function. It only works on platforms where 'fork' is
# available (Linux, Docker). On macOS the default context is 'spawn' which
# does not inherit monkeypatched module state.
@pytest.mark.skipif(
    'fork' not in mp.get_all_start_methods(),
    reason='fork context required for monkeypatch to propagate to workers',
)
def test_evaluate_candidate_tasks_returns_results_in_submission_order(monkeypatch):
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

    rows = _evaluate_candidate_tasks(
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

    # Results should be sorted in submission order regardless of pool scheduling.
    assert [row['variant_id'] for row in rows] == [task.variant_id for task in candidate_tasks]
