"""Tests for Pareto front computation and angle-based pruning."""

import numpy as np
import pandas as pd
import pytest

from polyis.pareto import (
    _prune_pareto_points,
    compute_pareto_front,
    compute_pareto_fronts_by_group,
    extract_stage_params,
)


# ---------------------------------------------------------------------------
# _prune_pareto_points unit tests
# ---------------------------------------------------------------------------


class TestPruneNoPruningNeeded:
    """Fewer points than target should keep all points."""

    def test_fewer_than_target(self):
        # 3 points, target 5: all kept.
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.3, 0.2, 0.1])
        mask = _prune_pareto_points(x, y, num_points=5)
        assert mask.all()

    def test_exact_count(self):
        # 5 points, target 5: all kept.
        x = np.arange(5, dtype=float)
        y = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        mask = _prune_pareto_points(x, y, num_points=5)
        assert mask.all()


class TestPruneTwoPoints:
    """Only endpoints present — cannot prune below 2."""

    def test_two_points_target_one(self):
        # 2 points with target 1: both kept (endpoints are never removed).
        x = np.array([0.0, 10.0])
        y = np.array([1.0, 0.0])
        mask = _prune_pareto_points(x, y, num_points=1)
        assert mask.all()


class TestPruneRemovesCollinearFirst:
    """A collinear interior point should be removed before a sharp corner."""

    def test_collinear_removed_first(self):
        # Points: (0,0), (1,1), (2,2), (3,3), (4,0).
        # Points 1, 2, 3 are collinear between endpoints.  Point 3 at (3,3) is
        # a corner adjacent to the sharp drop to (4,0), while point 1 at (1,1) is
        # collinear.  Pruning to 4 should remove one of the collinear interior
        # points (1 or 2), NOT the corner point.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        mask = _prune_pareto_points(x, y, num_points=4)
        kept = np.where(mask)[0]
        # Endpoints (0 and 4) must always be kept.
        assert 0 in kept
        assert 4 in kept
        assert len(kept) == 4
        # The corner point at index 3 (3, 3) should still be present.
        assert 3 in kept


class TestPrunePreservesEndpoints:
    """Pruning to 2 should leave only the first and last points."""

    def test_prune_to_two(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 0.9, 0.5, 0.2, 0.0])
        mask = _prune_pareto_points(x, y, num_points=2)
        kept = np.where(mask)[0].tolist()
        assert kept == [0, 4]


class TestPruneNormalizationMatters:
    """Without normalization the large x-range would dominate angle computation."""

    def test_different_scales(self):
        # x ranges [0, 1000], y ranges [0, 1].
        # Middle point is equidistant in normalized coords but NOT in raw coords.
        # With normalization both axes contribute; without, x dominates.
        x = np.array([0.0, 100.0, 500.0, 900.0, 1000.0])
        y = np.array([1.0, 0.9, 0.5, 0.1, 0.0])
        mask = _prune_pareto_points(x, y, num_points=3)
        kept = np.where(mask)[0]
        # Endpoints must survive.
        assert 0 in kept
        assert 4 in kept
        assert len(kept) == 3


class TestPruneDegenerateSingleAxis:
    """All x (or y) identical — normalization handles zero-range gracefully."""

    def test_all_same_x(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mask = _prune_pareto_points(x, y, num_points=2)
        kept = np.where(mask)[0].tolist()
        assert kept == [0, 3]

    def test_all_same_y(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        mask = _prune_pareto_points(x, y, num_points=2)
        kept = np.where(mask)[0].tolist()
        assert kept == [0, 3]


# ---------------------------------------------------------------------------
# compute_pareto_front integration tests
# ---------------------------------------------------------------------------


class TestComputeParetoFrontWithNumPoints:
    """Integration: DataFrame in, pruned DataFrame out."""

    def test_pruned_result(self):
        # Build a DataFrame with tradeoff points: higher time (better for
        # maximize_x) comes with lower accuracy, so all are non-dominated.
        df = pd.DataFrame({
            'time': list(range(1, 11)),
            'acc':  [1.0 - 0.08 * i for i in range(10)],
        })
        result = compute_pareto_front(df, 'time', 'acc', num_points=3)
        # Should have exactly 3 rows.
        assert len(result) == 3
        # Endpoints must be present.
        assert result['time'].min() == 1
        assert result['time'].max() == 10


class TestComputeParetoFrontNumPointsNone:
    """Backward compatibility: num_points=None returns all Pareto points."""

    def test_no_pruning(self):
        # Tradeoff data: higher time (better for maximize_x) with lower
        # accuracy, so all 5 points are non-dominated.
        df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'acc':  [1.0, 0.9, 0.8, 0.7, 0.6],
        })
        result = compute_pareto_front(df, 'time', 'acc', num_points=None)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# compute_pareto_fronts_by_group integration tests
# ---------------------------------------------------------------------------


class TestComputeParetoFrontsByGroupPassesNumPoints:
    """Per-group pruning should limit each group independently."""

    def test_per_group_pruning(self):
        # Two groups, each with 6 tradeoff points (higher time with lower acc).
        rows = []
        for ds in ['A', 'B']:
            for i in range(6):
                rows.append({'dataset': ds, 'time': float(i + 1), 'acc': 1.0 - i * 0.1})
        df = pd.DataFrame(rows)
        # Default minimize_x=False + decreasing acc with increasing time → all non-dominated.
        result = compute_pareto_fronts_by_group(df, ['dataset'], 'time', 'acc', num_points=3)
        # Each group should have at most 3 points.
        for ds in ['A', 'B']:
            group = result[result['dataset'] == ds]
            assert len(group) <= 3
            assert len(group) >= 2  # At least endpoints.


class TestComputeParetoFrontsByGroupNumPointsNone:
    """Passing num_points=None should override the default of 16."""

    def test_no_pruning_override(self):
        # Single group with 20 tradeoff points (higher time with lower acc).
        df = pd.DataFrame({
            'dataset': ['X'] * 20,
            'time': list(range(1, 21)),
            'acc': [1.0 - i * 0.04 for i in range(20)],
        })
        # Default minimize_x=False + decreasing acc with increasing time → all non-dominated.
        result = compute_pareto_fronts_by_group(df, ['dataset'], 'time', 'acc', num_points=None)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# extract_stage_params unit tests
# ---------------------------------------------------------------------------


def _make_pareto_df() -> pd.DataFrame:
    """Build a small synthetic Pareto DataFrame for extract_stage_params tests."""
    return pd.DataFrame({
        'classifier': ['SimpleCNN', 'SimpleCNN', 'SimpleCNN', 'SimpleCNN'],
        'tilesize': [60, 60, 60, 60],
        'sample_rate': [1, 1, 1, 1],
        'tilepadding': ['nopad', 'nopad', 'nopad', 'nopad'],
        'canvas_scale': [1.0, 1.0, 1.0, 1.0],
        'tracker': ['ocsortcython', 'bytetrackcython', 'ocsortcython', 'bytetrackcython'],
        # First two rows: threshold=NaN (no pruning); last two: threshold=0.4.
        'tracking_accuracy_threshold': [float('nan'), float('nan'), 0.4, 0.4],
    })


class TestExtractStageParamsDefault:
    """Default behavior (collapse_tracker_when_no_threshold=False) preserves tracker."""

    def test_no_collapse(self):
        # With default flag, tracker values are preserved even for NaN threshold rows.
        df = _make_pareto_df()
        columns = ['classifier', 'tilesize', 'sample_rate', 'tilepadding',
                    'canvas_scale', 'tracker', 'tracking_accuracy_threshold']
        result = extract_stage_params(df, columns)
        # All 4 rows are unique (different tracker x threshold combos).
        assert len(result) == 4
        # NaN threshold rows keep their real tracker values.
        assert ('SimpleCNN', 60, 1, 'nopad', 1.0, 'ocsortcython', None) in result
        assert ('SimpleCNN', 60, 1, 'nopad', 1.0, 'bytetrackcython', None) in result

    def test_nan_normalized_to_none(self):
        # Verify NaN threshold is converted to None for tuple equality.
        df = _make_pareto_df()
        columns = ['tracking_accuracy_threshold']
        result = extract_stage_params(df, columns)
        assert (None,) in result
        assert (0.4,) in result


class TestExtractStageParamsCollapseTracker:
    """collapse_tracker_when_no_threshold=True normalizes tracker for NaN threshold rows."""

    def test_tracker_set_to_none_when_threshold_nan(self):
        # With collapse flag, NaN-threshold rows get tracker=None.
        df = _make_pareto_df()
        columns = ['classifier', 'tilesize', 'sample_rate', 'tilepadding',
                    'canvas_scale', 'tracker', 'tracking_accuracy_threshold']
        result = extract_stage_params(df, columns, collapse_tracker_when_no_threshold=True)
        # The two NaN-threshold rows (ocsort, bytetrack) collapse to one (tracker=None).
        assert ('SimpleCNN', 60, 1, 'nopad', 1.0, None, None) in result
        # Real-threshold rows keep their tracker values.
        assert ('SimpleCNN', 60, 1, 'nopad', 1.0, 'ocsortcython', 0.4) in result
        assert ('SimpleCNN', 60, 1, 'nopad', 1.0, 'bytetrackcython', 0.4) in result
        # Total: 1 collapsed + 2 real-threshold = 3.
        assert len(result) == 3

    def test_deduplication_after_collapse(self):
        # Multiple tracker values with NaN threshold should deduplicate to one tuple.
        df = pd.DataFrame({
            'tracker': ['a', 'b', 'c'],
            'tracking_accuracy_threshold': [float('nan'), float('nan'), float('nan')],
        })
        columns = ['tracker', 'tracking_accuracy_threshold']
        result = extract_stage_params(df, columns, collapse_tracker_when_no_threshold=True)
        # All three rows collapse to a single (None, None) tuple.
        assert result == {(None, None)}

    def test_no_effect_without_tracker_column(self):
        # When tracker is not in columns, the flag has no effect.
        df = _make_pareto_df()
        columns = ['classifier', 'tilesize', 'sample_rate']
        result_default = extract_stage_params(df, columns)
        result_collapse = extract_stage_params(df, columns, collapse_tracker_when_no_threshold=True)
        assert result_default == result_collapse

    def test_no_effect_without_threshold_column(self):
        # When tracking_accuracy_threshold is not in columns, the flag has no effect.
        df = _make_pareto_df()
        columns = ['classifier', 'tilesize', 'tracker']
        result_default = extract_stage_params(df, columns)
        result_collapse = extract_stage_params(df, columns, collapse_tracker_when_no_threshold=True)
        assert result_default == result_collapse
