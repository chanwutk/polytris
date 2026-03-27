"""Tests for Pareto front computation and angle-based pruning."""

import numpy as np
import pandas as pd
import pytest

from polyis.pareto import (
    _prune_pareto_points,
    compute_pareto_front,
    compute_pareto_fronts_by_group,
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
        # Build a DataFrame where many points lie on the Pareto front.
        # With default minimize_x=False, both x and y increasing → all on front.
        df = pd.DataFrame({
            'time': list(range(1, 11)),
            'acc':  [0.1 * i for i in range(1, 11)],
        })
        result = compute_pareto_front(df, 'time', 'acc', num_points=3)
        # Should have exactly 3 rows.
        assert len(result) == 3
        # Endpoints must be present.
        assert result['time'].min() == 1
        assert result['time'].max() == 10
        # Result must still be sorted by time (descending for minimize_x=False).
        assert result['time'].is_monotonic_decreasing


class TestComputeParetoFrontNumPointsNone:
    """Backward compatibility: num_points=None returns all Pareto points."""

    def test_no_pruning(self):
        # With default minimize_x=False, both x and y increasing → all on front.
        df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'acc':  [0.6, 0.7, 0.8, 0.9, 1.0],
        })
        result = compute_pareto_front(df, 'time', 'acc', num_points=None)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# compute_pareto_fronts_by_group integration tests
# ---------------------------------------------------------------------------


class TestComputeParetoFrontsByGroupPassesNumPoints:
    """Per-group pruning should limit each group independently."""

    def test_per_group_pruning(self):
        # Two groups, each with 6 Pareto-optimal points (increasing time → increasing acc).
        rows = []
        for ds in ['A', 'B']:
            for i in range(6):
                rows.append({'dataset': ds, 'time': float(i + 1), 'acc': 0.5 + i * 0.1})
        df = pd.DataFrame(rows)
        # Note: default minimize_x=False + increasing acc with increasing time → all are Pareto.
        result = compute_pareto_fronts_by_group(df, ['dataset'], 'time', 'acc', num_points=3)
        # Each group should have at most 3 points.
        for ds in ['A', 'B']:
            group = result[result['dataset'] == ds]
            assert len(group) <= 3
            assert len(group) >= 2  # At least endpoints.


class TestComputeParetoFrontsByGroupNumPointsNone:
    """Passing num_points=None should override the default of 16."""

    def test_no_pruning_override(self):
        # Single group with 20 Pareto-optimal points (increasing time → increasing acc).
        df = pd.DataFrame({
            'dataset': ['X'] * 20,
            'time': list(range(1, 21)),
            'acc': [0.2 + i * 0.04 for i in range(20)],
        })
        # With default minimize_x=False and both x,y increasing, all 20 are Pareto-optimal.
        result = compute_pareto_fronts_by_group(df, ['dataset'], 'time', 'acc', num_points=None)
        assert len(result) == 20
