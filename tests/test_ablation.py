"""Tests for ablation condition filtering logic."""

import pandas as pd
import pytest

from evaluation.ablation import (
    ABLATION_CONDITIONS,
    AblationCondition,
    filter_by_ablation_condition,
)


def _make_tradeoff_df() -> pd.DataFrame:
    """Build a synthetic tradeoff DataFrame spanning multiple sample_rate/threshold combos."""
    rows = []
    for sr in [1, 2, 4, 8]:
        for threshold in [None, 0.4, 0.8]:
            rows.append({
                'classifier': 'ShuffleNet05',
                'tilesize': 60,
                'sample_rate': sr,
                'tracking_accuracy_threshold': threshold,
                'tilepadding': 'none',
                'canvas_scale': 1.0,
                'tracker': 'sortcython',
                'time': 10.0 / sr,
                'HOTA_HOTA': 0.8 - (0.01 * sr),
            })
    return pd.DataFrame(rows)


class TestFullConditionKeepsAll:
    """The 'full' ablation condition should pass all rows through unchanged."""

    def test_full_condition_no_filter(self):
        df = _make_tradeoff_df()
        full = ABLATION_CONDITIONS[0]
        assert full.name == 'full'
        result = filter_by_ablation_condition(df, full)
        assert len(result) == len(df)

    def test_full_condition_preserves_columns(self):
        df = _make_tradeoff_df()
        full = ABLATION_CONDITIONS[0]
        result = filter_by_ablation_condition(df, full)
        assert list(result.columns) == list(df.columns)


class TestNoSamplingCondition:
    """The 'no_sampling' condition should restrict to sample_rate=1 only."""

    def test_only_sample_rate_one(self):
        df = _make_tradeoff_df()
        no_sampling = ABLATION_CONDITIONS[1]
        assert no_sampling.name == 'no_sampling'
        result = filter_by_ablation_condition(df, no_sampling)
        # Only sample_rate=1 rows survive; all 3 threshold values remain.
        assert (result['sample_rate'] == 1).all()
        assert len(result) == 3

    def test_all_thresholds_preserved(self):
        df = _make_tradeoff_df()
        no_sampling = ABLATION_CONDITIONS[1]
        result = filter_by_ablation_condition(df, no_sampling)
        # The null threshold is represented as NaN.
        non_null = result['tracking_accuracy_threshold'].dropna().tolist()
        assert sorted(non_null) == [0.4, 0.8]
        assert result['tracking_accuracy_threshold'].isna().sum() == 1


class TestNoBothCondition:
    """The 'no_both' condition should restrict to sample_rate=1 AND threshold=null."""

    def test_single_row_survives(self):
        df = _make_tradeoff_df()
        no_both = ABLATION_CONDITIONS[2]
        assert no_both.name == 'no_both'
        result = filter_by_ablation_condition(df, no_both)
        assert len(result) == 1
        assert result.iloc[0]['sample_rate'] == 1
        assert pd.isna(result.iloc[0]['tracking_accuracy_threshold'])


class TestEmptyDataFrame:
    """Filtering an empty DataFrame should return an empty DataFrame."""

    def test_empty_input(self):
        df = pd.DataFrame(columns=['sample_rate', 'tracking_accuracy_threshold'])
        for condition in ABLATION_CONDITIONS:
            result = filter_by_ablation_condition(df, condition)
            assert result.empty


class TestMissingColumns:
    """Conditions should be skipped when the target column is absent."""

    def test_no_sample_rate_column(self):
        df = pd.DataFrame({'tracking_accuracy_threshold': [None, 0.4, 0.8]})
        no_sampling = ABLATION_CONDITIONS[1]
        # sample_rate column missing: the sample_rate filter is not applied.
        result = filter_by_ablation_condition(df, no_sampling)
        assert len(result) == 3

    def test_no_threshold_column(self):
        df = pd.DataFrame({'sample_rate': [1, 2, 4]})
        # Use a custom condition that restricts thresholds, so we can verify
        # the filter is skipped when the threshold column is absent.
        threshold_only = AblationCondition('threshold_only', 'Threshold Only', None, [None])
        result = filter_by_ablation_condition(df, threshold_only)
        assert len(result) == 3


class TestCustomCondition:
    """Verify that custom AblationCondition instances work correctly."""

    def test_custom_sample_rates(self):
        df = _make_tradeoff_df()
        custom = AblationCondition('custom', 'Custom', [2, 4], None)
        result = filter_by_ablation_condition(df, custom)
        assert sorted(result['sample_rate'].unique()) == [2, 4]

    def test_custom_thresholds_with_null(self):
        df = _make_tradeoff_df()
        custom = AblationCondition('custom', 'Custom', None, [None, 0.4])
        result = filter_by_ablation_condition(df, custom)
        non_null = result['tracking_accuracy_threshold'].dropna().unique().tolist()
        assert non_null == [0.4]
        assert result['tracking_accuracy_threshold'].isna().sum() > 0


class TestAblationConditionsOrdering:
    """Verify the canonical ordering and labels of ABLATION_CONDITIONS."""

    def test_three_conditions_defined(self):
        assert len(ABLATION_CONDITIONS) == 3

    def test_condition_names(self):
        names = [c.name for c in ABLATION_CONDITIONS]
        assert names == ['full', 'no_sampling', 'no_both']

    def test_condition_labels(self):
        labels = [c.label for c in ABLATION_CONDITIONS]
        assert labels == ['Polytris', 'Polytris (-Sampling)', 'Polytris (-Sampling, -Pruning)']
