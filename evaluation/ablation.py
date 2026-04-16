"""Ablation condition definitions for incremental optimization evaluation.

Defines parameter-space restrictions that produce separate Pareto curves per
dataset, each representing the system with one or more optimizations disabled.
Both p135 (Pareto extraction) and p201 (comparison visualization) import
these definitions so the ablation logic stays in one place.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AblationCondition:
    """A named parameter-space restriction for ablation evaluation."""

    # Internal identifier (e.g. 'no_sampling').
    name: str
    # Display label for chart legends (e.g. 'Polytris (-Sampling)').
    label: str
    # Allowed sample_rate values, or None to keep all configured rates.
    sample_rates: list[int] | None
    # Allowed tracking_accuracy_threshold values, or None to keep all.
    # Use [None] to restrict to the no-pruning configuration only.
    tracking_accuracy_thresholds: list[float | None] | None


# Ordered list: full system -> incrementally disable optimizations.
ABLATION_CONDITIONS: list[AblationCondition] = [
    AblationCondition('full',        'Polytris',              None, None),
    AblationCondition('no_sampling', 'Polytris (-Sampling)',  [1],  None),
    AblationCondition('no_pruning',  'Polytris (-Pruning)',   None, [None]),
    AblationCondition('no_both',     'Polytris (-Both)',      [1],  [None]),
]


def filter_by_ablation_condition(
    df: pd.DataFrame,
    condition: AblationCondition,
) -> pd.DataFrame:
    """
    Filter DataFrame rows to match an ablation condition's parameter constraints.

    Only filters on sample_rate and tracking_accuracy_threshold; other
    parameter dimensions (classifier, tilepadding, etc.) pass through unchanged.

    Args:
        df: DataFrame with optional sample_rate and tracking_accuracy_threshold columns.
        condition: Ablation condition whose constraints to apply.

    Returns:
        Filtered copy of the DataFrame.
    """
    filtered = df.copy()

    # Filter by sample_rate when the condition restricts it.
    if condition.sample_rates is not None and 'sample_rate' in filtered.columns:
        filtered = filtered[filtered['sample_rate'].isin(condition.sample_rates)]

    # Filter by tracking_accuracy_threshold when the condition restricts it.
    if condition.tracking_accuracy_thresholds is not None and 'tracking_accuracy_threshold' in filtered.columns:
        allowed = [t for t in condition.tracking_accuracy_thresholds if t is not None]
        include_null = any(t is None for t in condition.tracking_accuracy_thresholds)
        mask = filtered['tracking_accuracy_threshold'].isin(allowed)
        if include_null:
            mask = mask | filtered['tracking_accuracy_threshold'].isna()
        filtered = filtered[mask]

    return filtered
