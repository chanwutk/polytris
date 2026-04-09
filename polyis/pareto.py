"""Pareto front utilities shared across evaluation and pipeline scripts."""

import os

import numpy as np
import pandas as pd

from polyis.io import cache


# Parameter columns used to transfer valid-Pareto parameter sets to test points.
PARETO_PARAM_COLS = [
    'variant_id',
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
]


def val_gte(val: float, best_val: float) -> bool:
    return val >= best_val


def val_lte(val: float, best_val: float) -> bool:
    return val <= best_val


def _prune_pareto_points(
    x: np.ndarray,
    y: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Prune Pareto front points to *num_points* by iteratively removing
    the interior point with the widest vertex angle (most collinear).

    Coordinates are min-max normalized to [0, 1] before angle computation
    so that both axes contribute equally regardless of scale.

    Args:
        x: 1-D array of x coordinates (must be sorted ascending).
        y: 1-D array of y coordinates.
        num_points: Target number of points to keep (>= 2).

    Returns:
        Boolean mask of length ``len(x)``; True for points to keep.
    """
    n = len(x)
    # Nothing to prune when we already have few enough points or only endpoints.
    if n <= num_points or n <= 2:
        return np.ones(n, dtype=bool)

    # Min-max normalize both axes to [0, 1] so both contribute equally.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    xn = (x - x.min()) / (x_range if x_range > 0 else 1.0)
    yn = (y - y.min()) / (y_range if y_range > 0 else 1.0)

    # Maintain a list of active point indices throughout the iterative pruning.
    active = list(range(n))

    while len(active) > num_points:
        # Compute the cosine of the vertex angle at each interior point.
        worst_cos = 2.0  # Cosine is in [-1, 1]; 2.0 is above any valid value.
        worst_j = -1
        for j in range(1, len(active) - 1):
            # Vectors from the interior point to its two neighbors.
            i_prev, i_curr, i_next = active[j - 1], active[j], active[j + 1]
            v1x = xn[i_prev] - xn[i_curr]
            v1y = yn[i_prev] - yn[i_curr]
            v2x = xn[i_next] - xn[i_curr]
            v2y = yn[i_next] - yn[i_curr]

            # Magnitudes of both vectors.
            mag1 = np.sqrt(v1x * v1x + v1y * v1y)
            mag2 = np.sqrt(v2x * v2x + v2y * v2y)

            # Cosine of the vertex angle via dot product.
            if mag1 > 0 and mag2 > 0:
                cos_angle = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
            else:
                # Degenerate zero-length vector: treat as maximally collinear.
                cos_angle = -1.0

            # Track the point with the most negative cosine (widest angle).
            if cos_angle < worst_cos:
                worst_cos = cos_angle
                worst_j = j

        # Remove the most collinear interior point from the active list.
        active.pop(worst_j)

    # Build a boolean mask from the surviving active indices.
    mask = np.zeros(n, dtype=bool)
    mask[active] = True
    return mask


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str,
                         minimize_x: bool = False, maximize_y: bool = True,
                         num_points: int | None = None) -> pd.DataFrame:
    """
    Compute Pareto-optimal points from DataFrame.

    For minimize_x=True and maximize_y=True (the default):
    A point is Pareto-optimal if no other point has both lower x AND higher y.

    Args:
        df: DataFrame with data points
        x_col: Column name for x-axis (e.g., 'time')
        y_col: Column name for y-axis (e.g., 'HOTA_HOTA')
        minimize_x: If True, lower x is better; if False, higher x is better
        maximize_y: If True, higher y is better; if False, lower y is better
        num_points: If not None, prune the Pareto front to at most this many
            points by iteratively removing the most collinear interior point.

    Returns:
        DataFrame containing only Pareto-optimal points, sorted by x_col
    """
    # Drop rows with NaN in x or y columns
    df_clean = df.dropna(subset=[x_col, y_col]).copy()

    if df_clean.empty:
        return df_clean

    # Sort by x_col (ascending if minimizing x, descending if maximizing)
    # Sort so that the "best" x values come last (reversed iteration sweeps best→worst).
    df_sorted = df_clean.sort_values(x_col, ascending=not minimize_x).reset_index(drop=True)

    # Build Pareto front using cumulative max/min approach
    # Track the best y value seen so far from the "expensive" end (high x if minimizing x)
    pareto_indices = []

    better_y = val_gte if maximize_y else val_lte
    best_y = float('-inf') if maximize_y else float('inf')
    for idx in reversed(df_sorted.index):
        y_val = df_sorted.loc[idx, y_col]
        if better_y(y_val, best_y):
            pareto_indices.append(idx)
            best_y = y_val

    # Reverse to maintain sorted order by x
    pareto_indices = list(reversed(pareto_indices))

    # Collect Pareto-optimal points into a DataFrame.
    pareto_df = df_sorted.loc[pareto_indices].reset_index(drop=True)

    # Optionally prune to a target number of points by removing the most collinear interior points.
    if num_points is not None and len(pareto_df) > num_points:
        mask = _prune_pareto_points(
            pareto_df[x_col].values,
            pareto_df[y_col].values,
            num_points,
        )
        return pareto_df[mask].reset_index(drop=True)

    return pareto_df


def interpolate_pareto_line(pareto_df: pd.DataFrame, x_col: str, y_col: str,
                            query_points: np.ndarray,
                            extrapolate: bool = False) -> pd.DataFrame:
    """
    Linearly interpolate Pareto front at specified query points.

    Args:
        pareto_df: DataFrame with Pareto front points (sorted by x_col)
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values to interpolate
        query_points: Array of x values at which to interpolate
        extrapolate: If False, return NaN for points outside the data range

    Returns:
        DataFrame with 'query_point' and 'interpolated_value' columns
    """
    if pareto_df.empty:
        return pd.DataFrame({
            'query_point': query_points,
            'interpolated_value': [np.nan] * len(query_points)
        })

    # Get x and y values from Pareto front
    x_vals = pareto_df[x_col].values
    y_vals = pareto_df[y_col].values

    # Interpolate using numpy
    if extrapolate:
        # Allow extrapolation outside data range
        interpolated = np.interp(query_points, x_vals, y_vals)
    else:
        # Return NaN for points outside the data range
        interpolated = np.interp(query_points, x_vals, y_vals)
        # Mask values outside the range
        mask_below = query_points < x_vals.min()
        mask_above = query_points > x_vals.max()
        interpolated[mask_below | mask_above] = np.nan

    return pd.DataFrame({
        'query_point': query_points,
        'interpolated_value': interpolated
    })


def compute_pareto_fronts_by_group(df: pd.DataFrame, group_cols: list[str],
                                   x_col: str, y_col: str,
                                   num_points: int = 10) -> pd.DataFrame:
    """
    Compute Pareto fronts for each group in the DataFrame.

    Args:
        df: DataFrame with data points
        group_cols: Columns to group by (e.g., ['dataset', 'system'])
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        num_points: Maximum number of Pareto points per group. Pass None to
            disable pruning and keep all Pareto-optimal points.

    Returns:
        DataFrame with Pareto-optimal points for each group
    """
    # Apply Pareto front computation (with optional pruning) to each group.
    pareto_groups = (
        df.groupby(group_cols, group_keys=True, dropna=False)
        .apply(lambda g: compute_pareto_front(g, x_col, y_col, num_points=num_points),
               include_groups=False)
    )

    # Restore grouping keys from the MultiIndex as regular columns.
    pareto_groups = pareto_groups.reset_index(level=group_cols)

    return pareto_groups.reset_index(drop=True)


def select_test_points_from_valid_pareto(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    accuracy_col: str,
    time_col: str = 'time',
) -> pd.DataFrame:
    """
    Select test points whose parameter sets appear on the valid Pareto front.

    Args:
        valid_df: Videoset-level Polytris rows for valid split
        test_df: Videoset-level Polytris rows for test split
        accuracy_col: Accuracy metric column used for Pareto computation
        time_col: Runtime metric column used for Pareto computation

    Returns:
        Test rows matched to valid Pareto parameter sets
    """
    # Short-circuit when no test rows are available.
    if test_df.empty:
        print("  Warning: No test rows found; nothing to select")
        return pd.DataFrame(columns=test_df.columns)

    # Fall back to all test rows when no valid rows are available.
    if valid_df.empty:
        print("  Warning: No valid rows found; using all test rows")
        return test_df.copy()

    # Keep only parameter columns available in both valid and test DataFrames.
    match_cols = [col for col in PARETO_PARAM_COLS if col in valid_df.columns and col in test_df.columns]
    if not match_cols:
        print("  Warning: No shared parameter columns between valid and test; using all test rows")
        return test_df.copy()

    # Keep valid rows with metric/time values required by Pareto computation.
    valid_metric_df = valid_df.dropna(subset=[time_col, accuracy_col]).copy()
    if valid_metric_df.empty:
        print("  Warning: Valid rows have no metric/time values; using all test rows")
        return test_df.copy()

    # Compute valid Pareto points per dataset using DataFrame groupby+apply.
    # No pruning here: keep all Pareto points for complete parameter matching.
    valid_pareto_df = compute_pareto_fronts_by_group(
        valid_metric_df,
        ['dataset'],
        time_col,
        accuracy_col,
        num_points=None,
    )
    if valid_pareto_df.empty:
        print("  Warning: Valid Pareto is empty; using all test rows")
        return test_df.copy()

    # Build unique valid Pareto parameter sets per dataset.
    merge_cols = ['dataset', *match_cols]
    valid_param_sets = valid_pareto_df[merge_cols].drop_duplicates().copy()

    # Build merge keys that treat NaN values as equal for parameter matching.
    def build_merge_keys(df_in: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
        df_out = df_in.copy()
        key_cols: list[str] = []
        for col in cols:
            key_col = f'__{prefix}_{col}'
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[key_col] = df_out[col].fillna(-1_000_000_000.0)
            else:
                df_out[key_col] = df_out[col].astype('object').where(df_out[col].notna(), '__NA__')
            key_cols.append(key_col)
        return df_out, key_cols

    # Construct merge-key DataFrames for valid-parameter sets and test rows.
    valid_param_sets_keyed, merge_key_cols = build_merge_keys(valid_param_sets, merge_cols, 'valid')
    test_df_keyed, _ = build_merge_keys(test_df, merge_cols, 'valid')
    # Select test rows by matching merge keys against valid Pareto parameter sets.
    matched_test_df = test_df_keyed.merge(
        valid_param_sets_keyed[merge_key_cols].drop_duplicates(),
        on=merge_key_cols,
        how='inner'
    )
    # Drop temporary merge-key columns after match selection.
    matched_test_df = matched_test_df.drop(columns=merge_key_cols)

    # Fall back per-dataset to all test rows where no parameter match exists.
    matched_datasets = matched_test_df['dataset'].dropna().unique().tolist() \
        if 'dataset' in matched_test_df.columns else []
    fallback_test_df = test_df[~test_df['dataset'].isin(matched_datasets)].copy()
    selected_df = pd.concat([matched_test_df, fallback_test_df], ignore_index=True)

    # Print a compact per-dataset summary using DataFrame joins.
    summary_df = (
        test_df.groupby('dataset').size().rename('all_test_rows').to_frame()
        .join(valid_pareto_df.groupby('dataset').size().rename('valid_pareto_points'), how='left')
        .join(valid_param_sets.groupby('dataset').size().rename('valid_param_sets'), how='left')
        .join(matched_test_df.groupby('dataset').size().rename('matched_test_rows'), how='left')
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    print(summary_df.to_string(index=False))

    # Warn once when fallback rows were required for unmatched datasets.
    if not fallback_test_df.empty:
        fallback_datasets = sorted(fallback_test_df['dataset'].dropna().astype(str).unique().tolist())
        print(f"  Warning: No valid-parameter match for datasets {fallback_datasets}; used all test rows")

    return selected_df


def save_pareto_params(dataset: str, pareto_df: pd.DataFrame) -> None:
    """Save Pareto-optimal parameter sets for a dataset to the evaluation cache."""
    # Construct the output path using the evaluation cache helper.
    output_path = cache.eval(dataset, 'pareto-params') / 'pareto_params.csv'
    # Ensure the parent directory exists before writing.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the Pareto parameter sets as a CSV file.
    pareto_df.to_csv(output_path, index=False)
    print(f"Saved Pareto params for {dataset}: {len(pareto_df)} rows to {output_path}")


def load_pareto_params(dataset: str) -> pd.DataFrame:
    """Load Pareto-optimal parameter sets for a dataset from the evaluation cache."""
    # Construct the expected path using the evaluation cache helper.
    path = cache.eval(dataset, 'pareto-params') / 'pareto_params.csv'
    # Load and return the Pareto parameter sets CSV.
    return pd.read_csv(path)


def pareto_params_exist(dataset: str) -> bool:
    """Check whether Pareto parameter sets have been saved for a dataset."""
    # Check whether the Pareto params CSV exists for the given dataset.
    path = cache.eval(dataset, 'pareto-params') / 'pareto_params.csv'
    return path.exists()


def build_pareto_combo_filter(
    datasets: list[str],
    selected_videosets: list[str],
    columns: list[str],
    collapse_tracker_when_no_threshold: bool = False,
) -> dict[str, set[tuple]] | None:
    """
    Build per-dataset sets of allowed parameter combos for the test pass, or return None.

    Returns None when the test videoset is not selected or when Pareto params have
    not been extracted yet (i.e. the test pass has not been run).  When a non-None
    dict is returned, callers should skip any parameter combo not in the set for
    the current dataset.

    Args:
        datasets: Dataset names to load Pareto params for
        selected_videosets: Videosets being processed in this run
        columns: Pareto DataFrame columns to project to (defines the combo shape)
        collapse_tracker_when_no_threshold: Passed through to extract_stage_params.
            When True, tracker is set to None for rows where threshold is NaN/None.

    Returns:
        Dict mapping dataset name to its set of allowed parameter tuples,
        or None if filtering does not apply
    """
    # Only filter when explicitly running the test pass.
    if 'test' not in selected_videosets:
        return None
    for ds in datasets:
        if not pareto_params_exist(ds):
            raise Exception(f'Pareto params have not been extracted for {ds}')
    # Build per-dataset allowed-combo sets so each dataset is filtered independently.
    result: dict[str, set[tuple]] = {}
    for ds in datasets:
        result[ds] = extract_stage_params(
            load_pareto_params(ds), columns,
            collapse_tracker_when_no_threshold=collapse_tracker_when_no_threshold,
        )
    total = sum(len(v) for v in result.values())
    print(f"Pareto filter enabled: {total} total allowed combos for test pass")
    return result


def extract_stage_params(pareto_df: pd.DataFrame, columns: list[str],
                         collapse_tracker_when_no_threshold: bool = False) -> set[tuple]:
    """
    Project Pareto DataFrame to stage-relevant columns and return as a set of tuples.

    Normalizes NaN values to None so tuple equality works correctly for
    membership checks across numeric and nullable columns.

    Args:
        pareto_df: Pareto parameter DataFrame (output of load_pareto_params)
        columns: Column names to project to (must exist in pareto_df)
        collapse_tracker_when_no_threshold: When True and both 'tracker' and
            'tracking_accuracy_threshold' are in columns, set tracker to None
            for rows where threshold is NaN/None.  This matches the execution
            convention in p030-p050 where tracker is irrelevant when no
            pruning threshold is active.

    Returns:
        Set of tuples representing allowed parameter combinations
    """
    # Project the Pareto DataFrame to the requested stage-relevant columns.
    subset = pareto_df[columns].copy()
    # When upstream stages don't use the tracker dimension for configurations
    # without pruning, normalize tracker to None so the combo filter matches
    # the execution loop's convention (tracker=None when threshold=None).
    if (collapse_tracker_when_no_threshold
            and 'tracker' in columns
            and 'tracking_accuracy_threshold' in columns):
        no_threshold = subset['tracking_accuracy_threshold'].isna()
        subset.loc[no_threshold, 'tracker'] = None
    # Collect normalized tuples, converting NaN to None for hashable comparison.
    result: set[tuple] = set()
    for row in subset.itertuples(index=False):
        # Normalize float NaN to None so comparisons with None loop variables work.
        normalized = tuple(
            None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            for v in row
        )
        result.add(normalized)
    return result
