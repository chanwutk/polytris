#!/usr/local/bin/python

import argparse
import os
from typing import Mapping

import pandas as pd

from evaluation.p200_compare_compute import load_sota_tradeoff_data
from polyis.pareto import compute_pareto_front
from polyis.utilities import get_config, load_tradeoff_data, split_tradeoff_variants


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
ACCURACY_COL = 'HOTA_HOTA'
THROUGHPUT_COL = 'throughput_fps'
DEFAULT_THRESHOLDS = [threshold / 100.0 for threshold in range(1, 11)]
DETAIL_THRESHOLDS = [0.05, 0.10]
PRIOR_SYSTEMS = ['otif', 'leap']
OUTPUT_DIR = os.path.join('paper', 'figures', 'generated')
OUTPUT_TEX_PATH = os.path.join(OUTPUT_DIR, 'p203_compare_stats.tex')


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame with the same columns as the input."""
    # Return an empty slice so the caller keeps the exact same schema.
    return df.iloc[0:0].copy()


def _threshold_label(threshold: float) -> str:
    """Format a threshold ratio like 0.05 as a human-readable percent label."""
    # Convert the fractional threshold into an integer percent string.
    return f'{int(round(threshold * 100))}%'


def _format_optional_float(value: object, digits: int = 3) -> str:
    """Format an optional numeric value for CLI tables."""
    # Render missing values as a stable placeholder.
    if pd.isna(value):
        return 'NA'

    # Format present values with the requested precision.
    return f'{float(value):.{digits}f}'


def _format_optional_percent(value: object, digits: int = 2) -> str:
    """Format an optional fractional value as a percentage string."""
    # Render missing values as a stable placeholder.
    if pd.isna(value):
        return 'NA'

    # Scale the fractional value to human-readable percent units.
    return f'{float(value) * 100:.{digits}f}%'


def filter_pareto_by_dataset(
    df: pd.DataFrame,
    throughput_col: str = THROUGHPUT_COL,
    accuracy_col: str = ACCURACY_COL,
) -> pd.DataFrame:
    """Keep only Pareto-optimal rows for each dataset on throughput vs accuracy."""
    # Short-circuit empty inputs so downstream code can keep the same schema.
    if df.empty:
        return _empty_like(df)

    # Collect one Pareto-filtered DataFrame per dataset.
    pareto_frames: list[pd.DataFrame] = []

    # Process each dataset independently so fronts never mix across datasets.
    for dataset in sorted(df['dataset'].dropna().unique()):
        # Keep only the current dataset rows with both required metrics present.
        dataset_df = df[df['dataset'] == dataset].dropna(
            subset=[throughput_col, accuracy_col],
        ).copy()

        # Skip datasets with no usable points.
        if dataset_df.empty:
            continue

        # Compute the Pareto front while maximizing both throughput and accuracy.
        pareto_df = compute_pareto_front(
            dataset_df,
            throughput_col,
            accuracy_col,
            minx=False,
            miny=False,
        )

        # Append the dataset-local front when it contains at least one row.
        if not pareto_df.empty:
            pareto_frames.append(pareto_df)

    # Return an empty frame with the original schema when no Pareto rows exist.
    if not pareto_frames:
        return _empty_like(df)

    # Combine the dataset-local fronts into one shared DataFrame.
    return pd.concat(pareto_frames, ignore_index=True)


def add_loss_pct(
    df: pd.DataFrame,
    oracle_hota: float,
    accuracy_col: str = ACCURACY_COL,
) -> pd.DataFrame:
    """Attach a clamped relative HOTA-loss column to a system DataFrame."""
    # Fail fast when the oracle score is invalid for percentage scaling.
    assert oracle_hota > 0, f"oracle_hota must be positive, got {oracle_hota}"

    # Work on a copy so callers can safely reuse the original DataFrame.
    result = df.copy()

    # Compute the relative loss and clamp negative values to zero.
    loss_pct = (oracle_hota - result[accuracy_col]).clip(lower=0) / oracle_hota
    # Round the ratio so threshold-boundary rows stay stable under floating-point noise.
    result['loss_pct'] = loss_pct.round(12)

    return result


def select_best_feasible_row(
    df: pd.DataFrame,
    threshold: float,
    throughput_col: str = THROUGHPUT_COL,
) -> pd.Series | None:
    """Return the fastest row whose loss stays within the requested threshold."""
    # Keep only rows that satisfy the threshold bound.
    feasible_df = df[df['loss_pct'] <= threshold].copy()

    # Return no selection when the threshold admits no rows.
    if feasible_df.empty:
        return None

    # Resolve the row with the highest throughput among feasible points.
    best_idx = feasible_df[throughput_col].idxmax()

    # Return a detached copy so later edits do not alias the original frame.
    return feasible_df.loc[best_idx].copy()


def select_best_prior_row(
    prior_dfs_by_system: Mapping[str, pd.DataFrame],
    threshold: float,
) -> pd.Series | None:
    """Return the fastest feasible prior-system row across all available systems."""
    # Track the current best prior-system row across all systems.
    best_row: pd.Series | None = None

    # Evaluate each prior system independently before comparing them.
    for system_name, prior_df in prior_dfs_by_system.items():
        # Select the fastest feasible point for the current prior system.
        system_row = select_best_feasible_row(prior_df, threshold)

        # Skip systems that have no feasible point under the requested threshold.
        if system_row is None:
            continue

        # Annotate the selected row with the display name of the prior system.
        system_row = system_row.copy()
        system_row['system'] = system_name

        # Keep the first feasible system or any strictly faster feasible system.
        if best_row is None or float(system_row[THROUGHPUT_COL]) > float(best_row[THROUGHPUT_COL]):
            best_row = system_row

    return best_row


def load_polytris_and_naive_test_data(datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical test-split Polytris and naive tradeoff rows for all datasets."""
    # Collect one Polytris tradeoff DataFrame per dataset.
    polytris_frames: list[pd.DataFrame] = []
    # Collect one naive tradeoff DataFrame per dataset.
    naive_frames: list[pd.DataFrame] = []

    # Load each dataset through the shared tradeoff loader.
    for dataset in datasets:
        # Load the canonical split-level tradeoff table.
        tradeoff_df = load_tradeoff_data(dataset)

        # Backfill the dataset column when older cache rows omitted it.
        if 'dataset' not in tradeoff_df.columns:
            tradeoff_df = tradeoff_df.copy()
            tradeoff_df['dataset'] = dataset

        # Restrict the summary to the test split only.
        tradeoff_df = tradeoff_df[tradeoff_df['videoset'] == 'test'].copy()

        # Split the canonical table into Polytris rows and the naive baseline row.
        polytris_df, naive_df = split_tradeoff_variants(tradeoff_df)

        # Exclude the Perfect classifier so only deployable Polytris rows remain.
        if 'classifier' in polytris_df.columns:
            polytris_df = polytris_df[polytris_df['classifier'] != 'Perfect'].copy()

        # Append the dataset-local rows to the shared collections.
        polytris_frames.append(polytris_df)
        naive_frames.append(naive_df)

    # Combine all Polytris rows into one shared DataFrame.
    polytris_all_df = pd.concat(polytris_frames, ignore_index=True) if polytris_frames else pd.DataFrame()
    # Combine all naive rows into one shared DataFrame.
    naive_all_df = pd.concat(naive_frames, ignore_index=True) if naive_frames else pd.DataFrame()

    return polytris_all_df, naive_all_df


def build_frame_count_lookup(polytris_df: pd.DataFrame, naive_df: pd.DataFrame) -> dict[str, float]:
    """Resolve one frame-count value per dataset for throughput normalization."""
    # Combine Polytris and naive rows so the lookup works even if one side is sparse.
    combined_df = pd.concat([polytris_df, naive_df], ignore_index=True)

    # Keep the first non-null frame count for each dataset.
    frame_count_lookup = (
        combined_df
        .dropna(subset=['frame_count'])
        .groupby('dataset')['frame_count']
        .first()
        .to_dict()
    )

    return frame_count_lookup


def load_prior_test_data(
    datasets: list[str],
    frame_count_lookup: Mapping[str, float],
) -> dict[str, pd.DataFrame]:
    """Load prior-system tradeoff rows and derive throughput from dataset frame counts."""
    # Collect one tradeoff DataFrame per configured prior system.
    prior_dfs: dict[str, pd.DataFrame] = {}

    # Load each prior system through the shared SOTA tradeoff loader.
    for system_name in PRIOR_SYSTEMS:
        # Load the system-local tradeoff data across all datasets.
        prior_df = load_sota_tradeoff_data(datasets, system_name)

        # Skip systems that have no cached tradeoff rows.
        if prior_df.empty:
            continue

        # Restrict the summary to the test split only.
        prior_df = prior_df[prior_df['videoset'] == 'test'].copy()

        # Map each prior row to the dataset frame count used by Polytris.
        prior_df['frame_count'] = prior_df['dataset'].map(frame_count_lookup)

        # Derive throughput from the shared frame count and the measured runtime.
        prior_df[THROUGHPUT_COL] = prior_df['frame_count'] / prior_df['time']

        # Store the normalized prior-system rows under the display name.
        prior_dfs[system_name.upper()] = prior_df

    return prior_dfs


def build_threshold_detail_table(
    datasets: list[str],
    polytris_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    prior_dfs_by_system: Mapping[str, pd.DataFrame],
    threshold: float,
) -> pd.DataFrame:
    """Build one per-dataset detail table for a single HOTA-loss threshold."""
    # Collect one detail row per dataset.
    rows: list[dict[str, object]] = []

    # Process datasets in the configured order so CLI tables stay stable.
    for dataset in datasets:
        # Resolve the dedicated naive baseline row used as the oracle reference.
        oracle_df = naive_df[naive_df['dataset'] == dataset].copy()
        assert not oracle_df.empty, f"Missing naive oracle row for dataset {dataset}"
        oracle_row = oracle_df.iloc[0]
        oracle_hota = float(oracle_row[ACCURACY_COL])

        # Keep only Pareto-filtered Polytris rows for the current dataset.
        dataset_polytris_df = polytris_df[polytris_df['dataset'] == dataset].copy()
        dataset_polytris_df = add_loss_pct(dataset_polytris_df, oracle_hota)

        # Select the fastest feasible Polytris point under the current threshold.
        polytris_row = select_best_feasible_row(dataset_polytris_df, threshold)

        # Build one loss-annotated candidate table per prior system for this dataset.
        dataset_prior_dfs: dict[str, pd.DataFrame] = {}
        for system_name, prior_df in prior_dfs_by_system.items():
            # Keep only rows for the current dataset.
            dataset_prior_df = prior_df[prior_df['dataset'] == dataset].copy()

            # Skip systems that do not cover the current dataset.
            if dataset_prior_df.empty:
                continue

            # Attach loss percentages relative to the dataset-local oracle.
            dataset_prior_dfs[system_name] = add_loss_pct(dataset_prior_df, oracle_hota)

        # Select the fastest feasible prior-system point under the current threshold.
        prior_row = select_best_prior_row(dataset_prior_dfs, threshold)

        # Compute the Polytris-over-prior speedup when both sides are available.
        if (
            polytris_row is not None
            and prior_row is not None
            and float(prior_row[THROUGHPUT_COL]) > 0
        ):
            speedup_x = float(polytris_row[THROUGHPUT_COL]) / float(prior_row[THROUGHPUT_COL])
        else:
            speedup_x = pd.NA

        # Compute the Polytris-over-naive speedup when a feasible Polytris point exists.
        if polytris_row is not None and float(oracle_row[THROUGHPUT_COL]) > 0:
            naive_speedup_x = float(polytris_row[THROUGHPUT_COL]) / float(oracle_row[THROUGHPUT_COL])
        else:
            naive_speedup_x = pd.NA

        # Materialize the dataset-local detail row with explicit missing markers.
        rows.append({
            'threshold': threshold,
            'dataset': dataset,
            'oracle_hota': oracle_hota,
            'naive_throughput_fps': oracle_row[THROUGHPUT_COL],
            'polytris_variant_id': polytris_row['variant_id'] if polytris_row is not None else pd.NA,
            'polytris_hota': polytris_row[ACCURACY_COL] if polytris_row is not None else pd.NA,
            'polytris_loss_pct': polytris_row['loss_pct'] if polytris_row is not None else pd.NA,
            'polytris_throughput_fps': polytris_row[THROUGHPUT_COL] if polytris_row is not None else pd.NA,
            'prior_system': prior_row['system'] if prior_row is not None else pd.NA,
            'prior_hota': prior_row[ACCURACY_COL] if prior_row is not None else pd.NA,
            'prior_loss_pct': prior_row['loss_pct'] if prior_row is not None else pd.NA,
            'prior_throughput_fps': prior_row[THROUGHPUT_COL] if prior_row is not None else pd.NA,
            'speedup_x': speedup_x,
            'naive_speedup_x': naive_speedup_x,
        })

    # Return the detail table in one shared DataFrame.
    return pd.DataFrame.from_records(rows)


def build_threshold_summary_table(
    detail_df: pd.DataFrame,
    thresholds: list[float],
    dataset_count: int,
) -> pd.DataFrame:
    """Aggregate per-dataset detail rows into one threshold-level summary table."""
    # Collect one summary row per threshold.
    rows: list[dict[str, object]] = []

    # Aggregate the detail table independently for each threshold.
    for threshold in thresholds:
        # Keep only the rows for the current threshold.
        threshold_df = detail_df[detail_df['threshold'] == threshold].copy()

        # Count datasets where Polytris has at least one feasible point.
        polytris_meet_count = int(threshold_df['polytris_variant_id'].notna().sum())
        # Count datasets where at least one prior system has a feasible point.
        prior_meet_count = int(threshold_df['prior_system'].notna().sum())
        # Count datasets where no prior system meets the requested threshold.
        prior_fail_count = dataset_count - prior_meet_count

        # Keep only datasets where both Polytris and a prior system are comparable.
        speedup_values = threshold_df['speedup_x'].dropna()
        # Keep only datasets where Polytris can be compared to the naive pipeline.
        naive_speedup_values = threshold_df['naive_speedup_x'].dropna()

        # Record the threshold-level summary metrics.
        rows.append({
            'threshold': threshold,
            'polytris_meet_count': polytris_meet_count,
            'prior_meet_count': prior_meet_count,
            'prior_fail_count': prior_fail_count,
            'speedup_min_x': speedup_values.min() if not speedup_values.empty else pd.NA,
            'speedup_max_x': speedup_values.max() if not speedup_values.empty else pd.NA,
            'naive_speedup_min_x': (
                naive_speedup_values.min() if not naive_speedup_values.empty else pd.NA
            ),
            'naive_speedup_max_x': (
                naive_speedup_values.max() if not naive_speedup_values.empty else pd.NA
            ),
        })

    # Return the threshold summary table in one shared DataFrame.
    return pd.DataFrame.from_records(rows)


def build_threshold_reports(
    datasets: list[str],
    polytris_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    prior_dfs_by_system: Mapping[str, pd.DataFrame],
    thresholds: list[float] = DEFAULT_THRESHOLDS,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Build the full threshold summary table plus per-threshold detail tables."""
    # Collect the detail table for each requested threshold.
    detail_tables: dict[float, pd.DataFrame] = {}

    # Build each threshold-specific detail table independently.
    for threshold in thresholds:
        detail_tables[threshold] = build_threshold_detail_table(
            datasets,
            polytris_df,
            naive_df,
            prior_dfs_by_system,
            threshold,
        )

    # Combine all threshold detail tables into one summary input frame.
    combined_detail_df = pd.concat(detail_tables.values(), ignore_index=True)

    # Aggregate the combined detail rows into the threshold-level summary.
    summary_df = build_threshold_summary_table(
        combined_detail_df,
        thresholds,
        len(datasets),
    )

    return summary_df, detail_tables


def _format_prior_label(row: pd.Series) -> str:
    """Format a prior-system row as a short human-readable label."""
    # Resolve each component defensively so missing fields do not crash the renderer.
    system = str(row.get('system', 'PRIOR'))
    param_id = row.get('param_id', pd.NA)
    sample_rate = row.get('sample_rate', pd.NA)

    # Render param_id with a '?' fallback when absent (older caches).
    if pd.isna(param_id):
        param_segment = '#?'
    else:
        param_segment = f'#{int(param_id)}'

    # Render sample_rate with an empty suffix when absent.
    if pd.isna(sample_rate):
        sr_segment = ''
    else:
        sr_segment = f' (sr={int(sample_rate)})'

    return f'{system}{param_segment}{sr_segment}'


def build_combined_prior_pareto(
    prior_dfs_by_system: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine all prior-system rows into one per-dataset Pareto front with labels."""
    # Skip the combine step when no prior systems are available.
    if not prior_dfs_by_system:
        return pd.DataFrame()

    # Concatenate all prior rows; each row already carries its 'system' name.
    combined_df = pd.concat(list(prior_dfs_by_system.values()), ignore_index=True)

    # Pareto-filter the combined frame per dataset on throughput vs HOTA.
    pareto_df = filter_pareto_by_dataset(combined_df)

    # Short-circuit when the combined front has no rows so downstream code gets a stable schema.
    if pareto_df.empty:
        return pareto_df

    # Build a readable per-row label combining system, param_id, and sample rate.
    pareto_df = pareto_df.copy()
    pareto_df['prior_label'] = pareto_df.apply(_format_prior_label, axis=1)

    return pareto_df


def select_best_polytris_for_prior_point(
    polytris_pareto_df: pd.DataFrame,
    prior_row: pd.Series,
    throughput_col: str = THROUGHPUT_COL,
    accuracy_col: str = ACCURACY_COL,
) -> pd.Series | None:
    """Return the highest-HOTA Polytris Pareto point that is at least as fast as a prior row."""
    # Restrict the candidate pool to the prior row's dataset.
    dataset = prior_row['dataset']
    dataset_df = polytris_pareto_df[polytris_pareto_df['dataset'] == dataset]

    # Keep only Polytris points that match or beat the prior row's throughput.
    feasible_df = dataset_df[dataset_df[throughput_col] >= float(prior_row[throughput_col])]

    # Return no selection when no Polytris point is at least as fast.
    if feasible_df.empty:
        return None

    # Resolve the row with the highest HOTA among the at-least-as-fast points.
    best_idx = feasible_df[accuracy_col].idxmax()

    # Return a detached copy so later edits do not alias the shared frame.
    return feasible_df.loc[best_idx].copy()


def build_dominance_detail_table(
    datasets: list[str],
    polytris_pareto_df: pd.DataFrame,
    combined_prior_pareto_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one per-(dataset, prior Pareto point) dominance table against Polytris."""
    # Collect one row per (dataset, prior point).
    rows: list[dict[str, object]] = []

    # Process datasets in the configured order so CLI tables stay stable.
    for dataset in datasets:
        # Keep only the combined-front rows for the current dataset.
        dataset_prior_df = combined_prior_pareto_df[
            combined_prior_pareto_df['dataset'] == dataset
        ].copy()

        # Skip datasets that have no prior-system Pareto points.
        if dataset_prior_df.empty:
            continue

        # Sort by prior throughput to traverse the combined front slowest → fastest.
        dataset_prior_df = dataset_prior_df.sort_values(THROUGHPUT_COL).reset_index(drop=True)

        # Emit one detail row per prior Pareto point.
        for _, prior_row in dataset_prior_df.iterrows():
            # Pick the fastest-feasible-and-most-accurate Polytris point for this prior row.
            polytris_row = select_best_polytris_for_prior_point(polytris_pareto_df, prior_row)

            # Extract the Polytris-side fields, using NA when no point is at least as fast.
            if polytris_row is not None:
                polytris_variant_id = polytris_row['variant_id']
                polytris_hota: object = float(polytris_row[ACCURACY_COL])
                polytris_throughput: object = float(polytris_row[THROUGHPUT_COL])
                hota_delta: object = float(polytris_row[ACCURACY_COL]) - float(prior_row[ACCURACY_COL])
            else:
                polytris_variant_id = pd.NA
                polytris_hota = pd.NA
                polytris_throughput = pd.NA
                hota_delta = pd.NA

            # Materialize the per-prior-point detail row.
            rows.append({
                'dataset': dataset,
                'prior_label': prior_row['prior_label'],
                'prior_system': prior_row['system'],
                'prior_hota': float(prior_row[ACCURACY_COL]),
                'prior_throughput_fps': float(prior_row[THROUGHPUT_COL]),
                'polytris_variant_id': polytris_variant_id,
                'polytris_hota': polytris_hota,
                'polytris_throughput_fps': polytris_throughput,
                'hota_delta': hota_delta,
            })

    return pd.DataFrame.from_records(rows)


def _compute_dominance_summary_row(
    detail_df: pd.DataFrame,
    dataset: str | None = None,
) -> dict[str, object]:
    """Compute a dominance summary row over a slice of the detail table."""
    # Count all prior Pareto points in the slice (reachable or not).
    num_prior_points = int(len(detail_df))

    # Drop NA deltas so the summary stats reflect only reachable prior points.
    deltas = detail_df['hota_delta'].dropna() if 'hota_delta' in detail_df.columns else pd.Series(dtype=float)
    num_reachable = int(len(deltas))

    # Assemble the output dict, tagging with the dataset key when present.
    row: dict[str, object] = {}
    if dataset is not None:
        row['dataset'] = dataset
    row['num_prior_points'] = num_prior_points
    row['num_reachable'] = num_reachable
    row['hota_delta_min'] = float(deltas.min()) if not deltas.empty else pd.NA
    row['hota_delta_median'] = float(deltas.median()) if not deltas.empty else pd.NA
    row['hota_delta_max'] = float(deltas.max()) if not deltas.empty else pd.NA
    return row


def build_dominance_dataset_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dominance detail table into one summary row per dataset."""
    # Return an empty frame with the expected columns when there is nothing to aggregate.
    if detail_df.empty:
        return pd.DataFrame(columns=[
            'dataset', 'num_prior_points', 'num_reachable',
            'hota_delta_min', 'hota_delta_median', 'hota_delta_max',
        ])

    # Preserve the dataset order already present in the detail table.
    rows = [
        _compute_dominance_summary_row(detail_df[detail_df['dataset'] == dataset], dataset=dataset)
        for dataset in detail_df['dataset'].drop_duplicates().tolist()
    ]

    return pd.DataFrame.from_records(rows)


def build_dominance_global_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dominance detail table into one global summary row."""
    # Return an empty frame with the expected columns when there is nothing to aggregate.
    if detail_df.empty:
        return pd.DataFrame(columns=[
            'num_prior_points', 'num_reachable',
            'hota_delta_min', 'hota_delta_median', 'hota_delta_max',
        ])

    return pd.DataFrame.from_records([_compute_dominance_summary_row(detail_df)])


def format_summary_for_cli(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format the threshold summary table for human-readable CLI output."""
    # Work on a copy so callers can still access the numeric summary frame.
    formatted_df = summary_df.copy()

    # Render the threshold as a percent label.
    formatted_df['threshold'] = formatted_df['threshold'].map(_threshold_label)
    # Render the speedup columns with a consistent fixed precision.
    formatted_df['speedup_min_x'] = formatted_df['speedup_min_x'].map(_format_optional_float)
    formatted_df['speedup_max_x'] = formatted_df['speedup_max_x'].map(_format_optional_float)
    formatted_df['naive_speedup_min_x'] = formatted_df['naive_speedup_min_x'].map(_format_optional_float)
    formatted_df['naive_speedup_max_x'] = formatted_df['naive_speedup_max_x'].map(_format_optional_float)

    return formatted_df


def format_detail_for_cli(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Format one per-threshold detail table for human-readable CLI output."""
    # Work on a copy so callers can still access the numeric detail frame.
    formatted_df = detail_df.copy()

    # Render the threshold as a percent label.
    formatted_df['threshold'] = formatted_df['threshold'].map(_threshold_label)
    # Render HOTA columns with a consistent fixed precision.
    formatted_df['oracle_hota'] = formatted_df['oracle_hota'].map(_format_optional_float)
    formatted_df['polytris_hota'] = formatted_df['polytris_hota'].map(_format_optional_float)
    formatted_df['prior_hota'] = formatted_df['prior_hota'].map(_format_optional_float)
    # Render loss columns as human-readable percentages.
    formatted_df['polytris_loss_pct'] = formatted_df['polytris_loss_pct'].map(_format_optional_percent)
    formatted_df['prior_loss_pct'] = formatted_df['prior_loss_pct'].map(_format_optional_percent)
    # Render throughput and speedup columns with a consistent fixed precision.
    formatted_df['naive_throughput_fps'] = formatted_df['naive_throughput_fps'].map(_format_optional_float)
    formatted_df['polytris_throughput_fps'] = formatted_df['polytris_throughput_fps'].map(_format_optional_float)
    formatted_df['prior_throughput_fps'] = formatted_df['prior_throughput_fps'].map(_format_optional_float)
    formatted_df['speedup_x'] = formatted_df['speedup_x'].map(_format_optional_float)
    formatted_df['naive_speedup_x'] = formatted_df['naive_speedup_x'].map(_format_optional_float)

    return formatted_df


def format_dominance_detail_for_cli(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Format the dominance detail table for human-readable CLI output."""
    # Work on a copy so callers can still access the numeric detail frame.
    formatted_df = detail_df.copy()

    # Render HOTA, throughput, and delta columns with a consistent fixed precision.
    formatted_df['prior_hota'] = formatted_df['prior_hota'].map(_format_optional_float)
    formatted_df['prior_throughput_fps'] = formatted_df['prior_throughput_fps'].map(_format_optional_float)
    formatted_df['polytris_hota'] = formatted_df['polytris_hota'].map(_format_optional_float)
    formatted_df['polytris_throughput_fps'] = formatted_df['polytris_throughput_fps'].map(_format_optional_float)
    formatted_df['hota_delta'] = formatted_df['hota_delta'].map(_format_optional_float)

    return formatted_df


def format_dominance_summary_for_cli(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format a dominance summary table for human-readable CLI output."""
    # Work on a copy so callers can still access the numeric summary frame.
    formatted_df = summary_df.copy()

    # Render the aggregate delta columns with a consistent fixed precision.
    for column in ('hota_delta_min', 'hota_delta_median', 'hota_delta_max'):
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column].map(_format_optional_float)

    return formatted_df


def save_tex_macros(
    summary_df: pd.DataFrame,
    dominance_detail_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save the abstract-ready 5% and 10% summary values plus the max-HOTA-delta dominance macros as TeX macros."""
    # Map each abstract-ready threshold to the suffix used in macro names.
    threshold_suffixes = [(0.05, 'FivePct'), (0.10, 'TenPct')]

    # Map each integer-valued summary column to its macro-name prefix.
    int_fields = [
        ('polytris_meet_count', 'comparePolytrisMeet'),
        ('prior_meet_count', 'comparePriorMeet'),
        ('prior_fail_count', 'comparePriorFailDatasets'),
    ]

    # Map each float-valued summary column to its macro-name prefix.
    float_fields = [
        ('speedup_min_x', 'compareSpeedupMin'),
        ('speedup_max_x', 'compareSpeedupMax'),
        ('naive_speedup_min_x', 'compareNaiveSpeedupMin'),
        ('naive_speedup_max_x', 'compareNaiveSpeedupMax'),
    ]

    # Ensure the output directory exists before writing the macro file.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the auto-generated macro file in the same style as other paper helpers.
    with open(output_path, 'w') as f:
        # Identify the generating script for future debugging.
        f.write('% Auto-generated by evaluation/p203_compare_stats.py\n')

        # Emit one macro block per abstract-ready threshold.
        for threshold, suffix in threshold_suffixes:
            # Resolve the summary row for the current threshold.
            row = summary_df.loc[summary_df['threshold'] == threshold].iloc[0]

            # Write each integer-valued macro for the current threshold.
            for column, prefix in int_fields:
                f.write(
                    f'\\newcommand{{\\{prefix}{suffix}}}{{'
                    f'\\autogen{{{int(row[column])}}}}}\n'
                )

            # Write each float-valued macro for the current threshold.
            for column, prefix in float_fields:
                f.write(
                    f'\\newcommand{{\\{prefix}{suffix}}}{{'
                    f'\\autogen{{{float(row[column]):.1f}}}}}\n'
                )

        # Resolve the row with the highest HOTA delta in the dominance detail table.
        reachable_df = dominance_detail_df.dropna(subset=['hota_delta'])
        assert not reachable_df.empty, (
            'Dominance detail has no reachable rows; cannot emit max-HOTA-delta macros.'
        )
        best_row = reachable_df.loc[reachable_df['hota_delta'].idxmax()]

        # Emit the max-HOTA-delta macro rounded to 2 decimal digits for paper prose.
        f.write(
            f'\\newcommand{{\\compareMaxHotaImprovement}}{{'
            f'\\autogen{{{float(best_row["hota_delta"]):.2f}}}}}\n'
        )

        # Floor the matched prior-system throughput to the nearest 100 fps for the paper prose.
        prior_fps_floor = int(float(best_row['prior_throughput_fps']) // 100) * 100
        f.write(
            f'\\newcommand{{\\compareMaxHotaImprovementFps}}{{'
            f'\\autogen{{{prior_fps_floor}}}}}\n'
        )


def print_reports(
    summary_df: pd.DataFrame,
    detail_tables: Mapping[float, pd.DataFrame],
    dominance_global_summary_df: pd.DataFrame,
    dominance_dataset_summary_df: pd.DataFrame,
    dominance_detail_df: pd.DataFrame,
) -> None:
    """Print the threshold summary, the selected detail tables, and the dominance tables."""
    # Print the threshold-level summary first so the headline numbers are easy to scan.
    print('\n=== Threshold Summary ===')
    print(format_summary_for_cli(summary_df).to_string(index=False))

    # Print the detail tables in the requested order for abstract verification.
    for threshold in DETAIL_THRESHOLDS:
        print(f'\n=== {_threshold_label(threshold)} Detail ===')
        print(format_detail_for_cli(detail_tables[threshold]).to_string(index=False))

    # Print the dominance global summary so the headline dominance numbers come first.
    print('\n=== Dominance Global Summary ===')
    print(format_dominance_summary_for_cli(dominance_global_summary_df).to_string(index=False))

    # Print the dominance per-dataset summary for asymmetry across datasets.
    print('\n=== Dominance Per-Dataset Summary ===')
    print(format_dominance_summary_for_cli(dominance_dataset_summary_df).to_string(index=False))

    # Print the full per-point dominance table last for detailed inspection.
    print('\n=== Dominance Detail (per prior Pareto point) ===')
    print(format_dominance_detail_for_cli(dominance_detail_df).to_string(index=False))


def main() -> None:
    # Parse the CLI arguments before resolving the supported split.
    args = parse_args()

    # Skip unsupported validation requests because this summary is test-only.
    if args.valid:
        print('Skipping: abstract comparison stats only support the test split.')
        return

    # Log the datasets covered by the summary run.
    print(f'Processing datasets: {DATASETS}')

    # Load the canonical test-split Polytris and naive tradeoff data.
    polytris_df, naive_df = load_polytris_and_naive_test_data(DATASETS)

    # Fail fast when the expected Polytris or naive data is missing.
    assert not polytris_df.empty, 'No Polytris tradeoff rows found for the test split.'
    assert not naive_df.empty, 'No naive oracle rows found for the test split.'

    # Build the dataset-local frame-count lookup for prior-system throughput.
    frame_count_lookup = build_frame_count_lookup(polytris_df, naive_df)

    # Load and normalize the prior-system tradeoff data.
    prior_raw_dfs = load_prior_test_data(DATASETS, frame_count_lookup)

    # Pareto-filter the Polytris tradeoff rows by dataset.
    polytris_pareto_df = filter_pareto_by_dataset(polytris_df)

    # Pareto-filter each prior system by dataset.
    prior_pareto_dfs = {
        system_name: filter_pareto_by_dataset(prior_df)
        for system_name, prior_df in prior_raw_dfs.items()
    }

    # Build the threshold summary plus the per-threshold detail tables.
    summary_df, detail_tables = build_threshold_reports(
        DATASETS,
        polytris_pareto_df,
        naive_df,
        prior_pareto_dfs,
        DEFAULT_THRESHOLDS,
    )

    # Build the combined OTIF ∪ LEAP Pareto front per dataset for the dominance analysis.
    combined_prior_pareto_df = build_combined_prior_pareto(prior_raw_dfs)

    # Build the per-(dataset, prior Pareto point) dominance detail table against Polytris.
    dominance_detail_df = build_dominance_detail_table(
        DATASETS,
        polytris_pareto_df,
        combined_prior_pareto_df,
    )

    # Aggregate the dominance detail into per-dataset and global summaries.
    dominance_dataset_summary_df = build_dominance_dataset_summary(dominance_detail_df)
    dominance_global_summary_df = build_dominance_global_summary(dominance_detail_df)

    # Print the computed tables so the results are visible directly in the CLI.
    print_reports(
        summary_df,
        detail_tables,
        dominance_global_summary_df,
        dominance_dataset_summary_df,
        dominance_detail_df,
    )

    # Save the abstract-ready macros for the paper draft.
    save_tex_macros(summary_df, dominance_detail_df, OUTPUT_TEX_PATH)
    print(f'\nSaved {OUTPUT_TEX_PATH}')


if __name__ == '__main__':
    main()
