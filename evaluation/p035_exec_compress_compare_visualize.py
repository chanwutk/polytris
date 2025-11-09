#!/usr/bin/env python3
"""
Script to visualize and compare compression results across different methods.

This script reads compression comparison data from the evaluation directory:
- 083_compress

The 'stage' column in each CSV differentiates the compression methods.

Creates grouped bar chart visualizations comparing:
1. Number of compressed images (num_images)
2. Tile occupancy ratios (occupancy_ratio)
3. Runtime per operation
4. Total runtime across all operations
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


# Configuration for compression evaluation directory
COMPRESSION_EVAL_DIR = '083_compress'

STAGE_METHOD_MAP = {
    '030_compressed_frames': 'Pack Append',
    '031_compressed_frames': 'FFD Python',
    '032_compressed_frames': 'BFD Python',
    '033_compressed_frames': 'FFD C'
}

# Detection time per image (milliseconds) for different datasets
DETECTION_TIME_MS = {
    'caldot1': 40,
    'caldot2': 40,
    'jnc0': 100,
    'jnc2': 100,
    'jnc6': 100,
    'jnc7': 100,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize and compare compression results across different methods'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DATASETS_TO_TEST,
        help='Datasets to analyze (default: all datasets in DATASETS_TO_TEST)'
    )
    parser.add_argument(
        '--output-dir',
        default='output/compression_comparisons',
        help='Output directory for visualizations (default: output/compression_comparisons)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    return parser.parse_args()


def load_compression_data(dataset: str, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load image counts, tile counts, and runtime data from the compression evaluation directory.
    The 'stage' column in each CSV differentiates the compression methods.

    Args:
        dataset: Dataset name
        verbose: Whether to print verbose output

    Returns:
        Dictionary with keys 'image_counts', 'tile_counts', and 'runtime', each containing a DataFrame
        with data from all compression methods (uses 'stage' column to differentiate methods)
    """
    # Get evaluation directory path
    eval_dir = Path(CACHE_DIR) / dataset / 'evaluation' / COMPRESSION_EVAL_DIR

    result = {}

    # Load image counts CSV
    image_csv_path = eval_dir / 'image_counts_comparison.csv'
    image_df = pd.read_csv(image_csv_path)
    # Rename 'stage' column to 'method' and map stage names to method names
    image_df = image_df.rename(columns={'stage': 'method'})
    image_df['method'] = image_df['method'].map(STAGE_METHOD_MAP)
    result['image_counts'] = image_df
    if verbose:
        print(f"  Loaded image counts: {len(image_df)} rows")
        if 'method' in image_df.columns:
            print(f"    Methods: {image_df['method'].unique().tolist()}")

    # Load tile counts CSV
    tile_csv_path = eval_dir / 'tile_counts_comparison.csv'
    tile_df = pd.read_csv(tile_csv_path)
    # Rename 'stage' column to 'method' and map stage names to method names
    tile_df = tile_df.rename(columns={'stage': 'method'})
    tile_df['method'] = tile_df['method'].map(STAGE_METHOD_MAP)
    result['tile_counts'] = tile_df
    if verbose:
        print(f"  Loaded tile counts: {len(tile_df)} rows")
        if 'method' in tile_df.columns:
            print(f"    Methods: {tile_df['method'].unique().tolist()}")

    # Load runtime CSV
    runtime_csv_path = eval_dir / 'runtime_comparison.csv'
    runtime_df = pd.read_csv(runtime_csv_path)
    # Rename 'stage' column to 'method' and map stage names to method names
    runtime_df = runtime_df.rename(columns={'stage': 'method'})
    runtime_df['method'] = runtime_df['method'].map(STAGE_METHOD_MAP)
    result['runtime'] = runtime_df
    if verbose:
        print(f"  Loaded runtime data: {len(runtime_df)} rows")
        if 'method' in runtime_df.columns:
            print(f"    Methods: {runtime_df['method'].unique().tolist()}")

    return result


def create_config_label(row: pd.Series) -> str:
    """
    Create a configuration label from classifier, tilesize, and tilepadding.

    Args:
        row: DataFrame row with classifier, tilesize, tilepadding columns

    Returns:
        Configuration label string
    """
    return f"{row['classifier']}_{row['tilesize']}_{row['tilepadding']}"


def create_num_images_chart(df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a grouped bar chart comparing num_images across compression methods.

    Args:
        df: DataFrame with image counts data (must have 'method' column)
        dataset: Dataset name for the title
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    # Create configuration label
    df = df.copy()
    df['config'] = df.apply(create_config_label, axis=1)

    # Aggregate by config and method: sum num_images across all videos
    grouped_df = df.groupby(['config', 'method'], as_index=False)['num_images'].sum()
    assert isinstance(grouped_df, pd.DataFrame)

    if verbose:
        print(f"Creating num_images chart with {len(grouped_df)} data points")
        print(f"Configurations: {grouped_df['config'].nunique()}")
        print(f"Methods: {grouped_df['method'].unique().tolist()}")

    # Create grouped bar chart
    chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X('config:N',
                title='Configuration (Classifier_TileSize_Padding)',
                axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y('num_images:Q',
                title='Total Number of Compressed Images'),
        color=alt.Color('method:N',
                       title='Compression Method',
                       scale=alt.Scale(scheme='category10')),
        xOffset='method:N'
    ).properties(
        width=600,
        height=400,
        title=f'Comparison of Compressed Image Counts - {dataset}'
    ).configure_axis(
        gridOpacity=0.3
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )

    # Save the chart
    chart.save(output_path)
    print(f"Saved num_images chart: {output_path}")


def create_occupancy_ratio_chart(df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a grouped bar chart comparing occupancy_ratio across compression methods.

    Args:
        df: DataFrame with tile counts data (must have 'method' column)
        dataset: Dataset name for the title
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    # Create configuration label
    df = df.copy()
    df['config'] = df.apply(create_config_label, axis=1)

    # Aggregate by config and method: sum tiles across all videos, then recalculate occupancy ratio
    grouped_df = df.groupby(['config', 'method'], as_index=False).agg({
        'empty_tiles': 'sum',
        'occupied_tiles': 'sum',
        'total_tiles': 'sum'
    })

    # Recalculate occupancy ratio from aggregated tiles
    grouped_df['occupancy_ratio'] = grouped_df['occupied_tiles'] / grouped_df['total_tiles'].where(
        grouped_df['total_tiles'] > 0, 1
    )

    if verbose:
        print(f"Creating occupancy_ratio chart with {len(grouped_df)} data points")
        print(f"Configurations: {grouped_df['config'].nunique()}")
        print(f"Methods: {grouped_df['method'].unique().tolist()}")

    # Create grouped bar chart
    chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X('config:N',
                title='Configuration (Classifier_TileSize_Padding)',
                axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y('occupancy_ratio:Q',
                title='Tile Occupancy Ratio',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('method:N',
                       title='Compression Method',
                       scale=alt.Scale(scheme='category10')),
        xOffset='method:N'
    ).properties(
        width=600,
        height=400,
        title=f'Comparison of Tile Occupancy Ratios - {dataset}'
    ).configure_axis(
        gridOpacity=0.3
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )

    # Save the chart
    chart.save(output_path)
    print(f"Saved occupancy_ratio chart: {output_path}")


def create_runtime_chart(df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a grouped bar chart comparing runtime per operation across compression methods.

    Args:
        df: DataFrame with runtime data (must have 'method' column and 'op', 'runtime' columns)
        dataset: Dataset name for the title
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    if df.empty:
        if verbose:
            print(f"No runtime data available for {dataset}")
        return

    # Create configuration label
    df = df.copy()
    df['config'] = df.apply(create_config_label, axis=1)

    # Aggregate by config, method, and operation: sum runtime across all videos
    grouped_df = df.groupby(['config', 'method', 'op'], as_index=False)['runtime'].sum()
    assert isinstance(grouped_df, pd.DataFrame)

    # Exclude non-core operations from visualization
    exclude_ops = {'save_canvas', 'save_mapping_files', 'save_collage', 'read_frame'}
    if verbose:
        # Print all available operations before filtering
        print(f"Available operations before filtering ({len(grouped_df['op'].unique())}): {sorted(grouped_df['op'].unique().tolist())}")
    grouped_df = grouped_df[~grouped_df['op'].isin(exclude_ops)]

    if verbose:
        print(f"Creating runtime chart with {len(grouped_df)} data points")
        print(f"Configurations: {grouped_df['config'].nunique()}")
        print(f"Methods: {grouped_df['method'].unique().tolist()}")
        print(f"Operations (filtered): {grouped_df['op'].unique().tolist()}")

    # Create grouped bar chart with faceting by operation
    chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X('config:N',
                title='Configuration (Classifier_TileSize_Padding)',
                axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y('runtime:Q',
                title='Total Runtime (seconds)'),
        color=alt.Color('method:N',
                       title='Compression Method',
                       scale=alt.Scale(scheme='category10')),
        xOffset='method:N',
        column=alt.Column('op:N',
                         title='Operation')
    ).properties(
        width=400,
        height=400,
        title=f'Comparison of Runtime by Operation - {dataset}'
    ).configure_axis(
        gridOpacity=0.3
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )

    # Save the chart
    chart.save(output_path)
    print(f"Saved runtime chart: {output_path}")


def create_total_runtime_chart(df: pd.DataFrame, image_counts_df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a grouped bar chart comparing total runtime across compression methods.
    Includes both compression runtime and estimated detection runtime.

    Args:
        df: DataFrame with runtime data (must have 'method' column and 'op', 'runtime' columns)
        image_counts_df: DataFrame with image counts data (must have 'method', 'config', 'num_images' columns)
        dataset: Dataset name for the title
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    # Create configuration label for runtime data
    df = df.copy()
    df['config'] = df.apply(create_config_label, axis=1)

    # Aggregate by config and method: sum runtime across all videos and operations
    compression_df = df.groupby(['config', 'method'], as_index=False)['runtime'].sum()
    assert isinstance(compression_df, pd.DataFrame)
    compression_df = compression_df.rename(columns={'runtime': 'compression_runtime'})

    # Create configuration label for image counts data
    image_counts_df = image_counts_df.copy()
    image_counts_df['config'] = image_counts_df.apply(create_config_label, axis=1)

    # Aggregate image counts by config and method
    images_df = image_counts_df.groupby(['config', 'method'], as_index=False)['num_images'].sum()

    # Merge compression runtime with image counts
    grouped_df = compression_df.merge(images_df, on=['config', 'method'], how='left')

    # Calculate detection runtime based on dataset
    detection_time_ms = DETECTION_TIME_MS.get(dataset, 40)  # Default to 40ms if dataset not found
    grouped_df['detection_runtime'] = grouped_df['num_images'] * detection_time_ms / 1000.0  # Convert ms to seconds

    # Create separate dataframes for compression and detection
    compression_bars = grouped_df[['config', 'method', 'compression_runtime']].copy()
    compression_bars['runtime_type'] = 'Compression'
    compression_bars = compression_bars.rename(columns={'compression_runtime': 'runtime'})

    detection_bars = grouped_df[['config', 'method', 'detection_runtime']].copy()
    detection_bars['runtime_type'] = 'Detection'
    detection_bars = detection_bars.rename(columns={'detection_runtime': 'runtime'})

    # Combine both dataframes
    chart_df = pd.concat([compression_bars, detection_bars], ignore_index=True)

    if verbose:
        print(f"Creating total runtime chart with {len(chart_df)} data points")
        print(f"Configurations: {chart_df['config'].nunique()}")
        print(f"Methods: {chart_df['method'].unique().tolist()}")
        print(f"Detection time per image: {detection_time_ms}ms")

    # Create grouped bar chart with stacked bars
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('config:N',
                title='Configuration (Classifier_TileSize_Padding)',
                axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y('runtime:Q',
                title='Total Runtime (seconds)'),
        color=alt.Color('runtime_type:N',
                       title='Runtime Type',
                       scale=alt.Scale(scheme='category10')),
        xOffset=alt.XOffset('method:N',
                           title='Compression Method'),
        order=alt.Order('runtime_type:N', sort='descending')
    ).properties(
        width=600,
        height=400,
        title=f'Comparison of Total Runtime (Compression + Detection) - {dataset}'
    ).configure_axis(
        gridOpacity=0.3
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )

    # Save the chart
    chart.save(output_path)
    print(f"Saved total runtime chart: {output_path}")


def process_dataset(dataset: str, output_dir: Path, verbose: bool = False):
    """
    Process a single dataset and create all visualizations.

    Args:
        dataset: Dataset name
        output_dir: Output directory for visualizations
        verbose: Whether to print verbose output
    """
    if verbose:
        print(f"\nProcessing dataset: {dataset}")

    # Load compression data from all methods
    data = load_compression_data(dataset, verbose)

    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Create num_images comparison chart
    num_images_path = dataset_output_dir / f'{dataset}_num_images_comparison.png'
    create_num_images_chart(data['image_counts'], dataset, str(num_images_path), verbose)

    # Create occupancy_ratio comparison chart
    occupancy_ratio_path = dataset_output_dir / f'{dataset}_occupancy_ratio_comparison.png'
    create_occupancy_ratio_chart(data['tile_counts'], dataset, str(occupancy_ratio_path), verbose)

    # Create runtime comparison chart (per operation)
    runtime_path = dataset_output_dir / f'{dataset}_runtime_comparison.png'
    create_runtime_chart(data['runtime'], dataset, str(runtime_path), verbose)

    # Create total runtime comparison chart (compression + detection)
    total_runtime_path = dataset_output_dir / f'{dataset}_total_runtime_comparison.png'
    create_total_runtime_chart(data['runtime'], data['image_counts'], dataset, str(total_runtime_path), verbose)


def main(args):
    """
    Main function to create compression comparison visualizations.

    Args:
        args: Parsed command line arguments
    """
    if args.verbose:
        print("Starting compression comparison visualization...")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Output directory: {args.output_dir}")
        print(f"Datasets to analyze: {args.datasets}")
        print(f"Evaluation directory: {COMPRESSION_EVAL_DIR}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset in args.datasets:
        process_dataset(dataset, output_dir, args.verbose)

    print(f"\nVisualization complete. Charts saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
