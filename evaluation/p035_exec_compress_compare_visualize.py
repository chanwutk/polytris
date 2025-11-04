#!/usr/bin/env python3
"""
Script to visualize and compare compression results across different methods.

This script reads compression comparison data from multiple evaluation directories:
- 083_compress
- 084_compress
- 085_compress_single

And creates grouped bar chart visualizations comparing:
1. Number of compressed images (num_images)
2. Tile occupancy ratios (occupancy_ratio)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


# Configuration for the three compression method directories
COMPRESSION_METHODS = {
    '083_compress': 'FFD',
    '084_compress': 'BFD',
    '085_compress_single': 'Online'
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
    Load image counts and tile counts data from all available compression method directories.

    Args:
        dataset: Dataset name
        verbose: Whether to print verbose output

    Returns:
        Dictionary with keys 'image_counts' and 'tile_counts', each containing a DataFrame
        with data from all compression methods (includes 'method' column)
    """
    # Initialize lists to collect DataFrames
    image_dfs = []
    tile_dfs = []

    # Load data from each compression method directory
    for method_dir, method_label in COMPRESSION_METHODS.items():
        eval_dir = Path(CACHE_DIR) / dataset / 'evaluation' / method_dir

        if not eval_dir.exists():
            if verbose:
                print(f"  Directory not found: {eval_dir}")
            continue

        # Load image counts CSV
        image_csv_path = eval_dir / 'image_counts_comparison.csv'
        if image_csv_path.exists():
            image_df = pd.read_csv(image_csv_path)
            image_df['method'] = method_label
            image_dfs.append(image_df)
            if verbose:
                print(f"  Loaded image counts from {method_dir}: {len(image_df)} rows")
        else:
            if verbose:
                print(f"  Image counts CSV not found: {image_csv_path}")

        # Load tile counts CSV
        tile_csv_path = eval_dir / 'tile_counts_comparison.csv'
        if tile_csv_path.exists():
            tile_df = pd.read_csv(tile_csv_path)
            tile_df['method'] = method_label
            tile_dfs.append(tile_df)
            if verbose:
                print(f"  Loaded tile counts from {method_dir}: {len(tile_df)} rows")
        else:
            if verbose:
                print(f"  Tile counts CSV not found: {tile_csv_path}")

    # Combine all DataFrames
    result = {}

    if image_dfs:
        result['image_counts'] = pd.concat(image_dfs, ignore_index=True)
    else:
        result['image_counts'] = pd.DataFrame()

    if tile_dfs:
        result['tile_counts'] = pd.concat(tile_dfs, ignore_index=True)
    else:
        result['tile_counts'] = pd.DataFrame()

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
    if df.empty:
        if verbose:
            print(f"No image count data available for {dataset}")
        return

    # Create configuration label
    df = df.copy()
    df['config'] = df.apply(create_config_label, axis=1)

    # Aggregate by config and method: sum num_images across all videos
    grouped_df = df.groupby(['config', 'method'], as_index=False)['num_images'].sum()

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
    if df.empty:
        if verbose:
            print(f"No tile count data available for {dataset}")
        return

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

    if data['image_counts'].empty and data['tile_counts'].empty:
        print(f"No data found for dataset {dataset}")
        return

    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Create num_images comparison chart
    if not data['image_counts'].empty:
        num_images_path = dataset_output_dir / f'{dataset}_num_images_comparison.png'
        create_num_images_chart(data['image_counts'], dataset, str(num_images_path), verbose)
    else:
        if verbose:
            print("  No image counts data available")

    # Create occupancy_ratio comparison chart
    if not data['tile_counts'].empty:
        occupancy_ratio_path = dataset_output_dir / f'{dataset}_occupancy_ratio_comparison.png'
        create_occupancy_ratio_chart(data['tile_counts'], dataset, str(occupancy_ratio_path), verbose)
    else:
        if verbose:
            print("  No tile counts data available")


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
        print(f"Compression methods: {list(COMPRESSION_METHODS.values())}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset in args.datasets:
        try:
            process_dataset(dataset, output_dir, args.verbose)
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\nVisualization complete. Charts saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
