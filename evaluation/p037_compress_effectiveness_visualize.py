#!/usr/bin/env python3
"""
Script to visualize the effectiveness of the compression algorithm.

This script reads computed compression effectiveness data from CSV files
(created by p036_compress_effectiveness_compute.py) and creates visualizations comparing:
1. Number of empty, occupied (non-padding), and padding tiles in compressed images for each configuration
2. Normalized percentage of empty, occupied (non-padding), and padding tiles for each configuration
3. Number of compressed images per configuration

Padding tiles are tiles that are present in the index_map (occupied) but were not
in the original relevancy bitmap (below threshold), added by the tilepadding operation.

One visualization per dataset, comparing across classifier, tilesize, and tilepadding.
Configurations are loaded from configs/global.yaml using get_config.
"""

import os
from pathlib import Path

import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_DIR, get_config, get_video_frame_count


def get_total_original_frames(dataset: str) -> int:
    """
    Calculate the total number of original frames across all videos in a dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Total number of frames across all videos in the test set
    """
    videoset_dir = os.path.join(DATASETS_DIR, dataset, 'test')
    if not os.path.exists(videoset_dir):
        print(f"  Videoset directory {videoset_dir} does not exist")
        return 0
    
    # Get all video files
    video_files = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if len(video_files) == 0:
        print(f"  No video files found in {videoset_dir}")
        return 0
    
    # Sum frame counts for all videos
    total_frames = 0
    for video_file in video_files:
        frame_count = get_video_frame_count(dataset, video_file)
        total_frames += frame_count
    
    print(f"  Total original frames for {dataset}: {total_frames} (from {len(video_files)} videos)")
    
    return total_frames


def _create_tile_comparison_chart_base(df: pd.DataFrame, dataset: str, normalized: bool = False, include_config: bool = True) -> alt.Chart | None:
    """
    Base function to create a stacked bar chart comparing empty, occupied (non-padding), and padding tiles.
    
    Args:
        df: Aggregated DataFrame with tile counts
        dataset: Dataset name
        normalized: If True, normalize bars to percentages (0-100%)
        include_config: If True, apply configure_axis (for standalone charts). If False, skip (for concatenated charts).
        
    Returns:
        Altair Chart object or None if no data
    """
    if len(df) == 0:
        return None
    
    # Calculate total tiles and occupancy ratio
    df = df.copy()
    df['total_tiles'] = df['empty_tiles'] + df['occupied_tiles']
    df['occupancy_ratio'] = df['occupied_tiles'] / df['total_tiles'].replace(0, 1)
    df['non_padding_occupied'] = df['occupied_tiles'] - df['padding_tiles']
    df['config_label'] = df.apply(
        lambda row: f"{row['classifier'][:4]}_{row['tilesize']}_{row['sample_rate']}_{row['tilepadding'][:4]}_s{int(row['canvas_scale'] * 100)}", axis=1
    )
    
    # Sort by total tiles for better visualization
    df = df.sort_values('total_tiles', ascending=False)
    
    # Prepare data for stacked bar chart with three categories
    chart_data = []
    for _, row in df.iterrows():
        # Add empty tiles
        chart_data.append({
            'classifier': row['classifier'],
            'tilesize': row['tilesize'],
            'sample_rate': row['sample_rate'],
            'tilepadding': row['tilepadding'],
            'canvas_scale': row['canvas_scale'],
            'tile_type': 'Empty',
            'count': row['empty_tiles'],
            'config_label': row['config_label'],
            'occupancy_ratio': row['occupancy_ratio']
        })
        # Add non-padding occupied tiles
        chart_data.append({
            'classifier': row['classifier'],
            'tilesize': row['tilesize'],
            'sample_rate': row['sample_rate'],
            'tilepadding': row['tilepadding'],
            'canvas_scale': row['canvas_scale'],
            'tile_type': 'Occupied',
            'count': row['non_padding_occupied'],
            'config_label': row['config_label'],
            'occupancy_ratio': row['occupancy_ratio']
        })
        # Add padding tiles
        chart_data.append({
            'classifier': row['classifier'],
            'tilesize': row['tilesize'],
            'sample_rate': row['sample_rate'],
            'tilepadding': row['tilepadding'],
            'canvas_scale': row['canvas_scale'],
            'tile_type': 'Padding',
            'count': row['padding_tiles'],
            'config_label': row['config_label'],
            'occupancy_ratio': row['occupancy_ratio']
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Configure stack mode and axis based on normalization
    if normalized:
        stack_mode = 'normalize'
        x_title = 'Percent (%)'
        x_axis = alt.Axis(format='.0%')
        title_suffix = ' (Normalized)'
    else:
        stack_mode = 'zero'
        x_title = 'Number of Tiles'
        x_axis = alt.Axis()
        title_suffix = ''
    
    # Create stacked bar chart with three categories
    title_suffix_full = f'{title_suffix} by Configuration' if include_config else title_suffix
    chart = alt.Chart(chart_df).mark_bar(opacity=0.8).encode(
        x=alt.X('count:Q', title=x_title, stack=stack_mode, axis=x_axis),
        y=alt.Y('config_label:N', title='Configuration', 
                sort=alt.SortField('count', order='descending')),
        color=alt.Color('tile_type:N', title='Tile Type',
                       scale=alt.Scale(domain=['Occupied', 'Padding', 'Empty'],
                                       range=['#4caf50', '#ff9800', '#e0e0e0'])),
        tooltip=['config_label', 'tile_type', 'count', 'occupancy_ratio:Q', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Empty vs Occupied vs Padding Tiles{title_suffix_full} - {dataset}'
    )
    
    # Apply configure_axis only for standalone charts
    if include_config:
        chart = chart.configure_axis(
            labelAngle=0,
            labelLimit=200
        )
    
    return chart


def create_tile_comparison_chart(df: pd.DataFrame, dataset: str, output_path: str, normalized: bool = False):
    """
    Create a stacked bar chart comparing empty, occupied (non-padding), and padding tiles.
    
    Args:
        df: Aggregated DataFrame with tile counts
        dataset: Dataset name
        output_path: Path to save the chart
        normalized: If True, normalize bars to percentages (0-100%)
    """
    if len(df) == 0:
        print(f"  No data to plot for {dataset}")
        return
    
    chart = _create_tile_comparison_chart_base(df, dataset, normalized, include_config=True)
    if chart is None:
        return
    
    # Save chart
    chart.save(output_path)
    
    chart_type = "normalized tile comparison" if normalized else "tile comparison"
    print(f"  Saved {chart_type} chart: {output_path}")


def _create_image_count_chart_base(df: pd.DataFrame, dataset: str, original_count: int | None = None, include_config: bool = True) -> alt.Chart | alt.LayerChart | None:
    """
    Base function to create a bar chart comparing number of compressed images.
    
    Args:
        df: Aggregated DataFrame with image counts
        dataset: Dataset name
        original_count: Optional total number of original frames (for baseline line)
        include_config: If True, apply configure_axis (for standalone charts). If False, skip (for concatenated charts).
        
    Returns:
        Altair Chart object or None if no data
    """
    if len(df) == 0:
        return None
    
    # Prepare data
    chart_df = df.copy()
    chart_df['config_label'] = chart_df.apply(
        lambda row: f"{row['classifier'][:4]}_{row['tilesize']}_{row['sample_rate']}_{row['tilepadding'][:4]}_s{int(row['canvas_scale'] * 100)}", axis=1
    )
    
    # Sort by number of images
    chart_df = chart_df.sort_values('num_images', ascending=False)
    
    # Create bar chart
    title_suffix = ' by Configuration' if include_config else ''
    chart = alt.Chart(chart_df).mark_bar(opacity=0.8).encode(
        x=alt.X('num_images:Q', title='Number of Compressed Images'),
        y=alt.Y('config_label:N', title='Configuration',
                sort=alt.SortField('num_images', order='descending')),
        color=alt.Color('classifier:N', title='Classifier'),
        tooltip=['config_label', 'num_images', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Number of Compressed Images{title_suffix} - {dataset}'
    )
    
    # Add vertical red line for original frame count if provided
    if original_count is not None and original_count > 0:
        # Create a rule chart for the baseline
        baseline_data = pd.DataFrame({'original_count': [original_count]})
        baseline = alt.Chart(baseline_data).mark_rule(
            color='red',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(
            x=alt.X('original_count:Q', title='Number of Compressed Images')
        )
        
        # Layer the baseline on top of the bar chart
        chart = alt.layer(chart, baseline)
    
    # Apply configure_axis only for standalone charts
    if include_config:
        chart = chart.configure_axis(
            labelAngle=0,
            labelLimit=200
        )
    
    return chart


def create_image_count_chart(df: pd.DataFrame, dataset: str, output_path: str):
    """
    Create a bar chart comparing number of compressed images.
    
    Args:
        df: Aggregated DataFrame with image counts
        dataset: Dataset name
        output_path: Path to save the chart
    """
    if len(df) == 0:
        print(f"  No data to plot for {dataset}")
        return
    
    # Calculate total original frames for baseline
    original_count = get_total_original_frames(dataset)
    
    chart = _create_image_count_chart_base(df, dataset, original_count=original_count, include_config=True)
    if chart is None:
        return
    
    # Save chart
    chart.save(output_path)
    
    print(f"  Saved image count chart: {output_path}")


def create_all_datasets_tile_comparison_chart(all_datasets_data: dict[str, pd.DataFrame], 
                                               output_path: str, normalized: bool = False):
    """
    Create a horizontally concatenated chart comparing tiles across all datasets.
    
    Args:
        all_datasets_data: Dictionary mapping dataset names to their aggregated DataFrames
        output_path: Path to save the concatenated chart
        normalized: If True, normalize bars to percentages (0-100%)
    """
    # Create charts for each dataset
    charts = []
    for dataset, df in all_datasets_data.items():
        chart = _create_tile_comparison_chart_base(df, dataset, normalized=normalized, include_config=False)
        if chart is not None:
            charts.append(chart)
    
    if len(charts) == 0:
        print(f"  No data to plot for all datasets concatenation")
        return
    
    # Arrange charts in 3 columns using hconcat for each row, then vconcat for rows
    rows = []
    for i in range(0, len(charts), 3):
        row_charts = charts[i:min(i+3, len(charts))]
        rows.append(alt.hconcat(*row_charts) if len(row_charts) > 1 else row_charts[0])
    
    # Vertically concatenate rows
    concatenated_chart = alt.vconcat(*rows).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save concatenated chart
    concatenated_chart.save(output_path)
    
    chart_type = "normalized tile comparison" if normalized else "tile comparison"
    print(f"  Saved all datasets {chart_type} chart: {output_path}")


def create_all_datasets_image_count_chart(all_datasets_data: dict[str, pd.DataFrame], 
                                          output_path: str):
    """
    Create a horizontally concatenated chart comparing image counts across all datasets.
    
    Args:
        all_datasets_data: Dictionary mapping dataset names to their aggregated DataFrames
        output_path: Path to save the concatenated chart
    """
    # Create charts for each dataset
    charts = []
    for dataset, df in all_datasets_data.items():
        original_count = get_total_original_frames(dataset)
        chart = _create_image_count_chart_base(df, dataset, original_count=original_count, include_config=False)
        if chart is not None:
            charts.append(chart)
    
    if len(charts) == 0:
        print(f"  No data to plot for all datasets concatenation")
        return
    
    # Arrange charts in 3 columns using hconcat for each row, then vconcat for rows
    rows = []
    for i in range(0, len(charts), 3):
        row_charts = charts[i:min(i+3, len(charts))]
        rows.append(alt.hconcat(*row_charts) if len(row_charts) > 1 else row_charts[0])
    
    # Vertically concatenate rows
    concatenated_chart = alt.vconcat(*rows).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save concatenated chart
    concatenated_chart.save(output_path)
    
    print(f"  Saved all datasets image count chart: {output_path}")


def load_dataset_data(dataset: str, data_dir: Path) -> pd.DataFrame | None:
    """
    Load computed compression effectiveness data for a dataset from CSV.
    
    Args:
        dataset: Dataset name
        data_dir: Directory containing the CSV files
        
    Returns:
        DataFrame with aggregated data, or None if file not found
    """
    dataset_dir = data_dir / dataset
    csv_path = dataset_dir / f'{dataset}_compression_data.csv'
    
    if not csv_path.exists():
        print(f"  CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded data from: {csv_path}")
    print(f"  Total configurations: {len(df)}")
    return df


def process_dataset(dataset: str, data_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    """
    Process a single dataset and create visualizations.
    
    Args:
        dataset: Dataset name
        data_dir: Directory containing the CSV data files
        output_dir: Output directory for visualizations
        
    Returns:
        Aggregated DataFrame for this dataset, or None if no data
    """
    print(f"\nProcessing dataset: {dataset}")
    
    # Load data from CSV
    df = load_dataset_data(dataset, data_dir)
    
    if df is None or len(df) == 0:
        print(f"  No data found for dataset: {dataset}")
        return None
    
    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tile comparison chart
    tile_chart_path = dataset_output_dir / f'{dataset}_tile_comparison.png'
    create_tile_comparison_chart(df, dataset, str(tile_chart_path), normalized=False)
    
    # Create normalized tile comparison chart
    tile_chart_normalized_path = dataset_output_dir / f'{dataset}_tile_comparison_normalized.png'
    create_tile_comparison_chart(df, dataset, str(tile_chart_normalized_path), normalized=True)
    
    # Create image count chart
    image_chart_path = dataset_output_dir / f'{dataset}_image_count.png'
    create_image_count_chart(df, dataset, str(image_chart_path))
    
    print(f"  Total configurations: {len(df)}")
    print(f"  Total images: {df['num_images'].sum()}")
    print(f"  Total empty tiles: {df['empty_tiles'].sum()}")
    print(f"  Total occupied tiles: {df['occupied_tiles'].sum()}")
    print(f"  Total padding tiles: {df['padding_tiles'].sum()}")
    if 'occupancy_ratio' in df.columns:
        print(f"  Average occupancy ratio: {df['occupancy_ratio'].mean():.3f}")
    if 'padding_ratio' in df.columns:
        print(f"  Average padding ratio: {df['padding_ratio'].mean():.3f}")
    
    return df


def main():
    """
    Main function to create compression effectiveness visualizations.
    """
    # Load configuration
    config = get_config()
    
    # Get datasets from config
    datasets = config['EXEC']['DATASETS']
    
    # Set data directory (where CSV files are stored) and output directory
    data_dir = Path(CACHE_DIR) / 'SUMMARY' / '036_compress_effectiveness'
    output_dir = Path(CACHE_DIR) / 'SUMMARY' / '036_compress_effectiveness'
    
    print("Compression Effectiveness Visualization")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset and collect aggregated dataframes
    all_datasets_data = {}
    for dataset in datasets:
        agg_df = process_dataset(dataset, data_dir, output_dir)
        if agg_df is not None and len(agg_df) > 0:
            all_datasets_data[dataset] = agg_df
    
    # Create concatenated visualizations for all datasets
    if len(all_datasets_data) > 0:
        print(f"\nCreating concatenated visualizations for all datasets...")
        
        # Create concatenated tile comparison chart
        all_tile_chart_path = output_dir / 'all_datasets_tile_comparison.png'
        create_all_datasets_tile_comparison_chart(all_datasets_data, str(all_tile_chart_path), normalized=False)
        
        # Create concatenated normalized tile comparison chart
        all_tile_chart_normalized_path = output_dir / 'all_datasets_tile_comparison_normalized.png'
        create_all_datasets_tile_comparison_chart(all_datasets_data, str(all_tile_chart_normalized_path), normalized=True)
        
        # Create concatenated image count chart
        all_image_chart_path = output_dir / 'all_datasets_image_count.png'
        create_all_datasets_image_count_chart(all_datasets_data, str(all_image_chart_path))
    
    print(f"\nVisualization complete. Charts saved to: {output_dir}")


if __name__ == '__main__':
    main()

