#!/usr/bin/env python3
"""
Script to visualize the effectiveness of the compression algorithm.

This script analyzes compressed frames and creates visualizations comparing:
1. Number of empty, occupied (non-padding), and padding tiles in compressed images for each configuration
2. Number of compressed images per configuration

Padding tiles are tiles that are present in the index_map (occupied) but were not
in the original relevancy bitmap (below threshold), added by the tilepadding operation.

One visualization per dataset, comparing across classifier, tilesize, and tilepadding.
Configurations are loaded from configs/global.yaml using get_config.
"""

import argparse
import json
import multiprocessing
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
from rich.progress import Progress, BarColumn, TextColumn

from polyis.utilities import CACHE_DIR, get_config, load_classification_results, scale_to_percent


# Compression stage directory to analyze
COMPRESSION_STAGE = '033_compressed_frames'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize compression effectiveness across configurations'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    return parser.parse_args()


def count_images_for_path(config_path: Path) -> int:
    """
    Count the number of compressed images in a configuration directory.
    
    Args:
        config_path: Path to configuration directory
        
    Returns:
        Number of .jpg files in the images subdirectory
    """
    images_dir = config_path / 'images'
    
    if not images_dir.exists():
        return 0
    
    # Count .jpg files using glob
    jpg_files = list(images_dir.glob('*.jpg'))
    return len(jpg_files)


def parse_index_map_filename(filename: str) -> Tuple[int, int] | None:
    """
    Parse index map filename to extract start and end frame indices.
    
    Filename format: {collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.npy
    where frame_idx is the end frame.
    
    Args:
        filename: Name of the index map file
        
    Returns:
        Tuple of (start_frame, end_frame) or None if parsing fails
    """
    # Remove .npy extension
    base = filename.replace('.npy', '')
    # Match format: 0000_0000_0099
    match = re.match(r'^\d{4}_(\d{4})_(\d{4})$', base)
    if match:
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        return start_frame, end_frame
    return None


def load_offset_lookup(offset_lookup_path: Path) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """
    Load offset lookup JSONL file.
    
    Each line contains: [[packed_y, packed_x], [original_y, original_x], frame_idx]
    
    Args:
        offset_lookup_path: Path to the offset lookup JSONL file
        
    Returns:
        List of ((packed_y, packed_x), (original_y, original_x), frame_idx)
    """
    if not offset_lookup_path.exists():
        return []
    
    offset_lookup = []
    with open(offset_lookup_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert to tuple format: ((packed_y, packed_x), (original_y, original_x), frame_idx)
                offset_lookup.append((tuple(data[0]), tuple(data[1]), data[2]))
    return offset_lookup


def count_tiles_for_path(config_path: Path, dataset: str, video: str, 
                        classifier: str, tilesize: int, threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Count empty, occupied, and padding tiles from index_maps in a configuration directory.
    
    Padding tiles are tiles that are present in the index_map (occupied) but were not
    in the original relevancy bitmap (below threshold).
    
    Args:
        config_path: Path to configuration directory
        dataset: Dataset name
        video: Video name
        classifier: Classifier name
        tilesize: Tile size
        threshold: Threshold for classification probability (default: 0.5)
        
    Returns:
        Tuple of (empty_tiles, occupied_tiles, padding_tiles)
    """
    index_maps_dir = config_path / 'index_maps'
    
    if not index_maps_dir.exists():
        return 0, 0, 0
    
    # Get all .npy files
    npy_files = list(index_maps_dir.glob('*.npy'))
    
    # Load classification results for this video
    try:
        classification_results = load_classification_results(CACHE_DIR, dataset, video, tilesize, classifier)
    except FileNotFoundError:
        # If classification results not found, can't compute padding tiles
        # Fall back to basic counting
        empty_tiles = 0
        occupied_tiles = 0
        for npy_file in npy_files:
            index_map = np.load(str(npy_file))
            empty_tiles += int(np.sum(index_map == 0))
            occupied_tiles += int(np.sum(index_map != 0))
        return int(empty_tiles), int(occupied_tiles), 0
    
    # Load all index maps and compute counts using vectorized numpy operations
    empty_tiles = 0
    occupied_tiles = 0
    padding_tiles = 0
    
    for npy_file in npy_files:
        # Load the index map
        index_map = np.load(str(npy_file))
        
        # Parse filename to get frame range
        filename = npy_file.name
        frame_range = parse_index_map_filename(filename)
        
        if frame_range is None:
            # If we can't parse the filename, just count basic tiles
            empty_tiles += int(np.sum(index_map == 0))
            occupied_tiles += int(np.sum(index_map != 0))
            continue
        
        start_frame, end_frame = frame_range
        
        # Count basic tiles
        empty_tiles += int(np.sum(index_map == 0))
        occupied_mask = (index_map != 0)
        occupied_tiles += int(np.sum(occupied_mask))
        
        # Load offset_lookup to map compressed positions back to original frame positions
        offset_lookup_path = config_path / 'offset_lookups' / npy_file.name.replace('.npy', '.jsonl')
        offset_lookup = load_offset_lookup(offset_lookup_path)
        
        if len(offset_lookup) == 0:
            # If offset_lookup not found, can't compute padding tiles accurately
            continue
        
        # Build relevancy bitmaps for each frame (indexed by frame_idx)
        frame_relevancy_bitmaps = {}
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(classification_results):
                continue
            
            frame_result = classification_results[frame_idx]
            classifications: str = frame_result['classification_hex']
            classification_size: tuple[int, int] = frame_result['classification_size']
            
            # Create relevancy bitmap from classification hex
            bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
            bitmap_frame = bitmap_frame > (threshold * 255)
            bitmap_frame = bitmap_frame.astype(np.uint8)
            
            frame_relevancy_bitmaps[frame_idx] = bitmap_frame
        
        # Count padding tiles by checking each occupied position in index_map
        # For each group_id in index_map, use offset_lookup to find original position and frame
        # Then check if that original position was relevant in the relevancy bitmap
        padding_count = 0
        
        # Get all unique group IDs (excluding 0)
        unique_group_ids = np.unique(index_map[index_map != 0])
        
        for group_id in unique_group_ids:
            # Find all positions with this group_id
            group_positions = np.argwhere(index_map == group_id)
            
            # Get offset_lookup entry for this group (group_id is 1-indexed, offset_lookup is 0-indexed)
            if group_id > len(offset_lookup):
                # Invalid group_id, skip
                continue
            
            offset_entry = offset_lookup[group_id - 1]  # Convert to 0-indexed
            (packed_y, packed_x), (original_y, original_x), frame_idx = offset_entry
            
            # Check if this frame has a relevancy bitmap
            if frame_idx not in frame_relevancy_bitmaps:
                # Frame not in range or not found, count as padding
                padding_count += len(group_positions)
                continue
            
            relevancy_bitmap = frame_relevancy_bitmaps[frame_idx]
            
            # Check each position in the group
            for py, px in group_positions:
                # Calculate relative position within the polyomino
                # The offset_lookup gives us the anchor point (packed_y, packed_x) and (original_y, original_x)
                # We need to find the relative offset from the anchor
                rel_y = py - packed_y
                rel_x = px - packed_x
                
                # Map to original frame coordinates
                orig_y = original_y + rel_y
                orig_x = original_x + rel_x
                
                # Check if this original position is within bounds and was relevant
                if (0 <= orig_y < relevancy_bitmap.shape[0] and 
                    0 <= orig_x < relevancy_bitmap.shape[1]):
                    if relevancy_bitmap[orig_y, orig_x] == 0:
                        # This tile was not relevant in the original frame, so it's a padding tile
                        padding_count += 1
                else:
                    # Position out of bounds, count as padding
                    padding_count += 1
        
        padding_tiles += padding_count
    
    return int(empty_tiles), int(occupied_tiles), int(padding_tiles)


def process_video_dir(args_tuple: Tuple[Path, str, List[str], List[int], List[str], List[float], bool]) -> Tuple[List[dict], dict]:
    """
    Process a single video directory and return records and statistics.
    
    This function is designed to be used with multiprocessing.Pool.
    
    Args:
        args_tuple: Tuple containing (video_dir, dataset, classifiers, tilesizes, tilepaddings, canvas_scales, verbose)
        
    Returns:
        Tuple of (records_list, stats_dict) where stats_dict contains:
            videos_with_compression, total_config_dirs, skipped_invalid_format, skipped_not_in_target
    """
    video_dir, dataset, classifiers, tilesizes, tilepaddings, canvas_scales, verbose = args_tuple
    video_name = video_dir.name
    
    records = []
    videos_with_compression = 0
    total_config_dirs = 0
    skipped_invalid_format = 0
    skipped_not_in_target = 0
    
    # Check for compression stage directory
    compressed_frames_dir = video_dir / COMPRESSION_STAGE
    
    if not compressed_frames_dir.exists():
        return records, {
            'videos_with_compression': videos_with_compression,
            'total_config_dirs': total_config_dirs,
            'skipped_invalid_format': skipped_invalid_format,
            'skipped_not_in_target': skipped_not_in_target
        }
    
    videos_with_compression = 1
    
    # Find all configuration directories
    try:
        config_dirs_list = list(compressed_frames_dir.iterdir())
    except (OSError, PermissionError):
        return records, {
            'videos_with_compression': videos_with_compression,
            'total_config_dirs': total_config_dirs,
            'skipped_invalid_format': skipped_invalid_format,
            'skipped_not_in_target': skipped_not_in_target
        }
    
    for config_dir in config_dirs_list:
        if not config_dir.is_dir():
            continue
        
        total_config_dirs += 1
        
        # Parse configuration name: classifier_tilesize_samplerate_tilepadding_s{pct}
        config_name = config_dir.name
        parts = config_name.split('_')
        
        if len(parts) != 5:
            skipped_invalid_format += 1
            if verbose:
                print(f"    Warning: Skipping invalid config name: {config_name} (parts: {parts})")
            continue
        
        classifier = parts[0]
        try:
            tilesize = int(parts[1])
        except ValueError:
            skipped_invalid_format += 1
            if verbose:
                print(f"    Warning: Invalid tilesize in {config_name}")
            continue
        try:
            sample_rate = int(parts[2])
        except ValueError:
            skipped_invalid_format += 1
            if verbose:
                print(f"    Warning: Invalid sample_rate in {config_name}")
            continue
        tilepadding = parts[3]
        scale_str = parts[4]
        # Extract canvas scale from scale string (e.g., 's100' -> 1.0)
        canvas_scale = int(scale_str[1:]) / 100.0
        
        # Check if this configuration is in our target list
        if classifier not in classifiers:
            skipped_not_in_target += 1
            if verbose:
                print(f"    Skipping {config_name}: classifier '{classifier}' not in target list")
            continue
        if tilesize not in tilesizes:
            skipped_not_in_target += 1
            if verbose:
                print(f"    Skipping {config_name}: tilesize {tilesize} not in target list")
            continue
        if tilepadding not in tilepaddings:
            skipped_not_in_target += 1
            if verbose:
                print(f"    Skipping {config_name}: tilepadding '{tilepadding}' not in target list")
            continue
        if canvas_scale not in canvas_scales:
            skipped_not_in_target += 1
            if verbose:
                print(f"    Skipping {config_name}: canvas_scale {canvas_scale} not in target list")
            continue
        
        # Count images and tiles
        num_images = count_images_for_path(config_dir)
        empty_tiles, occupied_tiles, padding_tiles = count_tiles_for_path(
            config_dir, dataset, video_name, classifier, tilesize
        )
        
        records.append({
            'dataset': dataset,
            'video': video_name,
            'classifier': classifier,
            'tilesize': tilesize,
            'sample_rate': sample_rate,
            'tilepadding': tilepadding,
            'canvas_scale': canvas_scale,
            'num_images': num_images,
            'empty_tiles': empty_tiles,
            'occupied_tiles': occupied_tiles,
            'padding_tiles': padding_tiles,
        })
    
    return records, {
        'videos_with_compression': videos_with_compression,
        'total_config_dirs': total_config_dirs,
        'skipped_invalid_format': skipped_invalid_format,
        'skipped_not_in_target': skipped_not_in_target
    }


def collect_dataset_data(dataset: str, classifiers: List[str], tilesizes: List[int], 
                         tilepaddings: List[str], canvas_scales: List[float],
                         verbose: bool = False) -> pd.DataFrame:
    """
    Collect compression effectiveness data for a dataset.
    
    Args:
        dataset: Dataset name
        classifiers: List of classifier names to analyze
        tilesizes: List of tile sizes to analyze
        tilepaddings: List of tile padding modes to analyze
        canvas_scales: List of canvas scale factors to analyze
        verbose: Whether to print verbose output
        
    Returns:
        DataFrame with columns: dataset, video, classifier, tilesize, sample_rate, tilepadding,
                               canvas_scale, num_images, empty_tiles, occupied_tiles, padding_tiles
    """
    dataset_cache_dir = Path(CACHE_DIR) / dataset / 'execution'
    
    if not dataset_cache_dir.exists():
        if verbose:
            print(f"Dataset cache directory not found: {dataset_cache_dir}")
        return pd.DataFrame()
    
    records = []
    
    # Find all video directories
    video_dirs = [d for d in dataset_cache_dir.iterdir() if d.is_dir()]
    
    if verbose:
        print(f"  Found {len(video_dirs)} video directories")
    
    videos_with_compression = 0
    total_config_dirs = 0
    skipped_invalid_format = 0
    skipped_not_in_target = 0
    
    # Prepare arguments for parallel processing
    process_args = [
        (video_dir, dataset, classifiers, tilesizes, tilepaddings, canvas_scales, verbose)
        for video_dir in video_dirs
    ]
    
    # Process video directories in parallel
    num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking with rich
        if verbose:
            with Progress(
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                transient=False
            ) as progress:
                task = progress.add_task(
                    f"Processing {dataset} videos",
                    total=len(video_dirs)
                )
                results = []
                for result in pool.imap(process_video_dir, process_args):
                    results.append(result)
                    progress.update(task, advance=1)
        else:
            results = list(pool.imap(process_video_dir, process_args))
    
    # Aggregate results from parallel processing
    for video_records, stats in results:
        records.extend(video_records)
        videos_with_compression += stats['videos_with_compression']
        total_config_dirs += stats['total_config_dirs']
        skipped_invalid_format += stats['skipped_invalid_format']
        skipped_not_in_target += stats['skipped_not_in_target']
    
    if verbose:
        print(f"  Videos with {COMPRESSION_STAGE}: {videos_with_compression}")
        print(f"  Total config directories found: {total_config_dirs}")
        print(f"  Skipped (invalid format): {skipped_invalid_format}")
        print(f"  Skipped (not in target): {skipped_not_in_target}")
        
        # Debug: Show what directories exist in first few videos
        if videos_with_compression == 0 and len(video_dirs) > 0:
            print(f"  Debug: Checking first video directory structure...")
            sample_video = video_dirs[0]
            if sample_video.exists():
                existing_dirs = [d.name for d in sample_video.iterdir() if d.is_dir()]
                print(f"  Sample video '{sample_video.name}' has directories: {sorted(existing_dirs)}")
                # Check if any compression-related directories exist
                compression_related = [d for d in existing_dirs if 'compress' in d.lower() or 'pack' in d.lower() or d.startswith('03')]
                if compression_related:
                    print(f"  Compression-related directories found: {compression_related}")
    
    df = pd.DataFrame(records)
    
    if verbose:
        print(f"  Collected {len(df)} configuration records")
        if len(df) == 0 and total_config_dirs > 0:
            # Show sample config names for debugging
            sample_configs = []
            for video_dir in video_dirs[:5]:  # Check first 5 videos
                compressed_frames_dir = video_dir / COMPRESSION_STAGE
                if compressed_frames_dir.exists():
                    try:
                        config_dirs = [d.name for d in compressed_frames_dir.iterdir() if d.is_dir()]
                        sample_configs.extend(config_dirs[:3])  # First 3 from each
                    except (OSError, PermissionError):
                        pass
            if sample_configs:
                print(f"  Sample config names found: {sample_configs[:10]}")
                print(f"  Target classifiers: {classifiers}")
                print(f"  Target tilesizes: {tilesizes}")
                print(f"  Target tilepaddings: {tilepaddings}")
    
    return df


def aggregate_dataset_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data across videos for each configuration.
    
    Args:
        df: DataFrame with per-video data
        
    Returns:
        DataFrame aggregated by dataset, classifier, tilesize, sample_rate, tilepadding, canvas_scale
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    # Aggregate across videos
    agg_df = df.groupby(['dataset', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']).agg({
        'num_images': 'sum',
        'empty_tiles': 'sum',
        'occupied_tiles': 'sum',
        'padding_tiles': 'sum',
    }).reset_index()
    
    return agg_df


def create_tile_comparison_chart(df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a stacked bar chart comparing empty, occupied (non-padding), and padding tiles.
    
    Args:
        df: Aggregated DataFrame with tile counts
        dataset: Dataset name
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    if len(df) == 0:
        if verbose:
            print(f"  No data to plot for {dataset}")
        return
    
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
    
    # Create stacked bar chart with three categories
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('count:Q', title='Number of Tiles', stack='zero'),
        y=alt.Y('config_label:N', title='Configuration', 
                sort=alt.SortField('count', order='descending')),
        color=alt.Color('tile_type:N', title='Tile Type',
                       scale=alt.Scale(domain=['Occupied', 'Padding', 'Empty'],
                                      range=['#4caf50', '#ff9800', '#e0e0e0'])),
        tooltip=['config_label', 'tile_type', 'count', 'occupancy_ratio:Q', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Empty vs Occupied vs Padding Tiles by Configuration - {dataset}'
    ).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save chart
    chart.save(output_path)
    
    if verbose:
        print(f"  Saved tile comparison chart: {output_path}")


def create_image_count_chart(df: pd.DataFrame, dataset: str, output_path: str, verbose: bool = False):
    """
    Create a bar chart comparing number of compressed images.
    
    Args:
        df: Aggregated DataFrame with image counts
        dataset: Dataset name
        output_path: Path to save the chart
        verbose: Whether to print verbose output
    """
    if len(df) == 0:
        if verbose:
            print(f"  No data to plot for {dataset}")
        return
    
    # Prepare data
    chart_df = df.copy()
    chart_df['config_label'] = chart_df.apply(
        lambda row: f"{row['classifier'][:4]}_{row['tilesize']}_{row['sample_rate']}_{row['tilepadding'][:4]}_s{int(row['canvas_scale'] * 100)}", axis=1
    )
    
    # Sort by number of images
    chart_df = chart_df.sort_values('num_images', ascending=False)
    
    # Create bar chart
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('num_images:Q', title='Number of Compressed Images'),
        y=alt.Y('config_label:N', title='Configuration',
                sort=alt.SortField('num_images', order='descending')),
        color=alt.Color('classifier:N', title='Classifier'),
        tooltip=['config_label', 'num_images', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Number of Compressed Images by Configuration - {dataset}'
    ).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save chart
    chart.save(output_path)
    
    if verbose:
        print(f"  Saved image count chart: {output_path}")


def create_tile_comparison_chart_for_dataset(df: pd.DataFrame, dataset: str) -> alt.Chart | None:
    """
    Create a stacked bar chart comparing empty, occupied (non-padding), and padding tiles for a single dataset.
    
    This is a helper function that returns an Altair chart object instead of saving it.
    Used for creating concatenated visualizations.
    
    Args:
        df: Aggregated DataFrame with tile counts for a single dataset
        dataset: Dataset name
        
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
    
    # Create stacked bar chart with three categories
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('count:Q', title='Number of Tiles', stack='zero'),
        y=alt.Y('config_label:N', title='Configuration', 
                sort=alt.SortField('count', order='descending')),
        color=alt.Color('tile_type:N', title='Tile Type',
                       scale=alt.Scale(domain=['Occupied', 'Padding', 'Empty'],
                                      range=['#4caf50', '#ff9800', '#e0e0e0'])),
        tooltip=['config_label', 'tile_type', 'count', 'occupancy_ratio:Q', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Empty vs Occupied vs Padding Tiles - {dataset}'
    ).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    return chart


def create_image_count_chart_for_dataset(df: pd.DataFrame, dataset: str) -> alt.Chart | None:
    """
    Create a bar chart comparing number of compressed images for a single dataset.
    
    This is a helper function that returns an Altair chart object instead of saving it.
    Used for creating concatenated visualizations.
    
    Args:
        df: Aggregated DataFrame with image counts for a single dataset
        dataset: Dataset name
        
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
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('num_images:Q', title='Number of Compressed Images'),
        y=alt.Y('config_label:N', title='Configuration',
                sort=alt.SortField('num_images', order='descending')),
        color=alt.Color('classifier:N', title='Classifier'),
        tooltip=['config_label', 'num_images', 'classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale']
    ).properties(
        width=800,
        height=400,
        title=f'Number of Compressed Images - {dataset}'
    ).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    return chart


def create_all_datasets_tile_comparison_chart(all_datasets_data: dict[str, pd.DataFrame], 
                                               output_path: str, verbose: bool = False):
    """
    Create a horizontally concatenated chart comparing tiles across all datasets.
    
    Args:
        all_datasets_data: Dictionary mapping dataset names to their aggregated DataFrames
        output_path: Path to save the concatenated chart
        verbose: Whether to print verbose output
    """
    # Create charts for each dataset
    charts = []
    for dataset, df in all_datasets_data.items():
        chart = create_tile_comparison_chart_for_dataset(df, dataset)
        if chart is not None:
            charts.append(chart)
    
    if len(charts) == 0:
        if verbose:
            print(f"  No data to plot for all datasets concatenation")
        return
    
    # Horizontally concatenate all charts
    concatenated_chart = alt.hconcat(*charts).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save concatenated chart
    concatenated_chart.save(output_path)
    
    if verbose:
        print(f"  Saved all datasets tile comparison chart: {output_path}")


def create_all_datasets_image_count_chart(all_datasets_data: dict[str, pd.DataFrame], 
                                          output_path: str, verbose: bool = False):
    """
    Create a horizontally concatenated chart comparing image counts across all datasets.
    
    Args:
        all_datasets_data: Dictionary mapping dataset names to their aggregated DataFrames
        output_path: Path to save the concatenated chart
        verbose: Whether to print verbose output
    """
    # Create charts for each dataset
    charts = []
    for dataset, df in all_datasets_data.items():
        chart = create_image_count_chart_for_dataset(df, dataset)
        if chart is not None:
            charts.append(chart)
    
    if len(charts) == 0:
        if verbose:
            print(f"  No data to plot for all datasets concatenation")
        return
    
    # Horizontally concatenate all charts
    concatenated_chart = alt.hconcat(*charts).configure_axis(
        labelAngle=0,
        labelLimit=200
    )
    
    # Save concatenated chart
    concatenated_chart.save(output_path)
    
    if verbose:
        print(f"  Saved all datasets image count chart: {output_path}")


def process_dataset(dataset: str, classifiers: List[str], tilesizes: List[int], 
                    tilepaddings: List[str], canvas_scales: List[float],
                    output_dir: Path, verbose: bool = False) -> pd.DataFrame | None:
    """
    Process a single dataset and create visualizations.
    
    Args:
        dataset: Dataset name
        classifiers: List of classifier names to analyze
        tilesizes: List of tile sizes to analyze
        tilepaddings: List of tile padding modes to analyze
        canvas_scales: List of canvas scale factors to analyze
        output_dir: Output directory for visualizations
        verbose: Whether to print verbose output
        
    Returns:
        Aggregated DataFrame for this dataset, or None if no data
    """
    if verbose:
        print(f"\nProcessing dataset: {dataset}")
    
    # Collect data
    df = collect_dataset_data(dataset, classifiers, tilesizes, tilepaddings, canvas_scales, verbose)
    
    if len(df) == 0:
        if verbose:
            print(f"  No data found for dataset: {dataset}")
        return None
    
    # Aggregate across videos
    agg_df = aggregate_dataset_data(df)
    
    if len(agg_df) == 0:
        if verbose:
            print(f"  No aggregated data for dataset: {dataset}")
        return None
    
    # Calculate additional metrics
    agg_df['total_tiles'] = agg_df['empty_tiles'] + agg_df['occupied_tiles']
    agg_df['occupancy_ratio'] = agg_df['occupied_tiles'] / agg_df['total_tiles'].replace(0, 1)
    agg_df['non_padding_occupied'] = agg_df['occupied_tiles'] - agg_df['padding_tiles']
    agg_df['padding_ratio'] = agg_df['padding_tiles'] / agg_df['occupied_tiles'].replace(0, 1)
    
    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tile comparison chart
    tile_chart_path = dataset_output_dir / f'{dataset}_tile_comparison.png'
    create_tile_comparison_chart(agg_df, dataset, str(tile_chart_path), verbose)
    
    # Create image count chart
    image_chart_path = dataset_output_dir / f'{dataset}_image_count.png'
    create_image_count_chart(agg_df, dataset, str(image_chart_path), verbose)
    
    # Save aggregated data as CSV
    csv_path = dataset_output_dir / f'{dataset}_compression_data.csv'
    agg_df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"  Saved data CSV: {csv_path}")
        print(f"  Total configurations: {len(agg_df)}")
        print(f"  Total images: {agg_df['num_images'].sum()}")
        print(f"  Total empty tiles: {agg_df['empty_tiles'].sum()}")
        print(f"  Total occupied tiles: {agg_df['occupied_tiles'].sum()}")
        print(f"  Total padding tiles: {agg_df['padding_tiles'].sum()}")
        print(f"  Average occupancy ratio: {agg_df['occupancy_ratio'].mean():.3f}")
        print(f"  Average padding ratio: {agg_df['padding_ratio'].mean():.3f}")
    
    return agg_df


def main(args):
    """
    Main function to create compression effectiveness visualizations.
    
    Args:
        args: Parsed command line arguments
    """
    # Load configuration
    config = get_config()
    
    # Get datasets, classifiers, tilesizes, tilepaddings, and canvas_scales from config
    datasets = config['EXEC']['DATASETS']
    classifiers = config['EXEC']['CLASSIFIERS']
    tilesizes = config['EXEC']['TILE_SIZES']
    tilepaddings = config['EXEC']['TILEPADDING_MODES']
    canvas_scales = [float(s) for s in config['EXEC']['CANVAS_SCALE']]
    
    # Set output directory to {CACHE_DIR}/SUMMARY/036_compress_effectiveness
    output_dir = Path(CACHE_DIR) / 'SUMMARY' / '036_compress_effectiveness'
    
    if args.verbose:
        print("Compression Effectiveness Visualization")
        print("=" * 80)
        print(f"Datasets: {datasets}")
        print(f"Classifiers: {classifiers}")
        print(f"Tile sizes: {tilesizes}")
        print(f"Tile paddings: {tilepaddings}")
        print(f"Canvas scales: {canvas_scales}")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset and collect aggregated dataframes
    all_datasets_data = {}
    for dataset in datasets:
        agg_df = process_dataset(dataset, classifiers, tilesizes, tilepaddings, canvas_scales, output_dir, args.verbose)
        if agg_df is not None and len(agg_df) > 0:
            all_datasets_data[dataset] = agg_df
    
    # Create concatenated visualizations for all datasets
    if len(all_datasets_data) > 0:
        if args.verbose:
            print(f"\nCreating concatenated visualizations for all datasets...")
        
        # Create concatenated tile comparison chart
        all_tile_chart_path = output_dir / 'all_datasets_tile_comparison.png'
        create_all_datasets_tile_comparison_chart(all_datasets_data, str(all_tile_chart_path), args.verbose)
        
        # Create concatenated image count chart
        all_image_chart_path = output_dir / 'all_datasets_image_count.png'
        create_all_datasets_image_count_chart(all_datasets_data, str(all_image_chart_path), args.verbose)
    
    print(f"\nVisualization complete. Charts saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

