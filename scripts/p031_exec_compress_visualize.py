#!/usr/bin/env python3

import argparse
import os
import re
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize compression empty-space ratios over time by reading "
            "index_maps saved by 030_exec_compress.py. Creates faceted histograms "
            "using Altair with 20 bins per classifier/tile combination."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=False,
        default=DATASETS_TO_TEST,
        help="Dataset names (matches directories in DATA_DIR and CACHE_DIR)",
    )
    return parser.parse_args()


def list_video_dirs(dataset: str) -> list[str]:
    dataset_cache_dir = os.path.join(CACHE_DIR, dataset, 'execution')
    if not os.path.isdir(dataset_cache_dir):
        raise FileNotFoundError(f"Dataset cache dir does not exist: {dataset_cache_dir}")
    # Only directories that contain a packing folder are relevant
    video_dirs: list[str] = []
    for entry in os.listdir(dataset_cache_dir):
        full_path = os.path.join(dataset_cache_dir, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, "030_compressed_frames")):
            video_dirs.append(full_path)
    return sorted(video_dirs)


def list_classifier_tile_dirs(video_cache_dir: str) -> list[str]:
    packing_dir = os.path.join(video_cache_dir, "030_compressed_frames")
    if not os.path.isdir(packing_dir):
        return []
    dirs: list[str] = []
    for entry in os.listdir(packing_dir):
        full_path = os.path.join(packing_dir, entry)
        if os.path.isdir(full_path):
            # We expect an index_maps subdir
            if os.path.isdir(os.path.join(full_path, "index_maps")):
                dirs.append(full_path)
    return sorted(dirs)


def parse_classifier_and_tile(dir_name: str) -> tuple[str, int]:
    # dir_name is like "SimpleCNN_64" or possibly classifier names with underscores
    # Split on the last underscore
    if "_" not in dir_name:
        return dir_name, -1
    classifier, tile = dir_name.rsplit("_", 1)
    try:
        tilesize = int(tile)
    except ValueError:
        tilesize = -1
    return classifier, tilesize


def list_index_map_files(classifier_tile_dir: str) -> list[str]:
    index_dir = os.path.join(classifier_tile_dir, "index_maps")
    if not os.path.isdir(index_dir):
        return []
    files = [
        os.path.join(index_dir, f)
        for f in os.listdir(index_dir)
        if f.endswith(".npy")
    ]
    return sorted(files)


def parse_start_end_from_filename(npy_path: str) -> tuple[int, int]:
    base = os.path.basename(npy_path)
    name, _ = os.path.splitext(base)
    # Expected: 00000000_00000099.npy
    match = re.match(r"^(\d{8})_(\d{8})$", name)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def compute_content_ratio(index_map: np.ndarray) -> float:
    # index_map shape: (grid_h, grid_w)
    if index_map.ndim != 2:
        raise ValueError(f"Unexpected index_map shape: {index_map.shape}")
    occupancy = index_map[:, :]
    total_tiles = occupancy.size
    filled_tiles = np.count_nonzero(occupancy > 0)
    return float(filled_tiles) / float(total_tiles) if total_tiles > 0 else 0.0


def ensure_summary_dirs(dataset: str) -> tuple[str, str]:
    base = os.path.join(CACHE_DIR, "summary", dataset, "compression")
    each = os.path.join(base, "each")
    os.makedirs(each, exist_ok=True)
    return base, each


def plot_series(
    x_values: list[int],
    y_values: list[float],
    avg_value: float,
    title: str,
    output_png_path: str,
) -> None:
    # Create DataFrame for Altair
    df = pd.DataFrame({
        'frame_index': x_values,
        'content_ratio': y_values
    })
    
    # Create line chart with average line
    line = alt.Chart(df).mark_line(color='#1f77b4').encode(
        x=alt.X('frame_index:Q', title='Frame index (end of packed span)'),
        y=alt.Y('content_ratio:Q', title='Content ratio (index_map[:,:,0] > 0)', scale=alt.Scale(domain=[0, 1]))
    )
    
    # Add average line
    avg_line = alt.Chart(pd.DataFrame({'avg': [avg_value]})).mark_rule(
        color='#ff7f0e',
        strokeDash=[5, 5]
    ).encode(
        y='avg:Q'
    )
    
    # Combine charts
    chart = (line + avg_line).properties(
        width=800,
        height=300,
        title=title
    ).configure_axis(
        gridOpacity=0.3
    )
    
    # Save chart
    chart.save(output_png_path)


def plot_histogram_facets(labels: list[str], datasets: list[list[float]], title: str, output_png_path: str) -> None:
    # Create faceted histogram plot with Altair
    if not datasets:
        return
    
    # Filter out values >= 1 and prepare data
    filtered_data = []
    for label, data in zip(labels, datasets):
        filtered_values = [x for x in data if x < 1]
        if filtered_values:
            for value in filtered_values:
                filtered_data.append({'label': label, 'content_ratio': value})
    
    if not filtered_data:
        return
    
    # Create DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Create faceted histogram with 20 bins
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('content_ratio:Q', 
                bin=alt.Bin(maxbins=20),
                title='Content ratio'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('label:N', legend=None)
    ).properties(
        width=600,
        height=80
    ).facet(
        row=alt.Row('label:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left'))
    ).resolve_scale(
        y='independent'
    ).properties(
        title=title
    ).configure_axis(
        gridOpacity=0.3
    )
    
    # Save chart
    chart.save(output_png_path)
    print(f"Saved faceted histogram: {output_png_path}")


def plot_histogram_facets_per_video(video_name: str, video_data: list[tuple[str, list[float], float]], 
                                   dataset: str, output_dir: str) -> None:
    """
    Create a faceted histogram plot for a single video, sorted by average content ratio.
    
    Args:
        video_name: Name of the video
        video_data: List of tuples (label, y_values, avg_value) for each classifier/tile combination
        dataset: Dataset name
        output_dir: Directory to save the plot
    """
    if not video_data:
        return
    
    # Sort by average content ratio (descending)
    video_data.sort(key=lambda x: x[2], reverse=True)
    
    # Prepare data for faceted histogram
    plot_data = []
    for label, y_values, avg_value in video_data:
        # Filter out values >= 1
        filtered_values = [x for x in y_values if x < 1]
        if filtered_values:
            for value in filtered_values:
                plot_data.append({
                    'label': label,
                    'content_ratio': value,
                    'avg': avg_value
                })
    
    if not plot_data:
        print(f"No valid data for video {video_name}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create faceted histogram with 20 bins
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('content_ratio:Q', 
                bin=alt.Bin(maxbins=20),
                title='Content ratio'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('label:N', legend=None)
    ).properties(
        width=600,
        height=80
    ).facet(
        row=alt.Row('label:N', 
                   title=None, 
                   header=alt.Header(labelAngle=0, labelAlign='left'),
                   sort=alt.SortField('avg', order='descending'))
    ).resolve_scale(
        y='independent'
    ).properties(
        title=f"Content ratios by classifier/tile - {video_name} ({dataset})"
    ).configure_axis(
        gridOpacity=0.3
    )
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_name}_compress_content_ratio_histogram.png")
    chart.save(plot_path)
    print(f"Saved faceted histogram for {video_name}: {plot_path}")


def process_series_for_dir(dataset: str, video_cache_dir: str, classifier_tile_dir: str,
                           idx: int) -> tuple[str, list[int], list[float], float, str, int]:
    npy_files = list_index_map_files(classifier_tile_dir)
    if not npy_files:
        return "", [], [], 0.0, "", -1

    # Build time series
    time_to_ratio: list[tuple[int, float]] = []
    for npy_path in tqdm(npy_files, desc=f"Reading index_maps ({os.path.basename(classifier_tile_dir)})", leave=False, position=idx):
        try:
            start_idx, end_idx = parse_start_end_from_filename(npy_path)
            index_map = np.load(npy_path)
            content_ratio = compute_content_ratio(index_map)
            x_coord = end_idx if end_idx >= 0 else len(time_to_ratio)
            time_to_ratio.append((x_coord, content_ratio))
        except Exception as e:
            print(f"Warning: failed to read {npy_path}: {e}")

    if not time_to_ratio:
        return "", [], [], 0.0, "", -1

    # Sort by x (time)
    time_to_ratio.sort(key=lambda t: t[0])
    x_values = [t for t, _ in time_to_ratio]
    y_values = [r for _, r in time_to_ratio]
    avg_value = float(np.mean(y_values)) if len(y_values) > 0 else 0.0

    # Output paths
    series_name = os.path.basename(classifier_tile_dir)
    video_name = os.path.basename(video_cache_dir)
    classifier, tilesize = parse_classifier_and_tile(series_name)

    # Prepare summary output locations
    base_dir, each_dir = ensure_summary_dirs(dataset)
    safe_video = video_name
    safe_classifier = classifier
    # Filenames include identifiers to keep one file per series
    plot_path = os.path.join(each_dir, f"{safe_video}__{safe_classifier}_{tilesize}__compress_content_ratio.png")
    json_path = os.path.join(each_dir, f"{safe_video}__{safe_classifier}_{tilesize}__compress_content_ratio.json")

    title = f"{video_name} | {classifier} | tile {tilesize}"
    plot_series(x_values, y_values, avg_value, title, plot_path)
    print(f"Saved plot: {plot_path}")

    metrics = {
        "dataset": dataset,
        "video": video_name,
        "series": series_name,
        "classifier": classifier,
        "tilesize": tilesize,
        "x_values": x_values,
        "content_ratios": y_values,
        "average_content_ratio": avg_value,
        "note": "Content ratio = fraction of tiles with index_map[:,:,0] > 0",
    }
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {json_path}")

    label = f"{video_name}|{classifier}|{tilesize}"
    return label, x_values, y_values, avg_value, classifier, tilesize


def process_video_worker(args_tuple: tuple[str, str, int]) -> list[tuple[str, list[int], list[float], float, str, int]]:
    """
    Worker function for multiprocessing that processes a single video directory.
    
    Returns:
        List of tuples containing (label, x_values, y_values, avg_value, classifier, tilesize)
        for each classifier/tile combination in the video.
    """
    dataset, video_cache_dir, idx = args_tuple
    results = []
    
    classifier_tile_dirs = list_classifier_tile_dirs(video_cache_dir)
    if not classifier_tile_dirs:
        return results
    
    for classifier_tile_dir in classifier_tile_dirs:
        label, xs, ys, avg, clf, tile = process_series_for_dir(
            dataset, video_cache_dir, classifier_tile_dir, idx
        )
        if label and ys:
            results.append((label, xs, ys, avg, clf, tile))
    
    return results


def main(args: argparse.Namespace) -> None:
    # Convert datasets to list if it's a single string
    datasets = args.datasets if isinstance(args.datasets, list) else [args.datasets]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*80}\n")
        
        video_dirs = list_video_dirs(dataset)

        if not video_dirs:
            print(f"No video directories found for dataset {dataset}")
            continue

        # Prepare worker arguments
        worker_args = [
            (dataset, video_cache_dir, idx)
            for idx, video_cache_dir in enumerate(video_dirs)
        ]

        # Process videos in parallel
        with Pool(processes=int(cpu_count() * 0.3)) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(process_video_worker, worker_args),
                total=len(worker_args),
                desc=f"Processing videos for {dataset}"
            ))

        # Create individual violin plots for each video
        base_dir, _each_dir = ensure_summary_dirs(dataset)
        
        for video_results in results:
            if not video_results:
                continue
                
            # Extract video name from the first result
            video_name = video_results[0][0].split('|')[0]  # Extract video name from label
            
            # Prepare data for this video (label, y_values, avg_value)
            video_data = []
            for label, _xs, ys, avg, _clf, _tile in video_results:
                video_data.append((label, ys, avg))
            
            # Create faceted histogram for this video
            plot_histogram_facets_per_video(video_name, video_data, dataset, base_dir)
    
    print(f"\n{'='*80}")
    print(f"Finished processing all datasets")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main(parse_args())


