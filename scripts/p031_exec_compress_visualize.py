#!/usr/bin/env python3

import argparse
import os
import re
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from polyis.utilities import CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize compression empty-space ratios over time by reading "
            "index_maps saved by 030_exec_compress.py."
        )
    )
    parser.add_argument(
        "--dataset",
        required=False,
        default="b3d",
        help="Dataset name (matches directory in DATA_DIR and CACHE_DIR)",
    )
    return parser.parse_args()


def list_video_dirs(dataset: str) -> list[str]:
    dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
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
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, label="Content ratio", color="#1f77b4")
    plt.axhline(avg_value, color="#ff7f0e", linestyle="--", label=f"Average: {avg_value:.3f}")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Frame index (end of packed span)")
    plt.ylabel("Content ratio (index_map[:,:,0] > 0)")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.8)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def plot_violin(labels: list[str], datasets: list[list[float]], title: str, output_png_path: str) -> None:
    if not datasets:
        return
    plt.figure(figsize=(12, max(10, len(labels) * 0.5)))
    y = [[*filter(lambda x: x < 1, d)] for d in datasets]
    x = list(range(1, len(labels) + 1))

    x = [_x for i, _x in enumerate(x) if len(y[i]) > 0]
    labels = [_label for i, _label in enumerate(labels) if len(y[i]) > 0]
    y = [_y for _y in y if len(_y) > 0]

    parts = plt.violinplot(y, positions=x, showmedians=True, vert=False)
    plt.yticks(ticks=x, labels=labels)
    plt.xlabel("Content ratio")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, axis='x')
    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def plot_violin_per_video(video_name: str, video_data: list[tuple[str, list[float], float]], 
                         dataset: str, output_dir: str) -> None:
    """
    Create a violin plot for a single video, sorted by average content ratio.
    
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
    
    labels = [item[0] for item in video_data]
    datasets = [item[1] for item in video_data]
    
    # Filter out empty datasets and values >= 1
    filtered_data = []
    filtered_labels = []
    for i, (label, data) in enumerate(zip(labels, datasets)):
        filtered_data_points = [x for x in data if x < 1]
        if filtered_data_points:
            filtered_data.append(filtered_data_points)
            filtered_labels.append(label)
    
    if not filtered_data:
        print(f"No valid data for video {video_name}")
        return
    
    plt.figure(figsize=(12, max(8, len(filtered_labels) * 0.6)))
    x = list(range(1, len(filtered_labels) + 1))
    
    parts = plt.violinplot(filtered_data, positions=x, showmedians=True, vert=False)
    plt.yticks(ticks=x, labels=filtered_labels)
    plt.xlabel("Content ratio")
    plt.title(f"Content ratios by classifier/tile - {video_name} ({dataset})")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, axis='x')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{video_name}_compress_content_ratio_violin.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot for {video_name}: {plot_path}")


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
    dataset = args.dataset
    video_dirs = list_video_dirs(dataset)

    if not video_dirs:
        print(f"No video directories found for dataset {dataset}")
        return

    # Prepare worker arguments
    worker_args = [
        (dataset, video_cache_dir, idx)
        for idx, video_cache_dir in enumerate(video_dirs)
    ]

    # Process videos in parallel
    with Pool(processes=cpu_count()) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_video_worker, worker_args),
            total=len(worker_args),
            desc="Processing videos"
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
        
        # Create violin plot for this video
        plot_violin_per_video(video_name, video_data, dataset, base_dir)


if __name__ == "__main__":
    main(parse_args())


