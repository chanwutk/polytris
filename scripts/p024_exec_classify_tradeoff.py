#!/usr/local/bin/python

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track
import seaborn as sns
from typing import Any, Dict, List, Tuple
import multiprocessing as mp

from polyis.utilities import CACHE_DIR, DATA_DIR, load_classification_results, load_detection_results, mark_detections


TILE_SIZES = [30, 60]  #, 120]


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification visualization (default: 0.5)
    """
    parser = argparse.ArgumentParser(description='Compare accuracy-throughput tradeoff of classifiers')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    return parser.parse_args()


def evaluate_classification_accuracy(classifications: np.ndarray,
                                     detections: list[list[float]], tile_size: int,
                                     threshold: float) -> dict[str, Any]:
    """
    Evaluate classification accuracy by comparing predictions with groundtruth detections.

    Args:
        classifications (np.ndarray): 2D grid of classification scores
        detections (list[dict]): list of detection dictionaries
        tile_size (int): Size of each tile
        threshold (float): Classification threshold

    Returns:
        dict: dictionary containing evaluation metrics
    """
    grid_height = classifications.shape[0]
    grid_width = classifications.shape[1] if grid_height > 0 else 0

    # Initialize counters
    tp = 0  # True Positive: predicted above threshold, has detection overlap
    tn = 0  # True Negative: predicted below threshold, no detection overlap
    fp = 0  # False Positive: predicted above threshold, no detection overlap
    fn = 0  # False Negative: predicted below threshold, has detection overlap

    # Create a bitmap for all detections first
    # Calculate the total image dimensions based on grid and tile size
    total_height = grid_height * tile_size
    total_width = grid_width * tile_size

    detection_bitmap = mark_detections(detections, total_width, total_height,
                                       tile_size, slice(-4, None))

    # Extract tile regions from the detection bitmap and calculate overlap ratios
    for i in range(grid_height):
        for j in range(grid_width):
            score = classifications[i][j]

            # Determine prediction
            predicted_positive = score >= threshold
            actual_positive = detection_bitmap[i, j] > 0

            # Count metrics
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and actual_positive:
                fn += 1
            else:
                tn += 1

    # Calculate precision, recall, and accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1_score,
        'total_tiles': grid_height * grid_width
    }


def load_throughput_data(cache_dir: str, dataset: str, video_file: str, 
                        classifier: str, tile_size: int) -> float:
    """
    Load throughput data from score.jsonl file.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        video_file (str): Video file name
        classifier (str): Classifier name
        tile_size (int): Tile size used
        
    Returns:
        float: Average throughput in frames per second
    """
    score_file = os.path.join(cache_dir, dataset, video_file, 'relevancy', 
                             f'{classifier}_{tile_size}', 'score', 'score.jsonl')
    
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"Score file not found: {score_file}")
    
    total_runtime_ms = 0.0
    frame_count = 0
    
    with open(score_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                frame_count += 1
                
                # Sum up all runtime operations for this frame
                runtime_data = data.get('runtime', [])
                frame_runtime_ms = 0.0
                for op_data in runtime_data:
                    if isinstance(op_data, dict) and 'time' in op_data:
                        frame_runtime_ms += float(op_data['time'])
                
                total_runtime_ms += frame_runtime_ms
    
    if frame_count == 0:
        return 0.0
    
    # Calculate average throughput in frames per second
    avg_runtime_per_frame_ms = total_runtime_ms / frame_count
    throughput_fps = 1000.0 / avg_runtime_per_frame_ms if avg_runtime_per_frame_ms > 0 else 0.0
    
    return throughput_fps


def _evaluate_classifier_tile_worker(args):
    """
    Worker function to evaluate a single classifier-tile size combination for multiprocessing.
    
    Args:
        args: Tuple containing (video_file, dataset_name, classifier_name, tile_size, threshold)
    
    Returns:
        dict: Evaluation results for this classifier-tile combination
    """
    video_file, dataset_name, classifier_name, tile_size, threshold = args
    
    # Load classification results
    results = load_classification_results(CACHE_DIR, dataset_name, video_file, tile_size, classifier_name)
    
    # Load groundtruth detections for comparison
    groundtruth_detections = load_detection_results(CACHE_DIR, dataset_name, video_file, tracking=True)
    
    # Load throughput data
    throughput_fps = load_throughput_data(CACHE_DIR, dataset_name, video_file, classifier_name, tile_size)
    
    # Evaluate accuracy metrics for all frames
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    for frame_result, frame_detections in zip(results, groundtruth_detections):
        classifications = frame_result['classification_hex']
        classification_size = frame_result['classification_size']
        classifications = (np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
            .reshape(classification_size)
            .astype(np.float32) / 255.0)
        
        # Evaluate this frame
        frame_eval = evaluate_classification_accuracy(
            classifications, frame_detections['tracks'], tile_size, threshold
        )
        
        total_tp += frame_eval['tp']
        total_tn += frame_eval['tn']
        total_fp += frame_eval['fp']
        total_fn += frame_eval['fn']
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'video_file': video_file,
        'classifier': classifier_name,
        'tile_size': tile_size,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'throughput_fps': throughput_fps
    }


def create_tradeoff_visualizations(video_file: str, results: List[Dict], output_dir: str):
    """
    Create accuracy-throughput tradeoff visualizations.
    
    Args:
        video_file (str): Name of the video file
        results (List[Dict]): List of evaluation results for each classifier-tile combination
        output_dir (str): Directory to save visualizations
    """
    print(f"Creating tradeoff visualizations for {video_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    assert len(valid_results) > 0, f"No valid results for {video_file}"
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_results)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create 4 subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['precision', 'recall', 'accuracy', 'f1_score']
    titles = ['Precision vs Throughput', 'Recall vs Throughput', 'Accuracy vs Throughput', 'F1-Score vs Throughput']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Get unique classifiers and create color mapping
        unique_classifiers = df['classifier'].unique()
        colors = sns.color_palette('tab20', n_colors=len(unique_classifiers))
        color_map = dict(zip(unique_classifiers, colors))
        
        # Create scatter plot with lines connecting same classifier
        sns.scatterplot(data=df, x='throughput_fps', y=metric, 
                       hue='classifier', size='tile_size', 
                       sizes=(50, 200), alpha=0.7, ax=ax,
                       palette=color_map)
        
        # Add lines connecting points with same classifier using matching colors
        for classifier in df['classifier'].unique():
            classifier_data = df[df['classifier'] == classifier].sort_values('tile_size')  # type: ignore
            if len(classifier_data) > 1:
                ax.plot(classifier_data['throughput_fps'], classifier_data[metric], 
                       '--', alpha=0.5, linewidth=1, color=color_map[classifier])
        
        ax.set_xlabel('Throughput (FPS)')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{title} - {video_file}')
        ax.grid(True, alpha=0.3)
        
        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'tradeoff_analysis_{video_file}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved tradeoff visualizations to: {plot_path}")


def _process_video_worker(args):
    """
    Worker function to process a single video for multiprocessing.
    
    Args:
        args: Tuple containing (video_file, dataset_name, tile_sizes, threshold)
    
    Returns:
        str: Status message about processing completion
    """
    video_file, dataset_name, tile_sizes, threshold = args
    
    # Get all classifier-tile combinations for this video
    relevancy_dir = os.path.join(CACHE_DIR, dataset_name, video_file, 'relevancy')
    if not os.path.exists(relevancy_dir):
        return f"Skipping {video_file}: No relevancy directory found"
    
    classifier_tilesizes: List[Tuple[str, int]] = []
    for file in os.listdir(relevancy_dir):
        if '_' in file:
            classifier_name = file.split('_')[0]
            tile_size = int(file.split('_')[1])
            if classifier_name != 'groundtruth' and tile_size in tile_sizes:
                classifier_tilesizes.append((classifier_name, tile_size))
    
    if not classifier_tilesizes:
        return f"Skipping {video_file}: No valid classifier tile sizes found"
    
    # Prepare arguments for parallel processing
    worker_args = [(video_file, dataset_name, classifier_name, tile_size, threshold)
                   for classifier_name, tile_size in classifier_tilesizes]
    
    # Use multiprocessing to evaluate classifier-tile combinations in parallel
    num_processes = min(mp.cpu_count() - 1, len(worker_args))
    with mp.Pool(processes=num_processes) as pool:
        results = list(track(
            pool.imap(_evaluate_classifier_tile_worker, worker_args),
            total=len(worker_args),
            description=f"Evaluating classifier-tile combinations"
        ))
    
    # Create output directory for this video
    output_dir = os.path.join(CACHE_DIR, dataset_name, video_file, 'tradeoff_analysis')
    
    # Create visualizations
    create_tradeoff_visualizations(video_file, results, output_dir)
    
    return f"Completed tradeoff analysis for {video_file} with {len(classifier_tilesizes)} combinations"


def main(args):
    """
    Main function that orchestrates the accuracy-throughput tradeoff analysis.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)
    """
    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")

    print(f"Using threshold: {args.threshold}")

    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    assert len(video_files) > 0, f"No video files found in {dataset_dir}"

    # Prepare arguments for parallel processing
    worker_args = [(video_file, args.dataset, tile_sizes_to_process, args.threshold)
                   for video_file in sorted(video_files)]

    # # Process videos in parallel
    # with mp.Pool(processes=num_processes) as pool:
    #     results = list(pool.imap(_process_video_worker, worker_args))
    results = list(map(_process_video_worker, worker_args))

    # Print results
    print("\n=== Processing Results ===")
    for result in results:
        print(result)


if __name__ == '__main__':
    main(parse_args())
