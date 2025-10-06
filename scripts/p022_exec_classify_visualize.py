#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
from typing import Any, Callable
import multiprocessing as mp
from functools import partial

from polyis.utilities import CACHE_DIR, DATA_DIR, load_classification_results, load_detection_results, mark_detections, ProgressBar


TILE_SIZES = [60] # 30, 120


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification visualization (default: 0.5)
    """
    parser = argparse.ArgumentParser(description='Visualize video tile classification results')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Specify a filter to limit analysis to specific runs (e.g., "none", "neighbor").')
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
        dict: dictionary containing evaluation metrics and error details
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

    # Vectorized operations for much better performance
    # Flatten arrays for easier processing
    classification_scores = classifications.flatten()
    actual_positives = (detection_bitmap > 0).flatten()
    
    # Create boolean masks for predictions
    predicted_positives = classification_scores >= threshold
    
    # Calculate confusion matrix components using vectorized operations
    tp = np.sum(predicted_positives & actual_positives)
    fp = np.sum(predicted_positives & ~actual_positives)
    fn = np.sum(~predicted_positives & actual_positives)
    tn = np.sum(~predicted_positives & ~actual_positives)
    
    # Create error map using vectorized operations
    error_map = np.zeros((grid_height, grid_width), dtype=int)
    
    # Reshape predictions back to 2D for error mapping
    predicted_positives_2d = predicted_positives.reshape(grid_height, grid_width)
    actual_positives_2d = actual_positives.reshape(grid_height, grid_width)
    
    # False positives: predicted positive but actual negative
    error_map[predicted_positives_2d & ~actual_positives_2d] = 1
    
    # False negatives: predicted negative but actual positive  
    error_map[~predicted_positives_2d & actual_positives_2d] = 2

    # Calculate precision, recall, and accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1_score,
        'error_map': error_map,
        # 'overlap_ratios': overlap_ratios,
        'classification_scores': classification_scores,
        'actual_positives': actual_positives,
        'total_tiles': grid_height * grid_width
    }


def _evaluate_frame_worker(args):
    """
    Worker function to evaluate a single frame for multiprocessing.

    Args:
        args: Tuple containing (frame_result, frame_detections, tile_size, threshold)

    Returns:
        dict: Frame evaluation results
    """
    frame_result, frame_detections, tile_size, threshold = args

    # Validate frame data
    assert 'tracks' in frame_detections, f"tracks not in frame_detections: {frame_detections}"

    classifications = frame_result['classification_hex']
    classification_size = frame_result['classification_size']
    classifications = (np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
        .reshape(classification_size)
        .astype(np.float32) / 255.0)

    # Evaluate this frame
    frame_eval = evaluate_classification_accuracy(
        classifications, frame_detections['tracks'], tile_size, threshold
    )

    return frame_eval       


def create_statistics_visualizations(video_file: str, results: list[dict],
                                     groundtruth_detections: list[dict],
                                     tile_size: int, threshold: float, output_dir: str,
                                     gpu_id: int, command_queue: mp.Queue):
    """
    Create comprehensive statistics visualizations comparing classification results with groundtruth.

    Args:
        video_file (str): Name of the video file
        results (list[dict]): Classification results
        groundtruth_detections (list[list[dict]]): Groundtruth detections per frame
        tile_size (int): Tile size used for classification
        threshold (float): Classification threshold
        output_dir (str): Directory to save visualizations
        gpu_id (int): GPU ID (unused but required for ProgressBar compatibility)
        command_queue (mp.Queue): Queue for progress updates
    """
    # print(f"Creating statistics visualizations for {video_file}")

    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect frame-by-frame metrics
    frame_metrics = []
    all_actual_positives = []
    all_classification_scores = []
    all_error_counts = []
    pruned_tile_props = [frame_result['pruned_tiles_prop'] for frame_result in results if 'pruned_tiles_prop' in frame_result]


    # print(f"Evaluating {len(results)} frames using multiprocessing")

    # Prepare arguments for parallel processing
    # Validate frame indices first
    for frame_idx, frame_result in enumerate(results):
        assert frame_idx == frame_result['frame_idx'], \
            f"frame_idx mismatch: {frame_idx} != {frame_result['frame_idx']}"

    # # Create arguments for worker function
    # worker_args = [(frame_result, frame_detections, tile_size, threshold)
    #                for frame_result, frame_detections in zip(results, groundtruth_detections)]

    # # Use multiprocessing to evaluate frames in parallel
    # num_processes = min(mp.cpu_count() - 1, len(results))
    # with mp.Pool(processes=num_processes) as pool:
    #     frame_evals = list(
    #         # track(
    #             pool.imap(_evaluate_frame_worker, worker_args),
    #         #     total=len(results),
    #         #     description=f"Evaluating frames {idx + 1}"
    #         # )
    #     )
    frame_evals: list[dict] = []
    command_queue.put((f'cuda:{gpu_id}', { 'completed': 0, 'total': len(results) }))
    mod = int(len(results) * 0.05)
    for frame_idx, (frame_result, frame_detections) in enumerate(zip(results, groundtruth_detections)):
        frame_eval = _evaluate_frame_worker((frame_result, frame_detections, tile_size, threshold))
        frame_evals.append(frame_eval)
        if frame_idx % mod == 0:
            command_queue.put((f'cuda:{gpu_id}', { 'completed': frame_idx + 1 }))

    # Collect results from parallel processing
    for frame_eval in frame_evals:
        frame_metrics.append(frame_eval)
        all_actual_positives.extend(frame_eval['actual_positives'])
        all_classification_scores.extend(frame_eval['classification_scores'])
        all_error_counts.append(frame_eval['error_map'])

    # Aggregate overall metrics
    total_tp = sum(m['tp'] for m in frame_metrics)
    total_tn = sum(m['tn'] for m in frame_metrics)
    total_fp = sum(m['fp'] for m in frame_metrics)
    total_fn = sum(m['fn'] for m in frame_metrics)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Save overall metrics to a JSON file for easy parsing
    summary_metrics = {
        'video_file': video_file,
        # Classifier and filter are part of the directory name, e.g., MobileNetS_60_neighbor
        'classifier': '_'.join(os.path.basename(os.path.dirname(os.path.dirname(output_dir))).split('_')[:-2]),
        'filter': os.path.basename(os.path.dirname(os.path.dirname(output_dir))).split('_')[-1],
        'tile_size': tile_size,
        'threshold': threshold,
        'total_tp': int(total_tp),
        'total_tn': int(total_tn),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'overall_f1': float(overall_f1),
        'avg_pruned_tiles_prop': float(np.mean(pruned_tile_props)) if pruned_tile_props else 0.0,
    }
    summary_metrics_path = os.path.join(output_dir, 'summary_metrics.json')
    with open(summary_metrics_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)

    # Create individual visualizations
    visualize_error_summary(
        total_tp, total_tn, total_fp, total_fn,
        overall_precision, overall_recall, overall_accuracy, overall_f1,
        tile_size, output_dir
    )
    
    visualize_error_over_time(
        frame_metrics, groundtruth_detections,
        overall_precision, overall_recall, overall_f1,
        tile_size, output_dir
    )
    
    visualize_spatial_misclassification(
        all_error_counts, tile_size, output_dir
    )
    
    # visualize_score_distribution(
    #     all_classification_scores, all_actual_positives,
    #     threshold, tile_size, output_dir
    # )
    visualize_pruned_tile_distribution(
        pruned_tile_props, tile_size, output_dir
    )

    # print(f"Saved statistics visualizations to: {output_dir}")
    # print(f"Overall Metrics - Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}, F1: {overall_f1:.3f}")


def visualize_error_summary(total_tp: int, total_tn: int, total_fp: int, total_fn: int,
                            overall_precision: float, overall_recall: float, 
                            overall_accuracy: float, overall_f1: float,
                            tile_size: int, output_dir: str) -> str:
    """
    Create overall classification error summary visualization with stacked bar charts and metrics.
    
    Args:
        total_tp (int): Total true positives
        total_tn (int): Total true negatives
        total_fp (int): Total false positives
        total_fn (int): Total false negatives
        overall_precision (float): Overall precision score
        overall_recall (float): Overall recall score
        overall_accuracy (float): Overall accuracy score
        overall_f1 (float): Overall F1 score
        tile_size (int): Tile size used for classification
        output_dir (str): Directory to save visualization
        
    Returns:
        str: Path to saved visualization file
    """
    fig, ((ax1, ax3, ax2)) = plt.subplots(1, 3, figsize=(15, 6))

    # First stacked bar chart: x-axis is Predicted, color is Actual
    x_labels = ['Predicted Negative', 'Predicted Positive']
    actual_negative_values = [total_tn, total_fp]  # TN, FP
    actual_positive_values = [total_fn, total_tp]  # FN, TP

    bars1 = ax1.bar(x_labels, actual_negative_values, label='Actual Negative', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x_labels, actual_positive_values, bottom=actual_negative_values, label='Actual Positive', color='lightgreen', alpha=0.8)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Label for Actual Negative (bottom)
        if actual_negative_values[i] > 0:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2,
                    str(actual_negative_values[i]), ha='center', va='center', fontweight='bold')

        # Label for Actual Positive (top)
        if actual_positive_values[i] > 0:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_y() + bar2.get_height()/2,
                    str(actual_positive_values[i]), ha='center', va='center', fontweight='bold')

    ax1.set_ylabel('Count')
    ax1.set_title(f'Classification Results by Prediction (Tile Size: {tile_size})')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Metrics bar chart
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    values = [overall_precision, overall_recall, overall_accuracy, overall_f1]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Score')
    ax2.set_title(f'Overall Classification Metrics (Tile Size: {tile_size})')
    ax2.set_ylim(0, 1)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    # Second stacked bar chart: x-axis is Actual, color is Predicted
    x_labels2 = ['Actual Negative', 'Actual Positive']
    predicted_negative_values = [total_tn, total_fn]  # TN, FN
    predicted_positive_values = [total_fp, total_tp]  # FP, TP

    bars3 = ax3.bar(x_labels2, predicted_negative_values, label='Predicted Negative', color='lightcoral', alpha=0.8)
    bars4 = ax3.bar(x_labels2, predicted_positive_values, bottom=predicted_negative_values, label='Predicted Positive', color='lightgreen', alpha=0.8)

    # Add value labels on bars
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        # Label for Predicted Negative (bottom)
        if predicted_negative_values[i] > 0:
            ax3.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height()/2,
                    str(predicted_negative_values[i]), ha='center', va='center', fontweight='bold')

        # Label for Predicted Positive (top)
        if predicted_positive_values[i] > 0:
            ax3.text(bar4.get_x() + bar4.get_width()/2, bar4.get_y() + bar4.get_height()/2,
                    str(predicted_positive_values[i]), ha='center', va='center', fontweight='bold')

    ax3.set_ylabel('Count')
    ax3.set_title(f'Classification Results by Actual (Tile Size: {tile_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    overall_summary_path = os.path.join(output_dir, f'010_overall_summary_tile{tile_size}.png')
    plt.savefig(overall_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return overall_summary_path


def visualize_error_over_time(frame_metrics: list[dict], groundtruth_detections: list[dict],
                              overall_precision: float, overall_recall: float, overall_f1: float,
                              tile_size: int, output_dir: str) -> str:
    """
    Create classification error over time visualization with 4 subplots.
    
    Args:
        frame_metrics (list[dict]): Frame-by-frame evaluation metrics
        groundtruth_detections (list[dict]): Groundtruth detections per frame
        overall_precision (float): Overall precision score
        overall_recall (float): Overall recall score
        overall_f1 (float): Overall F1 score
        tile_size (int): Tile size used for classification
        output_dir (str): Directory to save visualization
        
    Returns:
        str: Path to saved visualization file
    """
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(25, 12))

    frame_indices = list(range(len(frame_metrics)))
    error_rates = [(m['fp'] + m['fn']) / m['total_tiles'] for m in frame_metrics]
    precision_rates = [m['precision'] for m in frame_metrics]
    recall_rates = [m['recall'] for m in frame_metrics]
    f1_scores = [m['f1_score'] for m in frame_metrics]

    # Calculate number of objects per frame (detections with overlap > 0)
    objects_per_frame = []
    for frame_detection in groundtruth_detections:
        assert 'tracks' in frame_detection, f"Frame detection does not contain tracks: {frame_detection}"
        # Count detections that have any overlap with tiles
        object_count = len([det for det in frame_detection['tracks'] if len(det) == 5])
        objects_per_frame.append(object_count)

    # Get number of tiles per frame and metrics data
    num_tiles_per_frame = frame_metrics[0]['total_tiles']
    tp_counts = [m['tp'] for m in frame_metrics]
    tn_counts = [m['tn'] for m in frame_metrics]
    fp_counts = [m['fp'] for m in frame_metrics]
    fn_counts = [m['fn'] for m in frame_metrics]

    # First subplot: Error rate and F1 over time
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(frame_indices, error_rates, 'r-', linewidth=2, label='Error Rate')
    line2 = ax1.plot(frame_indices, f1_scores, 'orange', linewidth=2, label='F1-Score')
    mean_error = float(np.mean(error_rates))
    ax1.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean Error: {mean_error:.3f}')
    ax1.axhline(y=float(overall_f1), color='orange', linestyle='--', alpha=0.7, label=f'Overall F1: {overall_f1:.3f}')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Rate/Score')
    ax1.set_title(f'Error Rate and F1-Score Over Time (Tile Size: {tile_size})')
    ax1.grid(True, alpha=0.3)

    # Object count on secondary y-axis (only for the first subplot)
    line3 = ax1_twin.plot(frame_indices, objects_per_frame, 'purple', linewidth=2, label='Object Count', alpha=0.7)
    ax1_twin.set_ylabel('Object Count', color='purple')
    ax1_twin.tick_params(axis='y', labelcolor='purple')

    # Combine legends
    lines = line1 + line2 + line3
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Second subplot: Precision and Recall over time (no object count)
    line4 = ax2.plot(frame_indices, precision_rates, 'g-', linewidth=2, label='Precision')
    line5 = ax2.plot(frame_indices, recall_rates, 'b-', linewidth=2, label='Recall')
    ax2.axhline(y=float(overall_precision), color='green', linestyle='--', alpha=0.7, label=f'Overall Precision: {overall_precision:.3f}')
    ax2.axhline(y=float(overall_recall), color='blue', linestyle='--', alpha=0.7, label=f'Overall Recall: {overall_recall:.3f}')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Precision and Recall Over Time (Tile Size: {tile_size})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Third subplot: True Positive and True Negative over time (no object count)
    line7 = ax3.plot(frame_indices, tp_counts, 'g-', linewidth=2, label='True Positives')
    line8 = ax3.plot(frame_indices, tn_counts, 'b-', linewidth=2, label='True Negatives')
    ax3.set_yticklabels([f'{int(v)}\n({v * 100 / num_tiles_per_frame:.1f}%)' for v in ax3.get_yticks()])
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Count (% of Tiles)')
    ax3.set_title(f'True Positives and Negatives Over Time (Tile Size: {tile_size})')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # Fourth subplot: False Positive and False Negative over time (no object count)
    line10 = ax4.plot(frame_indices, fp_counts, 'r-', linewidth=2, label='False Positives')
    line11 = ax4.plot(frame_indices, fn_counts, 'orange', linewidth=2, label='False Negatives')
    ax4.set_yticklabels([f'{int(v)}\n({v * 100 / num_tiles_per_frame:.1f}%)' for v in ax4.get_yticks()])
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Count (% of Tiles)')
    ax4.set_title(f'False Positives and Negatives Over Time (Tile Size: {tile_size})')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    plt.tight_layout()
    time_series_path = os.path.join(output_dir, f'020_time_series_tile{tile_size}.png')
    plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return time_series_path


def visualize_spatial_misclassification(all_error_counts: list[np.ndarray], tile_size: int, output_dir: str) -> str:
    """
    Create heatmap visualization of error count for each tile.
    
    Args:
        all_error_counts (list[np.ndarray]): Error maps from all frames
        tile_size (int): Tile size used for classification
        output_dir (str): Directory to save visualization
        
    Returns:
        str: Path to saved visualization file
    """
    # Aggregate error maps across all frames
    grid_height = len(all_error_counts[0])
    grid_width = len(all_error_counts[0][0])
    aggregated_error_map = np.zeros((grid_height, grid_width), dtype=int)

    for error_map in all_error_counts:
        aggregated_error_map += error_map

    plt.figure(figsize=(12, 8))
    # Use matplotlib heatmap
    plt.imshow(aggregated_error_map, cmap='Reds', interpolation='nearest')
    for i in range(aggregated_error_map.shape[0]):
        for j in range(aggregated_error_map.shape[1]):
            if aggregated_error_map[i, j] == 0:
                plt.text(j, i, str(aggregated_error_map[i, j]),
                         ha='center', va='center', color='black', fontweight='bold')
    plt.colorbar(label='Error Count')

    plt.title(f'Cumulative Error Count per Tile (Tile Size: {tile_size})\nRed: False Positive, Orange: False Negative')
    plt.xlabel('Tile X Position')
    plt.ylabel('Tile Y Position')

    error_heatmap_path = os.path.join(output_dir, f'030_error_heatmap_tile{tile_size}.png')
    plt.savefig(error_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return error_heatmap_path


def visualize_score_distribution(all_classification_scores: list[float], all_actual_positives: list[bool],
                                 threshold: float, tile_size: int, output_dir: str) -> str:
    """
    Create histograms for classification score distribution split by correctness.
    
    Args:
        all_classification_scores (list[float]): All classification scores
        all_actual_positives (list[bool]): All actual positive labels
        threshold (float): Classification threshold
        tile_size (int): Tile size used for classification
        output_dir (str): Directory to save visualization
        
    Returns:
        str: Path to saved visualization file
    """
    # Separate data for correct and incorrect predictions
    correct_scores = []
    incorrect_scores = []
    actual_positive_scores = []
    actual_negative_scores = []

    for score, actual_positive in zip(all_classification_scores, all_actual_positives):
        predicted_positive = score >= threshold

        # Count metrics
        if predicted_positive == actual_positive:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)

        # Count actual positive/negative scores
        if actual_positive:
            actual_positive_scores.append(score)
        else:
            actual_negative_scores.append(score)

    # Create histograms for correct and incorrect predictions, plus actual positive/negative
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Create twin axes for dual y-axis functionality
    twin_axes = []
    for ax in axes:
        twin_ax = ax.twinx()
        twin_axes.append(twin_ax)
    
    # Define consistent colors for original axes (left y-axis) and synchronized axes (right y-axis)
    original_colors = '#4682B4'  # SteelBlue
    sync_colors = '#FFD700'     # Yellow
    sync_colors_edge = '#B8860B'  # Darker yellow for edge color
    
    # Define plot configurations as tuples: (scores, title)
    plot_configs = [
        (correct_scores, 'Correct Predictions'),
        (incorrect_scores, 'Incorrect Predictions'),
        (actual_positive_scores, 'Actual Positive Scores'),
        (actual_negative_scores, 'Actual Negative Scores')
    ]
    
    # Create plots for each configuration
    for ax, twin_ax, (scores, title) in zip(axes, twin_axes, plot_configs):
        if len(scores) > 0:
            # Plot on original axis with original color
            ax.hist(scores, bins=100, alpha=0.7, color=original_colors, edgecolor=original_colors)
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
            ax.set_xlabel('Classification Score')
            ax.set_ylabel('Count', color=original_colors)
            ax.tick_params(axis='y', labelcolor=original_colors)
            ax.set_title(f'{title} (Tile Size: {tile_size})\nTotal: {len(scores):,}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot on twin axis with synchronized color
            twin_ax.hist(scores, bins=100, color=sync_colors, edgecolor=sync_colors_edge)
            twin_ax.set_ylabel('Count (Synchronized)', color=sync_colors)
            twin_ax.tick_params(axis='y', labelcolor=sync_colors)
        else:
            ax.text(0.5, 0.5, f'No {title.lower()}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title} (Tile Size: {tile_size})')

    # Sync y-axes across all subplots for synchronized axes (right y-axis)
    y_min_sync = float('inf')
    y_max_sync = float('-inf')
    
    for twin_ax in twin_axes:
        if twin_ax.get_children():  # Check if twin subplot has content
            y_min_sync = min(y_min_sync, twin_ax.get_ylim()[0])
            y_max_sync = max(y_max_sync, twin_ax.get_ylim()[1])
    
    # Set the same y-limits for all synchronized axes (right y-axis)
    for twin_ax in twin_axes:
        twin_ax.set_ylim(y_min_sync, y_max_sync)

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, f'040_histogram_scores_tile{tile_size}.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return histogram_path

def visualize_pruned_tile_distribution(pruned_tile_props: list[float], tile_size: int, output_dir: str) -> str:
    """
    Visualize the distribution of pruned tiles across all frames for a given classifier/tile size.

    Args:
        pruned_tile_props (list[float]): A list containing the proportion (0.0 to 1.0)
                                         of pruned tiles for each frame.
        tile_size (int): Tile size used for classification
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to saved visualization file
    """
    if len(pruned_tile_props) == 0:
        print(f"No pruned tile data found for tile size {tile_size}")
        return ""

    plt.figure(figsize=(12, 6))
    plt.hist(pruned_tile_props, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Proportion of Pruned Tiles per Frame")
    plt.ylabel("Frame Count")
    plt.title(f"Distribution of Pruned Tiles per Frame (Tile Size: {tile_size})")
    plt.grid(axis='y', alpha=0.3)

    prune_hist_path = os.path.join(output_dir, f'050_pruned_tile_distribution_tile{tile_size}.png')
    plt.tight_layout()
    plt.savefig(prune_hist_path, dpi=300, bbox_inches='tight')
    plt.close()

    return prune_hist_path


def _process_classifier_tile_worker(video_file: str, dataset_name: str, classifier_name: str, 
                                   tile_size: int, threshold: float, gpu_id: int, command_queue: mp.Queue):
    """
    Worker function to process a single classifier-tile size combination for multiprocessing.
    
    Args:
        video_file (str): Name of the video file
        dataset_name (str): Name of the dataset
        classifier_name (str): Name of the classifier
        tile_size (int): Tile size to use
        threshold (float): Classification threshold
        gpu_id (int): GPU ID (unused but required for ProgressBar compatibility)
        command_queue (mp.Queue): Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    
    # Send initial progress update
    command_queue.put((device, {
        'description': f"{video_file} {tile_size:>3} {classifier_name}",
        'completed': 0,
        'total': 1
    }))

    # Load classification results
    results = load_classification_results(CACHE_DIR, dataset_name, video_file, tile_size, classifier_name)
    
    # Load groundtruth detections for comparison
    groundtruth_detections = load_detection_results(CACHE_DIR, dataset_name, video_file, tracking=True)
    
    # Create output directory for statistics visualizations
    stats_output_dir = os.path.join(CACHE_DIR, dataset_name, video_file, 'relevancy', f'{classifier_name}_{tile_size}', 'statistics')
    
    # Create statistics visualizations
    create_statistics_visualizations(
        video_file, results, groundtruth_detections,
        tile_size, threshold, stats_output_dir, gpu_id, command_queue
    )


def main(args):
    """
    Main function that orchestrates the video tile classification visualization process.

    This function serves as the entry point for the script. It: 1. Validates the dataset directory exists
    2. Iterates through all videos in the dataset directory
    3. For each video, loads the classification results for the specified tile size(s)
    4. Creates visualizations showing tile classifications and statistics for each tile size

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)

         Note:
         - The script expects classification results from 020_exec_classify.py in:
           {CACHE_DIR}/{dataset}/{video_file}/relevancy/score/proxy_{tile_size}/score.jsonl
         - Looks for score.jsonl files
         - Videos are read from {DATA_DIR}/{dataset}/
         - Visualizations are saved to {CACHE_DIR}/{dataset}/{video_file}/relevancy/proxy_{tile_size}/statistics/
         - Summary statistics and plots are also generated
    """
    mp.set_start_method('spawn', force=True)
    
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

    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return

    print(f"Found {len(video_files)} video files to process")

    # Collect all video-classifier-tile combinations for parallel processing
    all_tasks = []
    
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)
        
        # Get classifier tile sizes for this video
        relevancy_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'relevancy')
        if not os.path.exists(relevancy_dir):
            print(f"Skipping {video_file}: No relevancy directory found")
            continue
            
        classifier_tilesizes: list[tuple[str, int]] = []
        for file in os.listdir(relevancy_dir):
            if '_' in file:
                parts = file.split('_')
                if len(parts) >= 3: # Expects classifier_tilesize_filter
                    filter_type = parts[-1]
                    tile_size = int(parts[-2])
                    classifier_name = '_'.join(parts[:-2])
                    # If a filter is specified via args, only process matching directories
                    if args.filter is None or args.filter == filter_type:
                        classifier_tilesizes.append((classifier_name, tile_size, filter_type))
        
        classifier_tilesizes = sorted(list(set(classifier_tilesizes)))
        
        if not classifier_tilesizes:
            print(f"Skipping {video_file}: No classifier tile sizes found")
            continue
            
        print(f"Found {len(classifier_tilesizes)} classifier tile sizes for {video_file}: {classifier_tilesizes}")
        
        # Add tasks for each classifier-tile size combination
        for classifier_name, tile_size, filter_type in classifier_tilesizes:
            task_args = (video_file, args.dataset, classifier_name, tile_size, args.threshold, filter_type)
            all_tasks.append(task_args)

    if not all_tasks:
        print("No tasks to process")
        return

    print(f"Processing {len(all_tasks)} tasks in parallel...")

    # Create worker functions for ProgressBar
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for video_file, dataset_name, classifier_name, tile_size, threshold, filter_type in all_tasks:
        funcs.append(partial(_process_classifier_tile_worker, video_file, dataset_name, 
                            classifier_name, tile_size, threshold, filter_type))
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.5)
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Run all tasks with ProgressBar
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


def _process_classifier_tile_worker(video_file: str, dataset_name: str, classifier_name: str, 
                                   tile_size: int, threshold: float, filter_type: str, gpu_id: int, command_queue: mp.Queue):
    """
    Worker function to process a single classifier-tile size combination for multiprocessing.
    """
    device = f'cuda:{gpu_id}'
    
    # Send initial progress update
    command_queue.put((device, {
        'description': f"{video_file} {tile_size:>3} {classifier_name} ({filter_type})",
        'completed': 0,
        'total': 1
    }))

    # Load classification results
    results = load_classification_results(CACHE_DIR, dataset_name, video_file, tile_size, classifier_name, filter_type)
    groundtruth_detections = load_detection_results(CACHE_DIR, dataset_name, video_file, tracking=True)
    
    # Create output directory for statistics visualizations
    stats_output_dir = os.path.join(CACHE_DIR, dataset_name, video_file, 'relevancy', f'{classifier_name}_{tile_size}_{filter_type}', 'statistics')
    
    create_statistics_visualizations(video_file, results, groundtruth_detections, tile_size, threshold, stats_output_dir, gpu_id, command_queue)

if __name__ == '__main__':
    main(parse_args())
