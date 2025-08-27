#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any
import multiprocessing as mp

from scripts.utilities import CACHE_DIR, DATA_DIR, load_classification_results, load_detection_results


TILE_SIZES = [32, 64, 128]


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 32, 64, 128, 'all')
            - threshold (float): Threshold for classification visualization (default: 0.5)
            - groundtruth (bool): Whether to use groundtruth scores (score_correct.jsonl) instead of model scores (score.jsonl)
            - statistics (bool): Whether to compare classification results with groundtruth and generate statistics
    """
    parser = argparse.ArgumentParser(description='Visualize video tile classification results')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['32', '64', '128', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    parser.add_argument('--groundtruth', action='store_true',
                        help='Use groundtruth scores (score_correct.jsonl) instead of model scores (score.jsonl)')
    parser.add_argument('--statistics', action='store_true',
                        help='Compare classification results with groundtruth and generate statistics (no video output, ignores --groundtruth flag)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use for parallel processing (default: number of CPU cores)')
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
    # detection_bitmap = np.zeros((total_height, total_width), dtype=np.uint32)
    detection_bitmap = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Mark all detections on the bitmap
    for detection in detections:
        if len(detection) != 5:
            continue

        # bbox format: [track_id, x1, y1, x2, y2]
        _track_id, det_x1, det_y1, det_x2, det_y2 = detection

        # Convert to integer coordinates and ensure they're within bitmap bounds
        det_x1 = int(max(0, det_x1) // tile_size)
        det_y1 = int(max(0, det_y1) // tile_size)
        det_x2 = int(min(total_width - 1, det_x2) // tile_size)
        det_y2 = int(min(total_height - 1, det_y2) // tile_size)

        assert det_x2 >= det_x1 and det_y2 >= det_y1, f"Invalid detection: {detection}"
        detection_bitmap[det_y1:det_y2+1, det_x1:det_x2+1] = 1

    error_map = np.zeros((grid_height, grid_width), dtype=int)
    actual_positives = []
    classification_scores = []

    # Extract tile regions from the detection bitmap and calculate overlap ratios
    for i in range(grid_height):
        for j in range(grid_width):
            score = classifications[i][j]

            # Store data for scatter plot
            classification_scores.append(score)

            # Determine prediction
            predicted_positive = score >= threshold
            # actual_positive = overlap > 0.0
            actual_positive = detection_bitmap[i, j] > 0
            actual_positives.append(actual_positive)

            # Count metrics
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
                error_map[i, j] = 1  # False positive error
            elif not predicted_positive and actual_positive:
                fn += 1
                error_map[i, j] = 2  # False negative error
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
    classifications = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size).astype(np.float32) / 255.0

    # Evaluate this frame
    frame_eval = evaluate_classification_accuracy(
        classifications, frame_detections['tracks'], tile_size, threshold
    )

    return frame_eval


def create_statistics_visualizations(video_file: str, results: list[dict],
                                   groundtruth_detections: list[dict],
                                   tile_size: int, threshold: float,
                                   output_dir: str):
    """
    Create comprehensive statistics visualizations comparing classification results with groundtruth.

    Args:
        video_file (str): Name of the video file
        results (list[dict]): Classification results
        groundtruth_detections (list[list[dict]]): Groundtruth detections per frame
        tile_size (int): Tile size used for classification
        threshold (float): Classification threshold
        output_dir (str): Directory to save visualizations
    """
    print(f"Creating statistics visualizations for {video_file}")

    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect frame-by-frame metrics
    frame_metrics = []
    all_actual_positives = []
    all_classification_scores = []
    all_error_counts = []

    print(f"Evaluating {len(results)} frames using multiprocessing")

    # Prepare arguments for parallel processing
    # Validate frame indices first
    for frame_idx, (frame_result, frame_detections) in enumerate(zip(results, groundtruth_detections)):
        assert frame_idx == frame_result['frame_idx'], f"frame_idx mismatch: {frame_idx} != {frame_result['frame_idx']}"

    # Create arguments for worker function
    worker_args = [(frame_result, frame_detections, tile_size, threshold)
                   for frame_result, frame_detections in zip(results, groundtruth_detections)]

    # Use multiprocessing to evaluate frames in parallel
    num_processes = min(mp.cpu_count(), len(results))
    with mp.Pool(processes=num_processes) as pool:
        frame_evals = list(tqdm(
            pool.imap(_evaluate_frame_worker, worker_args),
            total=len(results),
            desc="Evaluating frames"
        ))

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

    # 1. Overall classification error summary
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

    # 2. Classification error over time - with 4 subplots
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

    # 3. Heatmap of error count for each tile
    # Aggregate error maps across all frames
    grid_height = len(frame_metrics[0]['error_map'])
    grid_width = len(frame_metrics[0]['error_map'][0])
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

    # 4. Histograms: Classification Score distribution (split by correctness)

    # Separate data for correct and incorrect predictions
    correct_scores = []
    incorrect_scores = []
    actual_positive_scores = []
    actual_negative_scores = []

    # for score, overlap in zip(all_classification_scores, all_overlap_ratios):
    #     predicted_positive = score >= threshold
    #     actual_positive = overlap > 0.0
    for score, actual_positive in zip(all_classification_scores, all_actual_positives):
        predicted_positive = score >= threshold
        if predicted_positive == actual_positive:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
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

    print(f"Saved statistics visualizations to: {output_dir}")
    print(f"Overall Metrics - Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}, F1: {overall_f1:.3f}")


def create_visualization_frame(frame: np.ndarray, classifications: np.ndarray,
                              tile_size: int, threshold: float) -> np.ndarray:
    """
    Create a visualization frame by adjusting tile brightness based on classification scores.

    Args:
        frame (np.ndarray): Original video frame (H, W, 3)
        classifications (np.ndarray): 2D grid of classification scores
        tile_size (int): Size of tiles used for classification
        threshold (float): Threshold value for visualization

    Returns:
        np.ndarray: Visualization frame with adjusted tile brightness
    """
    # Create a copy of the frame for visualization
    vis_frame = frame.copy().astype(np.float32)

    # Get grid dimensions
    grid_height = len(classifications)
    grid_width = len(classifications[0]) if grid_height > 0 else 0

    # Calculate frame dimensions after padding
    vis_height = grid_height * tile_size
    vis_width = grid_width * tile_size

    # Ensure frame is large enough (handle padding)
    if frame.shape[0] < vis_height or frame.shape[1] < vis_width:
        # Pad frame if necessary
        pad_height = max(0, vis_height - frame.shape[0])
        pad_width = max(0, vis_width - frame.shape[1])
        vis_frame = np.pad(vis_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    # Apply brightness adjustments to each tile
    for i in range(grid_height):
        for j in range(grid_width):
            # Get tile coordinates
            y_start = i * tile_size
            y_end = min(y_start + tile_size, vis_frame.shape[0])
            x_start = j * tile_size
            x_end = min(x_start + tile_size, vis_frame.shape[1])

            # Get classification score for this tile
            score = classifications[i][j]

            # Calculate brightness factor based on threshold
            if score < threshold:
                # Reduce brightness for tiles below threshold
                brightness_factor = 0.3 + (score / threshold) * 0.4  # Range: 0.3 to 0.7
            else:
                # Keep normal brightness for tiles above threshold
                brightness_factor = 1.0

            # Apply brightness adjustment
            vis_frame[y_start:y_end, x_start:x_end] *= brightness_factor

    # Clip values to valid range and convert back to uint8
    vis_frame = np.clip(vis_frame, 0, 255).astype(np.uint8)

    return vis_frame


def create_overlay_frame(frame: np.ndarray, classifications: np.ndarray,
                        tile_size: int, threshold: float) -> np.ndarray:
    """
    Create an overlay frame showing tile boundaries and classification scores.

    Args:
        frame (np.ndarray): Original video frame (H, W, 3)
        classifications (np.ndarray): 2D grid of classification scores
        tile_size (int): Size of tiles used for classification
        threshold (float): Threshold value for visualization

    Returns:
        np.ndarray: Overlay frame with tile boundaries and scores
    """
    # Create a copy of the frame for overlay
    overlay_frame = frame.copy()

    # Get grid dimensions
    grid_height = len(classifications)
    grid_width = len(classifications[0]) if grid_height > 0 else 0

    # Calculate frame dimensions after padding
    vis_height = grid_height * tile_size
    vis_width = grid_width * tile_size

    # Ensure frame is large enough (handle padding)
    if frame.shape[0] < vis_height or frame.shape[1] < vis_width:
        # Pad frame if necessary
        pad_height = max(0, vis_height - frame.shape[0])
        pad_width = max(0, vis_width - frame.shape[1])
        overlay_frame = np.pad(overlay_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    # Draw tile boundaries and add score text
    for i in range(grid_height):
        for j in range(grid_width):
            # Get tile coordinates
            y_start = i * tile_size
            y_end = min(y_start + tile_size, overlay_frame.shape[0])
            x_start = j * tile_size
            x_end = min(x_start + tile_size, overlay_frame.shape[1])

            # Get classification score for this tile
            score = classifications[i][j]

            # Determine color based on threshold
            if score < threshold:
                color = (0, 0, 255)  # Red for below threshold
            else:
                color = (0, 255, 0)  # Green for above threshold

            # Draw tile boundary
            cv2.rectangle(overlay_frame, (x_start, y_start), (x_end, y_end), color, 2)

            # Add score text (scaled for readability)
            score_text = f"{score:.2f}"
            font_scale = min(tile_size / 50.0, 0.8)  # Scale font based on tile size
            font_thickness = max(1, int(tile_size / 32))

            # Calculate text position (centered in tile)
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_start + (tile_size - text_size[0]) // 2
            text_y = y_start + (tile_size + text_size[1]) // 2

            # Draw text with background for better visibility
            cv2.putText(overlay_frame, score_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 1)
            cv2.putText(overlay_frame, score_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return overlay_frame


def save_visualization_frames(video_path: str, results: list, tile_size: int,
                            threshold: float, output_dir: str):
    """
    Save visualization frames for a video as a video file.

    Args:
        video_path (str): Path to the input video file
        results (list): list of classification results from load_classification_results
        tile_size (int): Tile size used for classification
        threshold (float): Threshold value for visualization
        output_dir (str): Directory to save visualization video

    Raises:
        ValueError: If the number of video frames doesn't match the length of results
    """
    print(f"Creating visualizations for video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get actual video frame count and properties
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results_frame_count = len(results)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate that video frame count matches results length
    if video_frame_count != results_frame_count:
        cap.release()
        raise ValueError(
            f"Frame count mismatch: Video has {video_frame_count} frames, "
            f"but results contain {results_frame_count} frames. "
            f"This suggests the classification results don't match the video file."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create video writer for brightness visualization
    brightness_video_path = os.path.join(output_dir, 'visualization.mp4')
    # Use MP4V codec for compatibility
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    brightness_writer = cv2.VideoWriter(brightness_video_path, fourcc, fps, (width, height))

    if not brightness_writer.isOpened():
        print(f"Error: Could not create video writer for {brightness_video_path}")
        cap.release()
        return

    print(f"Creating visualization video with {video_frame_count} frames at {fps} FPS")

    # Process all frames
    for frame_idx in tqdm(range(video_frame_count), desc="Creating visualization video"):
        # Get frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Get classification results for this frame
        frame_result = results[frame_idx]
        classifications = frame_result['classification_hex']
        classification_size = frame_result['classification_size']
        classifications = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size).astype(np.float32) / 255.0

        # Create brightness visualization frame
        brightness_frame = create_visualization_frame(frame, classifications, tile_size, threshold)

        # Write frame to video
        brightness_writer.write(brightness_frame)

    # Release resources
    cap.release()
    brightness_writer.release()

    print(f"Saved visualization video to: {brightness_video_path}")


def create_summary_visualization(results: list, tile_size: int, threshold: float,
                               output_dir: str):
    """
    Create a summary visualization showing classification statistics.

    Args:
        results (list): list of classification results
        tile_size (int): Tile size used for classification
        threshold (float): Threshold value for visualization
        output_dir (str): Directory to save summary visualization
    """
    print("Creating summary visualization...")

    # Collect all scores
    all_scores = []
    for result in results:
        classifications = result['classification_hex']
        classification_size = result['classification_size']
        classifications = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size).astype(np.float32) / 255.0
        all_scores.extend(classifications.flatten())

    all_scores = np.array(all_scores)

    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram of all scores
    ax1.hist(all_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax1.set_xlabel('Classification Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Classification Scores (Tile Size: {tile_size})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(all_scores)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax2.set_ylabel('Classification Score')
    ax2.set_title(f'Box Plot of Classification Scores (Tile Size: {tile_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Frame-by-frame average scores
    frame_avg_scores = []
    for result in results:
        classifications = result['classification_hex']
        classification_size = result['classification_size']
        classifications = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size).astype(np.float32) / 255.0
        frame_avg_scores.append(np.mean(classifications.flatten()))

    ax3.plot(frame_avg_scores, color='blue', linewidth=2)
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Average Classification Score')
    ax3.set_title(f'Frame-by-Frame Average Scores (Tile Size: {tile_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Statistics table
    stats_text = f"""
    Total Tiles: {len(all_scores):,}
    Mean Score: {np.mean(all_scores):.4f}
    Median Score: {np.median(all_scores):.4f}
    Std Dev: {np.std(all_scores):.4f}
    Min Score: {np.min(all_scores):.4f}
    Max Score: {np.max(all_scores):.4f}

    Above Threshold: {np.sum(all_scores >= threshold):,} ({(np.sum(all_scores >= threshold) / len(all_scores) * 100):.1f}%)
    Below Threshold: {np.sum(all_scores < threshold):,} ({(np.sum(all_scores < threshold) / len(all_scores) * 100):.1f}%)
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title(f'Statistics Summary (Tile Size: {tile_size})')
    ax4.axis('off')

    plt.tight_layout()

    # Save summary plot
    summary_path = os.path.join(output_dir, f'summary_tile{tile_size}.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved summary visualization to: {summary_path}")


def main(args):
    """
    Main function that orchestrates the video tile classification visualization process.

    This function serves as the entry point for the script. It: 1. Validates the dataset directory exists
    2. Iterates through all videos in the dataset directory
    3. For each video, loads the classification results for the specified tile size(s)
    4. Creates visualizations showing tile classifications and brightness adjustments
    5. If --statistics flag is set, compares results with groundtruth and generates statistics

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('32', '64', '128', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)
            - groundtruth (bool): Whether to use groundtruth scores (score_correct.jsonl) instead of model scores (score.jsonl)
            - statistics (bool): Whether to compare classification results with groundtruth and generate statistics

         Note:
         - The script expects classification results from 020_exec_classify.py in:
           {CACHE_DIR}/{dataset}/{video_file}/relevancy/score/proxy_{tile_size}/
         - When groundtruth=True, looks for score_correct.jsonl files
         - When groundtruth=False, looks for score.jsonl files
         - Videos are read from {DATA_DIR}/{dataset}/
         - Visualizations are saved to {CACHE_DIR}/{dataset}/{video_file}/relevancy/proxy_{tile_size}/
         - The script creates a video file (visualization.mp4) showing brightness-adjusted frames
         - Summary statistics and plots are also generated
         - When --statistics is set, groundtruth comparison visualizations are generated instead of video output
         - When --statistics is set, the --groundtruth flag is automatically ignored (always uses model predictions)
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

    if args.statistics:
        print("Running in statistics mode - comparing classification results with groundtruth")
        print("Note: --groundtruth flag is ignored in statistics mode (using model predictions)")
        use_model_scores = False
    else:
        use_model_scores = args.groundtruth

    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return

    print(f"Found {len(video_files)} video files to process")

    # Process each video file
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)

        print(f"\nProcessing video file: {video_file}")

        classifier_tilesizes: list[tuple[str, int]] = []
        for file in os.listdir(os.path.join(CACHE_DIR, args.dataset, video_file, 'relevancy')):
            classifier_name = file.split('_')[0]
            tile_size = int(file.split('_')[1])
            classifier_tilesizes.append((classifier_name, tile_size))
        classifier_tilesizes = sorted(classifier_tilesizes)
        print(f"Found {len(classifier_tilesizes)} classifier tile sizes: {classifier_tilesizes}")

        # Process each tile size for this video
        for classifier_name, tile_size in classifier_tilesizes:
            print(f"Processing tile size: {tile_size}")

            # Load classification results (model predictions for statistics mode)
            results = load_classification_results(CACHE_DIR, args.dataset, video_file, tile_size, classifier_name, use_model_scores)

            if args.statistics:
                # Load groundtruth detections for comparison
                groundtruth_detections = load_detection_results(CACHE_DIR, args.dataset, video_file, tracking=True)

                # Create output directory for statistics visualizations
                stats_output_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'relevancy', f'{classifier_name}_{tile_size}', 'statistics')

                # Create statistics visualizations
                create_statistics_visualizations(
                    video_file, results, groundtruth_detections,
                    tile_size, args.threshold, stats_output_dir
                )

                print(f"Completed statistics analysis for tile size {tile_size}")
            else:
                # Create output directory for visualizations
                vis_output_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'relevancy', f'{classifier_name}_{tile_size}')

                # Create visualizations
                save_visualization_frames(video_file_path, results, tile_size, args.threshold, vis_output_dir)

                # Create summary visualization
                create_summary_visualization(results, tile_size, args.threshold, vis_output_dir)

                print(f"Completed visualizations for tile size {tile_size}")


if __name__ == '__main__':
    main(parse_args())
