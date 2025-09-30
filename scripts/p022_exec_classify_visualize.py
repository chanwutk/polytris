#!/usr/local/bin/python

import argparse
import os
import shutil
import numpy as np
import altair as alt
from typing import Callable
import multiprocessing as mp
from functools import partial
import pandas as pd

from polyis.utilities import CACHE_DIR, DATA_DIR, load_classification_results, load_detection_results, mark_detections, ProgressBar, DATASETS_TO_TEST


TILE_SIZES = [30, 60]  #, 120]


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (List[str]): Dataset names to process (default: ['b3d'])
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification visualization (default: 0.5)
    """
    parser = argparse.ArgumentParser(description='Visualize video tile classification results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    return parser.parse_args()


def evaluate_classification_accuracy(args):
    """
    Evaluate classification accuracy by comparing predictions with groundtruth detections.

    Args:
        args: Tuple containing (frame_result, frame_detections, tile_size, threshold)

    Returns:
        dict: dictionary containing evaluation metrics and error details
    """
    frame_result, frame_detections, tile_size, threshold = args

    # Validate frame data
    assert 'tracks' in frame_detections, f"tracks not in frame_detections: {frame_detections}"

    classifications = frame_result['classification_hex']
    classification_size = frame_result['classification_size']
    classifications = (np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
        .reshape(classification_size)
        .astype(np.float32) / 255.0)

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

    detection_bitmap = mark_detections(frame_detections['tracks'], total_width,
                                       total_height, tile_size, slice(-4, None))

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


def create_statistics_visualizations(results: list[dict], groundtruth_detections: list[dict],
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
        frame_eval = evaluate_classification_accuracy((frame_result, frame_detections, tile_size, threshold))
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

    # Create individual visualizations
    visualize_error_summary(
        total_tp, total_tn, total_fp, total_fn,
        overall_precision, overall_recall, overall_accuracy, overall_f1,
        tile_size, output_dir
    )
    
    visualize_error_over_time(
        frame_metrics, groundtruth_detections,
        # overall_precision, overall_recall, overall_f1,
        tile_size, output_dir
    )
    
    visualize_spatial_misclassification(
        all_error_counts, tile_size, output_dir
    )
    
    # visualize_score_distribution(
    #     all_classification_scores, all_actual_positives,
    #     threshold, tile_size, output_dir
    # )

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
    # Prepare data for confusion matrix charts
    confusion_data = [
        {'Prediction': 'Predicted Negative', 'Actual': 'Actual Negative', 'Count': total_tn},
        {'Prediction': 'Predicted Negative', 'Actual': 'Actual Positive', 'Count': total_fn},
        {'Prediction': 'Predicted Positive', 'Actual': 'Actual Negative', 'Count': total_fp},
        {'Prediction': 'Predicted Positive', 'Actual': 'Actual Positive', 'Count': total_tp}
    ]
    
    confusion_df = pd.DataFrame(confusion_data)
    
    # Create confusion matrix chart by prediction
    chart1 = alt.Chart(confusion_df).mark_bar().encode(
        x='Prediction:N',
        y='Count:Q',
        color=alt.Color('Actual:N', scale=alt.Scale(domain=['Actual Negative', 'Actual Positive'], 
                                                   range=['lightcoral', 'lightgreen'])),
        tooltip=['Prediction', 'Actual', 'Count']
    ).properties(
        title=f'Classification Results by Prediction (Tile Size: {tile_size})',
        width=200,
        height=300
    )
    
    # Create confusion matrix chart by actual
    chart2 = alt.Chart(confusion_df).mark_bar().encode(
        x='Actual:N',
        y='Count:Q',
        color=alt.Color('Prediction:N', scale=alt.Scale(domain=['Predicted Negative', 'Predicted Positive'], 
                                                       range=['lightcoral', 'lightgreen'])),
        tooltip=['Prediction', 'Actual', 'Count']
    ).properties(
        title=f'Classification Results by Actual (Tile Size: {tile_size})',
        width=200,
        height=300
    )
    
    # Prepare metrics data
    metrics_data = [
        {'Metric': 'Precision', 'Score': overall_precision},
        {'Metric': 'Recall', 'Score': overall_recall},
        {'Metric': 'Accuracy', 'Score': overall_accuracy},
        {'Metric': 'F1-Score', 'Score': overall_f1}
    ]
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create metrics chart
    chart3 = alt.Chart(metrics_df).mark_bar().encode(
        x='Metric:N',
        y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Precision', 'Recall', 'Accuracy', 'F1-Score'],
                                                   range=['skyblue', 'lightgreen', 'lightcoral', 'gold'])),
        tooltip=['Metric', alt.Tooltip('Score:Q', format='.3f')]
    ).properties(
        title=f'Overall Classification Metrics (Tile Size: {tile_size})',
        width=200,
        height=300
    )
    
    # Combine charts horizontally
    combined_chart = alt.hconcat(chart1, chart2, chart3, spacing=20)
    
    # Save the chart
    overall_summary_path = os.path.join(output_dir, f'010_overall_summary_tile{tile_size}.png')
    combined_chart.save(overall_summary_path, scale_factor=2)
    
    return overall_summary_path


def visualize_error_over_time(frame_metrics: list[dict], groundtruth_detections: list[dict],
                              # overall_precision: float, overall_recall: float, overall_f1: float,
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
    # num_tiles_per_frame = frame_metrics[0]['total_tiles']
    tp_counts = [m['tp'] for m in frame_metrics]
    tn_counts = [m['tn'] for m in frame_metrics]
    fp_counts = [m['fp'] for m in frame_metrics]
    fn_counts = [m['fn'] for m in frame_metrics]

    # Prepare data for all charts
    chart_data = []
    for i in range(len(frame_indices)):
        chart_data.extend([
            {'Frame': frame_indices[i], 'Value': error_rates[i], 'Metric': 'Error Rate', 'Chart': 'Chart1'},
            {'Frame': frame_indices[i], 'Value': f1_scores[i], 'Metric': 'F1-Score', 'Chart': 'Chart1'},
            {'Frame': frame_indices[i], 'Value': objects_per_frame[i], 'Metric': 'Object Count', 'Chart': 'Chart1'},
            {'Frame': frame_indices[i], 'Value': precision_rates[i], 'Metric': 'Precision', 'Chart': 'Chart2'},
            {'Frame': frame_indices[i], 'Value': recall_rates[i], 'Metric': 'Recall', 'Chart': 'Chart2'},
            {'Frame': frame_indices[i], 'Value': tp_counts[i], 'Metric': 'True Positives', 'Chart': 'Chart3'},
            {'Frame': frame_indices[i], 'Value': tn_counts[i], 'Metric': 'True Negatives', 'Chart': 'Chart3'},
            {'Frame': frame_indices[i], 'Value': fp_counts[i], 'Metric': 'False Positives', 'Chart': 'Chart4'},
            {'Frame': frame_indices[i], 'Value': fn_counts[i], 'Metric': 'False Negatives', 'Chart': 'Chart4'}
        ])
    
    df = pd.DataFrame(chart_data)
    
    # Create individual charts
    chart1_data = df[df['Chart'] == 'Chart1']
    assert isinstance(chart1_data, pd.DataFrame)
    chart1 = alt.Chart(chart1_data).mark_line().encode(
        x='Frame:Q',
        y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
        color='Metric:N',
        strokeDash=alt.condition(alt.datum.Metric == 'Object Count', alt.value([5, 5]), alt.value([0, 0]))
    ).properties(
        title=f'Error Rate and F1-Score Over Time (Tile Size: {tile_size})',
        width=600,
        height=200
    ).resolve_scale(y='independent')
    
    chart2_data = df[df['Chart'] == 'Chart2']
    assert isinstance(chart2_data, pd.DataFrame)
    chart2 = alt.Chart(chart2_data).mark_line().encode(
        x='Frame:Q',
        y='Value:Q',
        color='Metric:N'
    ).properties(
        title=f'Precision and Recall Over Time (Tile Size: {tile_size})',
        width=600,
        height=200
    )
    
    chart3_data = df[df['Chart'] == 'Chart3']
    assert isinstance(chart3_data, pd.DataFrame)
    chart3 = alt.Chart(chart3_data).mark_line().encode(
        x='Frame:Q',
        y='Value:Q',
        color='Metric:N'
    ).properties(
        title=f'True Positives and Negatives Over Time (Tile Size: {tile_size})',
        width=600,
        height=200
    )
    
    chart4_data = df[df['Chart'] == 'Chart4']
    assert isinstance(chart4_data, pd.DataFrame)
    chart4 = alt.Chart(chart4_data).mark_line().encode(
        x='Frame:Q',
        y='Value:Q',
        color='Metric:N'
    ).properties(
        title=f'False Positives and Negatives Over Time (Tile Size: {tile_size})',
        width=600,
        height=200
    )
    
    # Combine charts vertically
    combined_chart = alt.vconcat(chart1, chart2, chart3, chart4, spacing=20)
    
    # Save the chart
    time_series_path = os.path.join(output_dir, f'020_time_series_tile{tile_size}.png')
    combined_chart.save(time_series_path, scale_factor=2)
    
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

    # Prepare data for heatmap
    heatmap_data = []
    for i in range(grid_height):
        for j in range(grid_width):
            heatmap_data.append({
                'x': j,
                'y': i,
                'error_count': int(aggregated_error_map[i, j])
            })
    
    df = pd.DataFrame(heatmap_data)
    
    # Create heatmap chart
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O', title='Tile X Position'),
        y=alt.Y('y:O', title='Tile Y Position', sort=alt.SortField('y', order='descending')),
        color=alt.Color('error_count:Q', scale=alt.Scale(scheme='reds'), title='Error Count'),
        tooltip=['x', 'y', 'error_count']
    ).properties(
        title=f'Cumulative Error Count per Tile (Tile Size: {tile_size})',
        width=600,
        height=400
    )
    
    # Save the chart
    error_heatmap_path = os.path.join(output_dir, f'030_error_heatmap_tile{tile_size}.png')
    chart.save(error_heatmap_path, scale_factor=2)
    
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

    # Prepare data for histograms
    histogram_data = []
    
    # Add correct predictions data
    for score in correct_scores:
        histogram_data.append({'Score': score, 'Category': 'Correct Predictions', 'Count': 1})
    
    # Add incorrect predictions data
    for score in incorrect_scores:
        histogram_data.append({'Score': score, 'Category': 'Incorrect Predictions', 'Count': 1})
    
    # Add actual positive scores data
    for score in actual_positive_scores:
        histogram_data.append({'Score': score, 'Category': 'Actual Positive Scores', 'Count': 1})
    
    # Add actual negative scores data
    for score in actual_negative_scores:
        histogram_data.append({'Score': score, 'Category': 'Actual Negative Scores', 'Count': 1})
    
    df = pd.DataFrame(histogram_data)
    
    # Create histogram charts for each category
    charts = []
    categories = ['Correct Predictions', 'Incorrect Predictions', 'Actual Positive Scores', 'Actual Negative Scores']
    
    for category in categories:
        category_data = df[df['Category'] == category]
        if len(category_data) > 0:
            assert isinstance(category_data, pd.DataFrame)
            chart = alt.Chart(category_data).mark_bar().encode(
                alt.X('Score:Q', bin=alt.Bin(maxbins=100), title='Classification Score'),
                y='count():Q',
                color=alt.value('#4682B4')
            ).properties(
                title=f'{category} (Tile Size: {tile_size}) - Total: {len(category_data):,}',
                width=300,
                height=200
            # ).add_selection(
            #     alt.selection_interval()
            ).add_layer(
                alt.Chart(pd.DataFrame([{'threshold': threshold}])).mark_rule(
                    color='red', strokeDash=[5, 5], strokeWidth=2
                ).encode(x='threshold:Q')
            )
            charts.append(chart)
        else:
            # Create empty chart with text
            empty_data = pd.DataFrame([{'text': f'No {category.lower()}'}])
            chart = alt.Chart(empty_data).mark_text(size=20).encode(
                text='text:N'
            ).properties(
                title=f'{category} (Tile Size: {tile_size})',
                width=300,
                height=200
            )
            charts.append(chart)
    
    # Combine charts in a 2x2 grid
    combined_chart = alt.vconcat(
        alt.hconcat(charts[0], charts[1]),
        alt.hconcat(charts[2], charts[3])
    )
    
    # Save the chart
    histogram_path = os.path.join(output_dir, f'040_histogram_scores_tile{tile_size}.png')
    combined_chart.save(histogram_path, scale_factor=2)
    
    return histogram_path


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

    # Load classification results from execution directory
    results = load_classification_results(CACHE_DIR, dataset_name, video_file, tile_size,
                                          classifier_name, execution_dir=True)
    
    # Load groundtruth detections for comparison
    groundtruth_detections = load_detection_results(CACHE_DIR, dataset_name, video_file, tracking=True)
    
    # Create output directory for statistics visualizations
    stats_output_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file,
                                    '020_relevancy', f'{classifier_name}_{tile_size}',
                                    'statistics')
    
    # Create statistics visualizations
    create_statistics_visualizations(results, groundtruth_detections, tile_size,
                                     threshold, stats_output_dir, gpu_id, command_queue)


def main(args):
    """
    Main function that orchestrates the video tile classification visualization process.

    This function serves as the entry point for the script. It: 1. Validates the dataset directories exist
    2. Iterates through all videos in each dataset directory
    3. For each video, loads the classification results for the specified tile size(s)
    4. Creates visualizations showing tile classifications and statistics for each tile size

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)

         Note:
         - The script expects classification results from 021_exec_classify_correct.py in:
           {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier_name}_{tile_size}/score/score.jsonl
         - Looks for score.jsonl files
         - Videos are read from {DATA_DIR}/{dataset}/
         - Visualizations are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier_name}_{tile_size}/statistics/
         - Summary statistics and plots are also generated
    """
    mp.set_start_method('spawn', force=True)
    
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

    # Collect all video-classifier-tile combinations for parallel processing
    all_tasks = []
    
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue
        
        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"No video files found in {dataset_dir}")
            continue
        
        print(f"Found {len(video_files)} video files in dataset {dataset_name}")
        
        for video_file in sorted(video_files):
            # Get classifier tile sizes for this video from execution directory
            relevancy_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file, '020_relevancy')
            if not os.path.exists(relevancy_dir):
                print(f"Skipping {video_file}: No relevancy directory found in execution folder")
                continue
                
            classifier_tilesizes: list[tuple[str, int]] = []
            for file in os.listdir(relevancy_dir):
                if '_' in file:
                    classifier_name = file.split('_')[0]
                    tile_size = int(file.split('_')[1])
                    classifier_tilesizes.append((classifier_name, tile_size))
            
            classifier_tilesizes = sorted(classifier_tilesizes)
            
            if not classifier_tilesizes:
                print(f"Skipping {video_file}: No classifier tile sizes found")
                continue
                
            print(f"Found {len(classifier_tilesizes)} classifier tile sizes for {video_file}: {classifier_tilesizes}")
            
            # Add tasks for each classifier-tile size combination
            for classifier_name, tile_size in classifier_tilesizes:
                task_args = (video_file, dataset_name, classifier_name, tile_size, args.threshold)
                all_tasks.append(task_args)

    assert len(all_tasks) > 0, 'No tasks to process'

    print(f"Processing {len(all_tasks)} tasks in parallel...")

    # Create worker functions for ProgressBar
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for video_file, dataset_name, classifier_name, tile_size, threshold in all_tasks:
        funcs.append(partial(_process_classifier_tile_worker, video_file, dataset_name, 
                            classifier_name, tile_size, threshold))
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.5)
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Run all tasks with ProgressBar
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
