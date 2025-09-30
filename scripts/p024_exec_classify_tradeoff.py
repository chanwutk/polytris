#!/usr/local/bin/python

import argparse
import json
import os
import numpy as np
import pandas as pd
import altair as alt
from rich.progress import track
from typing import Any, Dict, List
import multiprocessing as mp

from polyis.utilities import CACHE_DIR, DATA_DIR, get_accuracy, get_f1_score, get_precision, get_recall, load_classification_results, load_detection_results, mark_detections


TILE_SIZES = [30, 60]  #, 120]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare accuracy-throughput tradeoff of classifiers')
    parser.add_argument('--datasets', required=False,
                        default=['caldot1-yolov5', 'caldot2-yolov5'],
                        nargs='+',
                        help='Dataset names (space-separated)')
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
    # tp: True Positive: predicted above threshold, has detection overlap
    # tn: True Negative: predicted below threshold, no detection overlap
    # fp: False Positive: predicted above threshold, no detection overlap
    # fn: False Negative: predicted below threshold, has detection overlap

    # tp fn
    # fp tn
    metrics = np.zeros((2, 2), dtype=int)

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

            metrics[int(predicted_positive), int(actual_positive)] += 1
            # print(predicted_positive, actual_positive)
    
    tp, fp, fn, tn = map(int, metrics.flatten().tolist())
    assert tp + fp + fn + tn == grid_height * grid_width, \
        "tp + fp + fn + tn != grid_height * grid_width: " \
        f"{tp + fp + fn + tn} != {grid_height * grid_width}\n" \
        f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}\n" \
        f"grid_height: {grid_height}, grid_width: {grid_width}\n" \
        f"metrics:\n{metrics}"

    # Calculate precision, recall, and accuracy
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    accuracy = get_accuracy(tp, tn, fp, fn)
    f1_score = get_f1_score(tp, fp, fn)

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall,
        'accuracy': accuracy, 'f1_score': f1_score,
        'total_tiles': grid_height * grid_width
    }


def load_throughput_data(cache_dir: str, dataset: str, video_file: str, 
                        classifier: str, tile_size: int) -> tuple[float, float, int]:
    """
    Load throughput data from score.jsonl file.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        video_file (str): Video file name
        classifier (str): Classifier name
        tile_size (int): Tile size used
    Returns:
        tuple[float, float, int]: Average throughput in frames per second,
                                  total runtime in milliseconds, number of frames
    """
    score_file = os.path.join(cache_dir, dataset, 'execution', video_file, '020_relevancy', 
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
        return 0.0, total_runtime_ms, frame_count
    
    # Calculate average throughput in frames per second
    avg_runtime_per_frame_ms = total_runtime_ms / frame_count
    if avg_runtime_per_frame_ms == 0:
        throughput_fps = 0.0
    else:
        throughput_fps = 1000.0 / avg_runtime_per_frame_ms
    
    return throughput_fps, total_runtime_ms, frame_count


def evaluate_classifier_tile(args) -> dict:
    """
    Evaluate a single classifier-tile size combination on multiple videos.
    
    Args:
        args: Tuple containing (video_files, dataset_name, classifier_name, tile_size, threshold)
    
    Returns:
        dict: Evaluation results for this classifier-tile combination
    """
    video_files, dataset_name, classifier_name, tile_size, threshold = args
    
    # Evaluate accuracy metrics for all frames
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_runtime_ms = 0.0
    total_frame_count = 0

    for video_file in video_files:
        # Load classification results
        results = load_classification_results(CACHE_DIR, dataset_name, video_file,
                                              tile_size, classifier_name, execution_dir=True)
        
        # Load groundtruth detections for comparison
        groundtruth_detections = load_detection_results(CACHE_DIR, dataset_name, video_file, tracking=True)
        
        # Load throughput data
        _, runtime_ms, frame_count = load_throughput_data(CACHE_DIR, dataset_name, video_file,
                                                          classifier_name, tile_size)
        total_runtime_ms += runtime_ms
        total_frame_count += frame_count
        
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
    precision = get_precision(total_tp, total_fp)
    recall = get_recall(total_tp, total_fn)
    accuracy = get_accuracy(total_tp, total_tn, total_fp, total_fn)
    f1_score = get_f1_score(total_tp, total_fp, total_fn)
    
    return {
        'video_file': video_files,
        'classifier': classifier_name,
        'tile_size': tile_size,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'throughput_fps': total_runtime_ms * 1000. / total_frame_count,
        'total_runtime_ms': total_runtime_ms,
        'total_frame_count': total_frame_count
    }


def visualize_tradeoff(dataset_name: str, results: List[Dict], output_dir: str):
    """
    Create accuracy-throughput tradeoff visualizations.
    
    Args:
        dataset_name (str): Name of the dataset
        results (List[Dict]): List of evaluation results for each classifier-tile combination
        output_dir (str): Directory to save visualizations
    """
    print(f"Creating tradeoff visualizations for {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    assert len(valid_results) > 0, f"No valid results for {dataset_name}"
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_results)
    
    # Create individual charts for each metric
    metrics = ['precision', 'recall', 'accuracy', 'f1_score']
    titles = ['Precision vs Throughput', 'Recall vs Throughput', 'Accuracy vs Throughput', 'F1-Score vs Throughput']
    
    charts = []
    for metric, title in zip(metrics, titles):
        # Create scatter plot with lines connecting same classifier
        scatter = alt.Chart(df).mark_circle().encode(
            x='throughput_fps:Q',
            y=alt.Y(f'{metric}:Q', scale=alt.Scale(zero=False)),
            color='classifier:N',
            size=alt.Size('tile_size:O', scale=alt.Scale(range=[50, 200])),
            tooltip=['classifier', 'tile_size', 'throughput_fps', metric]
        ).properties(
            title=f'{title} - {dataset_name}',
            width=300,
            height=250
        )
        
        # Add lines connecting points with same classifier
        line_data = []
        for classifier in df['classifier'].unique():
            classifier_data = df[df['classifier'] == classifier]
            assert isinstance(classifier_data, pd.DataFrame), f"classifier_data is not a DataFrame: {classifier_data}"
            classifier_data = classifier_data.sort_values('tile_size')
            if len(classifier_data) > 1:
                for i in range(len(classifier_data) - 1):
                    line_data.append({
                        'x1': classifier_data.iloc[i]['throughput_fps'],
                        'y1': classifier_data.iloc[i][metric],
                        'x2': classifier_data.iloc[i+1]['throughput_fps'],
                        'y2': classifier_data.iloc[i+1][metric],
                        'classifier': classifier
                    })
        
        if line_data:
            line_df = pd.DataFrame(line_data)
            lines = alt.Chart(line_df).mark_rule(strokeDash=[5, 5], opacity=0.5).encode(
                x='x1:Q',
                y='y1:Q',
                x2='x2:Q',
                y2='y2:Q',
                color='classifier:N'
            )
            chart = scatter + lines
        else:
            chart = scatter
        
        charts.append(chart)
    
    # Combine charts in a 2x2 grid
    combined_chart = alt.vconcat(
        alt.hconcat(charts[0], charts[1]),
        alt.hconcat(charts[2], charts[3])
    )
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'classification_throughput_accuracy_tradeoff.png')
    combined_chart.save(plot_path, scale_factor=2)
    
    # print(f"Saved tradeoff visualizations to: {plot_path}")


def process_dataset(args):
    """
    Worker function to process a single video for multiprocessing.
    
    Args:
        args: Tuple containing (video_files, dataset_name, tile_sizes, threshold)
    
    Returns:
        str: Status message about processing completion
    """
    video_files, dataset_name, tile_sizes, threshold = args
    
    classifier_tilesizes: set[tuple[str, int]] | None = None
    for video_file in video_files:
        # Get all classifier-tile combinations for this video
        relevancy_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file, '020_relevancy')
        if not os.path.exists(relevancy_dir):
            return f"Skipping {video_file}: No relevancy directory found"
        
        _classifier_tilesizes: set[tuple[str, int]] = set()
        for file in os.listdir(relevancy_dir):
            if '_' in file:
                classifier_name = file.split('_')[0]
                tile_size = int(file.split('_')[1])
                if classifier_name != 'groundtruth' and tile_size in tile_sizes:
                    _classifier_tilesizes.add((classifier_name, tile_size))

        if classifier_tilesizes is None:
            classifier_tilesizes = _classifier_tilesizes
        assert classifier_tilesizes == _classifier_tilesizes, \
            f"classifier_tilesizes != _classifier_tilesizes: {classifier_tilesizes} != {_classifier_tilesizes}"

    assert classifier_tilesizes is not None, f"No classifier tilesizes found for {video_files}"
    assert len(classifier_tilesizes) > 0, f"No classifier tilesizes found for {video_files}"
    
    # Prepare arguments for parallel processing
    worker_args = [(video_files, dataset_name, classifier_name, tile_size, threshold)
                   for classifier_name, tile_size in classifier_tilesizes]
    
    # Use multiprocessing to evaluate classifier-tile combinations in parallel
    num_processes = min(mp.cpu_count() - 1, len(worker_args))
    with mp.Pool(processes=num_processes) as pool:
        results = list(track(
            pool.imap(evaluate_classifier_tile, worker_args),
            total=len(worker_args),
            description=f"Evaluating classifier-tile combinations"
        ))
    
    # Create output directory for this video
    output_dir = os.path.join(CACHE_DIR, 'summary', dataset_name, 'classifiers', 'tradeoff')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    visualize_tradeoff(dataset_name, results, output_dir)
    
    # return f"Completed tradeoff analysis for {video_file} with {len(_classifier_tilesizes)} combinations"


def main(args):
    """
    Main function that orchestrates the accuracy-throughput tradeoff analysis.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)
    """
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

    mp.set_start_method('spawn', force=True)

    dataset_args = []

    # Process each dataset
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing dataset: {dataset_name}")

        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if len(video_files) == 0:
            print(f"No video files found in {dataset_dir}, skipping...")
            continue

        # # Prepare arguments for parallel processing
        # worker_args = [(video_file, dataset_name, tile_sizes_to_process, args.threshold)
        #                for video_file in sorted(video_files)]
        dataset_args.append((video_files, dataset_name, tile_sizes_to_process, args.threshold))

    # # Process videos in parallel
    # with mp.Pool(processes=num_processes) as pool:
    #     results = list(pool.imap(_process_video_worker, worker_args))
    list(map(process_dataset, dataset_args))


if __name__ == '__main__':
    main(parse_args())
