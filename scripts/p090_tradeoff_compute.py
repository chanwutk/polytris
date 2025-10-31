#!/usr/local/bin/python

import argparse
import json
import os
from typing import Dict, List, Literal, Tuple, Any
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

from rich.progress import track
import numpy as np
import pandas as pd
import cv2

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATASETS_TO_TEST, DATA_DIR, METRICS


def get_video_frame_count(dataset: str, video_name: str) -> int:
    """
    Get the total number of frames in a video using OpenCV.

    Args:
        dataset (str): Dataset name
        video_name (str): Video name (with extension, e.g., 'te01.mp4')

    Returns:
        int: Total number of frames in the video
    """
    # Construct video path - videos are in /polyis-data/datasets/{dataset}/test/
    video_path = os.path.join('/polyis-data/datasets', dataset, 'test', video_name)
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found for {dataset}/{video_name}")
        return 0
    
    # Open video and get frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return frame_count


def parse_args():
    parser = argparse.ArgumentParser(description='Compute accuracy-throughput tradeoff data')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_accuracy_results(dataset: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load saved accuracy results from individual video result files and combined dataset results.
    
    Loads both individual video results and combined dataset results from the new evaluation 
    directory structure created by p070_accuracy_compute.py.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]: 
            - List of individual video evaluation results
            - Dictionary mapping classifier_tilesize to combined dataset results
    """
    # Construct path to evaluation directory for this dataset
    evaluation_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
    assert os.path.exists(evaluation_dir), f"Evaluation directory {evaluation_dir} does not exist"
    
    individual_results = []
    combined_results = {}
    
    # Iterate through all classifier-tilesize-tilepadding combinations
    for classifier_tilesize_tilepadding in os.listdir(evaluation_dir):
        combination_dir = os.path.join(evaluation_dir, classifier_tilesize_tilepadding)
        assert os.path.isdir(combination_dir), f"Combination directory {combination_dir} does not exist"
        
        # Load individual video result files (exclude DATASET.json)
        for filename in os.listdir(combination_dir):
            if filename.endswith('.json'):
                results_path = os.path.join(combination_dir, filename)
                assert os.path.exists(results_path), f"Results file {results_path} does not exist"
                
                with open(results_path, 'r') as f:
                    result_data = json.load(f)
                    if filename == 'DATASET.json':
                        combined_results[classifier_tilesize_tilepadding] = result_data
                    else:
                        individual_results.append(result_data)
    
    print(f"Loaded {len(individual_results)} individual accuracy evaluation results")
    print(f"Loaded {len(combined_results)} combined dataset accuracy results")
    
    # Debug: Print sample result structure
    if individual_results:
        print(f"Sample individual accuracy result structure: {list(individual_results[0].keys())}")
        if 'video_name' in individual_results[0]:
            print(f"Sample video_name: {individual_results[0]['video_name']}")
        if 'classifier' in individual_results[0]:
            print(f"Sample classifier: {individual_results[0]['classifier']}")
        if 'tilesize' in individual_results[0]:
            print(f"Sample tilesize: {individual_results[0]['tilesize']}")
    
    return individual_results, combined_results


def load_throughput_results(dataset: str) -> Dict[str, Any]:
    """
    Load throughput results from the measurements directory.

    Args:
        dataset (str): Dataset name

    Returns:
        Dict[str, Any]: Throughput measurement data with CSV DataFrames
    """
    measurements_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput', 'measurements')

    if not os.path.exists(measurements_dir):
        print(f"Throughput measurements directory {measurements_dir} does not exist")
        return {}

    # Load CSV files
    query_overall_file = os.path.join(measurements_dir, 'query_execution_overall.csv')
    if not os.path.exists(query_overall_file):
        print(f"Query execution overall file {query_overall_file} does not exist")
        return {}

    query_overall = pd.read_csv(query_overall_file)

    # Load metadata
    metadata_file = os.path.join(measurements_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    print(f"Loaded throughput data with {len(query_overall)} query execution records")
    return {
        'query_overall': query_overall,
        'metadata': metadata
    }


def calculate_naive_runtime(video_name: str, query_overall: pd.DataFrame, dataset: str) -> float:
    """
    Calculate naive runtime for a specific video.

    This matches the naive calculation in p082_throughput_visualize.py,
    which sums the preprocessing stages (detection and tracking).

    Args:
        video_name: Name of the video
        query_overall: Query execution overall timing DataFrame
        dataset: Dataset name

    Returns:
        float: Naive runtime in seconds
    """
    # Add preprocessing time (naive approach)
    preprocessing_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']

    # Filter for this video and preprocessing stages with tilesize 0
    naive_data = query_overall[
        (query_overall['video'] == video_name) &
        (query_overall['stage'].isin(preprocessing_stages)) &
        (query_overall['tilesize'] == 0)
    ]

    naive_runtime = naive_data['time'].sum()

    return naive_runtime


def calculate_query_execution_runtime(video_name: str, classifier: str, tilesize: int, tilepadding: str,
                                    query_overall: pd.DataFrame, dataset: str) -> float:
    """
    Calculate query execution runtime for a specific configuration.

    This matches the query execution portion of the bars in p082_throughput_visualize.py,
    ignoring index construction time.

    Args:
        video_name: Name of the video
        classifier: Classifier used
        tilesize: Tile size used
        tilepadding: Tile padding used
        query_overall: Query execution overall timing DataFrame
        dataset: Dataset name

    Returns:
        float: Query execution runtime in seconds
    """
    # Add query execution time (specific to this tile size)
    query_stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']

    # Filter for this configuration
    config_data = query_overall[
        (query_overall['video'] == video_name) &
        (query_overall['stage'].isin(query_stages)) &
        (query_overall['classifier'] == classifier) &
        (query_overall['tilesize'] == tilesize) &
        (query_overall['tilepadding'] == tilepadding)
    ]

    query_runtime = config_data['time'].sum()

    return query_runtime


def match_accuracy_throughput_data(
    accuracy_results: List[dict],
    throughput_data: dict,
    combined_results: Dict[str, dict],
    dataset: str,
) -> tuple[list[dict], list[dict]]:
    """
    Match accuracy and throughput data by video/classifier/tilesize combination.

    Args:
        accuracy_results: List of individual video accuracy evaluation results
        throughput_data: Throughput measurement data (contains query_overall DataFrame)
        combined_results: Dictionary mapping classifier_tilesize to combined dataset accuracy results

    Returns:
        tuple[list[dict], list[dict]]:
            - Individual video data points
            - Dataset-wide aggregated data points using actual combined accuracy scores
    """
    matched_data = []
    query_overall = throughput_data.get('query_overall')

    # Cache frame counts to avoid repeated OpenCV calls
    frame_count_cache = {}

    # Group data by classifier and tilesize for aggregation
    grouped_data = defaultdict(list)

    print(f"Processing {len(accuracy_results)} accuracy results for matching...")

    processed_count = 0
    matched_count = 0

    for result in accuracy_results:
        # Skip combined results (video_name is None) - we only want individual video results
        if result.get('video_name') is None:
            continue

        video_name = result['video_name']
        classifier = result['classifier']
        tilesize = result['tilesize']
        tilepadding = result['tilepadding']

        # Only include classifiers from CLASSIFIERS_TO_TEST
        if classifier not in CLASSIFIERS_TO_TEST + ['Perfect']:
            print(f"Skipping classifier {classifier} (not in CLASSIFIERS_TO_TEST + ['Perfect'])")
            continue

        processed_count += 1

        # Extract accuracy metrics
        metrics = result['metrics']
        hota_score = metrics.get('HOTA', {}).get('HOTA(0)', 0.0)
        mota_score = metrics.get('CLEAR', {}).get('MOTA', 0.0)

        # Get frame count (with caching)
        if video_name not in frame_count_cache:
            frame_count_cache[video_name] = get_video_frame_count(dataset, video_name)
        frame_count = frame_count_cache[video_name]

        # Calculate query execution runtime only
        query_runtime = calculate_query_execution_runtime(video_name, classifier, tilesize,
                                                          tilepadding, query_overall, dataset)

        # Calculate throughput (frames per second)
        throughput_fps = frame_count / query_runtime if query_runtime > 0 else 0.0

        matched_entry = {
            'video_name': video_name,
            'classifier': classifier,
            'tilesize': tilesize,
            'tilepadding': tilepadding,
            'hota_score': hota_score,
            'mota_score': mota_score,
            'total_query_time': query_runtime,  # Now same as query_runtime
            'query_runtime': query_runtime,
            'frame_count': frame_count,
            'throughput_fps': throughput_fps,
            'stage_times': {}  # No longer tracking individual stage times
        }

        matched_data.append(matched_entry)
        matched_count += 1

        # Group by classifier, tilesize, and tilepadding for aggregation
        group_key = (classifier, tilesize, tilepadding)
        grouped_data[group_key].append(matched_entry)

    # Create dataset-wide aggregated data using actual combined accuracy scores
    aggregated_data = []
    for (classifier, tilesize, tilepadding), entries in grouped_data.items():
        if not entries:
            continue

        # Get actual combined accuracy scores from DATASET.json
        combination_key = f"{classifier}_{tilesize}_{tilepadding}"
        assert combination_key in combined_results, \
            f"Combined results not found for {combination_key}"
        combined_metrics = combined_results[combination_key]['metrics']
        actual_hota = combined_metrics.get('HOTA', {}).get('HOTA(0)', 0.0)
        actual_mota = combined_metrics.get('CLEAR', {}).get('MOTA', 0.0)
        print(f"Using actual combined accuracy scores for {combination_key}: " \
            f"HOTA={actual_hota:.3f}, MOTA={actual_mota:.3f}")

        # Calculate combined runtime and throughput
        total_frames = sum([entry['frame_count'] for entry in entries])
        total_runtime = sum([entry['query_runtime'] for entry in entries])
        combined_throughput = total_frames / total_runtime if total_runtime > 0 else 0.0

        aggregated_entry = {
            'video_name': 'Dataset Average',
            'classifier': classifier,
            'tilesize': tilesize,
            'tilepadding': tilepadding,
            'hota_score': actual_hota,
            'mota_score': actual_mota,
            'total_query_time': total_runtime,
            'query_runtime': total_runtime,
            'frame_count': total_frames,
            'throughput_fps': combined_throughput,
            'stage_times': {}  # Not used for aggregated data
        }

        aggregated_data.append(aggregated_entry)

    print(f"Processed {processed_count} accuracy results")
    print(f"Matched {matched_count} individual accuracy-throughput data points")
    print(f"Created {len(aggregated_data)} dataset-wide aggregated data points")
    return matched_data, aggregated_data


def compute_tradeoff(matched_data: list[dict], aggregated_data: list[dict],
                       output_dir: str, metrics_list: list[str], query_overall: pd.DataFrame,
                       dataset: str, x_column: str, x_title: str,
                       naive_column: Literal['naive_runtime', 'naive_throughput'],
                       plot_suffix: str, csv_suffix: str):
    """
    Compute tradeoff data with configurable x-axis, including dataset-wide aggregated data.

    Args:
        matched_data: List of individual video accuracy-throughput data points
        aggregated_data: List of dataset-wide aggregated data points
        output_dir: Output directory for data files
        metrics_list: List of metrics to compute
        query_overall: Query execution overall timing DataFrame
        dataset: Dataset name
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for plot filename
        csv_suffix: Suffix for CSV filename
    """
    print(f"Computing {plot_suffix} tradeoff data...")

    assert len(matched_data) > 0, \
        f"No matched data available for {plot_suffix} computation"
    assert len(aggregated_data) > 0, \
        f"No aggregated data available for {plot_suffix} computation"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame for easier data handling
    df = pd.DataFrame(matched_data)
    df_agg = pd.DataFrame(aggregated_data)

    # Calculate naive values for each video
    unique_videos = sorted(df['video_name'].unique())
    naive_values = {}
    for video in unique_videos:
        naive_runtime = calculate_naive_runtime(video, query_overall, dataset)
        if naive_column == 'naive_runtime':
            naive_values[video] = naive_runtime
        elif naive_column == 'naive_throughput':
            df_video = df[df['video_name'] == video]
            assert isinstance(df_video, pd.DataFrame), \
                f"Expected DataFrame for video {video}, got {type(df_video)}"
            frame_counts = df_video['frame_count']
            # Assert that all frame_counts have the same value
            assert frame_counts.nunique() == 1, \
                f"All frame_counts for video {video} must have the same value, " \
                f"but found {frame_counts.nunique()} unique values: {frame_counts.unique()}"
            frame_count = frame_counts.iloc[0]
            assert naive_runtime > 0, \
                f"Naive runtime must be greater than 0, got {naive_runtime}"
            naive_throughput = frame_count / naive_runtime
            naive_values[video] = naive_throughput

    # Add naive data to dataframe
    df_with_naive = df.copy()
    df_with_naive[naive_column] = df_with_naive['video_name'].map(naive_values)  # type: ignore

    # Save matched data to CSV
    csv_file_path = os.path.join(output_dir, f'individual_accuracy_{csv_suffix}_tradeoff.csv')
    df_with_naive.to_csv(csv_file_path, index=False)
    print(f"Saved matched data to: {csv_file_path}")

    # Calculate dataset-wide naive values
    total_naive_runtime = sum([calculate_naive_runtime(video, query_overall, dataset) for video in unique_videos])
    if naive_column == 'naive_runtime':
        dataset_naive_value = total_naive_runtime
    elif naive_column == 'naive_throughput':
        total_frames = df_agg['frame_count'].iloc[0]
        assert total_naive_runtime > 0, \
            f"Total naive runtime must be greater than 0, got {total_naive_runtime}"
        dataset_naive_value = total_frames / total_naive_runtime

    # Add dataset naive value to aggregated dataframe
    df_agg_with_naive = df_agg.copy()
    df_agg_with_naive[naive_column] = dataset_naive_value

    # Save matched aggregated data to CSV
    csv_file_path_agg = os.path.join(output_dir, f'combined_accuracy_{csv_suffix}_tradeoff.csv')
    df_agg_with_naive.to_csv(csv_file_path_agg, index=False)
    print(f"Saved matched aggregated data to: {csv_file_path_agg}")

    print(f"Computed tradeoff data for {plot_suffix} - visualization skipped")


def compute_tradeoffs(matched_data: list[dict], aggregated_data: list[dict], output_dir: str,
                      metrics_list: list[str], query_overall: pd.DataFrame, dataset: str):
    """
    Compute both runtime and throughput tradeoff data.

    Args:
        matched_data: List of individual video accuracy-throughput data points
        aggregated_data: List of dataset-wide aggregated data points
        output_dir: Output directory for data files
        metrics_list: List of metrics to compute
        query_overall: Query execution overall timing DataFrame
        dataset: Dataset name
    """
    # Compute runtime data
    compute_tradeoff(
        matched_data, aggregated_data, output_dir, metrics_list, query_overall, dataset,
        x_column='query_runtime',
        x_title='Query Execution Runtime (seconds)',
        naive_column='naive_runtime',
        plot_suffix='runtime',
        csv_suffix='runtime'
    )

    # Compute throughput data
    compute_tradeoff(
        matched_data, aggregated_data, output_dir, metrics_list, query_overall, dataset,
        x_column='throughput_fps',
        x_title='Throughput (frames/second)',
        naive_column='naive_throughput',
        plot_suffix='throughput',
        csv_suffix='throughput'
    )


def process_dataset(dataset: str):
    """
    Process a single dataset for accuracy-query execution runtime tradeoff computation.
    
    This function loads accuracy and throughput results for a single dataset, matches them,
    and computes tradeoff data showing the relationship between accuracy and query execution runtime.
    
    Args:
        dataset: Dataset name to process
    """
    print(f"Starting accuracy-query execution runtime tradeoff computation for: {dataset}")
    
    # Load accuracy results (both individual and combined)
    print(f"Loading accuracy results for {dataset}...")
    accuracy_results, combined_results = load_accuracy_results(dataset)
    
    assert accuracy_results, \
        f"No accuracy results found for {dataset}. " \
        "Please run p070_accuracy_compute.py first."
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Load throughput results
    print(f"Loading throughput results for {dataset}...")
    throughput_data = load_throughput_results(dataset)
    
    assert throughput_data, \
        f"No throughput results found for {dataset}. Please run " \
        "p080_throughput_gather.py and p081_throughput_compute.py first."
    
    # Match accuracy and throughput data
    print(f"Matching accuracy and throughput data for {dataset}...")
    matched_data, aggregated_data = match_accuracy_throughput_data(accuracy_results, throughput_data, combined_results, dataset)
    
    assert len(matched_data) > 0, \
        f"No matching data points found between accuracy and throughput results for {dataset}."
    
    # Compute tradeoff data
    output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '090_tradeoff')
    compute_tradeoffs(matched_data, aggregated_data, output_dir,
                                   metrics_list, throughput_data['query_overall'], dataset)
    
    print(f"Completed processing dataset: {dataset}")


def main(args):
    """
    Main function that orchestrates the accuracy-throughput tradeoff computation.
    
    This function serves as the entry point for the script. It:
    1. Loads accuracy results from p070_accuracy_compute.py
    2. Loads throughput results from p081_throughput_compute.py
    3. Gets video frame counts using OpenCV
    4. Matches the data by video/classifier/tilesize combination
    5. Computes tradeoff data showing accuracy vs query execution runtime relationships
    6. Computes tradeoff data showing accuracy vs throughput (frames/second) relationships
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/{classifier}_{tilesize}/{video_name}.json
        - The script expects throughput results from p081_throughput_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/080_throughput/measurements/query_execution_summaries.json
        - Results are saved to: {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/
        - Video files are expected in {DATA_DIR}/{dataset}/{video_name}.mp4 (or other extensions)
        - Only query execution runtime is used (index construction time is ignored)
        - Metrics are automatically detected from the accuracy results
    """
    print(f"Processing datasets: {args.datasets}")
    
    # Process datasets in parallel with progress tracking
    with Pool() as pool:
        ires = pool.imap(process_dataset, args.datasets)
        # ires = map(process_dataset, args.datasets)
        
        # Process datasets in parallel using imap with rich track
        _ = [*track(ires, total=len(args.datasets))]


if __name__ == '__main__':
    main(parse_args())
