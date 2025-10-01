#!/usr/local/bin/python

import argparse
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import altair as alt
import cv2

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATASETS_TO_TEST, DATA_DIR


def get_video_frame_count(dataset: str, video_name: str) -> int:
    """
    Get the total number of frames in a video using OpenCV.
    
    Args:
        dataset (str): Dataset name
        video_name (str): Video name (without extension)
        
    Returns:
        int: Total number of frames in the video
    """
    # Construct video path - assume .mp4 extension
    video_path = os.path.join(DATA_DIR, dataset, f"{video_name}")
    
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
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (list): Dataset names to process (default: ['caldot1', 'caldot2'])
            - metrics (str): Comma-separated list of metrics to evaluate (default: 'HOTA,CLEAR')
    """
    parser = argparse.ArgumentParser(description='Visualize accuracy-throughput tradeoffs')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR',
                        help='Comma-separated list of metrics to evaluate')
    return parser.parse_args()


def load_accuracy_results(dataset: str) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from individual detailed_results.json files.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
    if not os.path.exists(dataset_cache_dir):
        print(f"Dataset cache directory {dataset_cache_dir} does not exist")
        return []
    
    results = []
    execution_dir = os.path.join(dataset_cache_dir, 'execution')
    if not os.path.exists(execution_dir):
        print(f"Execution directory {execution_dir} does not exist")
        return []
    
    for video_filename in os.listdir(execution_dir):
        video_dir = os.path.join(execution_dir, video_filename)
        if not os.path.isdir(video_dir):
            continue
            
        tracking_accuracy_dir = os.path.join(video_dir, '070_tracking_accuracy')
        if not os.path.exists(tracking_accuracy_dir):
            continue

        for classifier_tilesize in os.listdir(tracking_accuracy_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            results_path = os.path.join(tracking_accuracy_dir, f'{classifier}_{ts}',
                                        'accuracy', 'detailed_results.json')
            
            if os.path.exists(results_path):
                print(f"Loading accuracy results from {results_path}")
                with open(results_path, 'r') as f:
                    results.append(json.load(f))
    
    print(f"Loaded {len(results)} accuracy evaluation results")
    return results


def load_throughput_results(dataset: str) -> Dict[str, Any]:
    """
    Load throughput results from the measurements directory.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        Dict[str, Any]: Throughput measurement data
    """
    measurements_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'throughput', 'measurements')
    
    if not os.path.exists(measurements_dir):
        print(f"Throughput measurements directory {measurements_dir} does not exist")
        return {}
    
    # Load query execution summaries (aggregated timing data)
    query_summaries_file = os.path.join(measurements_dir, 'query_execution_summaries.json')
    if not os.path.exists(query_summaries_file):
        print(f"Query execution summaries file {query_summaries_file} does not exist")
        return {}
    
    with open(query_summaries_file, 'r') as f:
        query_summaries = json.load(f)
    
    # Load metadata
    metadata_file = os.path.join(measurements_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded throughput data with {len(query_summaries)} stage summaries")
    return {
        'summaries': query_summaries,
        'metadata': metadata
    }


def calculate_naive_runtime(video_name: str, query_summaries: Dict[str, Any], dataset: str) -> float:
    """
    Calculate naive runtime for a specific video.
    
    This matches the naive calculation in p082_throughput_visualize.py,
    which sums the preprocessing stages (detection and tracking).
    
    Args:
        video_name: Name of the video
        query_summaries: Query execution timing data
        dataset: Dataset name
        
    Returns:
        float: Naive runtime in seconds
    """
    config_key = f"{dataset}/{video_name}_groundtruth_0"  # Naive uses groundtruth with tile size 0
    naive_runtime = 0.0
    
    # Add preprocessing time (naive approach)
    preprocessing_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_summaries.items():
        if stage_name in preprocessing_stages and config_key in stage_summaries and stage_summaries[config_key]:
            stage_total = np.mean(stage_summaries[config_key])  # Use mean like in p082
            naive_runtime += stage_total
    
    return naive_runtime


def calculate_query_execution_runtime(video_name: str, classifier: str, tile_size: int, 
                                    query_summaries: Dict[str, Any], dataset: str) -> float:
    """
    Calculate query execution runtime for a specific configuration.
    
    This matches the query execution portion of the bars in p082_throughput_visualize.py,
    ignoring index construction time.
    
    Args:
        video_name: Name of the video
        classifier: Classifier used
        tile_size: Tile size used
        query_summaries: Query execution timing data
        dataset: Dataset name
        
    Returns:
        float: Query execution runtime in seconds
    """
    config_key = f"{dataset}/{video_name}_{classifier}_{tile_size}"
    query_runtime = 0.0
    
    # Add query execution time (specific to this tile size)
    query_stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    for stage_name, stage_summaries in query_summaries.items():
        if stage_name in query_stages and config_key in stage_summaries and stage_summaries[config_key]:
            stage_total = np.mean(stage_summaries[config_key])  # Use mean like in p082
            query_runtime += stage_total
    
    return query_runtime


def match_accuracy_throughput_data(accuracy_results: List[Dict[str, Any]], 
                                 throughput_data: Dict[str, Any], dataset: str = 'b3d') -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match accuracy and throughput data by video/classifier/tilesize combination.
    
    Args:
        accuracy_results: List of accuracy evaluation results
        throughput_data: Throughput measurement data
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: 
            - Individual video data points
            - Dataset-wide aggregated data points with frame-weighted accuracy scores
    """
    matched_data = []
    query_summaries = throughput_data.get('summaries', {})
    query_timings = throughput_data.get('timings', {})
    
    # Key stages for query execution throughput
    query_stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    
    # Cache frame counts to avoid repeated OpenCV calls
    frame_count_cache = {}
    
    # Group data by classifier and tile_size for aggregation
    grouped_data = defaultdict(list)
    
    for result in accuracy_results:
        if not result.get('success', False):
            continue
            
        video_name = result['video_name']
        classifier = result['classifier']
        tile_size = result['tile_size']
        
        # Only include classifiers from CLASSIFIERS_TO_TEST
        if classifier not in CLASSIFIERS_TO_TEST:
            continue
        
        # Create config key for throughput lookup (include dataset prefix)
        config_key = f"{dataset}/{video_name}_{classifier}_{tile_size}"
        
        # Extract accuracy metrics
        metrics = result['metrics']
        hota_score = metrics.get('HOTA', {}).get('HOTA(0)', 0.0)
        mota_score = metrics.get('CLEAR', {}).get('MOTA', 0.0)
        
        # Get frame count (with caching)
        if video_name not in frame_count_cache:
            frame_count_cache[video_name] = get_video_frame_count(dataset, video_name)
        frame_count = frame_count_cache[video_name]
        
        # Calculate total query execution time (for reference)
        total_query_time = 0.0
        stage_times = {}
        
        for stage in query_stages:
            if stage in query_summaries and config_key in query_summaries[stage]:
                # Get the first (and typically only) timing for this config
                stage_time = query_summaries[stage][config_key][0] if query_summaries[stage][config_key] else 0.0
                stage_times[stage] = stage_time
                total_query_time += stage_time
        
        # Calculate query execution runtime only
        query_runtime = calculate_query_execution_runtime(video_name, classifier, tile_size, 
                                                        query_summaries, dataset)
        
        # Calculate throughput (frames per second)
        throughput_fps = frame_count / query_runtime if query_runtime > 0 else 0.0
        
        matched_entry = {
            'video_name': video_name,
            'classifier': classifier,
            'tile_size': tile_size,
            'hota_score': hota_score,
            'mota_score': mota_score,
            'total_query_time': total_query_time,
            'query_runtime': query_runtime,
            'frame_count': frame_count,
            'throughput_fps': throughput_fps,
            'stage_times': stage_times
        }
        
        matched_data.append(matched_entry)
        
        # Group by classifier and tile_size for aggregation
        group_key = (classifier, tile_size)
        grouped_data[group_key].append(matched_entry)
    
    # Create dataset-wide aggregated data
    aggregated_data = []
    for (classifier, tile_size), entries in grouped_data.items():
        if not entries:
            continue
            
        # Calculate frame-weighted averages for accuracy scores
        total_frames = sum([entry['frame_count'] for entry in entries])
        if total_frames > 0:
            # Weight each accuracy score by its frame count
            weighted_hota = sum([entry['hota_score'] * entry['frame_count'] for entry in entries]) / total_frames
            weighted_mota = sum([entry['mota_score'] * entry['frame_count'] for entry in entries]) / total_frames
        else:
            weighted_hota = 0.0
            weighted_mota = 0.0
            
        # Calculate combined runtime and throughput
        total_runtime = sum([entry['query_runtime'] for entry in entries])
        combined_throughput = total_frames / total_runtime if total_runtime > 0 else 0.0
        
        aggregated_entry = {
            'video_name': 'Dataset Average',
            'classifier': classifier,
            'tile_size': tile_size,
            'hota_score': weighted_hota,
            'mota_score': weighted_mota,
            'total_query_time': sum([entry['total_query_time'] for entry in entries]),
            'query_runtime': total_runtime,
            'frame_count': total_frames,
            'throughput_fps': combined_throughput,
            'stage_times': {}  # Not used for aggregated data
        }
        
        aggregated_data.append(aggregated_entry)
    
    print(f"Matched {len(matched_data)} individual accuracy-throughput data points")
    print(f"Created {len(aggregated_data)} dataset-wide aggregated data points")
    return matched_data, aggregated_data


def create_tradeoff_visualization(matched_data: List[Dict[str, Any]], aggregated_data: List[Dict[str, Any]], 
                                 output_dir: str, metrics_list: List[str], query_summaries: Dict[str, Any], 
                                 dataset: str, x_column: str, x_title: str, naive_column: str, 
                                 plot_suffix: str, csv_suffix: str) -> None:
    """
    Create a single tradeoff visualization with configurable x-axis, including dataset-wide aggregated subplot.
    
    Args:
        matched_data: List of individual video accuracy-throughput data points
        aggregated_data: List of dataset-wide aggregated data points
        output_dir: Output directory for visualizations
        metrics_list: List of metrics to visualize
        query_summaries: Query execution timing data
        dataset: Dataset name
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for plot filename
        csv_suffix: Suffix for CSV filename
    """
    print(f"Creating {plot_suffix} tradeoff visualizations...")
    
    if not matched_data:
        print(f"No matched data available for {plot_suffix} visualization")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier data handling
    df = pd.DataFrame(matched_data)
    df_agg = pd.DataFrame(aggregated_data)
    
    # Save matched data to CSV
    csv_file_path = os.path.join(output_dir, f'accuracy_{csv_suffix}_tradeoff.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Saved matched data to: {csv_file_path}")
    
    # Create scatter plots for each metric using Altair
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'hota_score'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'mota_score'
            metric_name = 'MOTA'
        else:
            continue
        
        # Calculate naive values for each video
        unique_videos = sorted(df['video_name'].unique())
        naive_values = {}
        for video in unique_videos:
            naive_runtime = calculate_naive_runtime(video, query_summaries, dataset)
            if naive_column == 'naive_runtime':
                naive_values[video] = naive_runtime
            elif naive_column == 'naive_throughput':
                frame_count = df[df['video_name'] == video]['frame_count'].iloc[0]
                naive_throughput = frame_count / naive_runtime if naive_runtime > 0 else 0.0
                naive_values[video] = naive_throughput
        
        # Add naive data to dataframe
        df_with_naive = df.copy()
        df_with_naive[naive_column] = df_with_naive['video_name'].map(naive_values)  # type: ignore
        
        # Calculate dataset-wide naive values
        total_naive_runtime = sum([calculate_naive_runtime(video, query_summaries, dataset) for video in unique_videos])
        if naive_column == 'naive_runtime':
            dataset_naive_value = total_naive_runtime
        elif naive_column == 'naive_throughput':
            total_frames = sum([df[df['video_name'] == video]['frame_count'].iloc[0] for video in unique_videos])
            dataset_naive_value = total_frames / total_naive_runtime if total_naive_runtime > 0 else 0.0
        
        # Add dataset naive value to aggregated dataframe
        df_agg_with_naive = df_agg.copy()
        df_agg_with_naive[naive_column] = dataset_naive_value
        
        # Create base charts
        base_individual = alt.Chart(df_with_naive)
        base_aggregated = alt.Chart(df_agg_with_naive)
        
        # Create individual video scatter plot
        individual_scatter = base_individual.mark_circle(opacity=0.7).encode(
            x=alt.X(f'{x_column}:Q', title=x_title),
            y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('classifier:N', title='Classifier'),
            size=alt.Size('tile_size:O', title='Tile Size', scale=alt.Scale(range=[20, 200])),
            tooltip=['video_name', 'classifier', 'tile_size', x_column, accuracy_col]
        ).properties(
            width=200,
            height=200
        )
        
        # Create dataset-wide scatter plot
        aggregated_scatter = base_aggregated.mark_circle(opacity=0.8, size=300).encode(
            x=alt.X(f'{x_column}:Q', title=x_title),
            y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('classifier:N', title='Classifier'),
            size=alt.Size('tile_size:O', title='Tile Size', scale=alt.Scale(range=[50, 300])),
            tooltip=['video_name', 'classifier', 'tile_size', x_column, accuracy_col]
        ).properties(
            width=200,
            height=200
        )
        
        # Create naive baseline lines for individual videos
        naive_lines_individual = base_individual.mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=2,
            opacity=0.8
        ).encode(
            x=f'{naive_column}:Q'
        )
        
        # Create naive baseline line for dataset-wide plot
        naive_line_aggregated = base_aggregated.mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=3,
            opacity=0.9
        ).encode(
            x=f'{naive_column}:Q'
        )
        
        # Combine individual video charts
        individual_chart = (individual_scatter + naive_lines_individual).facet(
            facet=alt.Facet('video_name:N', title=None,
                            header=alt.Header(labelExpr="'Video: ' + datum.value")),
            columns=4
        ).resolve_scale(
            x='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff (By Video)',
        )
        
        # Create dataset-wide chart
        dataset_chart = (aggregated_scatter + naive_line_aggregated).properties(
            title=f'{metric_name} vs {x_title} Tradeoff (Dataset Average)',
            width=400,
            height=300
        )
        
        # Combine individual and dataset charts vertically
        combined_chart = alt.vconcat(
            individual_chart,
            dataset_chart
        ).resolve_scale(
            x='independent'
        )
        
        # Save the chart
        plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_tradeoff.png')
        combined_chart.save(plot_path, scale_factor=2)
        print(f"Saved {metric_name} {plot_suffix} tradeoff plot to: {plot_path}")


def create_tradeoff_visualizations(matched_data: List[Dict[str, Any]], aggregated_data: List[Dict[str, Any]], 
                                 output_dir: str, metrics_list: List[str], query_summaries: Dict[str, Any], dataset: str) -> None:
    """
    Create both runtime and throughput tradeoff visualizations.
    
    Args:
        matched_data: List of individual video accuracy-throughput data points
        aggregated_data: List of dataset-wide aggregated data points
        output_dir: Output directory for visualizations
        metrics_list: List of metrics to visualize
        query_summaries: Query execution timing data
        dataset: Dataset name
    """
    # Create runtime visualization
    create_tradeoff_visualization(
        matched_data, aggregated_data, output_dir, metrics_list, query_summaries, dataset,
        x_column='query_runtime',
        x_title='Query Execution Runtime (seconds)',
        naive_column='naive_runtime',
        plot_suffix='runtime',
        csv_suffix='runtime'
    )
    
    # Create throughput visualization
    create_tradeoff_visualization(
        matched_data, aggregated_data, output_dir, metrics_list, query_summaries, dataset,
        x_column='throughput_fps',
        x_title='Throughput (frames/second)',
        naive_column='naive_throughput',
        plot_suffix='throughput',
        csv_suffix='throughput'
    )


def main(args):
    """
    Main function that orchestrates the accuracy-throughput tradeoff visualization.
    
    This function serves as the entry point for the script. It:
    1. Loads accuracy results from p070_accuracy_compute.py
    2. Loads throughput results from p081_throughput_compute.py
    3. Gets video frame counts using OpenCV
    4. Matches the data by video/classifier/tilesize combination
    5. Creates visualizations showing accuracy vs query execution runtime tradeoffs
    6. Creates visualizations showing accuracy vs throughput (frames/second) tradeoffs
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/{video_file}/evaluation/{classifier}_{tile_size}/accuracy/detailed_results.json
        - The script expects throughput results from p081_throughput_compute.py in:
          {CACHE_DIR}/summary/{dataset}/throughput/measurements/query_execution_summaries.json
        - Video files are expected in {DATA_DIR}/{dataset}/{video_name}.mp4 (or other extensions)
        - Only query execution runtime is used (index construction time is ignored)
    """
    # Parse metrics
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    print(f"Processing metrics: {metrics_list}")
    
    for dataset in args.datasets:
        print(f"Starting accuracy-query execution runtime tradeoff visualization for dataset: {dataset}")
        
        # Load accuracy results
        print("Loading accuracy results...")
        accuracy_results = load_accuracy_results(dataset)
        
        if not accuracy_results:
            print(f"No accuracy results found for {dataset}. Please run p070_accuracy_compute.py first.")
            continue
        
        # Load throughput results
        print("Loading throughput results...")
        throughput_data = load_throughput_results(dataset)
        
        if not throughput_data:
            print(f"No throughput results found for {dataset}. Please run p080_throughput_gather.py and p081_throughput_compute.py first.")
            continue
        
        # Match accuracy and throughput data
        print("Matching accuracy and throughput data...")
        matched_data, aggregated_data = match_accuracy_throughput_data(accuracy_results, throughput_data, dataset)
        
        if not matched_data:
            print(f"No matching data points found between accuracy and throughput results for {dataset}.")
            continue
        
        # Create visualizations
        output_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'tradeoff')
        create_tradeoff_visualizations(matched_data, aggregated_data, output_dir, metrics_list, throughput_data['summaries'], dataset)
        
        print(f"Accuracy-throughput tradeoff visualization complete for {dataset}! Results saved to: {output_dir}")
    
    print(f"\nAll datasets processed!")


if __name__ == '__main__':
    main(parse_args())
