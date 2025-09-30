#!/usr/local/bin/python

import argparse
import json
import os
import multiprocessing as mp
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import altair as alt
import pandas as pd

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


TILE_SIZES = [30, 60]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval and create visualizations')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def find_saved_results(cache_dir: str, dataset: str) -> List[Tuple[str, str, int]]:
    """
    Find all video files with saved accuracy results for the specified dataset.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        List[Tuple[str, str, int]]: List of (video_name, classifier, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset, 'execution')
    assert os.path.exists(dataset_cache_dir), f"Dataset cache directory {dataset_cache_dir} does not exist"
    
    video_tile_combinations: list[tuple[str, str, int]] = []
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        assert os.path.isdir(video_dir), f"Video directory {video_dir} is not a directory"
            
        evaluation_dir = os.path.join(video_dir, '070_tracking_accuracy')
        assert os.path.exists(evaluation_dir), f"Evaluation directory {evaluation_dir} does not exist"

        for classifier_tilesize in os.listdir(evaluation_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            results_path = os.path.join(evaluation_dir, f'{classifier}_{ts}',
                                        'accuracy', 'detailed_results.json')
            
            assert os.path.exists(results_path), f"Results path {results_path} does not exist"
            video_tile_combinations.append((video_filename, classifier, ts))
    
    return video_tile_combinations


def load_saved_results(dataset: str) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from individual detailed_results.json files.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    # Find all saved results
    video_tile_combinations = find_saved_results(CACHE_DIR, dataset)
    assert len(video_tile_combinations) > 0, f"No saved results found for dataset {dataset}"

    results = []
    for video_name, classifier, tile_size in video_tile_combinations:
        results_path = os.path.join(CACHE_DIR, dataset, 'execution', video_name,
                                    '070_tracking_accuracy', f'{classifier}_{tile_size}',
                                    'accuracy', 'detailed_results.json')
        
        print(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            results.append(json.load(f))
    
    print(f"Loaded {len(results)} saved evaluation results")
    return results


def get_results(eval_task: Callable[[], dict], res_queue: "mp.Queue[tuple[int, dict]]", worker_id: int):
    result = eval_task()
    res_queue.put((worker_id, result))


def main(args):
    """
    Main function that orchestrates the tracking accuracy evaluation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with tracking results for the specified dataset and tile size(s)
    2. Runs accuracy evaluation using TrackEval's B3D evaluation methods
    3. Creates summary reports of the accuracy results
    4. Creates visualizations for each dataset
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tracking results from 060_exec_track.py in:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/proxy_{tile_size}/tracking.jsonl
        - Groundtruth data should be in:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/tracking.jsonl
        - Multiple metrics are evaluated: HOTA, CLEAR (MOTA)
    """
    print(f"Starting tracking accuracy evaluation for datasets: {args.datasets}")
    
    # Process each dataset separately
    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_results = load_saved_results(dataset)
        assert len(dataset_results) > 0, f"No results found for dataset {dataset}"
        
        # Print summary for this dataset
        successful_results = [r for r in dataset_results if r['success']]
        failed_results = [r for r in dataset_results if not r['success']]
        
        print(f"  Successful evaluations: {len(successful_results)}")
        print(f"  Failed evaluations: {len(failed_results)}")
        
        if failed_results:
            print("\nFailed evaluations:")
            for result in failed_results:
                error_msg = result.get('error', 'Unknown error')
                print(f"  {result['video_name']} (tile size {result['tile_size']}): {error_msg}")
        
        assert len(successful_results) > 0, f"No successful results for dataset {dataset}"
        # Create output directory for this dataset
        output_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'accuracy')
        
        # Create visualizations for this dataset
        visualize_tracking_accuracy(successful_results, output_dir)
        print(f"Results saved to: {output_dir}")


def visualize_compared_accuracy_bar(video_tile_groups: Dict[str, Dict[int, Dict[str, List]]], 
                                    sorted_videos: List[str], sorted_tile_sizes: List[int],
                                    num_videos: int, num_tile_sizes: int, score_field: str,
                                    metric_name: str, xlabel: str, output_path: str) -> None:
    """
    Create a comparison plot for tracking accuracy scores by video and tile size.
    
    Args:
        video_tile_groups: Grouped data by video and tile size
        sorted_videos: List of video names in sorted order
        sorted_tile_sizes: List of tile sizes in sorted order
        num_videos: Number of videos
        num_tile_sizes: Number of tile sizes
        score_field: Field name for scores ('hota_scores' or 'clear_scores')
        metric_name: Display name for the metric ('HOTA' or 'MOTA')
        xlabel: Label for x-axis
        output_path: Path to save the plot
    """
    # Prepare data for the chart
    chart_data = []
    for video_name in sorted_videos:
        for tile_size in sorted_tile_sizes:
            if tile_size in video_tile_groups[video_name]:
                group_data = video_tile_groups[video_name][tile_size]
                
                # Sort by scores (descending)
                sorted_indices = sorted(range(len(group_data[score_field])), 
                                       key=lambda x: group_data[score_field][x], reverse=True)
                
                sorted_labels = [group_data['labels'][idx] for idx in sorted_indices]
                sorted_scores = [group_data[score_field][idx] for idx in sorted_indices]
                
                for label, score in zip(sorted_labels, sorted_scores):
                    chart_data.append({
                        'Video': video_name,
                        'Tile_Size': tile_size,
                        'Classifier': label,
                        'Score': score
                    })
    
    df = pd.DataFrame(chart_data)
    
    # Create horizontal bar chart with text labels inside bars
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X('Score:Q', title=xlabel, scale=alt.Scale(domain=[0, 1])),
        # y=alt.Y('Classifier:N', sort='-x', axis=alt.Axis(labels=False, ticks=False, title=None)),
        tooltip=['Video', 'Tile_Size', 'Classifier', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(
        width=200,
        height=200
    )
    
    # Add text labels inside the bars
    text = alt.Chart(df).mark_text(
        align='right',
        baseline='middle',
        dx=-3,  # Small offset from the left edge of the bar
        color='white'
    ).transform_calculate(text='datum.Score > 0.01 ? format(datum.Score, ".2f") : ""').encode(
        x=alt.X('Score:Q'),
        # y=alt.Y('Classifier:N', sort='-x', axis=alt.Axis(labels=False, ticks=False, title=None)),
        text=alt.Text('text:N'),
    )

    labels = alt.Chart(df).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontWeight='bold',
        color='black'
    ).transform_calculate(Score2='datum.Score * 0.0001').encode(
        x=alt.X('Score2:Q'),
        text=alt.Text('Classifier:N'),
        # white if score > 0, red if score < 0
        color=alt.condition(alt.datum.Score > 0.1, alt.value('white'), alt.value('black'))
    )
    
    # Layer the charts first, then apply faceting
    chart = (bars + labels + text).encode(
        y=alt.Y('Classifier:N', sort='-x', axis=alt.Axis(labels=False, ticks=False, title=None)),
    ).resolve_scale(y='independent').facet(
        row=alt.Row('Tile_Size:O', title='Tile Size'),
        column=alt.Column('Video:N', title=None)
    ).resolve_scale(y='independent').properties(padding=0)
    
    # Save the chart
    chart.save(output_path, scale_factor=2)


def visualize_tracking_accuracy(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for tracking accuracy results using altair.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for visualizations
    """
    print("Creating visualizations...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful evaluation results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier data handling
    data = []
    for result in successful_results:
        metrics = result['metrics']
        data.append({
            'Video': result['video_name'],
            'Classifier': result['classifier'],
            'Tile_Size': result['tile_size'],
            'HOTA': metrics.get('HOTA', {}).get('HOTA(0)', 0.0),
            'MOTA': metrics.get('CLEAR', {}).get('MOTA', 0.0)
        })
    
    df = pd.DataFrame(data)
    
    # Save results to CSV
    csv_file_path = os.path.join(output_dir, 'accuracy_results.csv')
    df.to_csv(csv_file_path, index=False)
    
    # Group data by video and tile size for bar plots
    video_tile_groups = {}
    for _, row in df.iterrows():
        video_name = row['Video']
        tile_size = row['Tile_Size']
        if video_name not in video_tile_groups:
            video_tile_groups[video_name] = {}
        if tile_size not in video_tile_groups[video_name]:
            video_tile_groups[video_name][tile_size] = {
                'labels': [],
                'hota_scores': [],
                'clear_scores': []
            }
        video_tile_groups[video_name][tile_size]['labels'].append(row['Classifier'])
        video_tile_groups[video_name][tile_size]['hota_scores'].append(row['HOTA'])
        video_tile_groups[video_name][tile_size]['clear_scores'].append(row['MOTA'])
    
    # Sort videos and tile sizes for consistent ordering
    sorted_videos = sorted(video_tile_groups.keys())
    sorted_tile_sizes = sorted(df['Tile_Size'].unique())
    num_videos = len(sorted_videos)
    num_tile_sizes = len(sorted_tile_sizes)
    
    # Create comparison plots for both HOTA and MOTA scores
    visualize_compared_accuracy_bar(
        video_tile_groups, sorted_videos, sorted_tile_sizes, num_videos, num_tile_sizes,
        'hota_scores', 'HOTA', 'HOTA Score',
        os.path.join(output_dir, 'hota_comparison.png')
    )
    
    visualize_compared_accuracy_bar(
        video_tile_groups, sorted_videos, sorted_tile_sizes, num_videos, num_tile_sizes,
        'clear_scores', 'MOTA', 'MOTA Score',
        os.path.join(output_dir, 'mota_comparison.png')
    )
    
    # Tile size comparison using altair (if multiple tile sizes exist)
    if len(df['Tile_Size'].unique()) > 1:
        # HOTA comparison
        hota_chart = alt.Chart(df).mark_boxplot().encode(
            x='Tile_Size:O',
            y='HOTA:Q',
            color='Tile_Size:O'
        ).properties(
            title='HOTA by Tile Size',
            width=250,
            height=300
        )
        
        # MOTA comparison
        mota_chart = alt.Chart(df).mark_boxplot().encode(
            x='Tile_Size:O',
            y='MOTA:Q',
            color='Tile_Size:O'
        ).properties(
            title='MOTA by Tile Size',
            width=250,
            height=300
        )
        
        # Combine charts horizontally
        combined_chart = alt.hconcat(hota_chart, mota_chart, spacing=20)
        
        # Save the chart
        combined_chart.save(os.path.join(output_dir, 'tile_size_comparison.png'), scale_factor=2)


if __name__ == '__main__':
    main(parse_args())

