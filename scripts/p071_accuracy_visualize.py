#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import multiprocessing as mp
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import altair as alt
import pandas as pd

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


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
    parser = argparse.ArgumentParser(description='Create visualizations for tracking accuracy results from p070_accuracy_compute.py')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--clear', action='store_true',
                        help='Remove and recreate the 070_accuracy evaluation directory for each dataset')
    return parser.parse_args()


def find_saved_results(cache_dir: str, dataset: str) -> List[Tuple[str, int, str]]:
    """
    Find all classifier/tilesize/tilepadding combinations with saved accuracy results.
    
    Scans the evaluation directory to discover all classifier/tilesize/tilepadding combinations
    that have completed accuracy evaluation results available.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        List[Tuple[str, int, str]]: List of (classifier, tilesize, tilepadding) tuples
    """
    # Construct path to evaluation directory for this dataset
    evaluation_dir = os.path.join(cache_dir, dataset, 'evaluation', '070_accuracy')
    assert os.path.exists(evaluation_dir), f"Evaluation directory {evaluation_dir} does not exist"
    
    # Collect all classifier/tilesize/tilepadding combinations
    classifier_tile_combinations: list[tuple[str, int, str]] = []
    
    # Iterate through all classifier-tilesize-tilepadding directories
    for classifier_tilesize_tilepadding in os.listdir(evaluation_dir):
        # Parse classifier, tile size, and tilepadding from directory name
        parts = classifier_tilesize_tilepadding.split('_')
        assert len(parts) == 3, f"Expected format 'classifier_tilesize_tilepadding', got '{classifier_tilesize_tilepadding}'"
        classifier, tilesize, tilepadding = parts
        ts = int(tilesize)
        
        # Verify that the required DATASET.json file exists
        # This ensures the evaluation was completed successfully
        dataset_results_path = os.path.join(evaluation_dir, f'{classifier}_{ts}_{tilepadding}', 'DATASET.json')
        assert os.path.exists(dataset_results_path), f"Dataset results path {dataset_results_path} does not exist"
        
        # Add this combination to our list
        classifier_tile_combinations.append((classifier, ts, tilepadding))
    
    return classifier_tile_combinations


def load_saved_results(dataset: str, combined: bool = False) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from result files.
    
    Loads either individual video results or combined dataset results based on
    the combined parameter. Individual results are used for per-video visualizations,
    while combined results are used for dataset-level visualizations.
    
    Args:
        dataset (str): Dataset name
        combined (bool): Whether to load combined results (DATASET.json) or individual video results
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    # Find all classifier/tilesize combinations with available results
    classifier_tile_combinations = find_saved_results(CACHE_DIR, dataset)
    assert len(classifier_tile_combinations) > 0, f"No saved results found for dataset {dataset}"

    # Initialize results list
    results = []
    evaluation_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
    
    # Process each classifier/tilesize/tilepadding combination
    for classifier, tilesize, tilepadding in classifier_tile_combinations:
        combination_dir = os.path.join(evaluation_dir, f'{classifier}_{tilesize}_{tilepadding}')
        
        # Load result files based on the combined parameter
        for filename in os.listdir(combination_dir):
            # Load DATASET.json if combined=True, otherwise load individual video files
            if filename.endswith('.json') and (filename == 'DATASET.json') == combined:
                results_path = os.path.join(combination_dir, filename)
                
                print(f"Loading results from {results_path}")
                # Load and parse JSON result file
                with open(results_path, 'r') as f:
                    result_data = json.load(f)
                    results.append(result_data)
    
    print(f"Loaded {len(results)} saved evaluation results")
    return results


def get_results(eval_task: Callable[[], dict], res_queue: "mp.Queue[tuple[int, dict]]", worker_id: int):
    """
    Helper function for multiprocessing to collect results from worker processes.
    
    Args:
        eval_task: Function to execute
        res_queue: Queue to put results in
        worker_id: ID of the worker process
    """
    # Execute the task and put result in queue with worker ID
    result = eval_task()
    res_queue.put((worker_id, result))


def main(args):
    """
    Main function that orchestrates the tracking accuracy visualization process.
    
    This function serves as the entry point for the script. It:
    1. Finds all classifier/tilesize/tilepadding combinations with saved accuracy results
    2. Loads individual video result files from the new evaluation directory structure
    3. Creates visualizations comparing accuracy across videos, classifiers, tile sizes, and tilepadding values
    4. Generates summary reports and charts for each dataset
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/{classifier}_{tilesize}_{tilepadding}/
          ├── DATASET.json (combined results)
          ├── {video_name}.json (individual video results)
          └── LOG.txt (evaluation logs)
        - Multiple metrics are visualized: HOTA, CLEAR (MOTA)
        - Visualizations are saved to: {CACHE_DIR}/{dataset}/evaluation/071_accuracy_visualize/
    """
    print(f"Starting tracking accuracy visualization for datasets: {args.datasets}")
    
    # Process each dataset separately to create independent visualizations
    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Clear 070_accuracy directory if requested
        if args.clear:
            accuracy_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
            if os.path.exists(accuracy_dir):
                shutil.rmtree(accuracy_dir)
                print(f"Cleared existing 070_accuracy directory: {accuracy_dir}")
        
        # Load individual video results for per-video visualizations
        # This loads results for all classifier/tilesize/tilepadding combinations
        dataset_results = load_saved_results(dataset, combined=False)
        assert len(dataset_results) > 0, f"No results found for dataset {dataset}"
        
        # Create output directory for this dataset's visualizations
        output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '071_accuracy_visualize')
        
        # Clean and recreate output directory to ensure fresh results
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create per-video visualizations using individual results
        visualize_tracking_accuracy(dataset_results, output_dir, combined=False)
        
        # Create dataset-level visualizations using combined results
        combined_results = load_saved_results(dataset, combined=True)
        visualize_tracking_accuracy(combined_results, output_dir, combined=True)
        
        print(f"Results saved to: {output_dir}")


def visualize_compared_accuracy_bar(video_tile_tilepadding_groups: Dict[str, Dict[int, Dict[str, Dict[str, List]]]], 
                                    sorted_videos: List[str], sorted_tilesizes: List[int], sorted_tilepadding_values: List[str],
                                    score_field: str, xlabel: str, output_path: str):
    """
    Create a comparison plot for tracking accuracy scores by video, tile size, and tilepadding.
    
    Creates a faceted bar chart showing accuracy scores for different classifiers
    across videos, tile sizes, and tilepadding values. Each facet shows one video-tilesize-tilepadding combination,
    with bars representing different classifiers sorted by performance.
    
    Args:
        video_tile_tilepadding_groups: Grouped data by video, tile size, and tilepadding
        sorted_videos: List of video names in sorted order
        sorted_tilesizes: List of tile sizes in sorted order
        sorted_tilepadding_values: List of tilepadding values in sorted order
        score_field: Field name for scores ('hota_scores' or 'clear_scores')
        xlabel: Label for x-axis
        output_path: Path to save the plot
    """
    # Prepare data for the chart by flattening grouped data
    chart_data = []
    
    # Iterate through all video-tilesize-tilepadding combinations
    for video_name in sorted_videos:
        for tilesize in sorted_tilesizes:
            for tilepadding in sorted_tilepadding_values:
                # Check if this combination has data
                if (tilesize in video_tile_tilepadding_groups[video_name] and 
                    tilepadding in video_tile_tilepadding_groups[video_name][tilesize]):
                    group_data = video_tile_tilepadding_groups[video_name][tilesize][tilepadding]
                    
                    # Sort classifiers by their scores (descending order)
                    sorted_indices = sorted(range(len(group_data[score_field])), 
                                           key=lambda x: group_data[score_field][x], reverse=True)
                    
                    # Extract sorted labels and scores
                    sorted_labels = [group_data['labels'][idx] for idx in sorted_indices]
                    sorted_scores = [group_data[score_field][idx] for idx in sorted_indices]
                    
                    # Add each classifier-score pair to chart data
                    for label, score in zip(sorted_labels, sorted_scores):
                        chart_data.append({
                            'Video': video_name,
                            'Tile_Size': tilesize,
                            'Tile_Padding': tilepadding,
                            'Classifier': label,
                            'Classifier_Tile_Padding': f'{label}_{tilepadding}',
                            'Score': score
                        })
    
    # Convert to pandas DataFrame for Altair
    df = pd.DataFrame(chart_data)
    
    # Create horizontal bar chart with text labels inside bars
    # Main bars showing the scores
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X('Score:Q', title=xlabel, scale=alt.Scale(domain=[0, 1])),
        # yOffset=alt.YOffset('Dilate:N'),
        color=alt.Color('Tile_Padding:N', title='Tile Padding'),
        tooltip=['Video', 'Tile_Size', 'Tile_Padding', 'Classifier', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(
        width=200,
        height=200
    )
    
    # Add score text labels inside the bars (white text)
    text = alt.Chart(df).mark_text(
        align='right',
        baseline='middle',
        dx=-3,  # Small offset from the right edge of the bar
        color='white'
    ).transform_calculate(text='datum.Score > 0.01 ? format(datum.Score, ".2f") : ""').encode(
        x=alt.X('Score:Q'),
        text=alt.Text('text:N'),
        # yOffset=alt.YOffset('Dilate:N')
    )

    # Add classifier name labels on the left side of bars
    labels = alt.Chart(df).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontWeight='bold',
        color='black'
    ).transform_calculate(Score2='datum.Score * 0.0001').encode(
        x=alt.X('Score2:Q'),
        text=alt.Text('Classifier_Tile_Padding:N'),
        # Use white text for high scores, black for low scores
        color=alt.condition(alt.datum.Score > 0.1, alt.value('white'), alt.value('black')),
        # yOffset=alt.YOffset('Dilate:N')
    )
    
    # Layer the charts (bars + labels + text) and apply faceting
    # Facet by tile size (rows), tilepadding (columns), and video (sub-columns)
    chart = (bars + labels + text).encode(
        y=alt.Y('Classifier_Tile_Padding:N', sort='-x', axis=alt.Axis(labels=False, ticks=False, title=None)),
        # yOffset=alt.YOffset('Tile_Padding:N'),
        # detail=alt.Detail('Tile_Padding:N'),
        # color=alt.Color('Tile_Padding:N', title='Tile Padding')
    ).resolve_scale(y='independent').facet(
        row=alt.Row('Tile_Size:O', title='Tile Size'),
        column=alt.Column('Video:N', title=None),
        # column=alt.Column('Tile_Padding:N', title='Tile Padding'),
        # facet=alt.Facet('Video:N', title=None)
    ).resolve_scale(y='independent').properties(padding=0)
    
    # Save the chart as PNG with high resolution
    chart.save(output_path, scale_factor=2)


def visualize_tracking_accuracy(results: List[Dict[str, Any]], output_dir: str, combined: bool = False):
    """
    Create visualizations for tracking accuracy results using Altair.
    
    Processes evaluation results and creates comparison charts showing
    accuracy scores across different classifiers, videos, and tile sizes.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for visualizations
        combined (bool): Whether these are combined dataset results or individual video results
    """
    print("Creating visualizations...")
    
    # Convert results to DataFrame for easier data handling
    data = []
    for result in results:
        metrics = result['metrics']
        # Extract key metrics from the nested structure
        data.append({
            'Video': result['video_name'] or 'Combined',  # Use 'Combined' for dataset-level results
            'Classifier': result['classifier'],
            'Tile_Size': result['tilesize'],
            'Tile_Padding': result['tilepadding'],
            'HOTA': metrics.get('HOTA', {}).get('HOTA(0)', 0.0),  # Extract HOTA score
            'MOTA': metrics.get('CLEAR', {}).get('MOTA', 0.0)   # Extract MOTA score
        })
    
    df = pd.DataFrame(data)

    # Set prefix for output files based on whether these are combined results
    prefix = "combined_" if combined else ""
    
    # Save results to CSV for further analysis
    csv_file_path = os.path.join(output_dir, f'{prefix}accuracy_results.csv')
    df.to_csv(csv_file_path, index=False)
    
    # Group data by video, tile size, and tilepadding for bar plots
    # This creates a nested structure: video -> tilesize -> tilepadding -> {labels, scores}
    video_tile_tilepadding_groups = {}
    for _, row in df.iterrows():
        video_name = row['Video']
        tilesize = row['Tile_Size']
        tilepadding = row['Tile_Padding']
        
        # Initialize nested structure if needed
        if video_name not in video_tile_tilepadding_groups:
            video_tile_tilepadding_groups[video_name] = {}
        if tilesize not in video_tile_tilepadding_groups[video_name]:
            video_tile_tilepadding_groups[video_name][tilesize] = {}
        if tilepadding not in video_tile_tilepadding_groups[video_name][tilesize]:
            video_tile_tilepadding_groups[video_name][tilesize][tilepadding] = {
                'labels': [],
                'hota_scores': [],
                'clear_scores': []
            }
        
        # Add this classifier's data to the group
        video_tile_tilepadding_groups[video_name][tilesize][tilepadding]['labels'].append(row['Classifier'])
        video_tile_tilepadding_groups[video_name][tilesize][tilepadding]['hota_scores'].append(row['HOTA'])
        video_tile_tilepadding_groups[video_name][tilesize][tilepadding]['clear_scores'].append(row['MOTA'])
    
    # Sort videos, tile sizes, and tilepadding values for consistent ordering in visualizations
    sorted_videos = sorted(video_tile_tilepadding_groups.keys())
    sorted_tilesizes = sorted(df['Tile_Size'].unique())
    sorted_tilepadding_values = sorted(df['Tile_Padding'].unique())
    
    # Create comparison plots for both HOTA and MOTA scores
    # Each plot shows classifiers ranked by performance for each video-tilesize-tilepadding combination
    visualize_compared_accuracy_bar(video_tile_tilepadding_groups, sorted_videos, sorted_tilesizes, sorted_tilepadding_values,
        'hota_scores', 'HOTA Score', os.path.join(output_dir, f'{prefix}hota.png'))
    
    visualize_compared_accuracy_bar(video_tile_tilepadding_groups, sorted_videos, sorted_tilesizes, sorted_tilepadding_values,
        'clear_scores', 'MOTA Score', os.path.join(output_dir, f'{prefix}mota.png'))


if __name__ == '__main__':
    main(parse_args())
