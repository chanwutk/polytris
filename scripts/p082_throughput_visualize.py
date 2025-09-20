#!/usr/local/bin/python

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any
import tqdm

from polyis.utilities import CACHE_DIR

FORMATS = ['png']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize runtime breakdown of training configurations')
    parser.add_argument('--dataset', type=str, 
                        default='b3d',
                        help='Dataset name to process')
    return parser.parse_args()


def load_measurements(measurements_dir: str) -> tuple[dict, dict, dict]:
    """Load the processed measurement data."""
    index_file = os.path.join(measurements_dir, 'index_construction_measurements.json')
    query_timings_file = os.path.join(measurements_dir, 'query_execution_timings.json')
    query_summaries_file = os.path.join(measurements_dir, 'query_execution_summaries.json')
    metadata_file = os.path.join(measurements_dir, 'metadata.json')
    
    with open(index_file, 'r') as f:
        index_timings = json.load(f)
    
    with open(query_timings_file, 'r') as f:
        query_timings_data = json.load(f)
    
    with open(query_summaries_file, 'r') as f:
        query_summaries_data = json.load(f)
    
    # Reconstruct the query_timings structure
    query_timings = {
        'timings': query_timings_data,
        'summaries': query_summaries_data
    }
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded index construction measurements")
    print(f"Loaded query execution timings and summaries")
    print(f"Loaded metadata for {len(metadata['videos'])} videos")
    
    return index_timings, query_timings, metadata


def visualize_breakdown_query_execution(query_timings: dict, output_dir: str, dataset: str, 
                                        video: str, video_name: str):
    """Create query execution visualization for a specific video.
    
    Args:
        query_timings: Query timing data
        output_dir: Output directory for saving plots
        dataset: Dataset name
        video: Specific video for per-video analysis
        video_name: Display name for the video
    """
    
    # Prepare data for pipeline visualization
    stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    stage_names = ['Classify', 'Compress', 'Detect', 'Track']
    
    # Group by classifier and tile size 
    classifiers = ['SimpleCNN', 'groundtruth']  # Default classifiers
    tile_sizes = [30, 60]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (stage, stage_name) in enumerate(tqdm.tqdm(zip(stages, stage_names), total=len(stages))):
        ax = axes[i]
        
        # Collect data for this stage grouped by operation
        config_labels = []
        op_data = defaultdict(list)  # op_name -> list of values for each config
        
        for classifier in classifiers:
            for tile_size in tile_sizes:
                # Per-video analysis
                config_key = f"{video}_{classifier}_{tile_size}"
                config_labels.append(f"{classifier} {tile_size}")
                
                # Get individual timings for this config to group by operation
                config_ops = defaultdict(float)
                if config_key in query_timings['timings'][stage]:
                    for timing in query_timings['timings'][stage][config_key]:
                        op_name = timing.get('op', 'unknown')
                        config_ops[op_name] += timing['time']
                
                # Ensure all operations have a value for this config (0 if not present)
                all_ops = set()
                for other_classifier in classifiers:
                    for other_tile_size in tile_sizes:
                        other_config_key = f"{video}_{other_classifier}_{other_tile_size}"
                        if other_config_key in query_timings['timings'][stage]:
                            for timing in query_timings['timings'][stage][other_config_key]:
                                all_ops.add(timing.get('op', 'unknown'))
                
                for op_name in all_ops:
                    op_data[op_name].append(config_ops.get(op_name, 0))
        
        if config_labels and op_data:
            # Calculate total values for each config to sort by bar size
            config_totals = []
            for i in range(len(config_labels)):
                total = sum(op_data[op_name][i] for op_name in op_data.keys() if i < len(op_data[op_name]))
                config_totals.append(total)
            
            # Sort configs by total value (descending)
            sorted_indices = sorted(range(len(config_labels)), key=lambda i: config_totals[i], reverse=True)
            sorted_config_labels = [config_labels[i] for i in sorted_indices]
            
            # Sort operation data to match the sorted config order
            sorted_op_data = {}
            for op_name, values in op_data.items():
                sorted_op_data[op_name] = [values[i] for i in sorted_indices]
            
            # Create stacked horizontal bar chart
            y = np.arange(len(sorted_config_labels))
            height = 0.6
            left = np.zeros(len(sorted_config_labels))
            
            # Color palette for different operations
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Plot each operation as a layer in the stacked horizontal bar
            for j, (op_name, values) in enumerate(sorted(sorted_op_data.items())):
                if any(v > 0 for v in values):  # Only plot if there are non-zero values
                    color = colors[j % len(colors)]
                    ax.barh(y, values, height, left=left, label=op_name, color=color, alpha=0.8)
                    left += np.array(values)
            
            # Set title
            ax.set_title(f'{stage_name} Runtime by Operation')
            ax.set_xlabel('Runtime (seconds)')
            ax.set_yticks(y)
            ax.set_yticklabels(sorted_config_labels)
            ax.grid(True, alpha=0.3)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Save plot with appropriate naming and directory structure
    subtitle = video_name
    output_dir = os.path.join(output_dir, 'per_video')
    os.makedirs(output_dir, exist_ok=True)
    suffix = video_name.replace('/', '_').replace('.', '_') if video_name else 'unknown'

    plt.suptitle(f'Query Execution Pipeline Runtime Breakdown ({subtitle})', fontsize=16)
    plt.tight_layout()

    for fmt in FORMATS:
        plt.savefig(os.path.join(output_dir, f'query_execution_pipeline_{suffix}.{fmt}'), 
                    dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_overall_runtime(index_timings: dict, query_timings: dict, 
                              output_dir: str, video: str, video_name: str):
    """Create comparative analysis between index construction and query execution.
    
    Args:
        index_timings: Index construction timing data
        query_timings: Query execution timing data
        output_dir: Output directory for saving plots
        video: Specific video for per-video analysis
        video_name: Display name for the video
    """
    
    # Calculate breakdown for index construction stages
    index_stages = {
        'Detection': 0,
        'Create Training Data': 0,
        'Classifier Training': 0
    }
    
    # Index construction stage breakdown
    for stage_name, stage_summaries in index_timings['summaries'].items():
        stage_total = 0
        for k, times in stage_summaries.items():
            # For per-video analysis, only include configs for this video
            if not k.startswith(video + '_'):
                continue
            if times:
                stage_total += np.sum(times)
        
        if '011_tune_detect' in stage_name:
            index_stages['Detection'] += stage_total
        elif '012_tune_create_training_data' in stage_name:
            index_stages['Create Training Data'] += stage_total
        elif '013_tune_train_classifier' in stage_name:
            index_stages['Classifier Training'] += stage_total
    
    # Discover all classifiers present in the query timing data
    all_classifiers = set()
    for stage_name, stage_summaries in query_timings['summaries'].items():
        for config_key in stage_summaries.keys():
            # Extract classifier from config key format: video_classifier_tilesize
            parts = config_key.split('_')
            if len(parts) >= 3:
                # Find classifier part (between video and tile size)
                if config_key.startswith(video + '_'):
                    # Extract classifier by removing video prefix and tile size suffix
                    remaining = '_'.join(parts[1:])
                    # Remove tile size suffix (last part if it's numeric)
                    if remaining.split('_')[-1].isdigit():
                        classifier = '_'.join(remaining.split('_')[:-1])
                    else:
                        classifier = remaining
                    all_classifiers.add(classifier)
    
    # Sort classifiers for consistent ordering, with 'groundtruth' last
    sorted_classifiers = sorted([c for c in all_classifiers if c != 'groundtruth'])
    if 'groundtruth' in all_classifiers:
        sorted_classifiers.append('groundtruth')
    
    # Calculate breakdown for query execution stages per classifier
    classifier_query_stages = {}
    for classifier in sorted_classifiers:
        classifier_query_stages[classifier] = {
            'Classify': 0,
            'Compress': 0,
            'Detect': 0,
            'Track': 0
        }
    
    # Query execution stage breakdown per classifier
    excluded_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name in excluded_stages:
            continue
        
        for classifier in sorted_classifiers:
            # Per-video analysis
            stage_total = 0
            for config_key, times in stage_summaries.items():
                # Only include configs for this video and this classifier
                if not config_key.startswith(video + '_'):
                    continue
                # Extract classifier from config key
                parts = config_key.split('_')
                if len(parts) >= 3:
                    remaining = '_'.join(parts[1:])
                    if remaining.split('_')[-1].isdigit():
                        config_classifier = '_'.join(remaining.split('_')[:-1])
                    else:
                        config_classifier = remaining
                    if config_classifier == classifier and times:
                        stage_total += np.mean(times)
            
            if '020_exec_classify' in stage_name:
                classifier_query_stages[classifier]['Classify'] += stage_total
            elif '030_exec_compress' in stage_name:
                classifier_query_stages[classifier]['Compress'] += stage_total
            elif '040_exec_detect' in stage_name:
                classifier_query_stages[classifier]['Detect'] += stage_total
            elif '060_exec_track' in stage_name:
                classifier_query_stages[classifier]['Track'] += stage_total
    
    # Calculate breakdown for preprocessing stages only
    preprocessing_stages = {
        'Detect': 0.,
        'Track': 0.
    }
    
    # Preprocessing stage breakdown
    preprocessing_stage_names = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name not in preprocessing_stage_names:
            continue
        
        # Per-video analysis
        stage_total = 0
        for config_key, times in stage_summaries.items():
            if not config_key.startswith(video + '_'):
                continue
            if times:
                stage_total += np.mean(times)
        
        if '001_preprocess_groundtruth_detection' in stage_name:
            preprocessing_stages['Detect'] += float(stage_total)
        elif '002_preprocess_groundtruth_tracking' in stage_name:
            preprocessing_stages['Track'] += float(stage_total)
    
    # Prepare data for stacked bar chart
    width = 0.6
    
    # Colors for different operations - expand palette for more classifiers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
              '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', 
              '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']
    
    # Index construction stack
    index_values = list(index_stages.values())
    index_labels = list(index_stages.keys())
    
    # Get preprocessing values first (needed for category totals calculation)
    preprocessing_values = list(preprocessing_stages.values())
    preprocessing_labels = list(preprocessing_stages.keys())
    
    # Build categories dynamically based on available classifiers
    categories_with_index = ['Index Construction']
    categories_without_index = []
    
    # Add query execution categories for each classifier
    for classifier in sorted_classifiers:
        display_name = classifier.title() if classifier != 'groundtruth' else 'Groundtruth'
        categories_with_index.append(f'Query Execution\n(Classifier: {display_name})')
        categories_without_index.append(f'Query Execution\n(Classifier: {display_name})')
    
    # Add naive category
    categories_with_index.append('Naive')
    categories_without_index.append('Naive')
    
    # Create figure with dynamic width based on number of classifiers
    fig_width = max(20, 8 + 2 * len(sorted_classifiers))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 8))
    
    # Calculate totals for each category to sort by bar size
    category_totals = [float(sum(index_values))]  # Index Construction
    for classifier in sorted_classifiers:
        category_totals.append(float(sum(classifier_query_stages[classifier].values())))
    category_totals.append(float(sum(preprocessing_values)))  # Naive
    
    # Sort categories by total value (descending)
    sorted_indices1 = sorted(range(len(categories_with_index)), key=lambda i: category_totals[i], reverse=True)
    sorted_categories_with_index = [categories_with_index[i] for i in sorted_indices1]
    y1 = np.arange(len(sorted_categories_with_index))
    
    # Create stacked horizontal bars for subplot 1
    left_values = [0.0] * len(sorted_categories_with_index)
    
    # Map original indices to sorted positions
    pos_map = {orig_idx: sorted_indices1.index(orig_idx) for orig_idx in range(len(categories_with_index))}
    
    # Plot index construction stages
    for i, (value, label) in enumerate(zip(index_values, index_labels)):
        if value > 0:
            pos = pos_map[0]  # Index Construction is at position 0
            ax1.barh(y1[pos], value, width, left=left_values[pos], 
                   label=f'Index: {label}', color=colors[i], alpha=0.8)
            left_values[pos] += value
    
    # Plot query execution stages for each classifier
    stage_labels = ['Classify', 'Compress', 'Detect', 'Track']
    for classifier_idx, classifier in enumerate(sorted_classifiers):
        query_values = [classifier_query_stages[classifier][label] for label in stage_labels]
        for i, (value, label) in enumerate(zip(query_values, stage_labels)):
            if value > 0:
                pos = pos_map[1 + classifier_idx]  # Query Execution classifiers start at position 1
                color_idx = (i + len(index_values)) % len(colors)
                ax1.barh(y1[pos], value, width, left=left_values[pos], 
                       label=f'Query: {label}' if classifier_idx == 0 else '', 
                       color=colors[color_idx], alpha=0.8)
                left_values[pos] += value
    
    # Plot preprocessing stages
    for i, (value, label) in enumerate(zip(preprocessing_values, preprocessing_labels)):
        if value > 0:
            pos = pos_map[len(categories_with_index) - 1]  # Naive is at last position
            color_idx = (i + len(index_values) + len(stage_labels)) % len(colors)
            ax1.barh(y1[pos], value, width, left=left_values[pos], 
                   label=f'Naive: {label}', color=colors[color_idx], alpha=0.8)
            left_values[pos] += value
    
    ax1.set_xlabel('Runtime (seconds)')
    ax1.set_title('With Index Construction')
    ax1.set_yticks(y1)
    ax1.set_yticklabels(sorted_categories_with_index)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Add total value labels at the end of bars for subplot 1
    for i, total in enumerate(left_values):
        if total <= 0:
            continue
        ax1.text(total + max(left_values)*0.01, y1[i], f'{total:.1f}s', 
                ha='left', va='center', fontweight='bold', fontsize='small')
    
    # === SUBPLOT 2: Without Index Construction ===
    
    # Calculate totals for each category to sort by bar size (excluding index construction)
    category_totals2 = []
    for classifier in sorted_classifiers:
        category_totals2.append(float(sum(classifier_query_stages[classifier].values())))
    category_totals2.append(float(sum(preprocessing_values)))  # Naive
    
    # Sort categories by total value (descending)
    sorted_indices2 = sorted(range(len(categories_without_index)), key=lambda i: category_totals2[i], reverse=True)
    sorted_categories_without_index = [categories_without_index[i] for i in sorted_indices2]
    y2 = np.arange(len(sorted_categories_without_index))
    
    # Create stacked horizontal bars for subplot 2 (excluding index construction)
    left_values2 = [0.0] * len(sorted_categories_without_index)
    
    # Map original indices to sorted positions
    pos_map2 = {orig_idx: sorted_indices2.index(orig_idx) for orig_idx in range(len(categories_without_index))}
    
    # Plot query execution stages for each classifier
    for classifier_idx, classifier in enumerate(sorted_classifiers):
        query_values = [classifier_query_stages[classifier][label] for label in stage_labels]
        for i, (value, label) in enumerate(zip(query_values, stage_labels)):
            if value <= 0:
                continue
            pos = pos_map2[classifier_idx]  # Query Execution classifiers
            color_idx = (i + len(index_values)) % len(colors)
            ax2.barh(y2[pos], value, width, left=left_values2[pos], 
                    label=f'Query: {label}' if classifier_idx == 0 else '', 
                    color=colors[color_idx], alpha=0.8)
            left_values2[pos] += value
    
    # Plot preprocessing stages
    for i, (value, label) in enumerate(zip(preprocessing_values, preprocessing_labels)):
        if value <= 0:
            continue
        pos = pos_map2[len(categories_without_index) - 1]  # Naive is at last position
        color_idx = (i + len(index_values) + len(stage_labels)) % len(colors)
        ax2.barh(y2[pos], value, width, left=left_values2[pos], 
                label=f'Naive: {label}' if i == 0 else '', 
                color=colors[color_idx], alpha=0.8)
        left_values2[pos] += value
    
    ax2.set_xlabel('Runtime (seconds)')
    ax2.set_title('Without Index Construction')
    ax2.set_yticks(y2)
    ax2.set_yticklabels(sorted_categories_without_index)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # Add total value labels at the end of bars for subplot 2
    for i, total in enumerate(left_values2):
        if total <= 0:
            continue
        ax2.text(total + max(left_values2)*0.01, y2[i], f'{total:.1f}s', 
                ha='left', va='center', fontweight='bold', fontsize='small')
    
    # Set overall title
    fig.suptitle(f'Index Construction vs Query Execution Runtime Breakdown - {video_name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save plot with appropriate naming and directory structure
    # Create per-video subdirectory for per-video analysis
    video_output_dir = os.path.join(output_dir, 'per_video')
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Save plot with safe filename
    safe_video_name = video_name.replace('/', '_').replace('.', '_')
    for fmt in FORMATS:
        fig.savefig(os.path.join(video_output_dir, f'index_vs_query_comparison_{safe_video_name}.{fmt}'), 
                   dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def extract_video_names(query_timings: dict) -> list[str]:
    """Extract unique video names from the query timing data."""
    videos = set()
    for stage_timings in query_timings['timings'].values():
        for config_key in stage_timings.keys():
            # config_key format: dataset/video_classifier_tilesize
            parts = config_key.split('_')
            if len(parts) >= 3:
                # Extract video name (everything before the last two underscores)
                video_part = '_'.join(parts[:-2])  # Remove classifier and tilesize
                videos.add(video_part)
    return sorted(list(videos))


def visualize_breakdown_query_execution_all(query_timings: dict, output_dir: str, dataset: str = 'b3d'):
    """Create visualizations for query execution runtime breakdown."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all videos from the data
    videos = extract_video_names(query_timings)
    print(f"Found {len(videos)} videos: {videos}")
    
    # Create per-video visualizations
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        visualize_breakdown_query_execution(query_timings, output_dir, dataset, video, video_name)


def visualize_overal_runtime_all(index_timings: dict, query_timings: dict, output_dir: str):
    """Create comparative analysis between index construction and query execution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all videos from the data
    videos = extract_video_names(query_timings)
    print(f"Creating comparative analysis for {len(videos)} videos: {videos}")

    # Create per-video comparative analyses
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        visualize_overall_runtime(index_timings, query_timings, output_dir, video, video_name)


def main():
    """Main function to create runtime breakdown visualizations."""
    args = parse_args()
    
    print(f"Loading processed measurements for dataset: {args.dataset}")
    measurements_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'throughput', 'measurements')
    print(f"Measurements directory: {measurements_dir}")
    
    assert os.path.exists(measurements_dir), f"Error: Measurements directory {measurements_dir} does not exist."
    
    index_timings, query_timings, metadata = load_measurements(measurements_dir)
    
    print("Creating query execution visualizations...")
    throughput_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'throughput')
    visualize_breakdown_query_execution_all(query_timings, throughput_dir, args.dataset)
    
    print("Creating comparative analysis...")
    visualize_overal_runtime_all(index_timings, query_timings, throughput_dir)
    
    print(f"\nVisualization complete! Results saved to: {throughput_dir}")
    print("- Per-video visualizations saved in per_video/ subdirectory")


if __name__ == '__main__':
    main()
