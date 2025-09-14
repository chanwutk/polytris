#!/usr/local/bin/python

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Any
from collections import defaultdict
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


def load_data_tables(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load the index construction and query execution data tables."""
    index_file = os.path.join(data_dir, 'index_construction.json')
    query_file = os.path.join(data_dir, 'query_execution.json')
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"Loaded {len(index_data)} index construction entries")
    print(f"Loaded {len(query_data)} query execution entries")
    
    return index_data, query_data


def parse_runtime_file(file_path: str, stage: str, accessor: Callable[[Dict], List[Dict]] | None = None) -> List[Dict]:
    """Parse a runtime file and extract timing data."""
    if not os.path.exists(file_path):
        return []
    
    timings = []
    ignored_ops = ['total_frame_time', 'read_frame', 'save_canvas', 'save_mapping_files']
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if accessor is not None:
                        data = accessor(data)
                    if isinstance(data, dict):
                        data = [data]
                    assert isinstance(data, list), f"Expected a list of dictionaries, got {type(data)}, {stage}, {file_path}"
                    for item in data:
                        if 'time' in item and isinstance(item['time'], (int, float)):
                            # Convert from milliseconds to seconds
                            item_copy = item.copy()
                            item_copy['time'] = item['time'] / 1000.0
                            if item_copy['op'] in ignored_ops:
                                continue
                            timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
        elif file_path.endswith('.json'):
            # JSON format - array of objects
            data = json.load(f)
            assert isinstance(data, list), f"Expected a list of objects, got {type(data)}, {stage}, {stage}"
            for entry in data:
                if accessor is not None:
                    entry = accessor(entry)
                if isinstance(entry, dict):
                    entry = [entry]

                assert isinstance(entry, list), f"Expected a list of dictionaries, got {type(entry)}, {stage}"
                for item in entry:
                    if 'time' in item and isinstance(item['time'], (int, float)):
                        # Convert from milliseconds to seconds
                        item_copy = item.copy()
                        item_copy['time'] = item['time'] / 1000.0
                        if item_copy['op'] in ignored_ops:
                            continue
                        timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
    
    return timings


def parse_index_construction_timings(index_data: List[Dict]) -> Dict[str, Any]:
    """Parse timing data for index construction stages."""
    timings = {
        '011_tune_detect': defaultdict(list),
        '012_tune_create_training_data': defaultdict(list),
        '013_tune_train_classifier': defaultdict(list)
    }
    
    stage_summaries = {
        '011_tune_detect': defaultdict(list),
        '012_tune_create_training_data': defaultdict(list),
        '013_tune_train_classifier': defaultdict(list)
    }

    accessors = {
        '011_tune_detect': lambda row: row[3],
        '012_tune_create_training_data': lambda row: row,
        '013_tune_train_classifier': lambda row: row
    }
    
    for entry in index_data:
        dataset_video = entry['dataset/video']
        classifier = entry['classifier']
        config_key = f"{dataset_video}_{classifier}"
        
        for stage, file_path in entry['runtime_files']:
            file_timings = parse_runtime_file(file_path, stage, accessors[stage])
            
            if not file_timings:
                continue
            
            # Store individual timings
            timings[stage][config_key].extend(file_timings)
            
            # Calculate stage summary
            if stage == '011_tune_detect':
                # Detection stage - sum all operations
                total_time = sum(t['time'] for t in file_timings if isinstance(t['time'], (int, float)))
                stage_summaries[stage][config_key].append(total_time)
            
            elif stage == '012_tune_create_training_data':
                # Training data creation - group by operation and tile size
                for timing in file_timings:
                    op = timing.get('op', 'unknown')
                    tile_size = timing.get('tile_size', 'unknown')
                    time_val = timing.get('time', 0)
                    
                    op_key = f"{config_key}_{tile_size}_{op}"
                    stage_summaries[stage][op_key].append(time_val)
            
            elif stage == '013_tune_train_classifier':
                # Classifier training and testing - sum all training and testing epochs
                total_time = sum(t['time'] for t in file_timings)
                stage_summaries[stage][config_key].append(total_time)
    
    return {
        'timings': timings,
        'summaries': stage_summaries
    }


def parse_query_execution_timings(query_data: List[Dict]) -> Dict[str, Any]:
    """Parse timing data for query execution stages."""
    timings = {
        '001_preprocess_groundtruth_detection': defaultdict(list),
        '002_preprocess_groundtruth_tracking': defaultdict(list),
        '020_exec_classify': defaultdict(list),
        '030_exec_compress': defaultdict(list),
        '040_exec_detect': defaultdict(list),
        '060_exec_track': defaultdict(list)
    }
    
    stage_summaries = {
        '001_preprocess_groundtruth_detection': defaultdict(list),
        '002_preprocess_groundtruth_tracking': defaultdict(list),
        '020_exec_classify': defaultdict(list),
        '030_exec_compress': defaultdict(list),
        '040_exec_detect': defaultdict(list),
        '060_exec_track': defaultdict(list)
    }

    accessors = {
        '001_preprocess_groundtruth_detection': lambda row: row['runtime'],
        '002_preprocess_groundtruth_tracking': lambda row: row['runtime'],
        '020_exec_classify': lambda row: row['runtime'],
        '030_exec_compress': lambda row: row['runtime'],
        '040_exec_detect': lambda row: row,
        '060_exec_track': lambda row: row['runtime']
    }
    
    for entry in query_data:
        dataset_video = entry['dataset/video']
        classifier = entry['classifier']
        tile_size = entry['tile_size']
        config_key = f"{dataset_video}_{classifier}_{tile_size}"
        
        for stage, file_path in entry['runtime_files']:
            file_timings = parse_runtime_file(file_path, stage, accessors[stage])
            
            if not file_timings:
                continue
            
            # Store individual timings
            timings[stage][config_key].extend(file_timings)
            
            # Calculate stage summary
            total_time = sum(t['time'] for t in file_timings if isinstance(t['time'], (int, float)))
            stage_summaries[stage][config_key].append(total_time)
    
    return {
        'timings': timings,
        'summaries': stage_summaries
    }


def extract_videos_from_data(query_timings: Dict[str, Any]) -> List[str]:
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


def create_query_execution_visualizations(query_timings: Dict[str, Any], output_dir: str, dataset: str = 'b3d'):
    """Create visualizations for query execution runtime breakdown."""
    os.makedirs(output_dir, exist_ok=True)
    
    summaries = query_timings['summaries']
    
    # Extract all videos from the data
    videos = extract_videos_from_data(query_timings)
    print(f"Found {len(videos)} videos: {videos}")
    
    # Create average visualization across all videos
    create_average_query_execution_visualization(query_timings, output_dir, dataset, videos)
    
    # Create per-video visualizations
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        create_per_video_query_execution_visualization(query_timings, output_dir, dataset, video, video_name)


def create_average_query_execution_visualization(query_timings: Dict[str, Any], output_dir: str, dataset: str, videos: List[str]):
    """Create average query execution visualization across all videos."""
    # Prepare data for pipeline visualization
    stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    stage_names = ['Classify', 'Compress', 'Detect', 'Track']
    
    # Group by classifier and tile size 
    classifiers = ['SimpleCNN', 'groundtruth']
    tile_sizes = [30, 60, 120]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (stage, stage_name) in enumerate(tqdm.tqdm(zip(stages, stage_names), total=len(stages))):
        ax = axes[i]
        
        # Collect data for this stage grouped by operation
        config_labels = []
        op_data = defaultdict(list)  # op_name -> list of values for each config
        
        for classifier in classifiers:
            for tile_size in tile_sizes:
                
                # Average across all videos for this classifier/tile_size combination
                config_ops = defaultdict(list)  # op_name -> list of times across videos
                
                for video in videos:
                    config_key = f"{video}_{classifier}_{tile_size}"
                    if config_key in query_timings['timings'][stage]:
                        video_ops = defaultdict(float)
                        for timing in query_timings['timings'][stage][config_key]:
                            op_name = timing.get('op', 'unknown')
                            video_ops[op_name] += timing['time']
                        
                        # Add this video's operation times to the list
                        for op_name, time_val in video_ops.items():
                            config_ops[op_name].append(time_val)
                
                if len(config_ops) == 0:
                    continue

                config_labels.append(f"{classifier}\n{tile_size}")
                # Calculate average for each operation across videos
                for op_name, times in config_ops.items():
                    avg_time = np.mean(times) if times else 0
                    if op_name not in op_data:
                        op_data[op_name] = []
                    op_data[op_name].append(avg_time)
                
                # Ensure all operations have a value for this config (0 if not present)
                all_ops = set()
                for other_video in videos:
                    other_config_key = f"{other_video}_{classifier}_{tile_size}"
                    if other_config_key in query_timings['timings'][stage]:
                        for timing in query_timings['timings'][stage][other_config_key]:
                            all_ops.add(timing.get('op', 'unknown'))
                
                for op_name in all_ops:
                    if op_name not in config_ops:
                        if op_name not in op_data:
                            op_data[op_name] = []
                        op_data[op_name].append(0)
        
        if config_labels and op_data:
            # Create stacked bar chart
            x = np.arange(len(config_labels))
            width = 0.6
            bottom = np.zeros(len(config_labels))
            
            # Color palette for different operations
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Plot each operation as a layer in the stacked bar
            for j, (op_name, values) in enumerate(sorted(op_data.items())):
                if any(v > 0 for v in values):  # Only plot if there are non-zero values
                    color = colors[j % len(colors)]
                    ax.bar(x, values, width, bottom=bottom, label=op_name, color=color, alpha=0.8)
                    bottom += np.array(values)
            
            ax.set_title(f'{stage_name} Runtime by Operation (Average)')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_xticks(x)
            ax.set_xticklabels(config_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.suptitle('Query Execution Pipeline Runtime Breakdown (Average Across All Videos)', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    for fmt in FORMATS:
        plt.savefig(os.path.join(output_dir, f'query_execution_pipeline_average.{fmt}'), 
                   dpi=300, bbox_inches='tight')
    
    plt.close()


def create_per_video_query_execution_visualization(query_timings: Dict[str, Any], output_dir: str, dataset: str, video: str, video_name: str):
    """Create query execution visualization for a specific video."""
    # Prepare data for pipeline visualization
    stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    stage_names = ['Classify', 'Compress', 'Detect', 'Track']
    
    # Group by classifier and tile size 
    classifiers = ['SimpleCNN', 'groundtruth']
    tile_sizes = [30, 60, 120]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        ax = axes[i]
        
        # Collect data for this stage grouped by operation
        config_labels = []
        op_data = defaultdict(list)  # op_name -> list of values for each config
        
        for classifier in classifiers:
            for tile_size in tile_sizes:
                config_key = f"{video}_{classifier}_{tile_size}"
                config_labels.append(f"{classifier}\n{tile_size}")
                
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
            # Create stacked bar chart
            x = np.arange(len(config_labels))
            width = 0.6
            bottom = np.zeros(len(config_labels))
            
            # Color palette for different operations
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Plot each operation as a layer in the stacked bar
            for j, (op_name, values) in enumerate(sorted(op_data.items())):
                if any(v > 0 for v in values):  # Only plot if there are non-zero values
                    color = colors[j % len(colors)]
                    ax.bar(x, values, width, bottom=bottom, label=op_name, color=color, alpha=0.8)
                    bottom += np.array(values)
            
            ax.set_title(f'{stage_name} Runtime by Operation')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_xticks(x)
            ax.set_xticklabels(config_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.suptitle(f'Query Execution Pipeline Runtime Breakdown - {video_name}', fontsize=16)
    plt.tight_layout()
    
    # Create per-video subdirectory
    video_output_dir = os.path.join(output_dir, 'per_video')
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Save plot
    safe_video_name = video_name.replace('/', '_').replace('.', '_')
    for fmt in FORMATS:
        plt.savefig(os.path.join(video_output_dir, f'query_execution_pipeline_{safe_video_name}.{fmt}'), 
                   dpi=300, bbox_inches='tight')
    
    plt.close()


def create_comparative_analysis(index_timings: Dict[str, Any], query_timings: Dict[str, Any], 
                               output_dir: str):
    """Create comparative analysis between index construction and query execution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all videos from the data
    videos = extract_videos_from_data(query_timings)
    print(f"Creating comparative analysis for {len(videos)} videos: {videos}")
    
    # Create average comparative analysis across all videos
    create_average_comparative_analysis(index_timings, query_timings, output_dir, videos)
    
    # Create per-video comparative analyses
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        create_per_video_comparative_analysis(index_timings, query_timings, output_dir, video, video_name)


def create_comparative_analysis_chart(index_timings: Dict[str, Any], query_timings: Dict[str, Any], 
                                    output_dir: str, videos: List[str] | None = None, video: str | None = None, 
                                    video_name: str | None = None):
    """Create comparative analysis between index construction and query execution.
    
    Args:
        index_timings: Index construction timing data
        query_timings: Query execution timing data
        output_dir: Output directory for saving plots
        videos: List of videos for average analysis (if None, creates per-video analysis)
        video: Specific video for per-video analysis (used when videos is None)
        video_name: Display name for the video (used when videos is None)
    """
    assert (videos is None) != (video is None), "Either videos or video must be provided"

    # Determine if this is average or per-video analysis
    is_average = videos is not None
    target_videos = videos if is_average else [video]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
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
            if video is not None and not k.startswith(video + '_'):
                continue
            if times:
                stage_total += np.sum(times)
        
        if '011_tune_detect' in stage_name:
            index_stages['Detection'] += stage_total
        elif '012_tune_create_training_data' in stage_name:
            index_stages['Create Training Data'] += stage_total
        elif '013_tune_train_classifier' in stage_name:
            index_stages['Classifier Training'] += stage_total
    
    # Calculate breakdown for query execution stages
    query_stages = {
        'Classify': 0.0 if is_average else 0,
        'Compress': 0.0 if is_average else 0,
        'Detect': 0.0 if is_average else 0,
        'Track': 0.0 if is_average else 0
    }
    
    # Query execution stage breakdown (exclude preprocessing stages and groundtruth classifier)
    excluded_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name in excluded_stages:
            continue
        
        if is_average:
            # Average across videos
            stage_video_totals = []  # Collect totals for each video
            for target_video in target_videos:
                video_total = 0
                for config_key, times in stage_summaries.items():
                    # Only include configs for this video and exclude groundtruth classifier
                    assert target_video is not None
                    if not config_key.startswith(target_video + '_') or '_groundtruth_' in config_key:
                        continue
                    if times:
                        video_total += np.mean(times)
                if video_total > 0:
                    stage_video_totals.append(video_total)
            
            # Average across videos
            stage_total = np.mean(stage_video_totals) if stage_video_totals else 0
        else:
            # Per-video analysis
            stage_total = 0
            for config_key, times in stage_summaries.items():
                # Only include configs for this video and exclude groundtruth classifier
                assert video is not None
                if not config_key.startswith(video + '_') or '_groundtruth_' in config_key:
                    continue
                if times:
                    stage_total += np.mean(times)
        
        assert isinstance(stage_total, (int, float))
        if '020_exec_classify' in stage_name:
            query_stages['Classify'] += float(stage_total) if is_average else stage_total
        elif '030_exec_compress' in stage_name:
            query_stages['Compress'] += float(stage_total) if is_average else stage_total
        elif '040_exec_detect' in stage_name:
            query_stages['Detect'] += float(stage_total) if is_average else stage_total
        elif '060_exec_track' in stage_name:
            query_stages['Track'] += float(stage_total) if is_average else stage_total
    
    # Calculate breakdown for query execution stages with groundtruth classifier only
    query_groundtruth_stages = {
        'Classify': 0.0 if is_average else 0,
        'Compress': 0.0 if is_average else 0,
        'Detect': 0.0 if is_average else 0,
        'Track': 0.0 if is_average else 0
    }
    
    # Query execution stage breakdown for groundtruth classifier only
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name in excluded_stages:
            continue
        
        if is_average:
            # Average across videos
            stage_video_totals = []  # Collect totals for each video
            for target_video in target_videos:
                video_total = 0
                for config_key, times in stage_summaries.items():
                    # Only include configs for this video and only groundtruth classifier
                    assert target_video is not None
                    if not config_key.startswith(target_video + '_') or '_groundtruth_' not in config_key:
                        continue
                    if times:
                        video_total += np.mean(times)
                if video_total > 0:
                    stage_video_totals.append(video_total)
            
            # Average across videos
            stage_total = np.mean(stage_video_totals) if stage_video_totals else 0
        else:
            # Per-video analysis
            stage_total = 0
            for config_key, times in stage_summaries.items():
                # Only include configs for this video and only groundtruth classifier
                assert video is not None
                if not config_key.startswith(video + '_') or '_groundtruth_' not in config_key:
                    continue
                if times:
                    stage_total += np.mean(times)
        
        assert isinstance(stage_total, (int, float))
        if '020_exec_classify' in stage_name:
            query_groundtruth_stages['Classify'] += float(stage_total) if is_average else stage_total
        elif '030_exec_compress' in stage_name:
            query_groundtruth_stages['Compress'] += float(stage_total) if is_average else stage_total
        elif '040_exec_detect' in stage_name:
            query_groundtruth_stages['Detect'] += float(stage_total) if is_average else stage_total
        elif '060_exec_track' in stage_name:
            query_groundtruth_stages['Track'] += float(stage_total) if is_average else stage_total
    
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
        
        if is_average:
            # Average across videos
            stage_video_totals = []  # Collect totals for each video
            for target_video in target_videos:
                video_total = 0
                for config_key, times in stage_summaries.items():
                    assert target_video is not None
                    if not config_key.startswith(target_video + '_'):
                        continue
                    if times:
                        video_total += np.mean(times)
                if video_total > 0:
                    stage_video_totals.append(video_total)
            
            # Average across videos
            stage_total = np.mean(stage_video_totals) if stage_video_totals else 0
        else:
            # Per-video analysis
            stage_total = 0
            for config_key, times in stage_summaries.items():
                assert video is not None
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
    
    # Colors for different operations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Index construction stack
    index_values = list(index_stages.values())
    index_labels = list(index_stages.keys())
    
    # Query execution stack  
    query_labels = tuple(sorted(query_stages.keys()))
    query_values = [query_stages[label] for label in query_labels]
    
    # Query execution groundtruth stack
    query_groundtruth_labels = tuple(sorted(query_groundtruth_stages.keys()))
    query_groundtruth_values = [query_groundtruth_stages[label] for label in query_groundtruth_labels]

    assert query_labels == query_groundtruth_labels

    # Preprocessing stack
    preprocessing_values = list(preprocessing_stages.values())
    preprocessing_labels = list(preprocessing_stages.keys())
    
    # === SUBPLOT 1: With Index Construction ===
    categories_with_index = ['Index Construction', 'Query Execution\n(Classifier: SimpleCNN)', 'Query Execution\n(Classifier: Groundtruth)', 'Naive']
    x1 = np.arange(len(categories_with_index))
    
    # Create stacked bars for subplot 1
    bottom_index = 0
    bottom_query = 0
    bottom_query_groundtruth = 0
    bottom_preprocessing = 0
    
    # Plot index construction stages
    for i, (value, label) in enumerate(zip(index_values, index_labels)):
        if value > 0:
            ax1.bar(x1[0], value, width, bottom=bottom_index, 
                   label=f'Index: {label}', color=colors[i], alpha=0.8)
            bottom_index += value
    
    # Plot query execution stages (SimpleCNN)
    for i, (value, label) in enumerate(zip(query_values, query_labels)):
        if value > 0:
            ax1.bar(x1[1], value, width, bottom=bottom_query, 
                   label=f'Query: {label}', color=colors[i+len(index_values)], alpha=0.8)
            bottom_query += value
    
    # Plot query execution stages (Groundtruth)
    for i, (value, label) in enumerate(zip(query_groundtruth_values, query_groundtruth_labels)):
        if value > 0:
            ax1.bar(x1[2], value, width, bottom=bottom_query_groundtruth, 
                   color=colors[i+len(index_values)], alpha=0.8)
            bottom_query_groundtruth += value
    
    # Plot preprocessing stages
    for i, (value, label) in enumerate(zip(preprocessing_values, preprocessing_labels)):
        if value > 0:
            ax1.bar(x1[3], value, width, bottom=bottom_preprocessing, 
                   label=f'Naive: {label}', color=colors[(i+len(index_values)+len(query_values)) % len(colors)], alpha=0.8)
            bottom_preprocessing += value
    
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('With Index Construction')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(categories_with_index)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Add total value labels on top of bars for subplot 1
    totals1 = [bottom_index, bottom_query, bottom_query_groundtruth, bottom_preprocessing]
    for i, total in enumerate(totals1):
        if total > 0 or is_average:
            ax1.text(x1[i], total + max(totals1)*0.01, f'{total:.1f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize='small')
    
    # === SUBPLOT 2: Without Index Construction ===
    categories_without_index = ['Query Execution\n(Classifier: SimpleCNN)', 'Query Execution\n(Classifier: Groundtruth)', 'Naive']
    x2 = np.arange(len(categories_without_index))
    
    # Create stacked bars for subplot 2 (excluding index construction)
    bottom_query2 = 0
    bottom_query_groundtruth2 = 0
    bottom_preprocessing2 = 0
    
    # Plot query execution stages (SimpleCNN)
    for i, (value, label) in enumerate(zip(query_values, query_labels)):
        if value > 0:
            ax2.bar(x2[0], value, width, bottom=bottom_query2, 
                   label=f'Query: {label}', color=colors[i+len(index_values)], alpha=0.8)
            bottom_query2 += value
    
    # Plot query execution stages (Groundtruth)
    for i, (value, label) in enumerate(zip(query_groundtruth_values, query_groundtruth_labels)):
        if value > 0:
            ax2.bar(x2[1], value, width, bottom=bottom_query_groundtruth2, 
                   color=colors[i+len(index_values)], alpha=0.8)
            bottom_query_groundtruth2 += value
    
    # Plot preprocessing stages
    for i, (value, label) in enumerate(zip(preprocessing_values, preprocessing_labels)):
        if value > 0:
            ax2.bar(x2[2], value, width, bottom=bottom_preprocessing2, 
                   label=f'Naive: {label}', color=colors[(i+len(index_values)+len(query_values)) % len(colors)], alpha=0.8)
            bottom_preprocessing2 += value
    
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Without Index Construction')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories_without_index)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # Add total value labels on top of bars for subplot 2
    totals2 = [bottom_query2, bottom_query_groundtruth2, bottom_preprocessing2]
    for i, total in enumerate(totals2):
        if total > 0 or is_average:
            ax2.text(x2[i], total + max(totals2)*0.01, f'{total:.1f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize='small')
    
    # Set overall title based on analysis type
    if is_average:
        fig.suptitle('Index Construction vs Query Execution Runtime Breakdown (Average Across All Videos)', fontsize=16)
    else:
        fig.suptitle(f'Index Construction vs Query Execution Runtime Breakdown - {video_name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save plot with appropriate naming and directory structure
    if is_average:
        # Save in main output directory for average analysis
        for fmt in FORMATS:
            fig.savefig(os.path.join(output_dir, f'index_vs_query_comparison_average.{fmt}'), 
                       dpi=300, bbox_inches='tight')
    else:
        # Create per-video subdirectory for per-video analysis
        video_output_dir = os.path.join(output_dir, 'per_video')
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Save plot with safe filename
        assert video_name is not None
        safe_video_name = video_name.replace('/', '_').replace('.', '_')
        for fmt in FORMATS:
            fig.savefig(os.path.join(video_output_dir, f'index_vs_query_comparison_{safe_video_name}.{fmt}'), 
                       dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def create_average_comparative_analysis(index_timings: Dict[str, Any], query_timings: Dict[str, Any], 
                                      output_dir: str, videos: List[str]):
    """Create average comparative analysis between index construction and query execution."""
    create_comparative_analysis_chart(index_timings, query_timings, output_dir, 
                                    videos=videos)


def create_per_video_comparative_analysis(index_timings: Dict[str, Any], query_timings: Dict[str, Any], 
                                        output_dir: str, video: str, video_name: str):
    """Create comparative analysis between index construction and query execution for a specific video."""
    create_comparative_analysis_chart(index_timings, query_timings, output_dir, 
                                    video=video, video_name=video_name)


def main():
    """Main function to create runtime breakdown visualizations."""
    args = parse_args()
    
    print(f"Loading throughput data tables for dataset: {args.dataset}")
    data_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'throughput')
    print(f"Data directory: {data_dir}")
    index_data, query_data = load_data_tables(data_dir)
    
    print("Parsing index construction timing data...")
    index_timings = parse_index_construction_timings(index_data)
    
    print("Parsing query execution timing data...")
    query_timings = parse_query_execution_timings(query_data)
    
    print("Creating query execution visualizations...")
    throughput_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'throughput')
    create_query_execution_visualizations(query_timings, throughput_dir, args.dataset)
    
    print("Creating comparative analysis...")
    create_comparative_analysis(index_timings, query_timings, throughput_dir)
    
    print(f"\nVisualization complete! Results saved to: {throughput_dir}")
    print("- Average visualizations saved as *_average.png/pdf")
    print("- Per-video visualizations saved in per_video/ subdirectory")


if __name__ == '__main__':
    main()
