#!/usr/local/bin/python

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from typing import Any

from scripts.utilities import CACHE_DIR


OUTPUT_DIR = 'pipeline-stages/all-speed-results'
# TILE_SIZES = [32, 64, 128]
TILE_SIZES = [64]

# Pipeline stages and their runtime data locations
PIPELINE_STAGES = {
    'tune_detect': {
        'path_template': '{dataset}/{video_file}/segments/detection/detections.jsonl',
        'description': 'Detection on video segments (011_tune_detect.py)',
        'data_format': 'timing_in_detections'  # Timing data is embedded in detection results
    },
    'create_training_data': {
        'path_template': '{dataset}/{video_file}/training/runtime/tilesize_{tile_size}/create_training_data.jsonl',
        'description': 'Training data creation (012_tune_create_training_data.py)',
        'data_format': 'jsonl_operations'
    },
    'train_classifier': {
        'path_template': '{dataset}/{video_file}/training/results/SimpleCNN_{tile_size}/',
        'description': 'Classifier training (013_tune_train_classifier.py)',
        'data_format': 'training_logs'
    },
    'classify': {
        'path_template': '{dataset}/{video_file}/relevancy/SimpleCNN_{tile_size}/score/score.jsonl',
        'description': 'Tile classification (020_exec_classify.py)',
        'data_format': 'runtime_in_frames'
    },
    'compress': {
        'path_template': '{dataset}/{video_file}/packing/SimpleCNN_{tile_size}/runtime.jsonl',
        'description': 'Video compression/packing (030_exec_compress.py)',
        'data_format': 'jsonl_operations'
    },
    'detect_packed': {
        'path_template': '{dataset}/{video_file}/packed_detections/SimpleCNN_{tile_size}/runtimes.jsonl',
        'description': 'Detection on packed images (040_exec_detect.py)',
        'data_format': 'jsonl_operations'
    },
    'track': {
        'path_template': '{dataset}/{video_file}/uncompressed_tracking/SimpleCNN_{tile_size}/runtimes.jsonl',
        'description': 'Object tracking (060_exec_track.py)',
        'data_format': 'jsonl_operations'
    }
}


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (str): Tile size to use for evaluation (choices: '64', '128', 'all')
            - output_dir (str): Output directory for results (default: 'pipeline-stages/all-speed-results')
            - create_plots (bool): Whether to create visualization plots (default: False)
            - stages (list): List of pipeline stages to analyze (default: all stages)
    """
    parser = argparse.ArgumentParser(description='Analyze speed and performance statistics from all pipeline stages')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--tile_size', type=str, choices=['64', '128', 'all'], default='all',
                        help='Tile size to use for evaluation (or "all" for all tile sizes)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--create_plots', action='store_true', default=False,
                        help='Whether to create visualization plots')
    parser.add_argument('--stages', nargs='*', choices=list(PIPELINE_STAGES.keys()),
                        default=list(PIPELINE_STAGES.keys()),
                        help='Pipeline stages to analyze (default: all stages)')
    return parser.parse_args()


def find_pipeline_results(cache_dir: str, dataset: str, tile_size: str, stages: list[str]) -> list[tuple[str, int, dict]]:
    """
    Find all video files with pipeline results and runtime data for the specified dataset and tile size.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        tile_size (str): Tile size to look for ('64', '128', or 'all')
        stages (list[str]): List of pipeline stages to look for
        
    Returns:
        list[tuple[str, int, dict]]: list of (video_name, tile_size, available_stages) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset)
    if not os.path.exists(dataset_cache_dir):
        raise ValueError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    # Determine which tile sizes to process
    tile_sizes_to_process = TILE_SIZES if tile_size == 'all' else [int(tile_size)]
    
    video_tile_combinations = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            for ts in tile_sizes_to_process:
                available_stages = {}
                
                # Check each requested stage
                for stage in stages:
                    stage_info = PIPELINE_STAGES[stage]
                    stage_path = stage_info['path_template'].format(
                        dataset=dataset, 
                        video_file=item, 
                        tile_size=ts
                    )
                    full_path = os.path.join(cache_dir, stage_path)
                    
                    # Special handling for different data formats
                    if stage_info['data_format'] == 'training_logs':
                        # Check for training log files
                        train_logs = os.path.join(full_path, 'train_losses.json')
                        test_logs = os.path.join(full_path, 'test_losses.json')
                        if os.path.exists(train_logs) and os.path.exists(test_logs):
                            available_stages[stage] = full_path
                    else:
                        # Check for regular files
                        if os.path.exists(full_path):
                            available_stages[stage] = full_path
                
                if available_stages:  # Only include if at least one stage has data
                    video_tile_combinations.append((item, ts, available_stages))
                    print(f"Found pipeline results for {item} with tile size {ts}: {list(available_stages.keys())}")
    
    return video_tile_combinations


def load_runtime_data(file_path: str) -> list[dict[str, Any]]:
    """
    Load runtime data from JSONL file.
    
    Args:
        file_path (str): Path to the runtime JSONL file
        
    Returns:
        list[dict[str, Any]]: List of runtime data for each frame
    """
    runtime_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                runtime_data.append(data)
    
    return runtime_data


def load_tune_detect_runtime(file_path: str) -> list[dict[str, Any]]:
    """
    Load runtime data from tune_detect detections.jsonl file.
    Runtime data is embedded in detection results as the 4th element.
    
    Args:
        file_path (str): Path to detections.jsonl file
        
    Returns:
        list[dict[str, Any]]: List of runtime data for each frame
    """
    runtime_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Format: [frame_idx, bboxes, segment_idx, timing]
                if len(data) >= 4 and isinstance(data[3], list):
                    frame_idx = data[0]
                    timing = data[3]
                    runtime_entry = {
                        'frame_idx': frame_idx,
                        'step_times': {op['op']: op['time'] for op in timing}
                    }
                    runtime_data.append(runtime_entry)
    
    return runtime_data


def load_training_logs_runtime(dir_path: str) -> dict[str, Any]:
    """
    Load runtime data from training log files.
    
    Args:
        dir_path (str): Path to directory containing train_losses.json and test_losses.json
        
    Returns:
        dict[str, Any]: Aggregated training runtime data
    """
    train_logs_path = os.path.join(dir_path, 'train_losses.json')
    test_logs_path = os.path.join(dir_path, 'test_losses.json')
    
    total_train_time = 0
    total_test_time = 0
    num_train_epochs = 0
    num_test_epochs = 0
    
    # Load training logs
    if os.path.exists(train_logs_path):
        with open(train_logs_path, 'r') as f:
            train_data = json.load(f)
            for epoch in train_data:
                if 'time' in epoch:
                    total_train_time += epoch['time']
                    num_train_epochs += 1
    
    # Load test logs
    if os.path.exists(test_logs_path):
        with open(test_logs_path, 'r') as f:
            test_data = json.load(f)
            for epoch in test_data:
                if 'time' in epoch:
                    total_test_time += epoch['time']
                    num_test_epochs += 1
    
    return {
        'total_train_time': total_train_time,
        'total_test_time': total_test_time,
        'total_time': total_train_time + total_test_time,
        'avg_train_time_per_epoch': total_train_time / max(num_train_epochs, 1),
        'avg_test_time_per_epoch': total_test_time / max(num_test_epochs, 1),
        'num_train_epochs': num_train_epochs,
        'num_test_epochs': num_test_epochs
    }


def load_classify_runtime(file_path: str) -> list[dict[str, Any]]:
    """
    Load runtime data from classify score.jsonl file.
    Runtime data is embedded in frame results.
    
    Args:
        file_path (str): Path to score.jsonl file
        
    Returns:
        list[dict[str, Any]]: List of runtime data for each frame
    """
    runtime_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'runtime' in data and 'frame_idx' in data:
                    frame_idx = data['frame_idx']
                    runtime = data['runtime']
                    runtime_entry = {
                        'frame_idx': frame_idx,
                        'step_times': {op['op']: op['time'] for op in runtime}
                    }
                    runtime_data.append(runtime_entry)
    
    return runtime_data


def load_stage_runtime_data(stage: str, file_path: str) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Load runtime data for a specific pipeline stage.
    
    Args:
        stage (str): Pipeline stage name
        file_path (str): Path to the runtime data file/directory
        
    Returns:
        list[dict[str, Any]] | dict[str, Any]: Runtime data in appropriate format
    """
    stage_info = PIPELINE_STAGES[stage]
    data_format = stage_info['data_format']
    
    if data_format == 'timing_in_detections':
        return load_tune_detect_runtime(file_path)
    elif data_format == 'training_logs':
        return load_training_logs_runtime(file_path)
    elif data_format == 'runtime_in_frames':
        return load_classify_runtime(file_path)
    elif data_format == 'jsonl_operations':
        return load_runtime_data(file_path)
    else:
        raise ValueError(f"Unknown data format: {data_format}")


def analyze_pipeline_performance(video_name: str, tile_size: int, available_stages: dict, 
                               output_dir: str) -> dict[str, Any]:
    """
    Analyze runtime performance for a single video across all available pipeline stages.
    
    Args:
        video_name (str): Name of the video
        tile_size (int): Tile size used
        available_stages (dict): Dictionary of available stages and their file paths
        output_dir (str): Output directory for results
        
    Returns:
        dict[str, Any]: Runtime performance analysis results
    """
    print(f"Analyzing pipeline performance for {video_name} with tile size {tile_size}")
    
    try:
        performance_stats = {
            'video_name': video_name,
            'tile_size': tile_size,
            'stages': {},
            'success': True
        }
        
        total_pipeline_time = 0
        
        # Analyze each available stage
        for stage, file_path in available_stages.items():
            print(f"  Processing stage: {stage}")
            
            try:
                stage_data = load_stage_runtime_data(stage, file_path)
                stage_stats = analyze_stage_performance(stage, stage_data)
                performance_stats['stages'][stage] = stage_stats
                
                # Add to total pipeline time if available
                if 'total_time' in stage_stats:
                    total_pipeline_time += stage_stats['total_time']
                    
            except Exception as e:
                print(f"    Error processing stage {stage}: {str(e)}")
                performance_stats['stages'][stage] = {
                    'error': str(e),
                    'success': False
                }
        
        performance_stats['total_pipeline_time'] = total_pipeline_time
        performance_stats['num_stages'] = len([s for s in performance_stats['stages'].values() if s.get('success', False)])
        
        return performance_stats
        
    except Exception as e:
        print(f"Error analyzing pipeline performance for {video_name} with tile size {tile_size}: {str(e)}")
        return {
            'video_name': video_name,
            'tile_size': tile_size,
            'error': str(e),
            'success': False
        }


def analyze_stage_performance(stage: str, stage_data: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """
    Analyze performance data for a specific pipeline stage.
    
    Args:
        stage (str): Pipeline stage name
        stage_data: Runtime data for the stage
        
    Returns:
        dict[str, Any]: Stage performance statistics
    """
    stage_info = PIPELINE_STAGES[stage]
    data_format = stage_info['data_format']
    
    if data_format == 'training_logs':
        # Training data is already aggregated as dict
        if isinstance(stage_data, dict):
            return {
                'stage': stage,
                'description': stage_info['description'],
                'total_time': stage_data.get('total_time', 0),
                'total_train_time': stage_data.get('total_train_time', 0),
                'total_test_time': stage_data.get('total_test_time', 0),
                'num_train_epochs': stage_data.get('num_train_epochs', 0),
                'num_test_epochs': stage_data.get('num_test_epochs', 0),
                'avg_train_time_per_epoch': stage_data.get('avg_train_time_per_epoch', 0),
                'avg_test_time_per_epoch': stage_data.get('avg_test_time_per_epoch', 0),
                'success': True
            }
        else:
            return {
                'stage': stage,
                'description': stage_info['description'],
                'error': 'Expected dict for training logs',
                'success': False
            }
    
    elif isinstance(stage_data, list) and stage_data:
        # Frame-based data
        all_times = []
        step_times = {}
        frame_count = len(stage_data)
        
        for frame_data in stage_data:
            if 'step_times' in frame_data:
                frame_total = 0
                for step_name, step_time in frame_data['step_times'].items():
                    if step_name not in step_times:
                        step_times[step_name] = []
                    step_times[step_name].append(step_time)
                    frame_total += step_time
                
                # Use total frame time if available, otherwise sum of steps
                if 'total_frame_time' in frame_data['step_times']:
                    all_times.append(frame_data['step_times']['total_frame_time'])
                else:
                    all_times.append(frame_total)
        
        if all_times:
            stats = {
                'stage': stage,
                'description': stage_info['description'],
                'total_time': sum(all_times),
                'total_frames': frame_count,
                'avg_frame_time': np.mean(all_times),
                'std_frame_time': np.std(all_times),
                'min_frame_time': np.min(all_times),
                'max_frame_time': np.max(all_times),
                'median_frame_time': np.median(all_times),
                'fps': 1.0 / np.mean(all_times) if np.mean(all_times) > 0 else 0,
                'success': True
            }
            
            # Add step-by-step timing analysis
            for step_name, step_times_list in step_times.items():
                if step_times_list:
                    stats[f'{step_name}_avg'] = np.mean(step_times_list)
                    stats[f'{step_name}_std'] = np.std(step_times_list)
                    stats[f'{step_name}_total'] = sum(step_times_list)
            
            # Add percentiles
            percentiles = [25, 50, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'frame_time_p{p}'] = np.percentile(all_times, p)
            
            return stats
        else:
            return {
                'stage': stage,
                'description': stage_info['description'],
                'error': 'No timing data found',
                'success': False
            }
    
    else:
        return {
            'stage': stage,
            'description': stage_info['description'],
            'error': 'Invalid or empty data format',
            'success': False
        }


def create_pipeline_summary(results: list[dict[str, Any]], output_dir: str) -> None:
    """
    Create a comprehensive text summary of the pipeline performance results.
    
    Args:
        results (list[dict[str, Any]]): List of performance analysis results
        output_dir (str): Output directory for results
    """
    print("Creating pipeline performance summary...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful performance analysis results to summarize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to JSON
    with open(os.path.join(output_dir, 'detailed_pipeline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create text summary
    summary_file = os.path.join(output_dir, 'pipeline_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Pipeline Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Get all available stages across all results
        all_stages = set()
        for result in successful_results:
            if 'stages' in result:
                all_stages.update(result['stages'].keys())
        all_stages = sorted(all_stages)
        
        f.write("Individual Video Results:\n")
        f.write("-" * 30 + "\n")
        for result in successful_results:
            f.write(f"Video: {result['video_name']}\n")
            f.write(f"  Tile Size: {result['tile_size']}\n")
            f.write(f"  Total Pipeline Time: {result.get('total_pipeline_time', 0):.4f}s\n")
            f.write(f"  Number of Stages: {result.get('num_stages', 0)}\n")
            
            if 'stages' in result:
                f.write("  Stage Performance:\n")
                for stage in all_stages:
                    if stage in result['stages'] and result['stages'][stage].get('success', False):
                        stage_data = result['stages'][stage]
                        f.write(f"    {stage}: {stage_data.get('total_time', 0):.4f}s")
                        if 'total_frames' in stage_data:
                            f.write(f" ({stage_data['total_frames']} frames, {stage_data.get('avg_frame_time', 0):.4f}s/frame)")
                        elif 'num_train_epochs' in stage_data:
                            f.write(f" ({stage_data['num_train_epochs']} train epochs, {stage_data['num_test_epochs']} test epochs)")
                        f.write("\n")
                    else:
                        f.write(f"    {stage}: Not available\n")
            f.write("\n")
        
        f.write("Stage-wise Summary Statistics:\n")
        f.write("-" * 35 + "\n")
        
        for stage in all_stages:
            stage_results = []
            for result in successful_results:
                if 'stages' in result and stage in result['stages'] and result['stages'][stage].get('success', False):
                    stage_results.append(result['stages'][stage])
            
            if stage_results:
                f.write(f"\n{stage.upper()} ({PIPELINE_STAGES[stage]['description']}):\n")
                
                # Aggregate statistics
                total_times = [s.get('total_time', 0) for s in stage_results if 'total_time' in s]
                if total_times:
                    f.write(f"  Total Time - Mean: {np.mean(total_times):.4f}s, Std: {np.std(total_times):.4f}s\n")
                    f.write(f"              Min: {np.min(total_times):.4f}s, Max: {np.max(total_times):.4f}s\n")
                
                # Frame-based statistics
                frame_times = [s.get('avg_frame_time', 0) for s in stage_results if 'avg_frame_time' in s]
                if frame_times:
                    fps_values = [s.get('fps', 0) for s in stage_results if 'fps' in s]
                    f.write(f"  Frame Time - Mean: {np.mean(frame_times):.4f}s, Std: {np.std(frame_times):.4f}s\n")
                    f.write(f"  FPS - Mean: {np.mean(fps_values):.2f}, Std: {np.std(fps_values):.2f}\n")
                
                # Training-specific statistics
                if stage == 'train_classifier':
                    train_times = [s.get('total_train_time', 0) for s in stage_results if 'total_train_time' in s]
                    test_times = [s.get('total_test_time', 0) for s in stage_results if 'total_test_time' in s]
                    if train_times:
                        f.write(f"  Training Time - Mean: {np.mean(train_times):.4f}s, Std: {np.std(train_times):.4f}s\n")
                        f.write(f"  Testing Time - Mean: {np.mean(test_times):.4f}s, Std: {np.std(test_times):.4f}s\n")
        
        # Overall pipeline statistics
        f.write("\nOverall Pipeline Statistics:\n")
        f.write("-" * 30 + "\n")
        pipeline_times = [r.get('total_pipeline_time', 0) for r in successful_results]
        if pipeline_times:
            f.write(f"Total Pipeline Time - Mean: {np.mean(pipeline_times):.2f}s, Std: {np.std(pipeline_times):.2f}s\n")
            f.write(f"                     Min: {np.min(pipeline_times):.2f}s, Max: {np.max(pipeline_times):.2f}s\n")
        
        # Tile size analysis if multiple tile sizes
        if len(set([r['tile_size'] for r in successful_results])) > 1:
            f.write("\nTile Size Analysis:\n")
            f.write("-" * 20 + "\n")
            for tile_size in sorted(set([r['tile_size'] for r in successful_results])):
                tile_results = [r for r in successful_results if r['tile_size'] == tile_size]
                tile_pipeline_times = [r.get('total_pipeline_time', 0) for r in tile_results]
                
                f.write(f"Tile Size {tile_size}:\n")
                if tile_pipeline_times:
                    f.write(f"  Pipeline Time: Mean={np.mean(tile_pipeline_times):.2f}s, Std={np.std(tile_pipeline_times):.2f}s\n")
                f.write(f"  Number of Videos: {len(tile_results)}\n\n")
    
    print(f"Summary saved to {summary_file}")


def create_pipeline_visualizations(results: list[dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for pipeline performance results using matplotlib.
    
    Args:
        results (list[dict[str, Any]]): List of performance analysis results
        output_dir (str): Output directory for visualizations
    """
    print("Creating matplotlib pipeline visualizations...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful performance analysis results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    video_names = []
    tile_sizes = []
    fps_values = []
    frame_times = []
    total_times = []
    total_frames = []
    
    for result in successful_results:
        video_names.append(result['video_name'])
        tile_sizes.append(result['tile_size'])
        fps_values.append(result['fps'])
        frame_times.append(result['avg_frame_time'])
        total_times.append(result['total_processing_time'])
        total_frames.append(result['total_frames'])
    
    # Create DataFrame for pipeline results
    pipeline_data = []
    for result in successful_results:
        row = {
            'Video': result['video_name'],
            'Tile_Size': result['tile_size'],
            'Total_Pipeline_Time': result.get('total_pipeline_time', 0),
            'Num_Stages': result.get('num_stages', 0)
        }
        
        # Add stage-specific timings
        if 'stages' in result:
            for stage, stage_data in result['stages'].items():
                if stage_data.get('success', False):
                    row[f'{stage}_time'] = stage_data.get('total_time', 0)
                    if 'avg_frame_time' in stage_data:
                        row[f'{stage}_avg_frame_time'] = stage_data['avg_frame_time']
                    if 'fps' in stage_data:
                        row[f'{stage}_fps'] = stage_data['fps']
        
        pipeline_data.append(row)
    
    df = pd.DataFrame(pipeline_data)
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'pipeline_results.csv'), index=False)
    
    # Set matplotlib style
    plt.style.use('default')
    
    # 1. Pipeline performance overview dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pipeline Performance Overview', fontsize=16, fontweight='bold')
    
    # Get all available stages
    all_stages = set()
    for result in successful_results:
        if 'stages' in result:
            all_stages.update(result['stages'].keys())
    all_stages = sorted(all_stages)
    
    # Total pipeline time comparison
    pipeline_times = [result.get('total_pipeline_time', 0) for result in successful_results]
    video_names = [result['video_name'] for result in successful_results]
    
    bars1 = axes[0, 0].bar(range(len(video_names)), pipeline_times, color='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
    axes[0, 0].set_title('Total Pipeline Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Video', fontsize=12)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_xticks(range(len(video_names)))
    axes[0, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Number of stages comparison
    num_stages = [result.get('num_stages', 0) for result in successful_results]
    
    bars2 = axes[0, 1].bar(range(len(video_names)), num_stages, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    axes[0, 1].set_title('Number of Pipeline Stages', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Video', fontsize=12)
    axes[0, 1].set_ylabel('Number of Stages', fontsize=12)
    axes[0, 1].set_xticks(range(len(video_names)))
    axes[0, 1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Stage breakdown (stacked bar chart)
    stage_times_matrix = []
    for stage in all_stages:
        stage_times = []
        for result in successful_results:
            if 'stages' in result and stage in result['stages'] and result['stages'][stage].get('success', False):
                stage_times.append(result['stages'][stage].get('total_time', 0))
            else:
                stage_times.append(0)
        stage_times_matrix.append(stage_times)
    
    import matplotlib.cm as cm
    colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(all_stages)))  # Generate distinct colors
    bottom = np.zeros(len(video_names))
    
    for i, (stage, stage_times) in enumerate(zip(all_stages, stage_times_matrix)):
        axes[1, 0].bar(range(len(video_names)), stage_times, bottom=bottom, 
                      label=stage, alpha=0.8, color=colors[i])
        bottom += stage_times
    
    axes[1, 0].set_title('Pipeline Stage Breakdown', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Video', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_xticks(range(len(video_names)))
    axes[1, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Stage performance heatmap
    if all_stages and len(all_stages) > 0:
        stage_matrix = np.array(stage_times_matrix)
        im = axes[1, 1].imshow(stage_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('Stage Performance Heatmap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Video', fontsize=12)
        axes[1, 1].set_ylabel('Pipeline Stage', fontsize=12)
        axes[1, 1].set_xticks(range(len(video_names)))
        axes[1, 1].set_xticklabels(video_names, rotation=45, ha='right')
        axes[1, 1].set_yticks(range(len(all_stages)))
        axes[1, 1].set_yticklabels(all_stages)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Time (seconds)', fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, 'No stage data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Stage Performance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pipeline_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stage performance analysis for each video
    for result in successful_results:
        if 'stages' not in result or not result['stages']:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Stage Performance Analysis: {result["video_name"]} (Tile {result["tile_size"]})', 
                     fontsize=16, fontweight='bold')
        
        # Stage timing comparison
        available_stages = []
        stage_times = []
        stage_colors = []
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(all_stages)))
        
        for i, stage in enumerate(all_stages):
            if stage in result['stages'] and result['stages'][stage].get('success', False):
                available_stages.append(stage)
                stage_times.append(result['stages'][stage].get('total_time', 0))
                stage_colors.append(colors[i])
        
        if available_stages:
            bars = axes[0, 0].bar(range(len(available_stages)), stage_times, color=stage_colors, alpha=0.7)
            axes[0, 0].set_title('Stage Timing Comparison', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Pipeline Stage', fontsize=12)
            axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
            axes[0, 0].set_xticks(range(len(available_stages)))
            axes[0, 0].set_xticklabels(available_stages, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # Stage pie chart
        if available_stages and stage_times:
            axes[0, 1].pie(stage_times, labels=available_stages, autopct='%1.1f%%', colors=stage_colors)
            axes[0, 1].set_title('Stage Time Distribution', fontsize=14, fontweight='bold')
        
        # Frame-based stages performance
        frame_based_stages = ['tune_detect', 'classify', 'compress', 'detect_packed', 'track']
        frame_stage_fps = []
        frame_stage_names = []
        
        for stage in frame_based_stages:
            if stage in result['stages'] and result['stages'][stage].get('success', False):
                stage_data = result['stages'][stage]
                if 'fps' in stage_data:
                    frame_stage_fps.append(stage_data['fps'])
                    frame_stage_names.append(stage)
        
        if frame_stage_fps:
            axes[1, 0].bar(range(len(frame_stage_names)), frame_stage_fps, color='lightblue', alpha=0.7)
            axes[1, 0].set_title('Frame Processing Rate (FPS)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Pipeline Stage', fontsize=12)
            axes[1, 0].set_ylabel('FPS', fontsize=12)
            axes[1, 0].set_xticks(range(len(frame_stage_names)))
            axes[1, 0].set_xticklabels(frame_stage_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Pipeline summary text
        summary_text = f"Total Pipeline Time: {result.get('total_pipeline_time', 0):.2f}s\n"
        summary_text += f"Number of Stages: {result.get('num_stages', 0)}\n\n"
        summary_text += "Stage Details:\n"
        for stage in available_stages:
            stage_data = result['stages'][stage]
            summary_text += f"• {stage}: {stage_data.get('total_time', 0):.3f}s"
            if 'total_frames' in stage_data:
                summary_text += f" ({stage_data['total_frames']} frames)"
            summary_text += "\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Pipeline Summary', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        safe_filename = result["video_name"].replace('.', '_')
        plt.savefig(os.path.join(output_dir, f'stage_performance_{safe_filename}_tile{result["tile_size"]}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Tile size comparison (if multiple tile sizes exist)
    tile_sizes = [result['tile_size'] for result in successful_results]
    if len(set(tile_sizes)) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Pipeline Performance Comparison by Tile Size', fontsize=16, fontweight='bold')
        
        metrics = [('Pipeline Time', pipeline_times), ('Num Stages', num_stages)]
        
        for i, (metric_name, values) in enumerate(metrics):
            if i >= 2:  # Only create 2 plots for the available metrics
                break
                
            # Group by tile size
            tile_size_data = {}
            for tile_size, value in zip(tile_sizes, values):
                if tile_size not in tile_size_data:
                    tile_size_data[tile_size] = []
                tile_size_data[tile_size].append(value)
            
            # Create box plot
            bp = axes[i].boxplot([tile_size_data[ts] for ts in sorted(tile_size_data.keys())], 
                                labels=[f'Tile {ts}' for ts in sorted(tile_size_data.keys())],
                                patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'{metric_name} by Tile Size', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].grid(True, alpha=0.3, linestyle='--')
        
        # Stage-wise comparison by tile size
        if len(all_stages) > 0:
            # Pick the most common stage for comparison
            stage_counts = {}
            for result in successful_results:
                if 'stages' in result:
                    for stage in result['stages']:
                        if result['stages'][stage].get('success', False):
                            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            if stage_counts:
                most_common_stage = max(stage_counts.keys(), key=lambda k: stage_counts[k])
                stage_times_by_tile = {}
                
                for result in successful_results:
                    tile_size = result['tile_size']
                    if tile_size not in stage_times_by_tile:
                        stage_times_by_tile[tile_size] = []
                    
                    if ('stages' in result and most_common_stage in result['stages'] and 
                        result['stages'][most_common_stage].get('success', False)):
                        stage_times_by_tile[tile_size].append(result['stages'][most_common_stage].get('total_time', 0))
                
                if len(axes) > 2 and stage_times_by_tile:
                    bp = axes[2].boxplot([stage_times_by_tile[ts] for ts in sorted(stage_times_by_tile.keys()) if stage_times_by_tile[ts]], 
                                        labels=[f'Tile {ts}' for ts in sorted(stage_times_by_tile.keys()) if stage_times_by_tile[ts]],
                                        patch_artist=True)
                    
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[2].set_title(f'{most_common_stage} Time by Tile Size', fontsize=14, fontweight='bold')
                    axes[2].set_ylabel('Time (seconds)', fontsize=12)
                    axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tile_size_pipeline_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Matplotlib pipeline visualizations saved to {output_dir}")
    print("Generated files:")
    print("  - pipeline_overview.png")
    print("  - stage_performance_[video]_tile[size].png (for each video)")
    if len(set(tile_sizes)) > 1:
        print("  - tile_size_pipeline_comparison.png")
    print("  - pipeline_results.csv")


def main(args):
    """
    Main function that orchestrates the pipeline performance analysis process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with pipeline results and runtime data for the specified dataset and tile size(s)
    2. Analyzes runtime performance metrics from all available pipeline stages
    3. Creates comprehensive performance summaries and visualizations
    4. Saves results in multiple formats (JSON, TXT, CSV, PNG)
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script collects runtime data from all pipeline stages
        - Results are saved to the specified output directory
        - Performance metrics include timing for each pipeline stage and overall pipeline performance
    """
    print(f"Starting pipeline performance analysis for dataset: {args.dataset}")
    print(f"Tile size(s): {args.tile_size}")
    print(f"Pipeline stages: {args.stages}")
    print(f"Output directory: {args.output_dir}")
    
    # Find pipeline results with runtime data
    video_tile_combinations = find_pipeline_results(CACHE_DIR, args.dataset, args.tile_size, args.stages)
    
    if not video_tile_combinations:
        print("No pipeline results with runtime data found. Please ensure the pipeline scripts have been run.")
        return
    
    print(f"Found {len(video_tile_combinations)} video-tile size combinations to analyze")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze pipeline performance
    results = []
    
    # Prepare arguments for parallel processing
    analysis_args = []
    for video_name, tile_size, available_stages in video_tile_combinations:
        analysis_args.append((video_name, tile_size, available_stages, args.output_dir))
    
    # Run analysis in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(analyze_pipeline_performance, analysis_args)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nAnalysis completed:")
    print(f"  Successful analyses: {len(successful_results)}")
    print(f"  Failed analyses: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed analyses:")
        for result in failed_results:
            print(f"  {result['video_name']} (tile size {result['tile_size']}): {result['error']}")
    
    if successful_results:
        # Create summary
        create_pipeline_summary(successful_results, args.output_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        pipeline_times = [r.get('total_pipeline_time', 0) for r in successful_results]
        num_stages = [r.get('num_stages', 0) for r in successful_results]
        
        if pipeline_times:
            print(f"  Pipeline Time: Mean={np.mean(pipeline_times):.2f}s, Std={np.std(pipeline_times):.2f}s")
        if num_stages:
            print(f"  Stages per Video: Mean={np.mean(num_stages):.1f}, Max={np.max(num_stages)}")
        
        # Stage-wise statistics
        all_stages = set()
        for result in successful_results:
            if 'stages' in result:
                all_stages.update(result['stages'].keys())
        
        for stage in sorted(all_stages):
            stage_times = []
            for result in successful_results:
                if 'stages' in result and stage in result['stages'] and result['stages'][stage].get('success', False):
                    stage_times.append(result['stages'][stage].get('total_time', 0))
            
            if stage_times:
                print(f"  {stage}: Mean={np.mean(stage_times):.4f}s, Videos={len(stage_times)}")
        
        # Create visualizations if requested
        if args.create_plots:
            print("\nCreating pipeline performance visualizations...")
            create_pipeline_visualizations(successful_results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main(parse_args())
