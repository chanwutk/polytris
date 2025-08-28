#!/usr/local/bin/python

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import glob

from scripts.utilities import CACHE_DIR

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create comprehensive visualizations for each tracking output')
    parser.add_argument('--dataset', type=str, default='b3d',
                        help='Dataset name to analyze')
    parser.add_argument('--output_dir', type=str, 
                        default='pipeline-stages/tracking-analysis',
                        help='Output directory for visualizations')
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Whether to create visualization plots')
    return parser.parse_args()


def find_tracking_configurations(cache_dir: str, dataset: str) -> List[Tuple[str, str, int]]:
    """
    Find all tracking configurations by examining uncompressed_tracking directories.
    
    Returns:
        List of (video_name, classifier, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset)
    if not os.path.exists(dataset_cache_dir):
        raise ValueError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    tracking_configs = []
    
    for video_item in os.listdir(dataset_cache_dir):
        video_path = os.path.join(dataset_cache_dir, video_item)
        if not os.path.isdir(video_path):
            continue
            
        uncompressed_tracking_dir = os.path.join(video_path, 'uncompressed_tracking')
        if not os.path.exists(uncompressed_tracking_dir):
            continue
            
        # Look for classifier_tilesize directories
        for config_dir in os.listdir(uncompressed_tracking_dir):
            config_path = os.path.join(uncompressed_tracking_dir, config_dir)
            if not os.path.isdir(config_path):
                continue
                
            # Check if tracking results exist
            tracking_file = os.path.join(config_path, 'tracking.jsonl')
            runtime_file = os.path.join(config_path, 'runtimes.jsonl')
            
            if os.path.exists(tracking_file) and os.path.exists(runtime_file):
                # Parse classifier and tile_size from directory name
                if '_' in config_dir:
                    parts = config_dir.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        classifier = '_'.join(parts[:-1])
                        tile_size = int(parts[-1])
                        tracking_configs.append((video_item, classifier, tile_size))
                        print(f"Found tracking config: {video_item} -> {classifier}_{tile_size}")
    
    return tracking_configs


def load_stage_runtime_for_config(cache_dir: str, dataset: str, video_name: str, 
                                 classifier: str, tile_size: int) -> Dict[str, Any]:
    """
    Load runtime data for all pipeline stages leading to a specific tracking configuration.
    
    Args:
        cache_dir: Cache directory path
        dataset: Dataset name
        video_name: Video file name
        classifier: Classifier name
        tile_size: Tile size
        
    Returns:
        Dictionary with runtime data for each stage
    """
    video_path = os.path.join(cache_dir, dataset, video_name)
    stage_runtimes = {}
    
    # Stage 1: 010_tune_segment_videos.py (no runtime - shared across all configs)
    # This stage doesn't produce runtime data, but we note it as preparation
    stage_runtimes['segment_videos'] = {
        'stage': 'segment_videos',
        'description': 'Video segmentation (010_tune_segment_videos.py)',
        'total_time': 0,
        'note': 'No runtime measurement - shared across all configurations',
        'success': True
    }
    
    # Stage 2: 011_tune_detect.py (shared across all configs)
    detect_path = os.path.join(video_path, 'segments', 'detection', 'detections.jsonl')
    if os.path.exists(detect_path):
        try:
            runtime_data = load_tune_detect_runtime(detect_path)
            stage_runtimes['tune_detect'] = analyze_frame_stage_performance('tune_detect', runtime_data)
            stage_runtimes['tune_detect']['note'] = 'Shared across all configurations'
        except Exception as e:
            stage_runtimes['tune_detect'] = {
                'stage': 'tune_detect',
                'description': 'Detection on video segments (011_tune_detect.py)', 
                'error': str(e),
                'success': False
            }
    
    # Stage 3: 012_tune_create_training_data.py (shared by tile_size)
    training_data_path = os.path.join(video_path, 'training', 'runtime', f'tilesize_{tile_size}', 'create_training_data.jsonl')
    if os.path.exists(training_data_path):
        try:
            runtime_data = load_jsonl_runtime(training_data_path)
            stage_runtimes['create_training_data'] = analyze_operation_stage_performance('create_training_data', runtime_data)
            stage_runtimes['create_training_data']['note'] = f'Shared by tile_size {tile_size}'
        except Exception as e:
            stage_runtimes['create_training_data'] = {
                'stage': 'create_training_data',
                'description': 'Training data creation (012_tune_create_training_data.py)',
                'error': str(e),
                'success': False
            }
    
    # Stage 4: 013_tune_train_classifier.py (specific to classifier+tile_size)
    training_logs_dir = os.path.join(video_path, 'training', 'results', f'{classifier}_{tile_size}')
    if os.path.exists(training_logs_dir):
        try:
            runtime_data = load_training_logs_runtime(training_logs_dir)
            stage_runtimes['train_classifier'] = analyze_training_stage_performance('train_classifier', runtime_data)
        except Exception as e:
            stage_runtimes['train_classifier'] = {
                'stage': 'train_classifier',
                'description': 'Classifier training (013_tune_train_classifier.py)',
                'error': str(e),
                'success': False
            }
    
    # Stage 5: 020_exec_classify.py (specific to classifier+tile_size)
    classify_path = os.path.join(video_path, 'relevancy', f'{classifier}_{tile_size}', 'score', 'score.jsonl')
    if os.path.exists(classify_path):
        try:
            runtime_data = load_classify_runtime(classify_path)
            stage_runtimes['classify'] = analyze_frame_stage_performance('classify', runtime_data)
        except Exception as e:
            stage_runtimes['classify'] = {
                'stage': 'classify',
                'description': 'Tile classification (020_exec_classify.py)',
                'error': str(e),
                'success': False
            }
    
    # Stage 6: 030_exec_compress.py (specific to classifier+tile_size)
    compress_path = os.path.join(video_path, 'packing', f'{classifier}_{tile_size}', 'runtime.jsonl')
    if os.path.exists(compress_path):
        try:
            runtime_data = load_jsonl_runtime(compress_path)
            stage_runtimes['compress'] = analyze_operation_stage_performance('compress', runtime_data)
        except Exception as e:
            stage_runtimes['compress'] = {
                'stage': 'compress',
                'description': 'Video compression/packing (030_exec_compress.py)',
                'error': str(e),
                'success': False
            }
    
    # Stage 7: 040_exec_detect.py (specific to classifier+tile_size)
    detect_packed_path = os.path.join(video_path, 'packed_detections', f'{classifier}_{tile_size}', 'runtimes.jsonl')
    if os.path.exists(detect_packed_path):
        try:
            runtime_data = load_jsonl_runtime(detect_packed_path)
            stage_runtimes['detect_packed'] = analyze_operation_stage_performance('detect_packed', runtime_data)
        except Exception as e:
            stage_runtimes['detect_packed'] = {
                'stage': 'detect_packed',
                'description': 'Detection on packed images (040_exec_detect.py)',
                'error': str(e),
                'success': False
            }
    
    # Stage 8: 050_exec_uncompress.py (specific to classifier+tile_size) 
    # Note: This stage typically doesn't have separate runtime files, runtime is in the next stage
    
    # Stage 9: 060_exec_track.py (specific to classifier+tile_size)
    track_path = os.path.join(video_path, 'uncompressed_tracking', f'{classifier}_{tile_size}', 'runtimes.jsonl')
    if os.path.exists(track_path):
        try:
            runtime_data = load_jsonl_runtime(track_path)
            stage_runtimes['track'] = analyze_frame_stage_performance('track', runtime_data)
        except Exception as e:
            stage_runtimes['track'] = {
                'stage': 'track',
                'description': 'Object tracking (060_exec_track.py)',
                'error': str(e),
                'success': False
            }
    
    return stage_runtimes


def load_tune_detect_runtime(file_path: str) -> List[Dict[str, Any]]:
    """Load runtime data from tune_detect detections.jsonl file."""
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


def load_classify_runtime(file_path: str) -> List[Dict[str, Any]]:
    """Load runtime data from classify score.jsonl file."""
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


def load_jsonl_runtime(file_path: str) -> List[Dict[str, Any]]:
    """Load runtime data from generic JSONL file."""
    runtime_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                runtime_data.append(data)
    
    return runtime_data


def load_training_logs_runtime(dir_path: str) -> Dict[str, Any]:
    """Load runtime data from training log files."""
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


def analyze_frame_stage_performance(stage: str, stage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance data for frame-based stages."""
    if not stage_data:
        return {
            'stage': stage,
            'error': 'No timing data found',
            'success': False
        }
    
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
        
        return stats
    else:
        return {
            'stage': stage,
            'error': 'No valid timing data found',
            'success': False
        }


def analyze_operation_stage_performance(stage: str, stage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance data for operation-based stages."""
    if not stage_data:
        return {
            'stage': stage,
            'error': 'No timing data found',
            'success': False
        }
    
    total_time = 0
    operation_times = {}
    
    for entry in stage_data:
        if 'runtime' in entry and isinstance(entry['runtime'], list):
            # Handle format with runtime as list of operations
            for op in entry['runtime']:
                if 'op' in op and 'time' in op:
                    op_name = op['op']
                    op_time = op['time']
                    if op_name not in operation_times:
                        operation_times[op_name] = []
                    operation_times[op_name].append(op_time)
                    total_time += op_time
        elif 'time' in entry:
            # Handle simple time format
            total_time += entry['time']
    
    stats = {
        'stage': stage,
        'total_time': total_time,
        'num_operations': len(stage_data),
        'success': True
    }
    
    # Add operation-specific stats
    for op_name, op_times in operation_times.items():
        if op_times:
            stats[f'{op_name}_avg'] = np.mean(op_times)
            stats[f'{op_name}_total'] = sum(op_times)
    
    return stats


def analyze_training_stage_performance(stage: str, stage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance data for training stage."""
    return {
        'stage': stage,
        'total_time': stage_data.get('total_time', 0),
        'total_train_time': stage_data.get('total_train_time', 0),
        'total_test_time': stage_data.get('total_test_time', 0),
        'num_train_epochs': stage_data.get('num_train_epochs', 0),
        'num_test_epochs': stage_data.get('num_test_epochs', 0),
        'avg_train_time_per_epoch': stage_data.get('avg_train_time_per_epoch', 0),
        'avg_test_time_per_epoch': stage_data.get('avg_test_time_per_epoch', 0),
        'success': True
    }


def create_tracking_configuration_analysis(config: Tuple[str, str, int], stage_runtimes: Dict[str, Any], 
                                         output_dir: str) -> Dict[str, Any]:
    """
    Create comprehensive analysis for a single tracking configuration.
    
    Returns:
        Dictionary with analysis results
    """
    video_name, classifier, tile_size = config
    
    # Calculate total pipeline time
    total_pipeline_time = 0
    successful_stages = []
    
    for stage_name, stage_data in stage_runtimes.items():
        if stage_data.get('success', False):
            total_pipeline_time += stage_data.get('total_time', 0)
            successful_stages.append(stage_name)
    
    analysis = {
        'video_name': video_name,
        'classifier': classifier,
        'tile_size': tile_size,
        'configuration': f"{classifier}_{tile_size}",
        'total_pipeline_time': total_pipeline_time,
        'num_stages': len(successful_stages),
        'successful_stages': successful_stages,
        'stages': stage_runtimes
    }
    
    return analysis


def create_configuration_visualizations(analyses: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations for tracking configurations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create overview comparison
    create_configuration_overview(analyses, output_dir)
    
    # 2. Create detailed analysis for each configuration
    create_individual_configuration_charts(analyses, output_dir)
    
    # 3. Create stage comparison across configurations
    create_stage_comparison_charts(analyses, output_dir)
    
    # 4. Create summary report
    create_configuration_summary(analyses, output_dir)


def create_configuration_overview(analyses: List[Dict[str, Any]], output_dir: str):
    """Create overview comparison of all tracking configurations."""
    
    if not analyses:
        return
    
    # Extract data for plotting
    config_names = [f"{a['video_name']}\n{a['configuration']}" for a in analyses]
    pipeline_times = [a['total_pipeline_time'] / 1000 for a in analyses]  # Convert to seconds
    num_stages = [a['num_stages'] for a in analyses]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Tracking Configuration Performance Overview', fontsize=16, fontweight='bold')
    
    # Total pipeline time comparison
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    bars1 = ax1.bar(range(len(config_names)), pipeline_times, color=colors, alpha=0.8)
    
    ax1.set_title('Total Pipeline Time by Configuration', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Configuration (Video + Classifier_TileSize)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, time in zip(bars1, pipeline_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(pipeline_times)*0.01,
                f'{height:.0f}s', ha='center', va='bottom', fontsize=9)
    
    # Number of successful stages
    bars2 = ax2.bar(range(len(config_names)), num_stages, color=colors, alpha=0.8)
    
    ax2.set_title('Number of Successful Pipeline Stages', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Configuration (Video + Classifier_TileSize)', fontsize=12)
    ax2.set_ylabel('Number of Stages', fontsize=12)
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, count in zip(bars2, num_stages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tracking_configurations_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_individual_configuration_charts(analyses: List[Dict[str, Any]], output_dir: str):
    """Create detailed charts for each tracking configuration."""
    
    for analysis in analyses:
        config_name = f"{analysis['video_name']}_{analysis['configuration']}"
        
        # Get successful stages and their times
        stage_names = []
        stage_times = []
        stage_descriptions = []
        
        stage_order = ['segment_videos', 'tune_detect', 'create_training_data', 'train_classifier', 
                      'classify', 'compress', 'detect_packed', 'track']
        
        for stage in stage_order:
            if stage in analysis['stages'] and analysis['stages'][stage].get('success', False):
                stage_data = analysis['stages'][stage]
                stage_names.append(stage.replace('_', ' ').title())
                stage_times.append(stage_data.get('total_time', 0) / 1000)  # Convert to seconds
                stage_descriptions.append(stage_data.get('description', ''))
        
        if not stage_names:
            continue
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Pipeline Analysis: {analysis["video_name"]} - {analysis["configuration"]}', 
                     fontsize=16, fontweight='bold')
        
        # Stage timing bar chart
        colors = plt.cm.Set2(np.linspace(0, 1, len(stage_names)))
        bars = ax1.bar(range(len(stage_names)), stage_times, color=colors, alpha=0.8)
        
        ax1.set_title('Stage Processing Times', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Pipeline Stage', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_xticks(range(len(stage_names)))
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, time in zip(bars, stage_times):
            height = bar.get_height()
            if height > max(stage_times) * 0.01:  # Only label significant bars
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(stage_times)*0.02,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # Stage timing pie chart
        ax2.pie(stage_times, labels=stage_names, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Stage Time Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'config_analysis_{config_name}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_stage_comparison_charts(analyses: List[Dict[str, Any]], output_dir: str):
    """Create charts comparing each stage across different configurations."""
    
    # Get all unique stages
    all_stages = set()
    for analysis in analyses:
        all_stages.update(analysis['stages'].keys())
    
    all_stages = sorted(all_stages)
    
    for stage in all_stages:
        stage_data = []
        config_names = []
        
        for analysis in analyses:
            if stage in analysis['stages'] and analysis['stages'][stage].get('success', False):
                stage_info = analysis['stages'][stage]
                stage_data.append(stage_info.get('total_time', 0) / 1000)  # Convert to seconds
                config_names.append(f"{analysis['video_name']}\n{analysis['configuration']}")
        
        if not stage_data:
            continue
        
        # Create comparison chart for this stage
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
        bars = ax.bar(range(len(config_names)), stage_data, color=colors, alpha=0.8)
        
        ax.set_title(f'{stage.replace("_", " ").title()} - Performance Across Configurations', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, time in zip(bars, stage_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(stage_data)*0.01,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stage_comparison_{stage}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_configuration_summary(analyses: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive summary report."""
    
    summary_file = os.path.join(output_dir, 'tracking_configurations_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Tracking Configuration Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Configurations Analyzed: {len(analyses)}\n\n")
        
        # Overall statistics
        total_pipeline_times = [a['total_pipeline_time'] for a in analyses if a['total_pipeline_time'] > 0]
        if total_pipeline_times:
            f.write("Overall Pipeline Performance:\n")
            f.write("-" * 35 + "\n")
            f.write(f"  Mean Total Time: {np.mean(total_pipeline_times)/1000:.2f}s\n")
            f.write(f"  Std Total Time: {np.std(total_pipeline_times)/1000:.2f}s\n")
            f.write(f"  Min Total Time: {np.min(total_pipeline_times)/1000:.2f}s\n")
            f.write(f"  Max Total Time: {np.max(total_pipeline_times)/1000:.2f}s\n\n")
        
        # Individual configuration details
        f.write("Individual Configuration Results:\n")
        f.write("-" * 40 + "\n")
        
        for analysis in analyses:
            f.write(f"\nConfiguration: {analysis['video_name']} - {analysis['configuration']}\n")
            f.write(f"  Total Pipeline Time: {analysis['total_pipeline_time']/1000:.2f}s\n")
            f.write(f"  Successful Stages: {analysis['num_stages']}/{len(analysis['stages'])}\n")
            f.write("  Stage Breakdown:\n")
            
            # Stage details
            stage_order = ['segment_videos', 'tune_detect', 'create_training_data', 'train_classifier', 
                          'classify', 'compress', 'detect_packed', 'track']
            
            for stage in stage_order:
                if stage in analysis['stages']:
                    stage_data = analysis['stages'][stage]
                    if stage_data.get('success', False):
                        time_str = f"{stage_data.get('total_time', 0)/1000:.2f}s"
                        note = stage_data.get('note', '')
                        if note:
                            time_str += f" ({note})"
                        f.write(f"    ✓ {stage}: {time_str}\n")
                    else:
                        error = stage_data.get('error', 'Unknown error')
                        f.write(f"    ✗ {stage}: Failed - {error}\n")
        
        # Stage-wise analysis
        f.write(f"\n\nStage-wise Performance Analysis:\n")
        f.write("-" * 40 + "\n")
        
        all_stages = set()
        for analysis in analyses:
            all_stages.update(analysis['stages'].keys())
        
        for stage in sorted(all_stages):
            stage_times = []
            stage_configs = []
            
            for analysis in analyses:
                if stage in analysis['stages'] and analysis['stages'][stage].get('success', False):
                    stage_times.append(analysis['stages'][stage].get('total_time', 0))
                    stage_configs.append(f"{analysis['video_name']}_{analysis['configuration']}")
            
            if stage_times:
                f.write(f"\n{stage.upper()}:\n")
                f.write(f"  Successful in {len(stage_times)}/{len(analyses)} configurations\n")
                f.write(f"  Mean Time: {np.mean(stage_times)/1000:.2f}s\n")
                f.write(f"  Std Time: {np.std(stage_times)/1000:.2f}s\n")
                f.write(f"  Min Time: {np.min(stage_times)/1000:.2f}s\n")
                f.write(f"  Max Time: {np.max(stage_times)/1000:.2f}s\n")
    
    print(f"Configuration summary saved to: {summary_file}")
    
    # Also save detailed results as JSON
    json_file = os.path.join(output_dir, 'tracking_configurations_detailed.json')
    with open(json_file, 'w') as f:
        json.dump(analyses, f, indent=2)
    
    print(f"Detailed results saved to: {json_file}")


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Analyzing tracking configurations for dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    
    # Find all tracking configurations
    tracking_configs = find_tracking_configurations(CACHE_DIR, args.dataset)
    
    if not tracking_configs:
        print("No tracking configurations found")
        return
    
    print(f"Found {len(tracking_configs)} tracking configurations to analyze")
    
    # Analyze each configuration
    analyses = []
    
    for config in tracking_configs:
        video_name, classifier, tile_size = config
        print(f"\nAnalyzing configuration: {video_name} - {classifier}_{tile_size}")
        
        # Load runtime data for all stages leading to this tracking result
        stage_runtimes = load_stage_runtime_for_config(CACHE_DIR, args.dataset, video_name, 
                                                      classifier, tile_size)
        
        # Create analysis for this configuration
        analysis = create_tracking_configuration_analysis(config, stage_runtimes, args.output_dir)
        analyses.append(analysis)
        
        print(f"  Pipeline time: {analysis['total_pipeline_time']/1000:.2f}s")
        print(f"  Successful stages: {analysis['num_stages']}")
    
    # Create visualizations
    if args.create_plots and analyses:
        print(f"\nCreating visualizations...")
        create_configuration_visualizations(analyses, args.output_dir)
        
        print("Tracking configuration analysis complete!")
        print(f"Generated files in {args.output_dir}:")
        print("  - tracking_configurations_overview.png")
        print("  - config_analysis_[video]_[config].png (for each configuration)")
        print("  - stage_comparison_[stage].png (for each stage)")
        print("  - tracking_configurations_summary.txt")
        print("  - tracking_configurations_detailed.json")
    
    print(f"\nAnalysis completed for {len(analyses)} tracking configurations")


if __name__ == '__main__':
    main()
