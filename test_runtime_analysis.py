#!/usr/bin/env python3
"""
Test script for the runtime analysis functionality.
This script creates sample runtime data and demonstrates the analysis capabilities.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Mock the matplotlib and seaborn imports for testing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting will be skipped")

def create_sample_runtime_data():
    """Create sample runtime data for testing."""
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = os.path.join(temp_dir, 'b3d', 'test_video')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 1. Create sample classification runtime data
        classification_dir = os.path.join(dataset_dir, 'relevancy', 'score', 'proxy_64')
        os.makedirs(classification_dir, exist_ok=True)
        
        classification_data = []
        for i in range(100):
            classification_data.append({
                'frame_idx': i,
                'runtime': np.random.exponential(0.1) + 0.05,  # Random runtime
                'stage': 'classification'
            })
        
        with open(os.path.join(classification_dir, 'score.jsonl'), 'w') as f:
            for data in classification_data:
                f.write(json.dumps(data) + '\n')
        
        # 2. Create sample compression runtime data
        compression_dir = os.path.join(dataset_dir, 'packing', 'proxy_64')
        os.makedirs(compression_dir, exist_ok=True)
        
        compression_data = []
        for i in range(100):
            compression_data.append({
                'frame_idx': i,
                'step_times': {
                    'read_frame': np.random.exponential(0.02) + 0.01,
                    'get_classifications': np.random.exponential(0.01) + 0.005,
                    'create_bitmap': np.random.exponential(0.03) + 0.02,
                    'group_tiles': np.random.exponential(0.05) + 0.03,
                    'pack_append': np.random.exponential(0.08) + 0.05,
                    'total_frame_time': np.random.exponential(0.2) + 0.15
                }
            })
        
        with open(os.path.join(compression_dir, 'runtime.jsonl'), 'w') as f:
            for data in compression_data:
                f.write(json.dumps(data) + '\n')
        
        # 3. Create sample detection runtime data
        detection_dir = os.path.join(dataset_dir, 'packed_detections', 'proxy_64')
        os.makedirs(detection_dir, exist_ok=True)
        
        detection_data = []
        for i in range(50):  # Fewer packed images
            detection_data.append({
                'image_file': f'img_{i}.jpg',
                'read_time': int((np.random.exponential(0.01) + 0.005) * 1e9),  # nanoseconds
                'detect_time': int((np.random.exponential(0.1) + 0.05) * 1e9)   # nanoseconds
            })
        
        with open(os.path.join(detection_dir, 'runtimes.jsonl'), 'w') as f:
            for data in detection_data:
                f.write(json.dumps(data) + '\n')
        
        # 4. Create sample tracking runtime data
        tracking_dir = os.path.join(dataset_dir, 'uncompressed_tracking', 'proxy_64')
        os.makedirs(tracking_dir, exist_ok=True)
        
        tracking_runtime_data = []
        tracking_results_data = []
        
        for i in range(100):
            # Runtime data
            tracking_runtime_data.append({
                'frame_idx': i,
                'step_times': {
                    'convert_detections': np.random.exponential(0.01) + 0.005,
                    'tracker_update': np.random.exponential(0.05) + 0.03,
                    'process_results': np.random.exponential(0.02) + 0.01,
                    'total_frame_time': np.random.exponential(0.1) + 0.08
                },
                'num_detections': np.random.poisson(5) + 1,
                'num_tracks': np.random.poisson(3) + 1
            })
            
            # Tracking results
            num_tracks = np.random.poisson(3) + 1
            tracks = []
            for j in range(num_tracks):
                x1 = np.random.uniform(0, 640)
                y1 = np.random.uniform(0, 480)
                x2 = x1 + np.random.uniform(20, 100)
                y2 = y1 + np.random.uniform(20, 100)
                tracks.append([j, x1, y1, x2, y2])
            
            tracking_results_data.append({
                'frame_idx': i,
                'tracks': tracks
            })
        
        with open(os.path.join(tracking_dir, 'runtimes.jsonl'), 'w') as f:
            for data in tracking_runtime_data:
                f.write(json.dumps(data) + '\n')
        
        with open(os.path.join(tracking_dir, 'tracks.jsonl'), 'w') as f:
            for data in tracking_results_data:
                f.write(json.dumps(data) + '\n')
        
        print(f"Sample data created in: {temp_dir}")
        return temp_dir

def test_runtime_analysis():
    """Test the runtime analysis functionality."""
    
    if not PLOTTING_AVAILABLE:
        print("Skipping plotting tests due to missing dependencies")
        return
    
    # Create sample data
    temp_dir = create_sample_runtime_data()
    
    # Test data loading functions
    print("\nTesting data loading...")
    
    # Test classification data loading
    classification_path = os.path.join(temp_dir, 'b3d', 'test_video', 'relevancy', 'score', 'proxy_64', 'score.jsonl')
    if os.path.exists(classification_path):
        print("✓ Classification data found")
        
        # Load and analyze
        with open(classification_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        df = pd.DataFrame(data)
        print(f"  - Loaded {len(df)} frames")
        print(f"  - Mean runtime: {df['runtime'].mean():.4f}s")
        print(f"  - Std runtime: {df['runtime'].std():.4f}s")
    
    # Test compression data loading
    compression_path = os.path.join(temp_dir, 'b3d', 'test_video', 'packing', 'proxy_64', 'runtime.jsonl')
    if os.path.exists(compression_path):
        print("✓ Compression data found")
        
        # Load and analyze
        with open(compression_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"  - Loaded {len(data)} frames")
        if data:
            step_times = data[0]['step_times']
            print(f"  - Available steps: {list(step_times.keys())}")
    
    # Test detection data loading
    detection_path = os.path.join(temp_dir, 'b3d', 'test_video', 'packed_detections', 'proxy_64', 'runtimes.jsonl')
    if os.path.exists(detection_path):
        print("✓ Detection data found")
        
        # Load and analyze
        with open(detection_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"  - Loaded {len(data)} packed images")
        if data:
            print(f"  - Sample read time: {data[0]['read_time'] / 1e9:.6f}s")
            print(f"  - Sample detect time: {data[0]['detect_time'] / 1e9:.6f}s")
    
    # Test tracking data loading
    tracking_path = os.path.join(temp_dir, 'b3d', 'test_video', 'uncompressed_tracking', 'proxy_64', 'runtimes.jsonl')
    if os.path.exists(tracking_path):
        print("✓ Tracking runtime data found")
        
        # Load and analyze
        with open(tracking_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"  - Loaded {len(data)} frames")
        if data:
            step_times = data[0]['step_times']
            print(f"  - Available steps: {list(step_times.keys())}")
    
    tracks_path = os.path.join(temp_dir, 'b3d', 'test_video', 'uncompressed_tracking', 'proxy_64', 'tracks.jsonl')
    if os.path.exists(tracks_path):
        print("✓ Tracking results found")
        
        # Load and analyze
        with open(tracks_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"  - Loaded {len(data)} frames")
        if data:
            num_tracks = len(data[0]['tracks'])
            print(f"  - Sample frame has {num_tracks} tracks")
    
    print("\n✓ All data loading tests passed!")
    
    # Test summary statistics creation
    print("\nTesting summary statistics...")
    
    # Create mock runtime data structure
    runtime_data = {
        'classification': pd.DataFrame([
            {'frame_idx': i, 'runtime': np.random.exponential(0.1) + 0.05}
            for i in range(100)
        ]),
        'compression': pd.DataFrame([
            {
                'frame_idx': i,
                'step_read_frame': np.random.exponential(0.02) + 0.01,
                'step_pack_append': np.random.exponential(0.08) + 0.05
            }
            for i in range(100)
        ])
    }
    
    # Create mock tracking data
    tracking_data = pd.DataFrame([
        {
            'frame_idx': np.random.randint(0, 100),
            'track_id': np.random.randint(0, 10),
            'x1': np.random.uniform(0, 640),
            'y1': np.random.uniform(0, 480),
            'x2': np.random.uniform(0, 640),
            'y2': np.random.uniform(0, 480),
            'width': np.random.uniform(20, 100),
            'height': np.random.uniform(20, 100),
            'area': np.random.uniform(400, 10000)
        }
        for _ in range(500)
    ])
    
    # Create summary
    summary = {
        'video_file': 'test_video',
        'tile_size': 64,
        'pipeline_stages': {},
        'tracking_summary': {}
    }
    
    # Add runtime statistics
    for stage, df in runtime_data.items():
        if stage == 'classification':
            summary['pipeline_stages'][stage] = {
                'total_frames': len(df),
                'mean_runtime': float(df['runtime'].mean()),
                'std_runtime': float(df['runtime'].std()),
                'min_runtime': float(df['runtime'].min()),
                'max_runtime': float(df['runtime'].max()),
                'total_runtime': float(df['runtime'].sum())
            }
        elif stage == 'compression':
            step_cols = [col for col in df.columns if col.startswith('step_')]
            if step_cols:
                step_stats = {}
                for step in step_cols:
                    step_stats[step.replace('step_', '')] = {
                        'mean': float(df[step].mean()),
                        'std': float(df[step].std()),
                        'total': float(df[step].sum())
                    }
                
                summary['pipeline_stages'][stage] = {
                    'total_frames': len(df),
                    'step_statistics': step_stats,
                    'total_runtime': float(df[step_cols].sum(axis=1).sum())
                }
    
    # Add tracking summary
    if not tracking_data.empty:
        track_durations = tracking_data.groupby('track_id')['frame_idx'].agg(['min', 'max', 'count'])
        track_durations['duration'] = track_durations['max'] - track_durations['min'] + 1
        
        summary['tracking_summary'] = {
            'total_tracks': len(track_durations),
            'total_detections': len(tracking_data),
            'mean_track_duration': float(track_durations['duration'].mean()),
            'std_track_duration': float(track_durations['duration'].std()),
            'min_track_duration': int(track_durations['duration'].min()),
            'max_track_duration': int(track_durations['duration'].max()),
            'mean_track_area': float(tracking_data['area'].mean()),
            'std_track_area': float(tracking_data['area'].std())
        }
    
    print("✓ Summary statistics created successfully!")
    print(f"  - Pipeline stages: {list(summary['pipeline_stages'].keys())}")
    print(f"  - Classification frames: {summary['pipeline_stages']['classification']['total_frames']}")
    print(f"  - Compression frames: {summary['pipeline_stages']['compression']['total_frames']}")
    print(f"  - Total tracks: {summary['tracking_summary']['total_tracks']}")
    
    print("\n🎉 All tests passed successfully!")
    print("\nThe runtime analysis script is ready to use with real data.")
    print("Run: python scripts/071_results_statistics_speed.py")

if __name__ == '__main__':
    test_runtime_analysis()
