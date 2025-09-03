#!/usr/bin/env python3
"""
YOLOv3 Speed Test Experiment

This script tests the inference speed of YOLOv3 on different video resolutions:
- 480p (854x480)
- 720p (1280x720) 
- 1080p (1920x1080)

Only measures inference time, excluding video decoding and resizing.
"""

import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
import torch
from ultralytics import YOLO
import argparse
from tqdm import tqdm

# Video resolutions to test
RESOLUTIONS = {
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}

def load_yolov3_model():
    """Load YOLOv3 model from ultralytics"""
    try:
        # Load YOLOv3 model
        model = YOLO('yolov3.pt')
        print("Loaded YOLOv3 model successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLOv3 model: {e}")
        return None

def resize_frame(frame, target_resolution):
    """Resize frame to target resolution"""
    target_width, target_height = target_resolution
    return cv2.resize(frame, (target_width, target_height))

def test_inference_speed(model, frame, num_runs=100):
    """Test inference speed on a single frame"""
    if model is None:
        return None
    
    # Warm up
    for _ in range(10):
        _ = model(frame, verbose=False)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        results = model(frame, verbose=False)
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(inference_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }

def process_video(video_path, model, resolution_name, target_resolution):
    """Process video and measure inference speed"""
    print(f"Processing {resolution_name} resolution: {target_resolution}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} FPS, {frame_count} frames")
    
    # Sample frames for testing (every 10th frame to speed up testing)
    sample_interval = max(1, frame_count // 100)  # Test ~100 frames
    frame_indices = list(range(0, frame_count, sample_interval))
    
    results = {
        'resolution': resolution_name,
        'target_size': target_resolution,
        'fps': fps,
        'frame_count': frame_count,
        'sample_interval': sample_interval,
        'tested_frames': len(frame_indices),
        'inference_times': []
    }
    
    # Process sample frames
    for frame_idx in tqdm(frame_indices, desc=f"Testing {resolution_name}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize frame to target resolution
        resized_frame = resize_frame(frame, target_resolution)
        
        # Test inference speed
        speed_result = test_inference_speed(model, resized_frame, num_runs=10)
        if speed_result:
            speed_result['frame_idx'] = frame_idx
            results['inference_times'].append(speed_result)
    
    cap.release()
    
    # Calculate overall statistics
    if results['inference_times']:
        all_times = [t['mean_time'] for t in results['inference_times']]
        results['overall_stats'] = {
            'mean_inference_time': np.mean(all_times),
            'std_inference_time': np.std(all_times),
            'min_inference_time': np.min(all_times),
            'max_inference_time': np.max(all_times),
            'total_test_time': sum(all_times)
        }
    
    return results

def create_visualization(video_path, model, resolution_name, target_resolution, output_path):
    """Create visualization video with bounding boxes"""
    print(f"Creating visualization for {resolution_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)
    
    # Process frames
    for frame_idx in tqdm(range(frame_count), desc=f"Visualizing {resolution_name}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to target resolution
        resized_frame = resize_frame(frame, target_resolution)
        
        # Run inference
        results = model(resized_frame, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = resized_frame.copy()
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, 
                                (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(annotated_frame, label, 
                              (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLOv3 inference speed on different resolutions')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output-dir', default='./detection_experiments/results', 
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization videos with bounding boxes')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading YOLOv3 model...")
    model = load_yolov3_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Test each resolution
    all_results = {}
    
    for resolution_name, target_resolution in RESOLUTIONS.items():
        print(f"\n{'='*50}")
        print(f"Testing {resolution_name} resolution")
        print(f"{'='*50}")
        
        # Test inference speed
        speed_results = process_video(args.video, model, resolution_name, target_resolution)
        if speed_results:
            all_results[resolution_name] = speed_results
            
            # Save results
            results_file = output_dir / f"yolov3_{resolution_name}_speed_results.json"
            with open(results_file, 'w') as f:
                json.dump(speed_results, f, indent=2)
            print(f"Speed results saved to: {results_file}")
            
            # Print summary
            if 'overall_stats' in speed_results:
                stats = speed_results['overall_stats']
                print(f"Average inference time: {stats['mean_inference_time']:.2f} ms")
                print(f"Standard deviation: {stats['std_inference_time']:.2f} ms")
                print(f"Min time: {stats['min_inference_time']:.2f} ms")
                print(f"Max time: {stats['max_inference_time']:.2f} ms")
        
        # Create visualization if requested
        if args.visualize:
            viz_output = output_dir / f"yolov3_{resolution_name}_visualization.mp4"
            create_visualization(args.video, model, resolution_name, target_resolution, str(viz_output))
    
    # Save combined results
    combined_results_file = output_dir / "yolov3_all_resolutions_results.json"
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_results_file}")
    
    # Print final comparison
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    for resolution_name, results in all_results.items():
        if 'overall_stats' in results:
            stats = results['overall_stats']
            print(f"{resolution_name}: {stats['mean_inference_time']:.2f} ± {stats['std_inference_time']:.2f} ms")

if __name__ == "__main__":
    main()
