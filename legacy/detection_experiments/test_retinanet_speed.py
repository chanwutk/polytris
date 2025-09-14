#!/usr/bin/env python3
"""
RetinaNet Speed Test Experiment

This script tests the inference speed of RetinaNet on different video resolutions:
- 480p (854x480)
- 720p (1280x720) 
- 1080p (1920x1080)

Only measures inference time, excluding video decoding and resizing.
Uses detectron2 implementation based on the b3d/detrk.py pattern.

Note: This script requires detectron2 to be properly installed.
See installation instructions in the README.
"""

import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
import torch
import argparse
from tqdm import tqdm

# Video resolutions to test
RESOLUTIONS = {
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}

def check_detectron2_installation():
    """Check if detectron2 is properly installed"""
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        print("✅ Detectron2 imported successfully")
        return True
    except ImportError as e:
        print("❌ Detectron2 not available. Please install it first.")
        print("\nInstallation instructions:")
        print("1. Clone detectron2 repository:")
        print("   git clone https://github.com/facebookresearch/detectron2.git")
        print("2. Install from source:")
        print("   cd detectron2")
        print("   python -m pip install -e .")
        print("3. Or use conda:")
        print("   conda install -c conda-forge detectron2")
        print("\nFor more details, see: https://detectron2.readthedocs.io/en/latest/tutorials/install.html")
        return False

def load_retinanet_model():
    """Load RetinaNet model from detectron2"""
    if not check_detectron2_installation():
        return None, None
    
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        cfg = get_cfg()
        # Use RetinaNet R50-FPN configuration
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
        
        # Set device
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"
        
        predictor = DefaultPredictor(cfg)
        print(f"✅ Loaded RetinaNet model successfully on {cfg.MODEL.DEVICE}")
        return predictor, cfg
    except Exception as e:
        print(f"❌ Error loading RetinaNet model: {e}")
        return None, None

def resize_frame(frame, target_resolution):
    """Resize frame to target resolution"""
    target_width, target_height = target_resolution
    return cv2.resize(frame, (target_width, target_height))

def test_inference_speed(predictor, frame, num_runs=100):
    """Test inference speed on a single frame"""
    if predictor is None:
        return None
    
    # Warm up
    for _ in range(10):
        _ = predictor(frame)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        outputs = predictor(frame)
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

def process_video(video_path, predictor, resolution_name, target_resolution):
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
        speed_result = test_inference_speed(predictor, resized_frame, num_runs=10)
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

def create_visualization(video_path, predictor, resolution_name, target_resolution, output_path):
    """Create visualization video with bounding boxes"""
    if not check_detectron2_installation():
        print("Cannot create visualization without detectron2")
        return
    
    print(f"Creating visualization for {resolution_name}")
    
    try:
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)
        
        # Get metadata for visualization
        metadata = MetadataCatalog.get("coco_2017_val")
        
        # Process frames
        for frame_idx in tqdm(range(frame_count), desc=f"Visualizing {resolution_name}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to target resolution
            resized_frame = resize_frame(frame, target_resolution)
            
            # Run inference
            outputs = predictor(resized_frame)
            
            # Create visualizer
            v = Visualizer(resized_frame[:, :, ::-1], metadata=metadata, scale=1.0)
            
            # Draw predictions
            instances = outputs["instances"].to("cpu")
            v = v.draw_instance_predictions(instances)
            
            # Convert back to BGR for OpenCV
            annotated_frame = v.get_image()[:, :, ::-1]
            
            # Write frame
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test RetinaNet inference speed on different resolutions')
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
    print("Loading RetinaNet model...")
    predictor, cfg = load_retinanet_model()
    if predictor is None:
        print("❌ Failed to load model. Please install detectron2 first.")
        print("\nExiting. Please see installation instructions above.")
        return
    
    # Test each resolution
    all_results = {}
    
    for resolution_name, target_resolution in RESOLUTIONS.items():
        print(f"\n{'='*50}")
        print(f"Testing {resolution_name} resolution")
        print(f"{'='*50}")
        
        # Test inference speed
        speed_results = process_video(args.video, predictor, resolution_name, target_resolution)
        if speed_results:
            all_results[resolution_name] = speed_results
            
            # Save results
            results_file = output_dir / f"retinanet_{resolution_name}_speed_results.json"
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
            viz_output = output_dir / f"retinanet_{resolution_name}_visualization.mp4"
            create_visualization(args.video, predictor, resolution_name, target_resolution, str(viz_output))
    
    # Save combined results
    combined_results_file = output_dir / "retinanet_all_resolutions_results.json"
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
