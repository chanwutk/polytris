#!/usr/local/bin/python

import argparse
import json
import os
import cv2
import shutil
import time
import yaml
from typing import Dict, List, Tuple

from ultralytics import YOLO

from polyis.utilities import DATA_DIR, CACHE_DIR, format_time, load_tracking_results


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - model_name (str): YOLO model variant to use (default: 'yolov5x')
            - epochs (int): Number of training epochs (default: 100)
            - imgsz (int): Training image size (default: 640)
            - batch_size (int): Batch size for training (default: 16)
    """
    parser = argparse.ArgumentParser(description='Train YOLOv5 object detection models on video datasets')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--model_name', type=str, default='yolov5x',
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                        help='YOLO model variant to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Training image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    return parser.parse_args()


def extract_frames_with_detections(video_path: str, tracking_data: Dict[int, List[List[float]]], 
                                   output_dir: str) -> List[str]:
    """
    Extract frames from video that contain object detections.
    
    Args:
        video_path (str): Path to the video file
        tracking_data (Dict[int, List[List[float]]]): Tracking data mapping frame indices to detections
        output_dir (str): Directory to save extracted frames
        
    Returns:
        List[str]: List of paths to extracted frame images
    """
    print(f"Extracting frames with detections from: {video_path}")
    
    # Create images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []
    
    print(f"Processing {frame_count} frames, extracting frames with detections...")
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if this frame has detections
        if frame_idx in tracking_data and len(tracking_data[frame_idx]) > 0:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(images_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames with detections")
    return extracted_frames


def create_yolo_annotations(tracking_data: Dict[int, List[List[float]]], 
                            extracted_frames: List[str], output_dir: str, 
                            frame_width: int, frame_height: int) -> List[str]:
    """
    Create YOLO format annotation files for the extracted frames.
    
    Args:
        tracking_data: Tracking data mapping frame indices to detections
        extracted_frames: List of extracted frame paths
        output_dir: Directory to save annotations
        frame_width: Width of video frames
        frame_height: Height of video frames
        
    Returns:
        List[str]: List of paths to annotation files
    """
    print("Creating YOLO format annotations...")
    
    # Create labels directory
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    annotation_files = []
    
    for frame_path in extracted_frames:
        # Extract frame index from filename
        frame_filename = os.path.basename(frame_path)
        frame_idx = int(frame_filename.split('_')[1].split('.')[0])
        
        # Get detections for this frame
        detections = tracking_data.get(frame_idx, [])
        
        if not detections:
            continue
            
        # Create annotation file
        annotation_filename = frame_filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(labels_dir, annotation_filename)
        
        with open(annotation_path, 'w') as f:
            for detection in detections:
                # detection format: [track_id, x1, y1, x2, y2]
                if len(detection) >= 5:
                    _, x1, y1, x2, y2 = detection[:5]
                    
                    # Convert to YOLO format (class, x_center, y_center, width, height) normalized [0,1]
                    x_center = (x1 + x2) / 2.0 / frame_width
                    y_center = (y1 + y2) / 2.0 / frame_height
                    width = (x2 - x1) / frame_width
                    height = (y2 - y1) / frame_height
                    
                    # Class 0 for all objects (single class detection)
                    class_id = 0
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        annotation_files.append(annotation_path)
    
    print(f"Created {len(annotation_files)} annotation files")
    return annotation_files


def create_dataset_yaml(dataset_dir: str, video_file: str) -> str:
    """
    Create YOLO dataset configuration YAML file.
    
    Args:
        dataset_dir: Directory containing the dataset
        video_file: Name of the video file being processed
        
    Returns:
        str: Path to the created YAML file
    """
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    
    # Split data into train/val (80/20 split)
    images_dir = os.path.join(dataset_dir, 'images')
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create train.txt and val.txt
    train_txt_path = os.path.join(dataset_dir, 'train.txt')
    val_txt_path = os.path.join(dataset_dir, 'val.txt')
    
    with open(train_txt_path, 'w') as f:
        for img in train_images:
            f.write(f"{os.path.join(images_dir, img)}\n")
    
    with open(val_txt_path, 'w') as f:
        for img in val_images:
            f.write(f"{os.path.join(images_dir, img)}\n")
    
    # Create YAML configuration
    yaml_config = {
        'path': dataset_dir,
        'train': train_txt_path,
        'val': val_txt_path,
        'nc': 1,  # number of classes
        'names': ['object']  # class names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"Created dataset YAML: {yaml_path}")
    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
    
    return yaml_path


def train_yolo_model(dataset_yaml: str, model_name: str, epochs: int, imgsz: int, 
                    batch_size: int, output_dir: str) -> Tuple[str, List[Dict]]:
    """
    Train YOLO model on the prepared dataset.
    
    Args:
        dataset_yaml: Path to dataset YAML configuration
        model_name: YOLO model variant to use
        epochs: Number of training epochs
        imgsz: Training image size
        batch_size: Batch size for training
        output_dir: Directory to save training results
        
    Returns:
        Tuple[str, List[Dict]]: Path to trained model and runtime information
    """
    print(f"Starting YOLO {model_name} training...")
    start_time = time.time()
    
    # Load pretrained model
    model = YOLO(f"{model_name}.pt")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=output_dir,
        name='train',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        patience=20,     # Early stopping patience
        device='cuda' if os.path.exists('/proc/driver/nvidia/version') else 'cpu'
    )
    
    training_time = time.time() - start_time
    
    # Find the trained model path
    train_dir = os.path.join(output_dir, 'train')
    weights_dir = os.path.join(train_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best.pt')
    
    if not os.path.exists(best_model_path):
        # Fallback to last.pt if best.pt doesn't exist
        best_model_path = os.path.join(weights_dir, 'last.pt')
    
    runtime_info = format_time(training=training_time)
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model saved at: {best_model_path}")
    
    return best_model_path, runtime_info


def process_video_training(video_file: str, dataset: str, args) -> None:
    """
    Process training for a single video file.
    
    Args:
        video_file: Name of the video file
        dataset: Dataset name
        args: Command line arguments
    """
    print(f"\n=== Processing video: {video_file} ===")
    
    # Set up paths
    video_path = os.path.join(DATA_DIR, dataset, video_file)
    cache_video_dir = os.path.join(CACHE_DIR, dataset, video_file)
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return
    
    # Create detectors directory
    detectors_dir = os.path.join(cache_video_dir, 'detectors')
    os.makedirs(detectors_dir, exist_ok=True)
    
    # Check if model already exists
    model_output_path = os.path.join(detectors_dir, f"{args.model_name}.pt")
    if os.path.exists(model_output_path):
        print(f"Model already exists: {model_output_path}")
        return
    
    try:
        # Load tracking results
        tracking_data = load_tracking_results(CACHE_DIR, dataset, video_file)
        
        if not tracking_data:
            print(f"Warning: No tracking data found for {video_file}")
            return
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create temporary dataset directory
        temp_dataset_dir = os.path.join(cache_video_dir, 'temp_yolo_dataset')
        if os.path.exists(temp_dataset_dir):
            shutil.rmtree(temp_dataset_dir)
        os.makedirs(temp_dataset_dir)
        
        try:
            # Extract frames with detections
            extracted_frames = extract_frames_with_detections(
                video_path, tracking_data, temp_dataset_dir
            )
            
            if not extracted_frames:
                print(f"Warning: No frames with detections found in {video_file}")
                return
            
            # Create YOLO annotations
            create_yolo_annotations(
                tracking_data, extracted_frames, temp_dataset_dir,
                frame_width, frame_height
            )
            
            # Create dataset YAML
            dataset_yaml = create_dataset_yaml(temp_dataset_dir, video_file)
            
            # Train YOLO model
            trained_model_path, runtime_info = train_yolo_model(
                dataset_yaml, args.model_name, args.epochs, args.imgsz,
                args.batch_size, temp_dataset_dir
            )
            
            # Copy trained model to final location
            shutil.copy2(trained_model_path, model_output_path)
            
            # Save runtime information
            runtime_path = os.path.join(detectors_dir, f"{args.model_name}_training_runtime.jsonl")
            with open(runtime_path, 'w') as f:
                runtime_entry = {
                    'video_file': video_file,
                    'model_name': args.model_name,
                    'epochs': args.epochs,
                    'imgsz': args.imgsz,
                    'batch_size': args.batch_size,
                    'num_frames_extracted': len(extracted_frames),
                    'runtime': runtime_info
                }
                f.write(json.dumps(runtime_entry) + '\n')
            
            print(f"Successfully trained {args.model_name} for {video_file}")
            print(f"Model saved at: {model_output_path}")
            
        finally:
            # Clean up temporary dataset directory
            if os.path.exists(temp_dataset_dir):
                shutil.rmtree(temp_dataset_dir)
                print(f"Cleaned up temporary dataset directory")
    
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        raise


def main(args):
    """
    Main function to train YOLO detectors on video datasets.
    
    This function:
    1. Validates the dataset directory exists
    2. Processes each video file in the dataset
    3. Extracts frames with detections from tracking.jsonl
    4. Creates YOLO format dataset (images + annotations)
    5. Trains YOLOv5 model on the dataset
    6. Saves trained weights to {CACHE_DIR}/{dataset}/{video_file}/detectors/{model_name}.pt
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - model_name (str): YOLO model variant to use
            - epochs (int): Number of training epochs
            - imgsz (int): Training image size
            - batch_size (int): Batch size for training
            
    Note:
        - The script expects the following directory structure:
          {DATA_DIR}/{dataset}/ - contains video files (.mp4, .avi, .mov, .mkv)
          {CACHE_DIR}/{dataset}/{video_file}/000_groundtruth/tracking.jsonl - contains tracking data
        - For each video, creates a YOLO dataset from frames containing detections
        - Uses single-class detection (class 0 for all objects)
        - Splits dataset 80/20 for train/validation
        - Saves trained model weights and runtime information
    """
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")
    
    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Training {args.model_name} with {args.epochs} epochs, image size {args.imgsz}")
    
    # Process each video file
    for video_file in sorted(video_files):
        try:
            process_video_training(video_file, args.dataset, args)
        except Exception as e:
            print(f"Failed to process {video_file}: {e}")
            continue
    
    print("All videos processed!")


if __name__ == '__main__':
    main(parse_args())
