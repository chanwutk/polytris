#!/usr/local/bin/python

import argparse
import json
import os
import cv2
import torch
import numpy as np
import time
import shutil
import multiprocessing as mp

from polyis.images import splitHWC, padHWC
from scipy.ndimage import binary_dilation # check this import later

from scripts.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATA_DIR, format_time, progress_bars


TILE_SIZES = [60]  #, 30, 120]
MANUALLY_INCLUDE = {"jnc00.mp4": [18, 36, 54, 72, 90, 17, 35, 53, 133, 134, 152, 161, 179, 197, 215, 22, 23, 24, 25, 26]} # comment this out if not needed
def get_classifier_class(classifier_name: str):
    """
    Get the classifier class based on the classifier name.
    
    Args:
        classifier_name (str): Name of the classifier to use
        
    Returns:
        The classifier class
        
    Raises:
        ValueError: If the classifier is not supported
    """
    if classifier_name == 'SimpleCNN':
        from polyis.models.classifier.simple_cnn import SimpleCNN
        return SimpleCNN
    elif classifier_name == 'YoloN':
        from polyis.models.classifier.yolo import YoloN
        return YoloN
    elif classifier_name == 'YoloS':
        from polyis.models.classifier.yolo import YoloS
        return YoloS
    elif classifier_name == 'YoloM':
        from polyis.models.classifier.yolo import YoloM
        return YoloM
    elif classifier_name == 'YoloL':
        from polyis.models.classifier.yolo import YoloL
        return YoloL
    elif classifier_name == 'YoloX':
        from polyis.models.classifier.yolo import YoloX
        return YoloX
    elif classifier_name == 'ShuffleNet05':
        from polyis.models.classifier.shufflenet import ShuffleNet05
        return ShuffleNet05
    elif classifier_name == 'ShuffleNet20':
        from polyis.models.classifier.shufflenet import ShuffleNet20
        return ShuffleNet20
    elif classifier_name == 'MobileNetL':
        from polyis.models.classifier.mobilenet import MobileNetL
        return MobileNetL
    elif classifier_name == 'MobileNetS':
        from polyis.models.classifier.mobilenet import MobileNetS
        return MobileNetS
    elif classifier_name == 'WideResNet50':
        from polyis.models.classifier.wide_resnet import WideResNet50
        return WideResNet50
    elif classifier_name == 'WideResNet101':
        from polyis.models.classifier.wide_resnet import WideResNet101
        return WideResNet101
    elif classifier_name == 'ResNet152':
        from polyis.models.classifier.resnet import ResNet152
        return ResNet152
    elif classifier_name == 'ResNet101':
        from polyis.models.classifier.resnet import ResNet101
        return ResNet101
    elif classifier_name == 'ResNet18':
        from polyis.models.classifier.resnet import ResNet18
        return ResNet18
    elif classifier_name == 'EfficientNetS':
        from polyis.models.classifier.efficientnet import EfficientNetS
        return EfficientNetS
    elif classifier_name == 'EfficientNetL':
        from polyis.models.classifier.efficientnet import EfficientNetL
        return EfficientNetL
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - classifiers (List[str]): List of classifier models to use (default: multiple classifiers)
    """
    parser = argparse.ArgumentParser(description='Execute trained classifier models to classify video tiles')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--classifiers', required=False, nargs='+',
                        default=CLASSIFIERS_TO_TEST,
                        choices=['SimpleCNN', 'YoloN', 'YoloS', 'YoloM', 'YoloL',
                                 'YoloX', 'ShuffleNet05', 'ShuffleNet20', 'MobileNetL',
                                 'MobileNetS', 'WideResNet50', 'WideResNet101',
                                 'ResNet152', 'ResNet101', 'ResNet18', 'EfficientNetS',
                                 'EfficientNetL'],
                        help='Specific classifiers to analyze (if not specified, all classifiers will be analyzed)')
    return parser.parse_args()


def load_model(video_path: str, tile_size: int, classifier_name: str) -> "torch.nn.Module":
    """
    Load trained classifier model for the specified tile size from a specific video directory.
    
    This function searches for a trained model in the expected directory structure:
    {video_path}/training/results/{classifier_name}_{tile_size}/model.pth
    
    Args:
        video_path (str): Path to the specific video directory
        tile_size (int): Tile size for which to load the model (30, 60, or 120)
        classifier_name (str): Name of the classifier model to use (default: 'SimpleCNN')
        
    Returns:
        The loaded trained model for the specified tile size.
            The model is loaded to CUDA and set to evaluation mode.
            
    Raises:
        FileNotFoundError: If no trained model is found for the specified tile size
        ValueError: If the classifier is not supported
    """
    results_path = os.path.join(video_path, 'training', 'results', f'{classifier_name}_{tile_size}')
    model_path = os.path.join(results_path, 'model.pth')
    
    if os.path.exists(model_path):
        print(f"Loading {classifier_name} model for tile size {tile_size} from {model_path}")
        model = torch.load(model_path, map_location='cuda', weights_only=False)
        model.eval()
        return model
    
    raise FileNotFoundError(f"No trained model found for tile size {tile_size} in {video_path}")


def process_frame_tiles(frame: np.ndarray, model: torch.nn.Module, tile_size: int, relevant_indices: np.ndarray, device: str) -> tuple[np.ndarray, list[dict]]:
    """
    Process a single video frame with the specified tile size and return relevance scores and timing information.

    Args:
        frame (np.ndarray): The video frame to process.
        model (torch.nn.Module): The classification model.
        tile_size (int): The size of the square tiles.
        relevant_indices (np.ndarray): A 1D array of indices for the tiles to be processed.
        device (str): The device ('cuda' or 'cpu') to run the model on.

    Returns:
        tuple[np.ndarray, list[dict]]: A tuple containing:
            - A 2D array of relevance scores.
            - A dictionary of runtime metrics.
    """
    with torch.no_grad():
        start_time = (time.time_ns() / 1e6)
        
        frame_tensor = torch.from_numpy(frame).to(device).float()
        padded_frame = padHWC(frame_tensor, tile_size, tile_size)
        tiles = splitHWC(padded_frame, tile_size, tile_size)
        
        num_tiles = tiles.shape[0] * tiles.shape[1]
        # tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)
        tiles_flat = torch.flatten(tiles)
        
        if relevant_indices is None:
            relevant_indices = np.arange(num_tiles)
        if len(relevant_indices) == 0:
            transform_runtime = (time.time_ns() / 1e6) - start_time
            inference_runtime = 0.0
            relevance_grid = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=np.uint8)
            return relevance_grid, format_time(transform=transform_runtime, inference=inference_runtime)
            
            
        # tiles_to_process = tiles_flat[relevant_indices]
        tiles_to_process = torch.index_select(tiles_flat, 0, relevant_indices)
        tiles_to_process = tiles_to_process / 255.0
        tiles_nchw = tiles_to_process.permute(0, 3, 1, 2)
        transform_runtime = (time.time_ns() / 1e6) - start_time
        
        start_time = (time.time_ns() / 1e6)
        predictions = model(tiles_nchw)
        probabilities = (predictions * 255).to(torch.uint8).cpu().numpy().flatten()
    
        grid_height, grid_width = tiles.shape[:2]
        relevance_grid = np.zeros((grid_height * grid_width), dtype=np.uint8)

        relevance_grid[relevant_indices] = probabilities

        relevance_grid = relevance_grid.reshape(grid_height, grid_width)

        end_time = (time.time_ns() / 1e6)
        inference_runtime = end_time - start_time

    return relevance_grid, format_time(transform=transform_runtime, inference=inference_runtime)

def mark_neighbor_tiles(relevance_grid, threshold):
    """
    Creates a new grid where tiles meeting the threshold, plus their 
    adjacent neighbors (including diagonals), are marked as 1.

    Args:
        relevance_grid (np.ndarray): A 2D grid of relevance scores.
        threshold (int): The relevance score threshold.

    Returns:
        np.ndarray: A new 2D array of 0s and 1s, where 1s indicate
                    relevant tiles or their neighbors. (Mask)
    """
    # Create a boolean mask for tiles above the threshold
    relevant_mask = (relevance_grid >= threshold)

    # Create kernel for all 8 neighboring tiles (TODO: should this be 4?)
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Apply dilation to expand the mask to include neighbors
    dilated_mask = binary_dilation(relevant_mask, structure=kernel)

    # convert to list of relevant indices
    relevant_indices = np.where(dilated_mask.flatten())[0]
    return relevant_indices

def process_video(video_path: str, model, tile_size: int, output_path: str, command_queue: mp.Queue, device: str, classifier: str):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using the trained classifier model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.
    
    Args:
        video_path (str): Path to the input video file to process
        model (ClassifyRelevance): Trained model for the specified tile size
        tile_size (int): Tile size used for processing (32, 64, or 128)
        output_path (str): Path where the output JSONL file will be saved
        command_queue (mp.Queue): Queue for progress updates
        device (str): Device to use for processing
        classifier (str): Classifier name to use
            
    Note:
        - Video is processed frame by frame to minimize memory usage
        - Progress is displayed using a progress bar
        - Results are flushed to disk after each frame for safety
        - Video metadata (FPS, dimensions, frame count) is extracted and logged
        - Each frame entry includes frame index, timestamp, frame dimensions, tile classifications, and runtime
        - The function handles various video formats (.mp4, .avi, .mov, .mkv)
        
    Output Format:
        Each line in the JSONL file contains a JSON object with:
        - frame_idx (int): Zero-based frame index
        - timestamp (float): Frame timestamp in seconds
        - frame_size (list[int]): [height, width] of the frame
        - tile_size (int): Tile size used for processing (30, 60, or 120)
        - tile_classifications (list[list[float]]): Relevance scores grid for the specified tile size
        - runtime (float): Runtime in seconds for the ClassifyRelevance model inference
    """
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    with open(output_path, 'w') as f:
        frame_idx = 0
        command_queue.put((device, {'description': f"{video_path.split('/')[-1]} {tile_size:>3} {classifier}",
                                    'completed': 0, 'total': frame_count}))
        prev_relevance_grid = None # no previous frame yet
        threshold = 0.5 # modify if necessary
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx == 0:
                # process entire first frame
                current_relevance_grid, runtime = process_frame_tiles(frame, model, tile_size, None, device)

            else:
                # we know prev_relevance_grid is not none 
                relevant_indices = mark_neighbor_tiles(prev_relevance_grid, threshold)
                manual_include = np.array([18, 36, 54, 72, 90, 17, 35, 53, 133, 134, 152, 161, 179, 197, 215, 22, 23, 24, 25, 26])
                relevant_indices = np.union1d(relevant_indices, manual_include) # manual include
                current_relevance_grid, runtime = process_frame_tiles(frame, model, tile_size, relevant_indices, device)

            
            # Update the relevance grid for the next loop iteration
            prev_relevance_grid = current_relevance_grid
            


            # Create result entry for this frame
            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "frame_size": [height, width],
                "tile_size": tile_size,
                "runtime": runtime,
                "classification_size": current_relevance_grid.shape,
                "classification_hex": current_relevance_grid.flatten().tobytes().hex(),
            }
            
            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            if frame_idx % 100 == 0:
                f.flush()
            
            frame_idx += 1
            command_queue.put((device, {'completed': frame_idx}))

    cap.release()
    print(f"Completed processing {frame_idx} frames. Results saved to {output_path}")


def process_video_task(video_file_path: str, cache_video_dir: str, classifier: str, 
                      tile_size: int, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video with a specific classifier and tile size.
    This function is designed to be called in parallel.
    
    Args:
        video_file_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
    """
    try:
        device = f'cuda:{gpu_id}'
        
        # Load the trained model for this specific video, classifier, and tile size
        model = load_model(cache_video_dir, tile_size, classifier)
        model = model.to(device)
        
        # Create output directory structure
        output_dir = os.path.join(cache_video_dir, 'relevancy')
        
        # Create score directory for this classifier and tile size
        classifier_dir = os.path.join(output_dir, f'{classifier}_{tile_size}')
        if os.path.exists(classifier_dir):
            shutil.rmtree(classifier_dir)
        os.makedirs(classifier_dir)
        
        # Create score directory for this tile size
        score_dir = os.path.join(classifier_dir, 'score')
        if os.path.exists(score_dir):
            shutil.rmtree(score_dir)
        os.makedirs(score_dir)
        output_path = os.path.join(score_dir, 'score.jsonl')
        
        # Process the video
        process_video(video_file_path, model, tile_size, output_path, command_queue, device, classifier)
        
    except Exception as e:
        print(f"Error processing {video_file_path} with {classifier} (tile_size={tile_size}): {e}")
        raise e


def _process_video_task(video_file_path: str, cache_video_dir: str, classifier: str, 
                       tile_size: int, gpu_id: int, gpu_id_queue: mp.Queue, 
                       command_queue: mp.Queue):
    """
    Wrapper function for process_video_task that handles GPU queue management.
    """
    try:
        process_video_task(video_file_path, cache_video_dir, classifier, 
                          tile_size, gpu_id, command_queue)
    finally:
        gpu_id_queue.put(gpu_id)


def main(args):
    """
    Main function that orchestrates the video tile classification process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Creates a list of all video/classifier/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves classification results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - classifiers (List[str]): List of classifier models to use (default: multiple classifiers)
            
    Note:
        - The script expects a specific directory structure:
          {DATA_DIR}/{dataset}/ - contains video files
          {DATA_CACHE}/{dataset}/{video_file_name}/training/results/{classifier_name}_{tile_size}/model.pth - contains trained models
          where DATA_DIR and DATA_CACHE are both /polyis-data/video-datasets-low
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - A separate model is loaded for each video directory, classifier, and tile size combination
        - When tile_size is 'all', all three tile sizes (30, 60, 120) are processed
        - Output files are saved in {DATA_CACHE}/{dataset}/{video_file_name}/relevancy/score/{classifier_name}_{tile_size}/score.jsonl
        - If no trained model is found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)
    
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")
    
    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")
    
    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    # Create tasks list with all video/classifier/tile_size combinations
    tasks: list[tuple[str, str, str, int]] = []
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)
        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)

        output_dir = os.path.join(cache_video_dir, 'relevancy')
        # TODO: Remove the dir if exists when the user specify the flag --clear
        os.makedirs(output_dir, exist_ok=True)
        
        for classifier in args.classifiers:
            for tile_size in tile_sizes_to_process:
                tasks.append((video_file_path, cache_video_dir, classifier, tile_size))
    
    print(f"Created {len(tasks)} tasks to process")
    
    # Set up multiprocessing
    num_gpus = 1
    assert num_gpus > 0, "No GPUs available"
    
    gpu_id_queue = mp.Queue(maxsize=num_gpus)
    for gpu_id in range(num_gpus):
        gpu_id_queue.put(gpu_id)
    
    command_queue = mp.Queue()
    progress_process = mp.Process(target=progress_bars,
                                  args=(command_queue, num_gpus, len(tasks)),
                                  daemon=True)
    progress_process.start()
    
    processes: list[mp.Process] = []
    for task in tasks:
        gpu_id = gpu_id_queue.get()
        command_queue.put(('overall', {'advance': 1}))
        process = mp.Process(target=_process_video_task,
                             args=(*task, gpu_id, gpu_id_queue, command_queue))
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
        process.terminate()
    
    command_queue.put(None)
    progress_process.join()
    progress_process.terminate()
    
    print("All tasks completed!")
            

if __name__ == '__main__':
    main(parse_args())
