#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable, List, Tuple
from dataclasses import dataclass, field
import cv2
import torch
import numpy as np
import time
import shutil
import multiprocessing as mp
from functools import partial

from polyis.images import splitHWC, padHWC
from scipy.ndimage import binary_dilation 

from polyis.utilities import CACHE_DIR, CLASSIFIERS_CHOICES, CLASSIFIERS_TO_TEST, DATA_DIR, format_time, ProgressBar


TILE_SIZES = [60]  #, 30, 120]

# hard coded entries below
MANUALLY_INCLUDE = {"jnc00.mp4": [18, 36, 54, 72, 90, 17, 35, 53, 133, 134, 152, 161, 179, 197, 215, 22, 23, 24, 25, 26],
                    "jnc02.mp4": [90, 108, 126, 6, 7, 8, 9, 10, 89, 107, 125, 143, 161, 204, 205, 206, 207, 208, 209],
                    "jnc06.mp4": [18, 36, 54, 72, 90, 108, 126, 7, 8, 9, 10, 11, 12, 53, 71, 89, 107, 125, 143, 161, 203, 204, 205, 206, 207, 208], # good
                    "jnc07.mp4": [109, 110, 111, 112, 72, 90, 108, 6, 7, 8, 9, 10, 11, 12, 107, 125, 210, 204, 205, 206, 207, 208, 209]
                    }


@dataclass
class TileBatch:
    """
    Accumulator for batching tiles from multiple frames for efficient GPU inference.
    Uses pre-allocated buffer to avoid repeated concatenations.
    Automatically flushes when buffer would overflow.
    """
    max_batch_size: int
    tile_size: int
    device: str
    model: torch.nn.Module
    tiles_buffer: torch.Tensor = field(init=False)
    current_size: int = field(default=0, init=False)
    frame_metadata: List[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize pre-allocated buffer on GPU."""
        self.tiles_buffer = torch.zeros(
            (self.max_batch_size, self.tile_size, self.tile_size, 3),
            dtype=torch.float32,
            device=self.device
        )
        self.current_size = 0
        self.frame_metadata = []

    def add_frame_tiles(self, frame_idx: int, tiles: torch.Tensor,
                       tile_indices: np.ndarray, grid_shape: Tuple[int, int]) -> List[Tuple[int, np.ndarray, float]]:
        """
        Add tiles from one frame to the batch.
        Automatically flushes if adding would overflow, then adds to fresh batch.

        Args:
            frame_idx: Index of the frame these tiles belong to
            tiles: Tensor of tiles (n_tiles, tile_size, tile_size, 3)
            tile_indices: Indices of these tiles in the frame's grid
            grid_shape: (grid_h, grid_w) of the full frame grid

        Returns:
            Results from flush if it occurred, empty list otherwise
        """
        n_tiles = tiles.shape[0]
        results = []

        # If adding would overflow, flush first
        if self.current_size + n_tiles > self.max_batch_size:
            results = self.flush()
            self.clear()

        # Now add tiles to buffer (guaranteed to fit)
        self.tiles_buffer[self.current_size:self.current_size + n_tiles] = tiles

        # Store metadata for result scattering
        self.frame_metadata.append({
            'frame_idx': frame_idx,
            'start_idx': self.current_size,
            'end_idx': self.current_size + n_tiles,
            'tile_indices': tile_indices,
            'grid_shape': grid_shape
        })

        self.current_size += n_tiles

        return results

    def flush(self) -> List[Tuple[int, np.ndarray, float, int]]:
        """
        Run inference on accumulated tiles and return results per frame.

        Returns:
            List of (frame_idx, relevance_grid, inference_time, num_processed_tiles) tuples
        """
        if self.current_size == 0:
            return []

        with torch.no_grad():
            # Extract valid tiles from buffer
            tiles_to_infer = self.tiles_buffer[:self.current_size]

            # Normalize to [0, 1]
            tiles_to_infer = tiles_to_infer / 255.0

            # Permute to NCHW format for model
            tiles_nchw = tiles_to_infer.permute(0, 3, 1, 2)

            # Run inference
            start_time = time.time_ns() / 1e6
            predictions = self.model(tiles_nchw)
            inference_time = (time.time_ns() / 1e6) - start_time

            # Convert to uint8 [0-255]
            probabilities = (predictions * 255).to(torch.uint8).cpu().numpy().flatten()

        # Scatter results back to individual frames
        results = []
        for metadata in self.frame_metadata:
            frame_idx = metadata['frame_idx']
            start_idx = metadata['start_idx']
            end_idx = metadata['end_idx']
            tile_indices = metadata['tile_indices']
            grid_h, grid_w = metadata['grid_shape']

            # Extract predictions for this frame
            frame_predictions = probabilities[start_idx:end_idx]

            # Create relevance grid
            relevance_grid = np.zeros((grid_h * grid_w), dtype=np.uint8)
            relevance_grid[tile_indices] = frame_predictions
            relevance_grid = relevance_grid.reshape(grid_h, grid_w)

            # Number of tiles actually processed for this frame
            num_processed_tiles = len(tile_indices)

            results.append((frame_idx, relevance_grid, inference_time / len(self.frame_metadata), num_processed_tiles))

        return results

    def clear(self):
        """Reset batch for next accumulation."""
        self.current_size = 0
        self.frame_metadata = []


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
            - num_readers (int): Number of parallel readers to use for video processing (default: 2)
            - batch_size (int): Maximum number of tiles to batch together for inference (default: 2048)
    """
    parser = argparse.ArgumentParser(description='Execute trained classifier models to classify video tiles')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--classifiers', required=False, nargs='+',
                        default=CLASSIFIERS_TO_TEST,
                        choices=CLASSIFIERS_CHOICES,
                        help='Specific classifiers to analyze (if not specified, all classifiers will be analyzed)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear the output directory before processing')
    parser.add_argument('--filter', type=str, default='none',
                        help='Frame-skipping filter to use (e.g., "none", "neighbor"). This also affects output directory naming.')
    parser.add_argument('--num-readers', type=int, default=2,
                        help='Number of parallel readers to use for video processing (default: 2)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Maximum number of tiles to batch together for inference (default: 2048)')
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

    raise FileNotFoundError(f"No trained model found for {classifier_name} tile size {tile_size} in {video_path}")


def position_video_reader(cap: cv2.VideoCapture, start_frame: int, video_path: str) -> cv2.VideoCapture:
    """
    Position video reader at start_frame using hybrid approach for maximum reliability.

    Uses 2-tier strategy:
    1. Try fast seek with cv2.CAP_PROP_POS_FRAMES and verify
    2. If not exact, reopen video and read sequentially from beginning

    Args:
        cap: Open VideoCapture object
        start_frame: Target frame index to position at
        video_path: Path to video file (for reopening if needed)

    Returns:
        VideoCapture positioned at start_frame (may be same or new instance)
    """
    if start_frame == 0:
        # Already at start, no positioning needed
        return cap

    # Tier 1: Try fast seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if actual_pos == start_frame:
        # Perfect! Fast seek worked
        return cap

    # Tier 2: Fast seek failed - use sequential read from beginning
    print(f"Warning: Fast seeking unreliable for {os.path.basename(video_path)}, "
          f"using sequential read to frame {start_frame}")
    cap.release()
    cap = cv2.VideoCapture(video_path)

    for _ in range(start_frame):
        ret, _ = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read to frame {start_frame} in {video_path}")

    return cap


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
        tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)
        
        if relevant_indices is None:
            relevant_indices = np.arange(num_tiles)
        if len(relevant_indices) == 0:
            transform_runtime = (time.time_ns() / 1e6) - start_time
            inference_runtime = 0.0
            relevance_grid = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=np.uint8)
            return relevance_grid, format_time(transform=transform_runtime, inference=inference_runtime)
            
            
        # tiles_to_process = tiles_flat[relevant_indices]
        relevant_indices_tensor = torch.from_numpy(relevant_indices).to(device)
        tiles_to_process = torch.index_select(tiles_flat, 0, relevant_indices_tensor)
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

def process_video_splice(video_path: str, start_frame: int, end_frame: int,
                        cache_video_dir: str, classifier: str, tile_size: int,
                        filter_type: str, batch_size: int, gpu_id: int,
                        command_queue: mp.Queue, splice_idx: int) -> List[dict]:
    """
    Process a splice of video frames with cross-frame tile batching.

    Args:
        video_path: Path to the video file
        start_frame: First frame index to process (inclusive)
        end_frame: Last frame index to process (exclusive)
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        filter_type: The type of filter to apply ('none', 'neighbor')
        batch_size: Maximum number of tiles to batch together
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
        splice_idx: Index of this splice (for progress tracking)

    Returns:
        List of frame result dictionaries in frame order
    """
    device = f'cuda:{gpu_id}'

    # Load the trained model
    model = load_model(cache_video_dir, tile_size, classifier)
    model = model.to(device)

    # Open video and position at start_frame
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    cap = position_video_reader(cap, start_frame, video_path)

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_in_splice = end_frame - start_frame

    # Create TileBatch for cross-frame batching
    tile_batch = TileBatch(
        max_batch_size=batch_size,
        tile_size=tile_size,
        device=device,
        model=model
    )

    # Progress tracking
    video_name = os.path.basename(video_path)
    description = f"{video_name} {tile_size:>3} {classifier} [splice {splice_idx}]"
    command_queue.put((device, {'description': description,
                                'completed': 0, 'total': num_frames_in_splice}))

    # Storage for completed frame results
    completed_results = {}
    prev_relevance_grid = None
    threshold = 0.5

    # Storage for per-frame transform times (for final flush)
    frame_transform_times = {}

    # Process frames in this splice
    for frame_offset in range(num_frames_in_splice):
        frame_idx = start_frame + frame_offset
        overall_start_time = time.time_ns() / 1e6

        ret, frame = cap.read()
        if not ret:
            break

        # Determine relevant indices
        relevant_indices = None
        pruned_tiles_prop = 0.0

        if filter_type == 'neighbor':
            if frame_offset > 0 and prev_relevance_grid is not None:
                # For subsequent frames in splice, use neighbor filter
                relevant_indices = mark_neighbor_tiles(prev_relevance_grid, threshold)
                video_name_only = os.path.basename(video_path)
                manual_include = np.array(MANUALLY_INCLUDE.get(video_name_only, []))
                if len(manual_include) > 0:
                    relevant_indices = np.union1d(relevant_indices, manual_include)
            # First frame of splice gets full classification (relevant_indices=None)

        # Prepare tiles
        transform_start_time = time.time_ns() / 1e6
        frame_tensor = torch.from_numpy(frame).to(device).float()
        padded_frame = padHWC(frame_tensor, tile_size, tile_size)
        tiles = splitHWC(padded_frame, tile_size, tile_size)

        grid_h, grid_w = tiles.shape[:2]
        num_tiles = grid_h * grid_w
        tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)

        # Select relevant tiles
        if relevant_indices is None:
            relevant_indices = np.arange(num_tiles)

        if len(relevant_indices) == 0:
            # No tiles to process - create empty relevance grid immediately
            current_relevance_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            transform_runtime = (time.time_ns() / 1e6) - transform_start_time
            inference_runtime = 0.0
            pruned_tiles_prop = 1.0
            overall_end_time = time.time_ns() / 1e6

            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "frame_size": [height, width],
                "tile_size": tile_size,
                "runtime": format_time(transform=transform_runtime, inference=inference_runtime,
                                      overall=overall_end_time - overall_start_time),
                "classification_size": current_relevance_grid.shape,
                "classification_hex": current_relevance_grid.flatten().tobytes().hex(),
                "pruned_tiles_prop": pruned_tiles_prop
            }
            completed_results[frame_idx] = frame_entry
            prev_relevance_grid = current_relevance_grid
        else:
            relevant_indices_tensor = torch.from_numpy(relevant_indices).to(device)
            tiles_to_process = torch.index_select(tiles_flat, 0, relevant_indices_tensor)
            transform_runtime = (time.time_ns() / 1e6) - transform_start_time

            # Store transform time for this frame
            frame_transform_times[frame_idx] = transform_runtime

            # Add tiles to batch - may trigger flush
            batch_results = tile_batch.add_frame_tiles(
                frame_idx=frame_idx,
                tiles=tiles_to_process,
                tile_indices=relevant_indices,
                grid_shape=(grid_h, grid_w)
            )

            # Process any flushed results
            for result_frame_idx, relevance_grid, inf_time, num_processed_tiles in batch_results:
                # pruned_tiles_prop should reflect tiles NOT processed, not tiles with low scores
                result_pruned_prop = (relevance_grid.size - num_processed_tiles) / relevance_grid.size if relevance_grid.size > 0 else 0.0

                result_transform_time = frame_transform_times.get(result_frame_idx, 0.0)
                overall_end_time = time.time_ns() / 1e6

                frame_entry = {
                    "frame_idx": result_frame_idx,
                    "timestamp": result_frame_idx / fps if fps > 0 else 0,
                    "frame_size": [height, width],
                    "tile_size": tile_size,
                    "runtime": format_time(transform=result_transform_time, inference=inf_time,
                                          overall=overall_end_time - overall_start_time),
                    "classification_size": relevance_grid.shape,
                    "classification_hex": relevance_grid.flatten().tobytes().hex(),
                    "pruned_tiles_prop": result_pruned_prop
                }
                completed_results[result_frame_idx] = frame_entry

                # Update prev_relevance_grid with most recent result
                if result_frame_idx >= start_frame + frame_offset - 1:
                    prev_relevance_grid = relevance_grid

        # Update progress
        command_queue.put((device, {'completed': frame_offset + 1}))

    # Flush any remaining tiles in batch
    final_results = tile_batch.flush()
    for result_frame_idx, relevance_grid, inf_time, num_processed_tiles in final_results:
        # pruned_tiles_prop should reflect tiles NOT processed, not tiles with low scores
        result_pruned_prop = (relevance_grid.size - num_processed_tiles) / relevance_grid.size if relevance_grid.size > 0 else 0.0

        result_transform_time = frame_transform_times.get(result_frame_idx, 0.0)
        overall_end_time = time.time_ns() / 1e6

        frame_entry = {
            "frame_idx": result_frame_idx,
            "timestamp": result_frame_idx / fps if fps > 0 else 0,
            "frame_size": [height, width],
            "tile_size": tile_size,
            "runtime": format_time(transform=result_transform_time, inference=inf_time,
                                  overall=overall_end_time - overall_start_time),
            "classification_size": relevance_grid.shape,
            "classification_hex": relevance_grid.flatten().tobytes().hex(),
            "pruned_tiles_prop": result_pruned_prop
        }
        completed_results[result_frame_idx] = frame_entry

    cap.release()

    # Return results in frame order
    sorted_results = [completed_results[frame_idx] for frame_idx in sorted(completed_results.keys())]
    return sorted_results


def mark_neighbor_tiles(relevance_grid: np.ndarray, threshold: float) -> np.ndarray:
    """
    relevance_grid: either float in [0,1] or uint8 in [0,255]
    threshold: if <=1, interpreted as [0,1] probability; if >1, treated as raw level
    """
    # normalize threshold to the grid's dtype/range
    if relevance_grid.dtype == np.uint8:
        thr = int(round(threshold * 255)) if threshold <= 1.0 else int(round(threshold))
    else:  # float grid assumed in [0,1]
        thr = float(threshold if threshold <= 1.0 else threshold / 255.0)

    mask = relevance_grid >= thr
    if not mask.any():
        return np.array([], dtype=int)

    dilated = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    return np.flatnonzero(dilated)

def pixel_difference(prev_frame: np.ndarray, current_frame: np.ndarray, tile_size: int, diff_threshold: int) -> np.ndarray:
    """
    Identifies relevant tiles based on the pixel difference between two consecutive frames.
    ...
    """
    # 1. Take the difference between current frame and previous frame
    abs_diff = cv2.absdiff(prev_frame, current_frame)
    
    # Convert the NumPy array to a PyTorch tensor and move it to the correct device
    abs_diff_tensor = torch.from_numpy(abs_diff).to('cpu').float() # Use the appropriate device, TODO: change this later 

    # Pad frames to be divisible by tile size
    padded_diff = padHWC(abs_diff_tensor, tile_size, tile_size)

    # Split the padded difference frame into tiles
    tiles = splitHWC(padded_diff, tile_size, tile_size)
    num_tiles = tiles.shape[0] * tiles.shape[1]
    tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)

    # 2. Calculate the total difference for each tile (sum up pixel difference values)
    # The sum is done across height, width, and color channels
    tile_diff_sums = torch.sum(tiles_flat, dim=(1, 2, 3)).cpu().numpy()

    # 3. Return a list of tile indices where the total difference is > threshold
    relevant_indices = np.where(tile_diff_sums > diff_threshold)[0]
    
    return relevant_indices

# def pixel_difference(prev_frame: np.ndarray, current_frame: np.ndarray, tile_size: int, diff_threshold: int) -> np.ndarray:
#     """
#     Identifies relevant tiles based on the pixel difference between two consecutive frames.

#     Args:
#         prev_frame (np.ndarray): The previous video frame.
#         current_frame (np.ndarray): The current video frame.
#         tile_size (int): The size of the tiles.
#         diff_threshold (int): The threshold for the total pixel difference per tile.

#     Returns:
#         np.ndarray: A 1D array of indices for the tiles that have a difference
#                     greater than the threshold.
#     """
#     # Take the difference between current frame and previous frame
#     abs_diff = cv2.absdiff(prev_frame, current_frame)
    
#     # Pad frames to be divisible by tile size
#     padded_diff = padHWC(abs_diff, tile_size, tile_size)

#     # Split the padded difference frame into tiles
#     tiles = splitHWC(padded_diff, tile_size, tile_size)
#     num_tiles = tiles.shape[0] * tiles.shape[1]
#     tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)

#     #  Calculate the total difference for each tile (sum up pixel difference values)
#     # The sum is done across height, width, and color channels
#     tile_diff_sums = np.sum(tiles_flat, axis=(1, 2, 3))

#     # Return a list of tile indices where the total difference is > threshold
#     relevant_indices = np.where(tile_diff_sums > diff_threshold)[0]
    
#     return relevant_indices

def merge_and_write_splice_results(splice_results_list: List[List[dict]], output_path: str):
    """
    Merge results from multiple splices and write to JSONL file.

    Args:
        splice_results_list: List of result lists, one per splice
        output_path: Path to output JSONL file
    """
    # Concatenate all splice results
    all_results = []
    for splice_results in splice_results_list:
        all_results.extend(splice_results)

    # Sort by frame_idx to ensure correct order
    all_results.sort(key=lambda x: x['frame_idx'])

    # Write to JSONL
    with open(output_path, 'w') as f:
        for frame_entry in all_results:
            f.write(json.dumps(frame_entry) + '\n')


def _run_splice_and_store_result(func: Callable, idx: int, results_dict: dict, gpu_id: int, command_queue: mp.Queue):
    """
    Wrapper function to run a splice task and store its result in a shared dictionary.
    Must be defined at module level for pickling.
    """
    result = func(gpu_id, command_queue)
    results_dict[idx] = result


def process_video_task(video_path: str, cache_video_dir: str, classifier: str,
                    tile_size: int, filter_type: str, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using the trained classifier model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.
    
    Args:
        video_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        filter_type (str): The type of filter to apply ('none', 'neighbor').
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
            
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
    device = f'cuda:{gpu_id}'
    
    # Load the trained model for this specific video, classifier, and tile size
    model = load_model(cache_video_dir, tile_size, classifier)
    model = model.to(device)
    # try:
    #     model.compile()
    #     # model = torch.compile(model)
    # except Exception as e:
    #     print(f"Failed to compile model: {e}")
    
    # Create output directory structure
    output_dir = os.path.join(cache_video_dir, 'relevancy') # This is now a base, specific dir is next
    
    # Create score directory for this classifier and tile size
    classifier_dir = os.path.join(output_dir, f'{classifier}_{tile_size}_{filter_type}')
    if os.path.exists(classifier_dir):
        shutil.rmtree(classifier_dir)
    os.makedirs(classifier_dir)
    
    # Create score directory for this tile size
    score_dir = os.path.join(classifier_dir, 'score')
    if os.path.exists(score_dir):
        shutil.rmtree(score_dir)
    os.makedirs(score_dir)
    output_path = os.path.join(score_dir, 'score.jsonl')
    
    # print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    with open(output_path, 'w') as f:
        frame_idx = 0
        description = f"{video_path.split('/')[-1]} {tile_size:>3} {classifier}"
        command_queue.put((device, {'description': description,
                                    'completed': 0, 'total': frame_count}))
        prev_relevance_grid = None # no previous frame yet
        prev_frame = None
        threshold = 0.5 # modify if necessary
        overall_start_time = time.time_ns() / 1e6

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # only use if flag is on
            relevant_indices = None
            pruned_tiles_prop = 0.0

            # get relevant indices for neighbor
            if filter_type == 'neighbor':
                if frame_idx > 0 and prev_relevance_grid is not None:
                    # For subsequent frames, determine relevant tiles based on the previous frame
                    relevant_indices = mark_neighbor_tiles(prev_relevance_grid, threshold)
                    video_name = os.path.basename(video_path)
                    manual_include = np.array(MANUALLY_INCLUDE.get(video_name, None))
                    if manual_include is not None:
                        relevant_indices = np.union1d(relevant_indices, manual_include)
            # 'none' filter or frame 0: relevant_indices remains None to process all tiles.
            
            # process tiles
            current_relevance_grid, runtime = process_frame_tiles(frame, model, tile_size, relevant_indices, device)

            if relevant_indices is not None:
                num_pruned = current_relevance_grid.size - len(relevant_indices)
                pruned_tiles_prop = num_pruned / current_relevance_grid.size if current_relevance_grid.size > 0 else 0.0
            
            # Update the relevance grid for the next loop iteration
            prev_relevance_grid = current_relevance_grid
            prev_frame = frame

            # Add overall time to runtime metrics before writing
            overall_end_time = time.time_ns() / 1e6
            runtime.append({'op': 'overall', 'time': overall_end_time - overall_start_time})

            num_tiles = (current_relevance_grid.shape[0] * current_relevance_grid.shape[1])
            # Create result entry for this frame
            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "frame_size": [height, width],
                "tile_size": tile_size,
                "runtime": runtime,
                "classification_size": current_relevance_grid.shape,
                "classification_hex": current_relevance_grid.flatten().tobytes().hex(),
                "pruned_tiles_prop": pruned_tiles_prop
            }
            
            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            if frame_idx % 100 == 0:
                f.flush()
            
            frame_idx += 1
            command_queue.put((device, {'completed': frame_idx}))

            # Reset overall timer for the next frame
            overall_start_time = time.time_ns() / 1e6

    cap.release()
    # print(f"Completed processing {frame_idx} frames. Results saved to {output_path}")


def main(args):
    """
    Main function that orchestrates the video tile classification process using parallel processing.

    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Creates splice tasks for each video/classifier/tile_size combination
    3. Uses multiprocessing to process splices in parallel across available GPUs
    4. Merges splice results and saves classification results

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - classifiers (List[str]): List of classifier models to use (default: multiple classifiers)
            - num_readers (int): Number of parallel readers per video (default: 2)
            - batch_size (int): Maximum tiles to batch together (default: 2048)

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

    assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"

    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")

    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # Get frame counts for each video and calculate splice boundaries
    video_metadata = {}
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_file}, skipping")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Adjust num_readers if video is shorter
        effective_num_readers = min(args.num_readers, max(1, frame_count))

        # Calculate splice boundaries (balanced distribution)
        frames_per_reader = frame_count // effective_num_readers
        remainder = frame_count % effective_num_readers

        splice_boundaries = []
        current_start = 0
        for i in range(effective_num_readers):
            # Distribute remainder across first readers
            splice_size = frames_per_reader + (1 if i < remainder else 0)
            splice_end = current_start + splice_size
            splice_boundaries.append((current_start, splice_end))
            current_start = splice_end

        video_metadata[video_file] = {
            'path': video_file_path,
            'frame_count': frame_count,
            'num_readers': effective_num_readers,
            'splice_boundaries': splice_boundaries
        }

    # Create splice tasks
    funcs: list[Callable[[int, mp.Queue], None]] = []
    task_metadata = []  # Track which task corresponds to which video/classifier/tile_size/splice

    for video_file in sorted(video_metadata.keys()):
        metadata = video_metadata[video_file]
        video_file_path = metadata['path']
        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)

        output_dir = os.path.join(cache_video_dir, 'relevancy')

        # Clear output directory if --clear flag is specified
        if args.clear and os.path.exists(output_dir):
            print(f"Clearing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for classifier in args.classifiers:
            for tile_size in tile_sizes_to_process:
                # Create one task per splice
                for splice_idx, (start_frame, end_frame) in enumerate(metadata['splice_boundaries']):
                    func = partial(
                        process_video_splice,
                        video_file_path,
                        start_frame,
                        end_frame,
                        cache_video_dir,
                        classifier,
                        tile_size,
                        args.filter,
                        args.batch_size,
                        splice_idx=splice_idx
                    )
                    funcs.append(func)
                    task_metadata.append({
                        'video_file': video_file,
                        'classifier': classifier,
                        'tile_size': tile_size,
                        'splice_idx': splice_idx,
                        'num_splices': metadata['num_readers']
                    })

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    print(f"Processing {len(video_metadata)} videos with {num_gpus} GPUs")
    print(f"Total splice tasks: {len(funcs)}")

    # Create a manager for shared results dictionary
    manager = mp.Manager()
    results_dict = manager.dict()

    # Wrap all functions with partial to bind idx and results_dict
    wrapped_funcs = [partial(_run_splice_and_store_result, func, idx, results_dict)
                     for idx, func in enumerate(funcs)]

    # Run all splice tasks in parallel
    ProgressBar(num_workers=num_gpus, num_tasks=len(wrapped_funcs)).run_all(wrapped_funcs)

    # Group results by (video, classifier, tile_size) and merge splices
    result_groups = {}
    for idx in range(len(funcs)):
        result = results_dict.get(idx)
        if result is None:
            print(f"Warning: No result for task {idx}")
            continue
        meta = task_metadata[idx]
        key = (meta['video_file'], meta['classifier'], meta['tile_size'])
        if key not in result_groups:
            result_groups[key] = {}
        result_groups[key][meta['splice_idx']] = result

    # Merge and write results for each group
    for (video_file, classifier, tile_size), splice_results in result_groups.items():
        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
        classifier_dir = os.path.join(cache_video_dir, 'relevancy', f'{classifier}_{tile_size}_{args.filter}')
        score_dir = os.path.join(classifier_dir, 'score')
        os.makedirs(score_dir, exist_ok=True)
        output_path = os.path.join(score_dir, 'score.jsonl')

        # Sort splices by index and merge
        sorted_splice_indices = sorted(splice_results.keys())
        sorted_splice_results = [splice_results[idx] for idx in sorted_splice_indices]

        merge_and_write_splice_results(sorted_splice_results, output_path)
        print(f"Completed {video_file} - {classifier}_{tile_size}: {output_path}")
            

if __name__ == '__main__':
    main(parse_args())
