#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass, field
import cv2
import torch
import numpy as np
import time
import shutil
import threading
import queue
from functools import partial

from polyis.images import splitHWC, padHWC
from scipy.ndimage import binary_dilation

from polyis.utilities import CACHE_DIR, CLASSIFIERS_CHOICES, CLASSIFIERS_TO_TEST, DATA_DIR, format_time

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


def load_model(video_path: str, tile_size: int, classifier_name: str, device: str = 'cpu') -> "torch.nn.Module":
    """
    Load trained classifier model for the specified tile size from a specific video directory.

    This function searches for a trained model in the expected directory structure:
    {video_path}/training/results/{classifier_name}_{tile_size}/model.pth

    Args:
        video_path (str): Path to the specific video directory
        tile_size (int): Tile size for which to load the model (30, 60, or 120)
        classifier_name (str): Name of the classifier model to use (default: 'SimpleCNN')
        device (str): Device to load the model to ('cpu' or 'cuda:X')

    Returns:
        The loaded trained model for the specified tile size.
            The model is loaded to the specified device and set to evaluation mode.

    Raises:
        FileNotFoundError: If no trained model is found for the specified tile size
        ValueError: If the classifier is not supported
    """
    results_path = os.path.join(video_path, 'training', 'results', f'{classifier_name}_{tile_size}')
    model_path = os.path.join(results_path, 'model.pth')

    if os.path.exists(model_path):
        print(f"Loading {classifier_name} model for tile size {tile_size} from {model_path}")
        # Load to CPU first to avoid CUDA initialization delays
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Model loaded to CPU, moving to {device}...")
        model = model.to(device)
        model.eval()
        print(f"Model ready on {device}")
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


def reader_thread_fn(video_path: str, start_frame: int, end_frame: int,
                     tile_size: int, filter_type: str, device: str,
                     tile_queue: queue.Queue, result_queue: queue.Queue,
                     splice_idx: int):
    """
    Reader thread that extracts tiles from a video splice and pushes them to a shared queue.
    Waits for inference results to perform neighbor filtering on subsequent frames.

    Args:
        video_path: Path to the video file
        start_frame: First frame index to process (inclusive)
        end_frame: Last frame index to process (exclusive)
        tile_size: Tile size to use
        filter_type: The type of filter to apply ('none', 'neighbor')
        device: GPU device string (e.g., 'cuda:0')
        tile_queue: Shared queue to push tiles to for inference
        result_queue: Queue to receive results from inference thread
        splice_idx: Index of this splice
    """
    # Open video and position at start_frame
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    cap = position_video_reader(cap, start_frame, video_path)

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.basename(video_path)
    print(f"Reader {splice_idx}: Processing {video_name} frames {start_frame}-{end_frame}")

    prev_relevance_grid = None
    threshold = 0.5

    # Process frames in this splice
    for frame_offset in range(end_frame - start_frame):
        frame_idx = start_frame + frame_offset

        ret, frame = cap.read()
        if not ret:
            break

        # Determine relevant indices
        relevant_indices = None

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

        transform_runtime = (time.time_ns() / 1e6) - transform_start_time

        if len(relevant_indices) == 0:
            # No tiles to process - create empty relevance grid immediately
            current_relevance_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

            # Push result directly to queue with special marker
            tile_queue.put({
                'type': 'empty_frame',
                'frame_idx': frame_idx,
                'relevance_grid': current_relevance_grid,
                'transform_time': transform_runtime,
                'fps': fps,
                'height': height,
                'width': width,
                'splice_idx': splice_idx
            })

            prev_relevance_grid = current_relevance_grid
        else:
            relevant_indices_tensor = torch.from_numpy(relevant_indices).to(device)
            tiles_to_process = torch.index_select(tiles_flat, 0, relevant_indices_tensor)

            # Push tiles to queue for inference
            tile_queue.put({
                'type': 'tiles',
                'frame_idx': frame_idx,
                'tiles': tiles_to_process,
                'tile_indices': relevant_indices,
                'grid_shape': (grid_h, grid_w),
                'transform_time': transform_runtime,
                'fps': fps,
                'height': height,
                'width': width,
                'splice_idx': splice_idx
            })

            # Wait for result from inference thread to update prev_relevance_grid
            while True:
                result = result_queue.get()
                if result['frame_idx'] == frame_idx:
                    prev_relevance_grid = result['relevance_grid']
                    break

    cap.release()

    print(f"Reader {splice_idx}: Completed {end_frame - start_frame} frames")

    # Signal completion by pushing sentinel
    tile_queue.put({'type': 'done', 'splice_idx': splice_idx})


def inference_thread_fn(model: torch.nn.Module, device: str, tile_size: int, batch_size: int,
                       tile_queue: queue.Queue, results_dict: dict, result_queues: dict,
                       num_readers: int, lock: threading.Lock):
    """
    Inference thread that pulls tiles from queue, batches them, and runs GPU inference.

    Args:
        model: The classification model to use
        device: GPU device string (e.g., 'cuda:0')
        tile_size: Tile size being processed
        batch_size: Maximum number of tiles to batch together
        tile_queue: Shared queue to pull tiles from
        results_dict: Dictionary to store completed frame results
        result_queues: Dict mapping splice_idx -> queue for sending results back to readers
        num_readers: Number of reader threads
        lock: Lock for thread-safe updates
    """
    # Create TileBatch for cross-frame batching
    tile_batch = TileBatch(
        max_batch_size=batch_size,
        tile_size=tile_size,
        device=device,
        model=model
    )

    print(f"Inference thread started, waiting for tiles from {num_readers} readers...")

    readers_done = 0
    frame_metadata = {}  # Track metadata for each frame
    frames_processed = 0

    while readers_done < num_readers:
        try:
            item = tile_queue.get(timeout=0.1)
        except queue.Empty:
            # Flush pending batch if queue is temporarily empty
            # This prevents deadlock when batch doesn't fill completely
            if tile_batch.current_size > 0:
                batch_results = tile_batch.flush()

                # Process flushed results
                for result_frame_idx, relevance_grid, inf_time, num_processed_tiles in batch_results:
                    meta = frame_metadata.get(result_frame_idx, {})
                    result_pruned_prop = (relevance_grid.size - num_processed_tiles) / relevance_grid.size if relevance_grid.size > 0 else 0.0

                    frame_entry = {
                        "frame_idx": result_frame_idx,
                        "timestamp": result_frame_idx / meta.get('fps', 30) if meta.get('fps', 30) > 0 else 0,
                        "frame_size": [meta.get('height', 0), meta.get('width', 0)],
                        "tile_size": tile_size,
                        "runtime": format_time(transform=meta.get('transform_time', 0.0), inference=inf_time,
                                              overall=meta.get('transform_time', 0.0) + inf_time),
                        "classification_size": relevance_grid.shape,
                        "classification_hex": relevance_grid.flatten().tobytes().hex(),
                        "pruned_tiles_prop": result_pruned_prop
                    }

                    with lock:
                        results_dict[result_frame_idx] = frame_entry

                    frames_processed += 1
                    if frames_processed % 100 == 0:
                        print(f"Inference: Processed {frames_processed} frames, {readers_done}/{num_readers} readers done")

                    # Send result back to the reader that owns this frame
                    splice_idx = meta.get('splice_idx')
                    if splice_idx is not None and splice_idx in result_queues:
                        result_queues[splice_idx].put({
                            'frame_idx': result_frame_idx,
                            'relevance_grid': relevance_grid
                        })
            continue

        item_type = item['type']

        if item_type == 'done':
            # One reader finished
            readers_done += 1
            continue

        elif item_type == 'empty_frame':
            # Frame with no tiles to process - store result immediately
            frame_idx = item['frame_idx']
            relevance_grid = item['relevance_grid']
            transform_time = item['transform_time']
            fps = item['fps']
            height = item['height']
            width = item['width']
            splice_idx = item['splice_idx']

            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "frame_size": [height, width],
                "tile_size": tile_size,
                "runtime": format_time(transform=transform_time, inference=0.0, overall=transform_time),
                "classification_size": relevance_grid.shape,
                "classification_hex": relevance_grid.flatten().tobytes().hex(),
                "pruned_tiles_prop": 1.0
            }

            with lock:
                results_dict[frame_idx] = frame_entry

            # Send result back to reader for neighbor filtering
            if splice_idx in result_queues:
                result_queues[splice_idx].put({
                    'frame_idx': frame_idx,
                    'relevance_grid': relevance_grid
                })

        elif item_type == 'tiles':
            # Store metadata for this frame
            frame_idx = item['frame_idx']
            frame_metadata[frame_idx] = {
                'transform_time': item['transform_time'],
                'fps': item['fps'],
                'height': item['height'],
                'width': item['width'],
                'splice_idx': item['splice_idx']
            }

            # Add to batch - may trigger flush
            batch_results = tile_batch.add_frame_tiles(
                frame_idx=frame_idx,
                tiles=item['tiles'],
                tile_indices=item['tile_indices'],
                grid_shape=item['grid_shape']
            )

            # Process any flushed results
            for result_frame_idx, relevance_grid, inf_time, num_processed_tiles in batch_results:
                meta = frame_metadata.get(result_frame_idx, {})
                result_pruned_prop = (relevance_grid.size - num_processed_tiles) / relevance_grid.size if relevance_grid.size > 0 else 0.0

                frame_entry = {
                    "frame_idx": result_frame_idx,
                    "timestamp": result_frame_idx / meta.get('fps', 30) if meta.get('fps', 30) > 0 else 0,
                    "frame_size": [meta.get('height', 0), meta.get('width', 0)],
                    "tile_size": tile_size,
                    "runtime": format_time(transform=meta.get('transform_time', 0.0), inference=inf_time,
                                          overall=meta.get('transform_time', 0.0) + inf_time),
                    "classification_size": relevance_grid.shape,
                    "classification_hex": relevance_grid.flatten().tobytes().hex(),
                    "pruned_tiles_prop": result_pruned_prop
                }

                with lock:
                    results_dict[result_frame_idx] = frame_entry

                frames_processed += 1
                if frames_processed % 100 == 0:
                    print(f"Inference: Processed {frames_processed} frames, {readers_done}/{num_readers} readers done")

                # Send result back to the reader that owns this frame
                splice_idx = meta.get('splice_idx')
                if splice_idx is not None and splice_idx in result_queues:
                    result_queues[splice_idx].put({
                        'frame_idx': result_frame_idx,
                        'relevance_grid': relevance_grid
                    })

    # Flush any remaining tiles
    print(f"Inference: All {num_readers} readers done, flushing final batch...")
    final_results = tile_batch.flush()
    for result_frame_idx, relevance_grid, inf_time, num_processed_tiles in final_results:
        meta = frame_metadata.get(result_frame_idx, {})
        result_pruned_prop = (relevance_grid.size - num_processed_tiles) / relevance_grid.size if relevance_grid.size > 0 else 0.0

        frame_entry = {
            "frame_idx": result_frame_idx,
            "timestamp": result_frame_idx / meta.get('fps', 30) if meta.get('fps', 30) > 0 else 0,
            "frame_size": [meta.get('height', 0), meta.get('width', 0)],
            "tile_size": tile_size,
            "runtime": format_time(transform=meta.get('transform_time', 0.0), inference=inf_time,
                                  overall=meta.get('transform_time', 0.0) + inf_time),
            "classification_size": relevance_grid.shape,
            "classification_hex": relevance_grid.flatten().tobytes().hex(),
            "pruned_tiles_prop": result_pruned_prop
        }

        with lock:
            results_dict[result_frame_idx] = frame_entry

        # Send result back to readers
        splice_idx = meta.get('splice_idx')
        if splice_idx is not None and splice_idx in result_queues:
            result_queues[splice_idx].put({
                'frame_idx': result_frame_idx,
                'relevance_grid': relevance_grid
            })

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

def process_video_threaded(video_path: str, cache_video_dir: str, classifier: str,
                           tile_size: int, filter_type: str, batch_size: int,
                           num_readers: int, gpu_id: int, splice_boundaries: List[Tuple[int, int]]):
    """
    Process a video using multiple reader threads feeding a single GPU inference thread.

    Args:
        video_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        filter_type: The type of filter to apply ('none', 'neighbor')
        batch_size: Maximum number of tiles to batch together
        num_readers: Number of reader threads to use
        gpu_id: GPU ID to use for processing
        splice_boundaries: List of (start_frame, end_frame) tuples for each reader

    Returns:
        Dictionary mapping frame_idx to frame result
    """
    device = f'cuda:{gpu_id}'

    # Load the trained model for this specific video, classifier, and tile size
    model = load_model(cache_video_dir, tile_size, classifier, device=device)

    # Create output directory structure
    output_dir = os.path.join(cache_video_dir, 'relevancy')
    classifier_dir = os.path.join(output_dir, f'{classifier}_{tile_size}_{filter_type}')
    if os.path.exists(classifier_dir):
        shutil.rmtree(classifier_dir)
    os.makedirs(classifier_dir)

    score_dir = os.path.join(classifier_dir, 'score')
    os.makedirs(score_dir, exist_ok=True)
    output_path = os.path.join(score_dir, 'score.jsonl')

    # Shared data structures
    tile_queue = queue.Queue(maxsize=batch_size * 4)  # Limit queue size to prevent memory issues
    result_queues = {i: queue.Queue() for i in range(num_readers)}
    results_dict = {}
    lock = threading.Lock()

    # Start inference thread
    inference_thread = threading.Thread(
        target=inference_thread_fn,
        args=(model, device, tile_size, batch_size, tile_queue, results_dict,
              result_queues, num_readers, lock),
        daemon=False
    )
    inference_thread.start()

    # Start reader threads
    reader_threads = []
    for splice_idx, (start_frame, end_frame) in enumerate(splice_boundaries):
        thread = threading.Thread(
            target=reader_thread_fn,
            args=(video_path, start_frame, end_frame, tile_size, filter_type,
                  device, tile_queue, result_queues[splice_idx], splice_idx),
            daemon=False
        )
        thread.start()
        reader_threads.append(thread)

    print(f"Started {num_readers} reader threads and 1 inference thread, processing frames...")

    # Wait for all readers to finish
    for thread in reader_threads:
        thread.join()

    # Wait for inference thread to finish
    inference_thread.join()

    # Write results to file
    sorted_results = [results_dict[frame_idx] for frame_idx in sorted(results_dict.keys())]
    with open(output_path, 'w') as f:
        for frame_entry in sorted_results:
            f.write(json.dumps(frame_entry) + '\n')

    print(f"Completed {os.path.basename(video_path)} - {classifier}_{tile_size}: {output_path}")

    return results_dict




def main(args):
    """
    Main function that orchestrates the video tile classification process using threading and GPU assignment.

    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Creates video tasks for each video/classifier/tile_size combination
    3. Assigns one GPU per video, processes videos sequentially (or in parallel if multiple GPUs)
    4. Each video uses multiple reader threads feeding a single GPU inference thread

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
    print("Starting main function...")

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

    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    print(f"Processing {len(video_metadata)} videos with {num_gpus} GPUs")
    print(f"Using {args.num_readers} reader threads per video")

    # Clear output directories if --clear flag is specified
    if args.clear:
        for video_file in video_metadata.keys():
            cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
            output_dir = os.path.join(cache_video_dir, 'relevancy')
            if os.path.exists(output_dir):
                print(f"Clearing output directory: {output_dir}")
                shutil.rmtree(output_dir)

    # Process each video/classifier/tile_size combination
    # Assign GPUs in round-robin fashion
    gpu_idx = 0
    for video_file in sorted(video_metadata.keys()):
        metadata = video_metadata[video_file]
        video_file_path = metadata['path']
        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)

        for classifier in args.classifiers:
            for tile_size in tile_sizes_to_process:
                # Assign GPU for this video
                assigned_gpu = gpu_idx % num_gpus
                gpu_idx += 1

                print(f"\nProcessing {video_file} - {classifier}_{tile_size} on GPU {assigned_gpu}")

                # Process video with threading
                process_video_threaded(
                    video_path=video_file_path,
                    cache_video_dir=cache_video_dir,
                    classifier=classifier,
                    tile_size=tile_size,
                    filter_type=args.filter,
                    batch_size=args.batch_size,
                    num_readers=metadata['num_readers'],
                    gpu_id=assigned_gpu,
                    splice_boundaries=metadata['splice_boundaries']
                )

    print("\nAll videos processed successfully!")
            

if __name__ == '__main__':
    main(parse_args())
