#!/usr/local/bin/python

import argparse
from enum import IntEnum
import json
import os
from typing import Callable, Literal, NamedTuple
import cv2
import numpy as np
import shutil
import time
import multiprocessing as mp
from functools import partial

import torch

from polyis import dtypes
from polyis.utilities import (
    CACHE_DIR, CLASSIFIERS_CHOICES,
    DATASETS_DIR, TILEPADDING_MODES, format_time,
    load_classification_results,
    CLASSIFIERS_TO_TEST, ProgressBar, DATASETS_TO_TEST, TILE_SIZES
)
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack_ffd import pack_all


class PackMode(IntEnum):
    """Packing mode options for bin packing algorithms."""
    Easiest_Fit = 0  # Pack into collage with most empty space
    First_Fit = 1    # Pack into first collage that fits
    Best_Fit = 2     # Pack into collage with least empty space that fits


class PolyominoPosition(NamedTuple):
    oy: int
    ox: int
    py: int
    px: int
    frame: int
    shape: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description='Execute compression of video tiles into images based on classification results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification probability (0.0 to 1.0)')
    parser.add_argument('--classifiers', required=False,
                        default=CLASSIFIERS_TO_TEST + ['Perfect'],
                        choices=CLASSIFIERS_CHOICES + ['Perfect'],
                        nargs='+',
                        help='Classifier names to use (can specify multiple): '
                             f'{", ".join(CLASSIFIERS_CHOICES)}. For example: '
                             '--classifiers YoloN ShuffleNet05 ResNet18')
    parser.add_argument('--clear', action='store_true',
                        help='Remove and recreate the compressed frames folder')
    parser.add_argument('--tilepadding', type=str, choices=['none', 'connected', 'disconnected'],
                        nargs='+', default=['none', 'connected', 'disconnected'],
                        help='Apply padding to the classification results (space-separated list of none/connected/disconnected)')
    parser.add_argument('--mode', type=lambda x: PackMode[x], 
                        default=PackMode.Best_Fit,
                        help='Packing mode for the pack_all function. Options: Easiest_Fit, First_Fit, Best_Fit (default: Best_Fit)')
    return parser.parse_args()


OffsetLookup = tuple[tuple[int, int], tuple[int, int], int]


def save_packed_image(canvas: dtypes.NPImage, index_map: dtypes.IndexMap, offset_lookup: list[OffsetLookup],
                      collage_idx: int, start_idx: int, frame_idx: int, output_dir: str, step_times: dict):
    """
    Save the packed image, index_map, and offset_lookup.
    
    Args:
        canvas: The canvas to save
        index_map: The index map to save
        offset_lookup: The offset lookup to save
        start_idx: The start index of the packed image
        frame_idx: The end index of the packed image
        output_dir: The directory to save the files
        step_times: The step times to update
    """
    image_dir = os.path.join(output_dir, 'images')
    index_map_dir = os.path.join(output_dir, 'index_maps')
    offset_lookup_dir = os.path.join(output_dir, 'offset_lookups')

    # Profile: Save canvas
    step_start = (time.time_ns() / 1e6)
    img_path = os.path.join(image_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.jpg')
    cv2.imwrite(img_path, canvas)
    step_times['save_canvas'] = (time.time_ns() / 1e6) - step_start

    # Profile: Save index_map and offset_lookup
    step_start = (time.time_ns() / 1e6)
    index_map_path = os.path.join(index_map_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.npy')
    np.save(index_map_path, index_map)

    offset_lookup_path = os.path.join(offset_lookup_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.jsonl')
    with open(offset_lookup_path, 'w') as f:
        for offset in offset_lookup:
            f.write(json.dumps(offset) + '\n')
    step_times['save_mapping_files'] = (time.time_ns() / 1e6) - step_start


# PolyominoPosition = tuple[int, int, int, int, int, int, np.ndarray]
Collage = list[PolyominoPosition]


def compress(video_file_path: str, cache_video_dir: str, classifier: str, tilesize: int,
             threshold: float, tilepadding: Literal['none', 'connected', 'disconnected'],
             mode: PackMode, gpu_id: int, command_queue: mp.Queue):
    """
    Compress a single video by batch processing all frames at once using pack_all.

    Args:
        video_file_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tilesize: Tile size to use
        threshold: Threshold for classification probability
        tilepadding: Whether to apply padding to classification results
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    video_name = os.path.basename(video_file_path)

    # Load classification results
    dataset = os.path.basename(os.path.dirname(os.path.dirname(cache_video_dir)))
    video_file = os.path.basename(cache_video_dir)
    results = load_classification_results(CACHE_DIR, dataset, video_file,
                                          tilesize, classifier, execution_dir=True)
    
    # Create output directory for compression results
    output_dir = os.path.join(cache_video_dir, '033_compressed_frames', f'{classifier}_{tilesize}_{tilepadding}')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir)
    index_map_dir = os.path.join(output_dir, 'index_maps')
    os.makedirs(index_map_dir)
    offset_lookup_dir = os.path.join(output_dir, 'offset_lookups')
    os.makedirs(offset_lookup_dir)

    # Send initial progress update
    description = f"{dataset} {video_name.split('.')[0]} {tilesize:>3} {classifier[:4]} {tilepadding[:4]}"
    # command_queue.put((device, {
    #     'description': description + ' grouping',
    #     'completed': 0,
    #     'total': len(results)
    # }))
    
    # Open video to get dimensions
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert num_frames_total == len(results), f"Expected {len(results)} frames, got {num_frames_total}"

    # Calculate grid dimensions
    grid_height = height // tilesize
    grid_width = width // tilesize

    # Step 1: Group tiles for all frames to get polyominoes
    timing_data = []

    polyominoes_stacks = np.empty(len(results), dtype=np.uint64)
    for frame_idx, frame_result in enumerate(results):
        step_times = {}

        # Get classification results
        step_start = (time.time_ns() / 1e6)
        classifications: str = frame_result['classification_hex']
        classification_size: tuple[int, int] = frame_result['classification_size']
        step_times['get_classifications'] = (time.time_ns() / 1e6) - step_start

        # Create bitmap from classifications
        step_start = (time.time_ns() / 1e6)
        bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
        bitmap_frame = bitmap_frame > (threshold * 255)
        bitmap_frame = bitmap_frame.astype(np.uint8)
        assert dtypes.is_bitmap(bitmap_frame), bitmap_frame.shape
        step_times['create_bitmap'] = (time.time_ns() / 1e6) - step_start

        # Group connected tiles into polyominoes
        step_start = (time.time_ns() / 1e6)
        polyominoes = group_tiles(bitmap_frame, TILEPADDING_MODES[tilepadding])
        polyominoes_stacks[frame_idx] = polyominoes
        step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start

        timing_data.append({'step': 'group_tiles', 'frame_idx': frame_idx, 'runtime': format_time(**step_times)})

        # # Update progress
        # if frame_idx % max(1, len(results) // 100) == 0:
        #     command_queue.put((device, {'description': description + ' grouping', 'completed': frame_idx}))

    # Step 2: Pack all polyominoes in batches (10 equal parts)
    num_batches = 1
    batch_size = len(polyominoes_stacks) // num_batches
    # Handle case where len(polyominoes_stacks) < num_batches
    if batch_size == 0:
        batch_size = 1
        num_batches = len(polyominoes_stacks)

    command_queue.put((device, {'description': description + ' packing', 'completed': 0, 'total': num_batches}))

    # Initialize empty list to store all collages from all batches
    collages = []
    total_pack_time = 0.0

    # Process each batch
    for batch_idx in range(num_batches):
        # Calculate batch boundaries
        start_idx = batch_idx * batch_size
        # For the last batch, include any remaining frames
        if batch_idx == num_batches - 1:
            end_idx = len(polyominoes_stacks)
        else:
            end_idx = start_idx + batch_size

        # Extract batch of polyominoes
        batch_polyominoes = polyominoes_stacks[start_idx:end_idx]

        # Pack this batch
        batch_start = (time.time_ns() / 1e6)
        batch_collages_ = pack_all(batch_polyominoes, grid_height, grid_width, int(mode))
        batch_pack_time = (time.time_ns() / 1e6) - batch_start
        total_pack_time += batch_pack_time

        # Adjust frame indices in batch_collages to be relative to the full video
        # pack_all returns frame indices relative to the batch (0-indexed within batch)
        # We need to offset them by start_idx to get the actual frame index
        batch_collages: list[list[PolyominoPosition]] = []
        for collage in batch_collages_:
            batch_collages.append([
                PolyominoPosition(oy=poly_pos.oy, ox=poly_pos.ox,
                                  py=poly_pos.py, px=poly_pos.px,
                                  frame=poly_pos.frame + start_idx,
                                  shape=poly_pos.shape)
                for poly_pos in collage
            ])

        # Merge batch collages into the overall collages list
        collages.extend(batch_collages)

        # Record timing for this batch
        timing_data.append({
            'step': f'pack_batch_{batch_idx}',
            'frames': f'{start_idx}-{end_idx-1}',
            'runtime': format_time(pack_batch=batch_pack_time)
        })

        # Update progress
        command_queue.put((device, {'description': description + ' packing', 'completed': batch_idx + 1}))

    # # Record total packing time
    # timing_data.append({'step': 'pack_all_total', 'runtime': format_time(pack_all_total=total_pack_time)})

    # Step 3: Read all frames from video
    command_queue.put((device, {'description': description + ' reading', 'completed': 0, 'total': num_frames_total}))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for idx in range(num_frames_total):
        ret, frame = cap.read()
        if not ret:
            break
        assert dtypes.is_np_image(frame), frame.shape
        frames.append(frame)
        if idx % max(1, num_frames_total // 100) == 0:
            command_queue.put((device, {'description': description + ' reading', 'completed': idx}))
    
    assert len(frames) == num_frames_total, f"Expected {num_frames_total} frames, got {len(frames)}"
    assert cap.read()[0] is False, "Expected no more frames"
    cap.release()

    # Step 4: Render and save each collage
    command_queue.put((device, {'description': description + ' rendering', 'completed': 0, 'total': len(collages)}))

    for collage_idx, collage in enumerate(collages):
        assert len(collage) > 0, f"Expected at least one polyomino in collage {collage_idx}"
        step_times = {}

        # Initialize canvas and metadata structures
        step_start = (time.time_ns() / 1e6)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        assert dtypes.is_np_image(canvas), canvas.shape
        index_map = np.zeros((grid_height, grid_width), dtype=np.uint16)
        assert dtypes.is_index_map(index_map), index_map.shape
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        step_times['initialize_canvas'] = (time.time_ns() / 1e6) - step_start

        # Get frame range for this collage
        frame_indices = [pos.frame for pos in collage]
        start_frame = min(frame_indices)
        end_frame = max(frame_indices)

        # Process each polyomino in this collage
        step_start = (time.time_ns() / 1e6)
        for gid, poly_pos in enumerate(collage, start=1):
            oy, ox, py, px, frame_idx, shape = poly_pos

            # Get source frame
            frame = frames[frame_idx]

            # Optimized tile rendering: vectorized coordinate computation with slice-based copying
            # Slice operations use optimized block memory copy (memcpy-like) which is faster than per-pixel
            i_coords = shape[:, 0]
            j_coords = shape[:, 1]
            
            # Compute all tile corner positions at once (vectorized)
            sy_coords = (oy + i_coords) * tilesize
            sx_coords = (ox + j_coords) * tilesize
            dy_coords = (py + i_coords) * tilesize
            dx_coords = (px + j_coords) * tilesize
            
            # Copy tiles using optimized slice operations (block memory copy)
            for idx in range(len(shape)):
                sy, sx = sy_coords[idx], sx_coords[idx]
                dy, dx = dy_coords[idx], dx_coords[idx]
                canvas[dy:dy+tilesize, dx:dx+tilesize] = frame[sy:sy+tilesize, sx:sx+tilesize]

            # Update index_map (vectorized)
            index_map[py + i_coords, px + j_coords] = gid

            # Update offset_lookup
            offset_lookup.append(((py, px), (oy, ox), frame_idx))
        step_times['render_tiles'] = (time.time_ns() / 1e6) - step_start

        # Save the collage
        step_start = (time.time_ns() / 1e6)
        save_packed_image(canvas, index_map, offset_lookup, collage_idx, start_frame, end_frame, output_dir, step_times)
        step_times['save_collage'] = (time.time_ns() / 1e6) - step_start

        timing_data.append({'step': 'process_collage', 'runtime': format_time(**step_times)})

        # Update progress
        command_queue.put((device, {'description': description + ' rendering', 'completed': collage_idx + 1}))

    # # Free polyomino stacks
    # print('free polyominoes')
    # step_start = (time.time_ns() / 1e6)
    # command_queue.put((device, {'description': description + ' freeing polyominoes', 'completed': 0, 'total': len(polyominoes_stacks)}))
    # for idx, polyominoes in enumerate(polyominoes_stacks):
    #     free_polyimino_stack(polyominoes)
    #     command_queue.put((device, {'description': description + ' freeing polyominoes', 'completed': idx}))
    # end_time = (time.time_ns() / 1e6)
    # timing_data.append({'step': 'free_polyominoes', 'runtime': format_time(free_polyominoes=end_time - step_start)})
    # print('free polyominoes done')

    # Save runtime data
    runtime_file = os.path.join(output_dir, 'runtime.jsonl')
    with open(runtime_file, 'w') as f:
        for data in timing_data:
            f.write(json.dumps(data) + '\n')

    command_queue.put((device, {'description': description + ' done', 'completed': len(collages)}))


def main(args):
    """
    Main function that orchestrates the video tile compression process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves compression results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tilesize (str): Tile size to use for compression ('30', '60', '120', or 'all')
            - threshold (float): Threshold for classification probability (0.0 to 1.0)
            - classifiers (list): List of classifier names to use (default: CLASSIFIERS_TO_TEST)
            - clear (bool): Whether to remove and recreate the compressed frames folder
            
    Note:
        - The script expects classification results from 020_exec_classify.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier}_{tilesize}/score/
        - Looks for score.jsonl files
        - Videos are read from {DATASETS_DIR}/{dataset}/
        - Compressed images are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/033_compressed_frames/{classifier}_{tilesize}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/033_compressed_frames/{classifier}_{tilesize}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/033_compressed_frames/{classifier}_{tilesize}/offset_lookups/
        - When tilesize is 'all', all tile sizes (30, 60, 120) are processed
        - When classifiers is not specified, all classifiers in CLASSIFIERS_TO_TEST are processed
        - If no classification results are found for a video, that video is skipped with a warning
        - Tiles with classification probability > threshold are considered relevant for compression
    """
    mp.set_start_method('spawn', force=True)
    
    # Create tasks list with all video/classifier/tilesize combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATASETS_DIR, dataset_name)

        for videoset in ['test']:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Videoset directory {videoset_dir} does not exist, skipping...")
                continue
            
            # Get all video files from the dataset directory
            video_files = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            for video_file in sorted(video_files):
                video_file_path = os.path.join(videoset_dir, video_file)
                cache_video_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file)

                compressed_frames_base_dir = os.path.join(cache_video_dir, '033_compressed_frames')
                if args.clear and os.path.exists(compressed_frames_base_dir):
                    shutil.rmtree(compressed_frames_base_dir)
                    print(f"Cleared existing compressed frames folder: {compressed_frames_base_dir}")
                
                for classifier in args.classifiers:
                    for tilesize in TILE_SIZES:
                        for tilepadding in TILEPADDING_MODES:
                            funcs.append(partial(compress, video_file_path, cache_video_dir,
                                                 classifier, tilesize, args.threshold, tilepadding, args.mode))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.3)
    # num_processes = max(1, torch.cuda.device_count() // 2)
    # num_processes = 1
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    ProgressBar(num_workers=torch.cuda.device_count(), num_tasks=len(funcs), refresh_per_second=10).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
