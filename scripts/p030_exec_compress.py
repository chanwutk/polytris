#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable
import cv2
import numpy as np
import shutil
import time
import multiprocessing as mp
from functools import partial

from polyis import dtypes
from polyis.utilities import (
    CACHE_DIR, CLASSIFIERS_CHOICES,
    DATA_DIR, format_time,
    load_classification_results,
    CLASSIFIERS_TO_TEST, ProgressBar
)
from lib.pack_append import pack_append
from lib.group_tiles import group_tiles


# TILE_SIZES = [30, 60, 120]
TILE_SIZES = [30, 60]


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (List[str]): Dataset names to process (default: ['b3d'])
            - tile_size (int | str): Tile size to use for compression (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification probability (default: 0.5)
            - classifiers (str): Classifier names to use
            - clear (bool): Whether to remove and recreate the compressed frames folder (default: False)
    """
    parser = argparse.ArgumentParser(description='Execute compression of video tiles into images based on classification results')
    parser.add_argument('--datasets', required=False,
                        default=['caldot1-yolov5', 'caldot2-yolov5'],
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for compression (or "all" for all tile sizes)')
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
    return parser.parse_args()


def render(canvas: dtypes.NPImage, positions: list[dtypes.PolyominoPositions], 
           frame: dtypes.NPImage, chunk_size: int) -> dtypes.NPImage:
    """
    Render packed polyominoes onto the canvas.
    
    Args:
        canvas: The canvas to render onto
        positions: List of packed polyominoe positions
        frame: Source frame
        chunk_size: Size of each tile/chunk
        
    Returns:
        np.ndarray: Updated canvas
    """
    # TODO: use Torch, Cython, or Numba.
    for y, x, mask, offset in positions:
        yfrom = y * chunk_size
        xfrom = x * chunk_size
        
        # Get mask indices where True
        for i, j in zip(*np.nonzero(mask)):
            patch = frame[
                (i + offset[0]) * chunk_size:(i + offset[0] + 1) * chunk_size,
                (j + offset[1]) * chunk_size:(j + offset[1] + 1) * chunk_size,
            ]
            canvas[
                yfrom + (chunk_size * i): yfrom + (chunk_size * i) + chunk_size,
                xfrom + (chunk_size * j): xfrom + (chunk_size * j) + chunk_size,
            ] = patch
    
    return canvas


def apply_pack(
    positions: list[dtypes.PolyominoPositions],
    canvas: dtypes.NPImage,
    index_map: dtypes.IndexMap,
    offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
    frame_idx: int,
    frame: dtypes.NPImage,
    tile_size: int,
    step_times: dict[str, float],
):
    """
    Apply packed results to the canvas, index_map, and offset_lookup.
    
    Args:
        positions: The positions of the packed polyominoes
        canvas: The canvas to render onto
        index_map: The index map to update
        offset_lookup: The offset lookup to update
        frame_idx: The index of the frame
        frame: The frame to render
        tile_size: The size of the tile
        step_times: The step times to update
        
    Returns:
        dtypes.NPImage: canvas
    """
    # Profile: Render packed polyominoes onto canvas
    step_start = (time.time_ns() / 1e6)
    canvas = render(canvas, positions, frame, tile_size)
    step_times['render_canvas'] = (time.time_ns() / 1e6) - step_start

    assert index_map is not None
    
    # Profile: Update index_map and det_info
    step_start = (time.time_ns() / 1e6)
    # Start gid from the current length of offset_lookup
    start_gid = len(offset_lookup)
    for i, (y, x, mask, offset) in enumerate(positions):
        gid = start_gid + i + 1
        h = mask.shape[0]
        w = mask.shape[1]
        # assert not np.any(index_map[y:y+h, x:x+w] & mask), \
        #     (index_map[y:y+h, x:x+w], mask)
        index_map[y:y+h, x:x+w] += mask.astype(np.uint16) * gid
        offset_lookup.append(((y, x), offset, frame_idx))
    step_times['update_mapping'] = (time.time_ns() / 1e6) - step_start

    return canvas


def save_packed_image(canvas: dtypes.NPImage, index_map: dtypes.IndexMap,
                      offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
                      start_idx: int, frame_idx: int, output_dir: str, step_times: dict):
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
    img_path = os.path.join(image_dir, f'{start_idx:08d}_{frame_idx:08d}.jpg')
    cv2.imwrite(img_path, canvas)
    step_times['save_canvas'] = (time.time_ns() / 1e6) - step_start

    # Profile: Save index_map and offset_lookup
    step_start = (time.time_ns() / 1e6)
    index_map_path = os.path.join(index_map_dir, f'{start_idx:08d}_{frame_idx:08d}.npy')
    np.save(index_map_path, index_map)

    offset_lookup_path = os.path.join(offset_lookup_dir, f'{start_idx:08d}_{frame_idx:08d}.jsonl')
    with open(offset_lookup_path, 'w') as f:
        for offset in offset_lookup:
            f.write(json.dumps(offset) + '\n')
    step_times['save_mapping_files'] = (time.time_ns() / 1e6) - step_start


def compress(video_file_path: str, cache_video_dir: str, classifier: str, 
             tile_size: int, threshold: float, gpu_id: int, command_queue: mp.Queue):
    """
    Compress a single video with a specific classifier and tile size.
    
    Args:
        video_file_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        threshold: Threshold for classification probability
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    video_name = os.path.basename(video_file_path)
    
    # Load classification results
    dataset = os.path.basename(os.path.dirname(os.path.dirname(cache_video_dir)))
    video_file = os.path.basename(cache_video_dir)
    results = load_classification_results(CACHE_DIR, dataset, video_file,
                                          tile_size, classifier, execution_dir=True)
    
    # Create output directory for compression results
    output_dir = os.path.join(cache_video_dir, '030_compressed_frames', f'{classifier}_{tile_size}')
    if os.path.exists(output_dir):
        # Remove the entire directory
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Send initial progress update
    command_queue.put((device, {
        'description': f"{video_name} {tile_size:>3} {classifier}",
        'completed': 0,
        'total': len(results)
    }))
    
    # # Process the video for compression
    # print(f"Processing video for compression: {video_file_path}")
    
    # Open video to get dimensions and initialize capture
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image_dir = os.path.join(output_dir, 'images')
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    index_map_dir = os.path.join(output_dir, 'index_maps')
    if os.path.exists(index_map_dir):
        shutil.rmtree(index_map_dir)
    os.makedirs(index_map_dir)

    offset_lookup_dir = os.path.join(output_dir, 'offset_lookups')
    if os.path.exists(offset_lookup_dir):
        shutil.rmtree(offset_lookup_dir)
    os.makedirs(offset_lookup_dir)
    
    # Calculate grid dimensions
    grid_height = height // tile_size
    grid_width = width // tile_size

    def init_compression_variables():
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        assert dtypes.is_np_image(canvas), canvas.shape
        occupied_tiles = np.zeros((grid_height, grid_width), dtype=np.uint8)
        assert dtypes.is_bitmap(occupied_tiles), occupied_tiles.shape
        index_map = np.zeros((grid_height, grid_width), dtype=np.uint16)
        assert dtypes.is_index_map(index_map), index_map.shape
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        return canvas, occupied_tiles, index_map, offset_lookup, True, False
    
    # Initialize compression variables
    canvas, occupied_tiles, index_map, offset_lookup, clean, full = init_compression_variables()
    start_idx = 0
    last_frame_idx = -1
    read_frame_idx = -1
    frame_idx = -1
    count_compressed_images = 0
    
    # Initialize profiling output file
    runtime_file = os.path.join(output_dir, 'runtime.jsonl')
    
    # print(f"Processing {len(results)} frames with tile size {tile_size}")

    with open(runtime_file, 'w') as f:
        # Process each frame
        mod = int(len(results) * 0.05)
        for frame_idx, frame_result in enumerate(results):
            # Start profiling for this frame
            # frame_start_time = (time.time_ns() / 1e6)
            step_times = {}
            
            # Assert that frame_idx is increasing
            assert frame_idx > last_frame_idx, f"Frame index must be increasing, got {frame_idx} after {last_frame_idx}"
            last_frame_idx = frame_idx
                
            # Profile: Read frame from video
            step_start = (time.time_ns() / 1e6)
            ret = False
            frame = None
            while read_frame_idx < frame_idx:
                ret = cap.grab()
                read_frame_idx += 1
            assert ret
            ret, frame = cap.retrieve()
            assert ret
            assert dtypes.is_np_image(frame), frame.shape
            step_times['read_frame'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Get classification results
            step_start = (time.time_ns() / 1e6)
            classifications: str = frame_result['classification_hex']
            classification_size: tuple[int, int] = frame_result['classification_size']
            step_times['get_classifications'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Create bitmap from classifications
            step_start = (time.time_ns() / 1e6)
            bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
            bitmap_frame = bitmap_frame > (threshold * 255)
            bitmap_frame = bitmap_frame.astype(np.uint8)
            assert dtypes.is_bitmap(bitmap_frame), bitmap_frame.shape
            step_times['create_bitmap'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Group connected tiles into polyominoes
            step_start = (time.time_ns() / 1e6)
            polyominoes = group_tiles(bitmap_frame)
            step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Sort polyominoes by size
            step_start = (time.time_ns() / 1e6)
            polyominoes = sorted(polyominoes, key=lambda x: x[0].sum(), reverse=True)
            step_times['sort_polyominoes'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Try compressing polyominoes
            step_start = (time.time_ns() / 1e6)
            positions = None if full else pack_append(polyominoes, grid_height, grid_width, occupied_tiles)

            if positions is not None:
                step_times['pack_append'] = (time.time_ns() / 1e6) - step_start
                canvas = apply_pack(positions, canvas, index_map, offset_lookup,
                                    frame_idx, frame, tile_size, step_times)
                clean = False
            else:
                # If compression fails, save current compressed image and start new one
                step_times['pack_append'] = (time.time_ns() / 1e6) - step_start
                
                # Profile: Save compressed image
                save_packed_image(canvas, index_map, offset_lookup, start_idx, frame_idx, output_dir, step_times)
                count_compressed_images += 1
                
                # Profile: Reset variables
                step_start = (time.time_ns() / 1e6)
                canvas, occupied_tiles, index_map, offset_lookup, clean, full = init_compression_variables()
                start_idx = frame_idx
                step_times['reset_variables'] = (time.time_ns() / 1e6) - step_start

                # Profile: Retry compression for current frame
                step_start = (time.time_ns() / 1e6)
                positions = pack_append(polyominoes, grid_height, grid_width, occupied_tiles)
                if positions is not None:
                    step_times['pack_append_retry'] = (time.time_ns() / 1e6) - step_start
                    canvas = apply_pack(positions, canvas, index_map, offset_lookup,
                                        frame_idx, frame, tile_size, step_times)
                    clean = False
                else:
                    step_times['pack_append_retry'] = (time.time_ns() / 1e6) - step_start

                    # If retry compression fails, save the entire frame as a single polyomino
                    print(f"Failed to compress frame {frame_idx} even after reset, saving entire frame as single polyomino")
                    
                    # Render the entire frame onto canvas
                    # canvas = frame
                    # step_times['render_canvas'] = 0

                    # Profile: Update index_map and det_info for the full frame polyomino
                    step_start = (time.time_ns() / 1e6)
                    # Use the next available group ID
                    # TODO: on this case, save index_map as something recognizable like ([0]),
                    # uncompresser should by pass uncompressing and use detection as is.
                    _index_map = np.ones((grid_height, grid_width), dtype=np.uint16)
                    assert dtypes.is_index_map(_index_map), _index_map.shape
                    _offset_lookup = [((0, 0), (0, 0), frame_idx)]
                    step_times['update_mapping'] = (time.time_ns() / 1e6) - step_start
                    
                    save_packed_image(frame, _index_map, _offset_lookup, start_idx, frame_idx, output_dir, step_times)
                    count_compressed_images += 1

            # Save profiling data for this frame
            profiling_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_polyominoes': len(polyominoes),
            }
            
            f.write(json.dumps(profiling_data) + '\n')
            if frame_idx % mod == 0:
                command_queue.put((device, {'completed': frame_idx}))
    
    # Release video capture
    cap.release()
    
    # Save final compressed image if exists
    if not clean:
        save_packed_image(canvas, index_map, offset_lookup, start_idx, frame_idx, output_dir, {})
        count_compressed_images += 1

    # print(f"Completed compression for video. Created {count_compressed_images} compressed images.")
    # print(f"Runtime profiling data saved to: {runtime_file}")


def main(args):
    """
    Main function that orchestrates the video tile compression process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves compression results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for compression ('30', '60', '120', or 'all')
            - threshold (float): Threshold for classification probability (0.0 to 1.0)
            - classifiers (list): List of classifier names to use (default: CLASSIFIERS_TO_TEST)
            - clear (bool): Whether to remove and recreate the compressed frames folder
            
    Note:
        - The script expects classification results from 020_exec_classify.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier}_{tile_size}/score/
        - Looks for score.jsonl files
        - Videos are read from {DATA_DIR}/{dataset}/
        - Compressed images are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tile_size}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tile_size}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tile_size}/offset_lookups/
        - When tile_size is 'all', all tile sizes (30, 60, 120) are processed
        - When classifiers is not specified, all classifiers in CLASSIFIERS_TO_TEST are processed
        - If no classification results are found for a video, that video is skipped with a warning
        - Tiles with classification probability > threshold are considered relevant for compression
    """
    mp.set_start_method('spawn', force=True)
    
    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")
    
    # Create tasks list with all video/classifier/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue
        
        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in sorted(video_files):
            video_file_path = os.path.join(dataset_dir, video_file)
            cache_video_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file)

            compressed_frames_base_dir = os.path.join(cache_video_dir, '030_compressed_frames')
            if args.clear and os.path.exists(compressed_frames_base_dir):
                shutil.rmtree(compressed_frames_base_dir)
                print(f"Cleared existing compressed frames folder: {compressed_frames_base_dir}")
            
            for classifier in args.classifiers:
                for tile_size in tile_sizes_to_process:

                    score_file = os.path.join(cache_video_dir, '020_relevancy',
                                              f'{classifier}_{tile_size}', 'score', 'score.jsonl')
                    if not os.path.exists(score_file):
                        print(f"No score file found for {video_file} {classifier} {tile_size}, skipping")
                        continue

                    funcs.append(partial(compress, video_file_path, cache_video_dir,
                                         classifier, tile_size, args.threshold))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.5)
    # num_processes = 8
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=2).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
