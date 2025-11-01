#!/usr/local/bin/python

import argparse
import json
import os
import sys
from typing import Callable, Literal
import cv2
import numpy as np
import shutil
import time
import multiprocessing as mp
import queue
from functools import partial

import torch
import torch.nn.functional as F

from polyis import dtypes
from polyis.utilities import (
    CACHE_DIR, CLASSIFIERS_CHOICES,
    DATASETS_DIR, TILEPADDING_MODES, format_time,
    load_classification_results,
    CLASSIFIERS_TO_TEST, ProgressBar, DATASETS_TO_TEST, TILE_SIZES
)
from polyis.binpack.pack_append import pack_append
from polyis.binpack.group_tiles import free_polyimino_stack, group_tiles


def parse_args():
    parser = argparse.ArgumentParser(description='Execute compression of video tiles into images based on classification results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tilesize', type=str, choices=['30', '60', '120', 'all'], default='all',
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
    parser.add_argument('--tilepadding', type=str, choices=['none', 'connected', 'disconnected'],
                        nargs='+', default=['none', 'connected', 'disconnected'],
                        help='Apply padding to the classification results (space-separated list of none/connected/disconnected)')
    return parser.parse_args()


def render(canvas: dtypes.NPImage, positions: list[dtypes.PolyominoPositions], 
           frame: dtypes.NPImage, tile_size: int):
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
    ts = tile_size

    # TODO: use Torch, Cython, or Numba.
    for y, x, mask, offset in positions:
        yfrom = y * ts
        xfrom = x * ts
        yoffset, xoffset = offset
        
        # Get mask indices where True
        for i, j in mask.reshape(-1, 2).astype(np.uint16):
            sy = (i + yoffset) * ts
            sx = (j + xoffset) * ts

            dy = yfrom + (ts * i)
            dx = xfrom + (ts * j)

            canvas[dy:dy+ts, dx:dx+ts] = frame[sy:sy+ts, sx:sx+ts]

def apply_pack(
    positions: list[dtypes.PolyominoPositions],
    canvas: dtypes.NPImage,
    index_map: dtypes.IndexMap,
    offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
    frame_idx: int,
    frame: dtypes.NPImage,
    tilesize: int,
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
        tilesize: The size of the tile
        step_times: The step times to update
        
    Returns:
        dtypes.NPImage: canvas
    """
    # Profile: Render packed polyominoes onto canvas
    step_start = (time.time_ns() / 1e6)
    render(canvas, positions, frame, tilesize)
    step_times['render_canvas'] = (time.time_ns() / 1e6) - step_start

    assert index_map is not None
    
    # Profile: Update index_map and det_info
    step_start = (time.time_ns() / 1e6)
    # Start gid from the current length of offset_lookup
    start_gid = len(offset_lookup)
    for i, (y, x, mask, offset) in enumerate(positions):
        gid = start_gid + i + 1
        # mask is a 1D array of positions in format [x, y, x, y, x, y, ...]
        # Reshape to get x and y coordinates as separate arrays
        coords = mask.reshape(-1, 2).astype(np.uint16)  # Shape: (n_coords, 2) where each row is [x, y]
        mask_y = coords[:, 0]  # All y coordinates
        mask_x = coords[:, 1]  # All x coordinates
        # Set the gid at all positions at once using vectorized operations
        try:
            index_map[y + mask_y, x + mask_x] = gid
        except IndexError as e:
            print(f"Error: {y}, {x}, {offset[0]}, {offset[1]}")
            print(f"Mask: {mask}")
            raise e
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


PolyominoPosition = tuple[int, int, int, int, int, int, np.ndarray]
Collage = list[PolyominoPosition]


def renderer(video_file: str, collage_queue: "queue.Queue[Collage]"):
    """
    Renderer process that consumes packed collages from a queue and saves them to disk.
    
    Args:
        video_file: The video file name for logging purposes
        collage_queue: Queue containing packed collages to render
    """
    cap = cv2.VideoCapture(video_file)
    frames: list[dtypes.NPImage] = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            assert idx == num_frames - 1, f"Failed to read frame {idx} of {video_file}"
            break
        assert dtypes.is_np_image(frame), frame.shape
        frames.append(frame)
    assert cap.read()[0] is False, "Expected no more frames"
    cap.release()

    while True:
        collage_data = collage_queue.get()
        if collage_data is None:
            break
        
        # Unpack collage data
        ox, oy, px, py, _rotation, frame, shape = collage_data
        
        # Save the packed image (implement saving logic here)
        # For example, save to a predefined output directory
        output_dir = os.path.join(CACHE_DIR, 'rendered_collages', os.path.basename(video_file))
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, f'collage_{start_idx:08d}_{end_idx:08d}.jpg')
        cv2.imwrite(img_path, canvas)
        print(f"Saved collage {img_path}")



def compress(video_file_path: str, cache_video_dir: str, classifier: str, tilesize: int,
             threshold: float, tilepadding: Literal['none', 'connected', 'disconnected'], gpu_id: int, command_queue: mp.Queue):
    """
    Compress a single video with a specific classifier and tile size.
    
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
    output_dir = os.path.join(cache_video_dir, '030_compressed_frames', f'{classifier}_{tilesize}_{tilepadding}')
    if os.path.exists(output_dir):
        # Remove the entire directory
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Send initial progress update
    description = f"{video_name} {tilesize:>3} {classifier} {tilepadding}"
    command_queue.put((device, {
        'description': description + ' 0',
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
    grid_height = height // tilesize
    grid_width = width // tilesize

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
    
    # print(f"Processing {len(results)} frames with tile size {tilesize}")

    add_margin = torch.tensor(
        [[[[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]]],
        dtype=torch.uint8,
        requires_grad=False,
    )

    with open(runtime_file, 'w') as f:
        # Process each frame
        fail_count = 0
        mod = int(len(results) * 0.01)
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
            polyominoes = group_tiles(bitmap_frame, TILEPADDING_MODES[tilepadding])
            step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start
            
            # # Profile: Sort polyominoes by size
            # step_start = (time.time_ns() / 1e6)
            # polyominoes = sorted(polyominoes, key=lambda x: x[0].sum(), reverse=True)
            # step_times['sort_polyominoes'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Try compressing polyominoes
            step_start = (time.time_ns() / 1e6)
            positions = None if full else pack_append(polyominoes, grid_height, grid_width, occupied_tiles)

            if positions is not None:
                step_times['pack_append'] = (time.time_ns() / 1e6) - step_start
                canvas = apply_pack(positions, canvas, index_map, offset_lookup,
                                    frame_idx, frame, tilesize, step_times)
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
                                        frame_idx, frame, tilesize, step_times)
                    clean = False
                else:
                    step_times['pack_append_retry'] = (time.time_ns() / 1e6) - step_start

                    # If retry compression fails, save the entire frame as a single polyomino
                    fail_count += 1
                    # print(f"Failed to compress frame {frame_idx} even after reset, saving entire frame as single polyomino")
                    
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

            num_polyominoes = free_polyimino_stack(polyominoes)

            # Save profiling data for this frame
            profiling_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_polyominoes': num_polyominoes,
            }
            
            f.write(json.dumps(profiling_data) + '\n')
            if frame_idx % mod == 0:
                command_queue.put((device, {'description': description + f' {fail_count:>3}', 'completed': frame_idx}))
    
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
        - Compressed images are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tilesize}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tilesize}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/030_compressed_frames/{classifier}_{tilesize}/offset_lookups/
        - When tilesize is 'all', all tile sizes (30, 60, 120) are processed
        - When classifiers is not specified, all classifiers in CLASSIFIERS_TO_TEST are processed
        - If no classification results are found for a video, that video is skipped with a warning
        - Tiles with classification probability > threshold are considered relevant for compression
    """
    mp.set_start_method('spawn', force=True)
    
    # Determine which tile sizes to process
    if args.tilesize == 'all':
        tilesizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tilesizes_to_process}")
    else:
        tilesizes_to_process = [int(args.tilesize)]
        print(f"Processing tile size: {tilesizes_to_process[0]}")
    
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

                compressed_frames_base_dir = os.path.join(cache_video_dir, '030_compressed_frames')
                if args.clear and os.path.exists(compressed_frames_base_dir):
                    shutil.rmtree(compressed_frames_base_dir)
                    print(f"Cleared existing compressed frames folder: {compressed_frames_base_dir}")
                
                for classifier in args.classifiers:
                    for tilesize in tilesizes_to_process:
                        score_file = os.path.join(cache_video_dir, '020_relevancy',
                                                f'{classifier}_{tilesize}', 'score', 'score.jsonl')
                        if not os.path.exists(score_file):
                            print(f"No score file found for {video_file} {classifier} {tilesize}, skipping")
                            continue

                        for tilepadding in set[Literal['none', 'connected', 'disconnected']](args.tilepadding):
                            funcs.append(partial(compress, video_file_path, cache_video_dir,
                                                classifier, tilesize, args.threshold, tilepadding))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.1)
    # num_processes = max(1, torch.cuda.device_count() // 2)
    # num_processes = 1
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=2).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
