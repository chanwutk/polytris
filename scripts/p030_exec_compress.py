#!/usr/local/bin/python

import argparse
from enum import IntEnum
import json
import os
from typing import Callable, NamedTuple
import cv2
import numpy as np
import shutil
import time
import multiprocessing as mp
from functools import partial

import torch

from polyis import dtypes
from polyis.utilities import format_time, load_classification_results, ProgressBar, get_config, TILEPADDING_MAPS, TilePadding
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = config['EXEC']['CLASSIFIERS']
DATASETS = config['EXEC']['DATASETS']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']
TILEPADDING_MODES = config['EXEC']['TILEPADDING_MODES']


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


OUTPUT_DIR_MAP = {
    PackMode.Best_Fit: '033_compressed_frames',
    PackMode.Easiest_Fit: '034_compressed_frames',
    PackMode.First_Fit: '035_compressed_frames',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Execute compression of video tiles into images based on classification results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification probability (0.0 to 1.0)')
    parser.add_argument('--mode', type=lambda x: PackMode[x], 
                        default=PackMode.Best_Fit,
                        help='Packing mode for the pack_all function. Options: Easiest_Fit, First_Fit, Best_Fit (default: Best_Fit)')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
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


def compress(dataset: str, videoset: str, video: str, classifier: str, tilesize: int,
             sample_rate: int, tilepadding: TilePadding, threshold: float, mode: PackMode,
             gpu_id: int, command_queue: mp.Queue):
    """
    Compress a single video by batch processing all sampled frames at once using pack_all.

    Args:
        dataset: Name of the dataset
        videoset: Videoset name (test, train, or valid)
        video: Name of the video
        classifier: Classifier name to use
        tilesize: Tile size to use
        tilepadding: Whether to apply padding to classification results
        sample_rate: Sample rate for frame sampling (1 = all frames)
        threshold: Threshold for classification probability
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    video_name = video
    cache_video_dir = os.path.join(CACHE_DIR, dataset, 'execution', video)
    video_path = os.path.join(DATASETS_DIR, dataset, videoset, video)

    # Load classification results for the specified sample rate
    results = load_classification_results(CACHE_DIR, dataset, video, tilesize, classifier, sample_rate)

    # Create output directory for compression results
    output_dir_name = OUTPUT_DIR_MAP[mode]
    output_dir = os.path.join(cache_video_dir, output_dir_name, f'{classifier}_{tilesize}_{sample_rate}_{tilepadding}')
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
    
    # Open video to get dimensions
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error: Could not open video {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Note: results contains only sampled frames (frame_idx % sample_rate == 0)
    # The total video frame count may be larger than len(results)

    # Calculate grid dimensions
    grid_height = height // tilesize
    grid_width = width // tilesize

    # Step 1: Group tiles for all frames to get polyominoes
    timing_data = []

    # Create mapping from array index to absolute frame index
    # This is CRITICAL: results contains only sampled frames, but we need absolute frame indices
    array_idx_to_frame_idx = {i: result['idx'] for i, result in enumerate(results)}

    polyominoes_stacks = np.empty(len(results), dtype=np.uint64)
    for array_idx, frame_result in enumerate(results):
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
        polyominoes = group_tiles(bitmap_frame, TILEPADDING_MAPS[tilepadding])
        polyominoes_stacks[array_idx] = polyominoes
        step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start

        # Use absolute frame index in timing data
        absolute_frame_idx = array_idx_to_frame_idx[array_idx]
        timing_data.append({'step': 'group_tiles', 'frame_idx': absolute_frame_idx, 'runtime': format_time(**step_times)})

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
        batch_collages_ = pack(batch_polyominoes, grid_height, grid_width, int(mode))
        batch_pack_time = (time.time_ns() / 1e6) - batch_start
        total_pack_time += batch_pack_time

        # Adjust frame indices in batch_collages to use absolute frame indices
        # pack_all returns frame indices relative to the batch (0-indexed within batch)
        # We need to map these through our array_idx_to_frame_idx mapping to get absolute frame indices
        batch_collages: list[list[PolyominoPosition]] = []
        for collage in batch_collages_:
            batch_collages.append([
                PolyominoPosition(oy=poly_pos.oy, ox=poly_pos.ox,
                                  py=poly_pos.py, px=poly_pos.px,
                                  # Map from batch-relative index to absolute frame index
                                  frame=array_idx_to_frame_idx[poly_pos.frame + start_idx],
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

    # Step 3: Read ONLY sampled frames from video (only frames in results)
    command_queue.put((device, {'description': description + ' reading', 'completed': 0, 'total': len(results)}))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create set of sampled frame indices for fast lookup
    sampled_indices_set = {result['idx'] for result in results}
    # Create mapping from absolute frame index to frame data
    # This allows us to access frames by their absolute index in the original video
    frame_idx_to_frame: dict[int, np.ndarray] = {}
    mod = max(1, len(results) // 20)
    frames_read = 0

    for idx in range(num_frames_total):
        ret, frame = cap.read()
        if not ret:
            break

        # Only store frames that are in the sampled set
        if idx in sampled_indices_set:
            assert dtypes.is_np_image(frame), frame.shape
            frame_idx_to_frame[idx] = frame
            frames_read += 1
            if frames_read % mod == 0:
                command_queue.put((device, {'description': description + ' reading',
                                           'completed': frames_read}))

    assert len(frame_idx_to_frame) == len(results), f"Expected {len(results)} sampled frames, got {len(frame_idx_to_frame)}"
    assert cap.read()[0] is False, "Expected no more frames"
    cap.release()

    # Step 4: Render and save each collage
    command_queue.put((device, {'description': description + ' rendering', 'completed': 0, 'total': len(collages)}))

    mod = max(1, len(collages) // 20)
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

            # Get source frame using absolute frame index
            frame = frame_idx_to_frame[frame_idx]

            # Tile rendering: scale grid positions to raw video resolution
            i_coords = shape[:, 0]
            j_coords = shape[:, 1]

            # Compute tile boundaries scaled to raw video resolution (vectorized)
            sy_starts = (oy + i_coords) * height // grid_height
            sx_starts = (ox + j_coords) * width // grid_width
            sy_ends = (oy + i_coords + 1) * height // grid_height
            sx_ends = (ox + j_coords + 1) * width // grid_width
            dy_starts = (py + i_coords) * height // grid_height
            dx_starts = (px + j_coords) * width // grid_width
            dy_ends = (py + i_coords + 1) * height // grid_height
            dx_ends = (px + j_coords + 1) * width // grid_width

            # Precompute tile sizes and padding (vectorized)
            dst_hs = (dy_ends - dy_starts).astype(int)
            dst_ws = (dx_ends - dx_starts).astype(int)
            src_hs = (sy_ends - sy_starts).astype(int)
            src_ws = (sx_ends - sx_starts).astype(int)
            pad_hs = np.maximum(0, dst_hs - src_hs)
            pad_ws = np.maximum(0, dst_ws - src_ws)
            needs_padding = (pad_hs > 0) | (pad_ws > 0)

            # Copy tiles, repeating last row/column if sizes differ
            for idx in range(len(shape)):
                src_tile = frame[sy_starts[idx]:sy_ends[idx], sx_starts[idx]:sx_ends[idx]]
                # Pad with repeated edge pixels if source tile is smaller than destination
                if needs_padding[idx]:
                    src_tile = np.pad(src_tile, ((0, pad_hs[idx]), (0, pad_ws[idx]), (0, 0)), mode='edge')
                canvas[dy_starts[idx]:dy_ends[idx], dx_starts[idx]:dx_ends[idx]] = src_tile[:dst_hs[idx], :dst_ws[idx]]

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
        if collage_idx % mod == 0:
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
            - threshold (float): Threshold for classification probability (0.0 to 1.0)
            - mode (PackMode): Packing mode for the pack_all function. Options: Easiest_Fit, First_Fit, Best_Fit (default: Best_Fit)
            
    Note:
        - The script expects classification results from 020_exec_classify.py in:
          {CACHE_DIR}/{dataset}/execution/{video}/020_relevancy/{classifier}_{tilesize}/score/
        - Looks for score.jsonl files
        - Videos are read from {DATASETS_DIR}/{dataset}/
        - Compressed images are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/offset_lookups/
        - If no classification results are found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)
    threshold = args.threshold
    mode = args.mode
    
    # Determine which videosets to process based on arguments
    videosets = []
    if args.test:
        videosets.append('test')
    if args.train:
        videosets.append('train')
    if args.valid:
        videosets.append('valid')
    
    # If no videosets are specified, default to all three
    if not videosets:
        videosets = ['test']
    
    # Create tasks list with all video/classifier/tilesize/sample_rate combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)

        for videoset in videosets:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Videoset directory {videoset_dir} does not exist, skipping...")
                continue

            # Get all video files from the dataset directory
            videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video in sorted(videos):
                for classifier in CLASSIFIERS:
                    for tilesize in TILE_SIZES:
                        for tilepadding in TILEPADDING_MODES:
                            for sample_rate in SAMPLE_RATES:
                                funcs.append(partial(compress, dataset, videoset, video, classifier, tilesize, sample_rate, tilepadding, threshold, mode))
    
    print(f"Created {len(funcs)} tasks to process")
    
    ProgressBar(num_workers=torch.cuda.device_count(), num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
