#!/usr/local/bin/python

import argparse
import json
import os
import cv2
import numpy as np
from rich.progress import track
import shutil
import time

from scripts.utilities import CACHE_DIR, DATA_DIR, format_time, load_classification_results
from lib.pack_append import pack_append
from lib.group_tiles import group_tiles


# TILE_SIZES = [30, 60, 120]
TILE_SIZES = [60]


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for packing (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification probability (default: 0.5)
            - classifier (str): Classifier name to use (default: 'SimpleCNN')
            - clear (bool): Whether to remove and recreate the packing folder (default: False)
    """
    parser = argparse.ArgumentParser(description='Execute packing of video tiles into images based on classification results')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for packing (or "all" for all tile sizes)')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification probability (0.0 to 1.0)')
    parser.add_argument('--classifier', type=str, default='SimpleCNN',
                        help='Classifier name to use (default: SimpleCNN)')
    parser.add_argument('--clear', action='store_true',
                        help='Remove and recreate the packing folder')
    return parser.parse_args()


def render(canvas: np.ndarray, positions: list[tuple[int, int, int, np.ndarray, tuple[int, int]]], 
           frame: np.ndarray, chunk_size: int) -> np.ndarray:
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
    for y, x, _groupid, mask, offset in positions:
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


def apply_pack(pack_results: tuple[np.ndarray, list], canvas: np.ndarray, index_map: np.ndarray,
               offset_lookup: dict, frame_idx: int, frame: np.ndarray, tile_size: int, step_times: dict):
    """
    Apply packed results to the canvas, index_map, and offset_lookup.
    
    Args:
        pack_results: Tuple of (bitmap, positions)
        canvas: The canvas to render onto
        index_map: The index map to update
        offset_lookup: The offset lookup to update
        frame_idx: The index of the frame
        frame: The frame to render
        tile_size: The size of the tile
        step_times: The step times to update
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (bitmap, canvas)
    """
    bitmap, positions = pack_results
    
    # Profile: Render packed polyominoes onto canvas
    step_start = (time.time_ns() / 1e6)
    canvas = render(canvas, positions, frame, tile_size)
    step_times['render_canvas'] = (time.time_ns() / 1e6) - step_start

    assert index_map is not None
    
    # Profile: Update index_map and det_info
    step_start = (time.time_ns() / 1e6)
    for gid, (y, x, mask, _offset) in enumerate(positions):
        assert not np.any(index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] & mask), (index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0], mask)
        index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * (gid + 1)
        index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * frame_idx
        offset_lookup[(int(frame_idx), int(gid + 1))] = ((y, x), _offset)
    step_times['update_mapping'] = (time.time_ns() / 1e6) - step_start

    return bitmap, canvas


def save_packed_image(canvas: np.ndarray, index_map: np.ndarray, offset_lookup: dict,
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

    offset_lookup_path = os.path.join(offset_lookup_dir, f'{start_idx:08d}_{frame_idx:08d}.json')
    with open(offset_lookup_path, 'w') as f:
        json.dump({str(k): v for k, v in offset_lookup.items()}, f, indent=2)
    step_times['save_mapping_files'] = (time.time_ns() / 1e6) - step_start


def compress_video(video_path: str, results: list, tile_size: int, output_dir: str, threshold: float = 0.5):
    """
    Process a single video for packing based on classification results.
    
    Args:
        video_path (str): Path to the input video file
        results (list): List of classification results
        tile_size (int): Tile size used for processing
        output_dir (str): Directory to save packing results
        threshold (float): Threshold for classification probability (0.0 to 1.0)
    """
    print(f"Processing video for packing: {video_path}")
    
    # Open video to get dimensions and initialize capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
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

    def init_packing_variables():
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        occupied_tiles = np.zeros((grid_height, grid_width), dtype=np.uint8)
        index_map = np.zeros((grid_height, grid_width, 2), dtype=np.int32)
        offset_lookup: dict = {}
        return canvas, occupied_tiles, index_map, offset_lookup, True, False
    
    # Initialize packing variables
    canvas, occupied_tiles, index_map, offset_lookup, clean, full = init_packing_variables()
    frame_cache = {}
    start_idx = 0
    last_frame_idx = -1
    read_frame_idx = -1
    frame_idx = -1
    
    # Initialize profiling output file
    runtime_file = os.path.join(output_dir, 'runtime.jsonl')
    
    print(f"Processing {len(results)} frames with tile size {tile_size}")

    with open(runtime_file, 'w') as f:
        # Process each frame
        for frame_idx, frame_result in enumerate(track(results, description="Packing frames")):
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
            frame_cache[frame_idx] = frame
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
            step_times['create_bitmap'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Group connected tiles into polyominoes
            step_start = (time.time_ns() / 1e6)
            polyominoes = group_tiles(bitmap_frame)
            step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Sort polyominoes by size
            step_start = (time.time_ns() / 1e6)
            polyominoes = sorted(polyominoes, key=lambda x: x[1].sum(), reverse=True)
            step_times['sort_polyominoes'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Try packing polyominoes
            step_start = (time.time_ns() / 1e6)
            positions = None if full else pack_append(polyominoes, grid_height, grid_width, occupied_tiles)

            if positions is not None:
                step_times['pack_append'] = (time.time_ns() / 1e6) - step_start
                pack_results = (occupied_tiles, positions)  # Create tuple for apply_pack
                occupied_tiles, canvas = apply_pack(pack_results, canvas, index_map, offset_lookup,
                                                    frame_idx, frame, tile_size, step_times)
                clean = False
            else:
                # If packing fails, save current packed image and start new one
                step_times['pack_append'] = (time.time_ns() / 1e6) - step_start
                
                # Profile: Save packed image
                save_packed_image(canvas, index_map, offset_lookup, start_idx, frame_idx, output_dir, step_times)
                
                # Profile: Reset variables
                step_start = (time.time_ns() / 1e6)
                canvas, occupied_tiles, index_map, offset_lookup, clean, full = init_packing_variables()
                frame_cache = {frame_idx: frame}
                start_idx = frame_idx
                step_times['reset_variables'] = (time.time_ns() / 1e6) - step_start

                # Profile: Retry packing for current frame
                step_start = (time.time_ns() / 1e6)
                positions = pack_append(polyominoes, grid_height, grid_width, occupied_tiles)
                if positions is not None:
                    step_times['pack_append_retry'] = (time.time_ns() / 1e6) - step_start
                    pack_results = (occupied_tiles, positions)  # Create tuple for apply_pack
                    occupied_tiles, canvas = apply_pack(pack_results, canvas, index_map, offset_lookup,
                                                        frame_idx, frame, tile_size, step_times)
                    clean = False
                else:
                    step_times['pack_append_retry'] = (time.time_ns() / 1e6) - step_start

                    # If retry packing fails, save the entire frame as a single polyomino
                    print(f"Failed to pack frame {frame_idx} even after reset, saving entire frame as single polyomino")
                    
                    # Render the entire frame onto canvas
                    # canvas = frame
                    # step_times['render_canvas'] = 0

                    # Profile: Update index_map and det_info for the full frame polyomino
                    step_start = (time.time_ns() / 1e6)
                    _index_map = np.ones((grid_height, grid_width, 2), dtype=np.int32)
                    _index_map[0:grid_height, 0:grid_width, 1] = frame_idx
                    _offset_lookup = {(frame_idx, 1): ((0, 0), (0, 0)) }
                    step_times['update_mapping'] = (time.time_ns() / 1e6) - step_start
                    
                    save_packed_image(frame, _index_map, _offset_lookup, start_idx, frame_idx, output_dir, step_times)
            
            # Save profiling data for this frame
            profiling_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_polyominoes': len(polyominoes),
            }
            
            f.write(json.dumps(profiling_data) + '\n')
    
    # Release video capture
    cap.release()
    
    # Save final packed image if exists
    if not clean:
        save_packed_image(canvas, index_map, offset_lookup, start_idx, frame_idx, output_dir, {})

    print(f"Completed packing for video. Created {len(results) - start_idx} packed images.")
    print(f"Runtime profiling data saved to: {runtime_file}")


def main(args):
    """
    Main function that orchestrates the video tile packing process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Iterates through all videos in the dataset directory
    3. For each video, loads classification results for the specified tile size(s)
    4. Groups connected tiles into polyominoes and packs them into images
    5. Saves packed images and their mappings
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for packing ('64', '128', or 'all')
            - threshold (float): Threshold for classification probability (0.0 to 1.0)
            - classifier (str): Classifier name to use (default: 'SimpleCNN')
            - clear (bool): Whether to remove and recreate the packing folder
            
    Note:
        - The script expects classification results from 020_exec_classify.py in:
          {CACHE_DIR}/{dataset}/{video_file}/relevancy/{classifier}_{tile_size}/score/
        - Looks for score.jsonl files
        - Videos are read from {DATA_DIR}/{dataset}/
        - Packed images are saved to {CACHE_DIR}/{dataset}/{video_file}/packing/{classifier}_{tile_size}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/{video_file}/packing/{classifier}_{tile_size}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/{video_file}/packing/{classifier}_{tile_size}/offset_lookups/
        - When tile_size is 'all', all two tile sizes (30, 60, 120) are processed
        - If no classification results are found for a video, that video is skipped with a warning
        - Tiles with classification probability > threshold are considered relevant for packing
    """
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
    
    # Process each video file
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)

        print(f"Processing video file: {video_file}")
        
        # Clear packing folder once per video if --clear flag is set
        if args.clear:
            packing_base_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'packing')
            if os.path.exists(packing_base_dir):
                shutil.rmtree(packing_base_dir)
                print(f"Cleared existing packing folder: {packing_base_dir}")
        
        # Process each tile size for this video
        for tile_size in tile_sizes_to_process:
            print(f"Processing tile size: {tile_size}")
            
            # Load classification results
            results = load_classification_results(CACHE_DIR, args.dataset, video_file, tile_size, args.classifier)
            
            # Create output directory for packing results
            packing_output_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'packing', f'{args.classifier}_{tile_size}')
            if os.path.exists(packing_output_dir):
                # Remove the entire directory
                shutil.rmtree(packing_output_dir)
            os.makedirs(packing_output_dir, exist_ok=True)
            
            # Process the video for packing
            compress_video(video_file_path, results, tile_size, packing_output_dir, args.threshold)
            
            print(f"Completed packing for tile size {tile_size}")


if __name__ == '__main__':
    main(parse_args())
