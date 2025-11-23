#!/usr/local/bin/python

import os
import json
import glob
import subprocess
import numpy as np
from functools import partial

from polyis.utilities import ProgressBar, get_config

config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_TO_TEST = config['EXEC']['DATASETS']
TILE_SIZES = config['EXEC']['TILE_SIZES']


def optimize_training_data(dataset_name: str, tile_size: int, gpu_id: int, command_queue):
    """
    Optimize training data by removing images for tiles that have never been relevant across the entire dataset.
    
    This function:
    1. Loads all always_relevant_tiles bitmaps for all videos in the dataset
    2. Combines them to find tiles that have never been relevant across the entire dataset
    3. Removes all training images for those tile positions from all videos in the dataset
    
    Args:
        dataset_name: Name of the dataset
        tile_size: Tile size to process
        command_queue: Queue for progress updates
    """
    # Construct paths
    cache_dir = os.path.join(CACHE_DIR, dataset_name)
    always_relevant_dir = os.path.join(cache_dir, 'indexing', 'always_relevant')
    training_base_dir = os.path.join(cache_dir, 'indexing', 'training')
    training_data_path = os.path.join(training_base_dir, 'data', f'tilesize_{tile_size}')
    
    # Check if always_relevant directory exists
    if not os.path.exists(always_relevant_dir):
        return
    
    # Get list of videos from always_relevant_tiles files
    relevancy_files = [
        f
        for f in os.listdir(always_relevant_dir)
        if f.endswith('.npy') and f.startswith(f'{tile_size}_')
    ]
    
    if len(relevancy_files) == 0:
        return
    
    # Load and combine all always_relevant_tiles bitmaps for the dataset
    combined_always_relevant_tiles = None
    for relevancy_file in relevancy_files:
        always_relevant_tiles_path = os.path.join(always_relevant_dir, relevancy_file)
        
        if not os.path.exists(always_relevant_tiles_path):
            continue
        
        # Load the always_relevant_tiles bitmap for this video
        relevancy = np.load(always_relevant_tiles_path)
        
        # Combine with other videos using OR operation
        if combined_always_relevant_tiles is None:
            combined_always_relevant_tiles = relevancy.copy()
        else:
            # Ensure shapes match (they should for same tile_size)
            assert combined_always_relevant_tiles.shape == relevancy.shape, f"Shapes do not match for {relevancy_file}"
            combined_always_relevant_tiles |= relevancy
    
    if combined_always_relevant_tiles is None:
        return
    
    # Find tile positions that have never been relevant across the entire dataset (0 values)
    never_relevant_positions = np.where(combined_always_relevant_tiles == 0)
    never_relevant_yx = list(zip(never_relevant_positions[0], never_relevant_positions[1]))
    
    if len(never_relevant_yx) == 0:
        # All tiles were relevant at some point across the dataset, nothing to remove
        return
    
    # Count files before removal
    num_files_before = 0
    for label in ['pos', 'neg']:
        label_dir = os.path.join(training_data_path, label)
        if os.path.exists(label_dir):
            num_files_before += int(subprocess.check_output(f'ls -1 {label_dir} | wc -l', shell=True, text=True).strip())
    
    # Process both pos and neg directories
    for label in ['pos', 'neg']:
        label_dir = os.path.join(training_data_path, label)
        
        if not os.path.exists(label_dir):
            continue
        
        # For each never-relevant tile position, find and remove matching files from all videos
        for y, x in never_relevant_yx:
            # Pattern: *_{y}_{x}.jpg matches any video_file and any frame_idx
            # The pattern matches: {video_file}_{frame_idx}_{y}_{x}.jpg
            pattern = os.path.join(label_dir, f'*_*_{y}_{x}.jpg')
            subprocess.run(f'rm {pattern}', shell=True, check=False)
    
    # Count files after removal
    num_files_after = 0
    for label in ['pos', 'neg']:
        label_dir = os.path.join(training_data_path, label)
        if os.path.exists(label_dir):
            num_files_after += int(subprocess.check_output(f'ls -1 {label_dir} | wc -l', shell=True, text=True).strip())
    
    # Save statistics to indexing directory
    stats_path = os.path.join(always_relevant_dir, f'{tile_size}_optimization_stats.json')
    
    # Update stats for this tile size
    stats = {
        'files_before': num_files_before,
        'files_after': num_files_after,
        'files_removed': num_files_before - num_files_after,
        'percentage_removed': (num_files_before - num_files_after) / num_files_before * 100,
        'never_relevant_tiles': len(never_relevant_yx)
    }
    
    # Save updated stats
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    np.save(os.path.join(always_relevant_dir, f'{tile_size}_all.npy'), combined_always_relevant_tiles)


def main():
    """
    Main function to optimize training data by removing images for tiles that have never been relevant.
    
    This function:
    1. Iterates through each dataset and tile size combination
    2. Loads all always_relevant_tiles bitmaps for all videos in each dataset
    3. Combines them to find tile positions that have never been relevant across the entire dataset
    4. Removes all training images for those positions from all videos in the dataset
    
    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/indexing/always_relevant/{video_file}.npy (always_relevant_tiles bitmap per video)
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/pos/ (positive training images)
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/neg/ (negative training images)
        
        Images are removed based on the naming convention: {video_file}_{frame_idx}_{y}_{x}.jpg
        where (y, x) are tile positions that have never been marked as relevant across the entire dataset.
    """
    funcs = []
    
    for dataset_name in DATASETS_TO_TEST:
        cache_dir = os.path.join(CACHE_DIR, dataset_name)
        always_relevant_dir = os.path.join(cache_dir, 'indexing', 'always_relevant')
        
        # Check if always_relevant directory exists
        if not os.path.exists(always_relevant_dir):
            continue
        
        # Create task functions for each dataset and tile size combination
        for tile_size in TILE_SIZES:
            funcs.append(partial(optimize_training_data, dataset_name, tile_size))
    
    if len(funcs) == 0:
        print("No training data to optimize. Make sure p012_tune_create_training_data.py has been run first.")
        return
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=20, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main()

