#!/usr/local/bin/python

import argparse
import os
import shutil
import re

from polyis.utilities import CACHE_DIR, DATASETS_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(description='Rename groundtruth_{tilesize} directories to Perfect_{tilesize}')
    parser.add_argument('--datasets', required=False, default=DATASETS_CHOICES, nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be renamed without actually doing it')
    return parser.parse_args()


def find_groundtruth_directories(dataset_dir: str) -> list[tuple[str, str, str, str, str]]:
    """
    Find all groundtruth_{tilesize} directories in both execution and evaluation directories.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        
    Returns:
        list[tuple[str, str, str, str, str]]: List of (dir_type, video_file, stage, old_dir_name, tilesize) tuples
    """
    groundtruth_dirs = []
    
    if not os.path.exists(dataset_dir):
        return groundtruth_dirs
    
    # Search execution directories (with video_file subdirectories)
    execution_dir = os.path.join(dataset_dir, 'execution')
    if os.path.exists(execution_dir):
        # Get all video files from the execution directory
        video_files = [f for f in os.listdir(execution_dir) 
                       if os.path.isdir(os.path.join(execution_dir, f))]
        
        for video_file in video_files:
            video_file_path = os.path.join(execution_dir, video_file)
            
            # Look for stage directories within each video file
            for stage in os.listdir(video_file_path):
                stage_path = os.path.join(video_file_path, stage)
                if os.path.isdir(stage_path):
                    # Look for groundtruth_{tilesize} directories within the stage
                    for item in os.listdir(stage_path):
                        item_path = os.path.join(stage_path, item)
                        if os.path.isdir(item_path):
                            # Match groundtruth_{tilesize} pattern
                            match = re.match(r'^groundtruth_(\d+)$', item)
                            if match:
                                tilesize = match.group(1)
                                groundtruth_dirs.append(('execution', video_file, stage, item, tilesize))
    
    # Search evaluation directories (direct stage subdirectories)
    evaluation_dir = os.path.join(dataset_dir, 'evaluation')
    if os.path.exists(evaluation_dir):
        # Look for stage directories directly in evaluation
        for stage in os.listdir(evaluation_dir):
            stage_path = os.path.join(evaluation_dir, stage)
            if os.path.isdir(stage_path):
                # Look for groundtruth_{tilesize} directories within the stage
                for item in os.listdir(stage_path):
                    item_path = os.path.join(stage_path, item)
                    if os.path.isdir(item_path):
                        # Match groundtruth_{tilesize} pattern
                        match = re.match(r'^groundtruth_(\d+)$', item)
                        if match:
                            tilesize = match.group(1)
                            groundtruth_dirs.append(('evaluation', None, stage, item, tilesize))
    
    return groundtruth_dirs


def rename_groundtruth_directories(dataset_name: str, dry_run: bool = False):
    """
    Rename groundtruth_{tilesize} directories to Perfect_{tilesize} in the dataset.
    
    Args:
        dataset_name (str): Name of the dataset to process
        dry_run (bool): Whether to only show what would be renamed without actually doing it
        
    Note:
        This function searches for directories matching the pattern groundtruth_{tilesize}
        and renames them to Perfect_{tilesize}. It reports directories that are not moved
        (either because they don't exist or because of errors during the move operation).
    """
    dataset_dir = os.path.join(CACHE_DIR, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist, skipping...")
        return
    
    print(f"Processing dataset: {dataset_name}")
    
    # Find all groundtruth_{tilesize} directories
    groundtruth_dirs = find_groundtruth_directories(dataset_dir)
    
    if not groundtruth_dirs:
        print(f"No groundtruth directories found in {dataset_dir}")
        return
    
    total_renames = 0
    total_not_moved = 0
    
    for dir_type, video_file, stage, old_dir_name, tilesize in groundtruth_dirs:
        dir_path = os.path.join(dataset_dir, dir_type)
        
        if video_file is not None:
            # Execution directory structure: dataset/execution/video_file/stage/
            video_file_path = os.path.join(dir_path, video_file)
            stage_path = os.path.join(video_file_path, stage)
            print(f"  Processing {dir_type}: {video_file}, stage: {stage}")
        else:
            # Evaluation directory structure: dataset/evaluation/stage/
            stage_path = os.path.join(dir_path, stage)
            print(f"  Processing {dir_type}: stage: {stage}")
        
        old_path = os.path.join(stage_path, old_dir_name)
        new_dir_name = f"Perfect_{tilesize}"
        new_path = os.path.join(stage_path, new_dir_name)
        
        if os.path.exists(old_path):
            # Check if target directory already exists
            if os.path.exists(new_path):
                print(f"    Target directory {new_dir_name} already exists, skipping {old_dir_name}")
                total_not_moved += 1
                continue
            
            if dry_run:
                print(f"    Would rename: {old_dir_name} -> {new_dir_name}")
            else:
                try:
                    shutil.move(old_path, new_path)
                    print(f"    Renamed: {old_dir_name} -> {new_dir_name}")
                except Exception as e:
                    print(f"    Error renaming {old_dir_name} -> {new_dir_name}: {e}")
                    total_not_moved += 1
                    continue
            total_renames += 1
        else:
            print(f"    Directory {old_dir_name} not found, skipping")
            total_not_moved += 1
    
    print(f"  Total directories {'would be' if dry_run else ''} renamed: {total_renames}")
    if total_not_moved > 0:
        print(f"  Total directories not {'would be' if dry_run else ''} moved: {total_not_moved}")


def main(args):
    """
    Main function that orchestrates the groundtruth to Perfect directory renaming process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset execution directories exist
    2. Finds all groundtruth_{tilesize} directories in each dataset
    3. Renames them to Perfect_{tilesize}
    4. Provides feedback on the renaming process
    5. Reports directories that are not moved (missing or errors)
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - dry_run (bool): Whether to only show what would be renamed
            
    Note:
        - The script looks for directories in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/{stage}/groundtruth_{tilesize}
          {CACHE_DIR}/{dataset}/evaluation/{stage}/groundtruth_{tilesize}
        - Renames them to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/{stage}/Perfect_{tilesize}
          {CACHE_DIR}/{dataset}/evaluation/{stage}/Perfect_{tilesize}
        - Use --dry-run to preview changes without making them
        - Existing directories with the new names will cause errors
        - Reports directories that are not moved with reasons (not found, errors)
    """
    print("Groundtruth to Perfect Directory Renaming Script")
    print("=" * 50)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
        print()
    
    for dataset_name in args.datasets:
        rename_groundtruth_directories(dataset_name, args.dry_run)
        print()
    
    if args.dry_run:
        print("Dry run completed. Use without --dry-run to perform actual renaming.")
    else:
        print("Directory renaming completed!")


if __name__ == '__main__':
    main(parse_args())