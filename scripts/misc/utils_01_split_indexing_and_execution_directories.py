#!/usr/local/bin/python

import argparse
import os
import shutil
from pathlib import Path

from polyis.utilities import CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description='Split b3d cache directory structure into indexing and execution directories')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be moved without actually doing it')
    return parser.parse_args()


def get_video_name_without_extension(video_file: str) -> str:
    """
    Extract video name without extension.
    
    Args:
        video_file (str): Video filename with extension
        
    Returns:
        str: Video name without extension
    """
    return Path(video_file).stem


def categorize_directories(directory_names: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorize directories into indexing and execution steps.
    
    Args:
        directory_names (list[str]): List of directory names to categorize
        
    Returns:
        tuple[list[str], list[str]]: (indexing_dirs, execution_dirs)
    """
    # Indexing steps: segments (renamed to segment) and training
    indexing_steps = {'segments', 'training'}
    
    # Everything else is execution steps
    indexing_dirs = []
    execution_dirs = []
    
    for dir_name in directory_names:
        if dir_name in indexing_steps:
            indexing_dirs.append(dir_name)
        else:
            execution_dirs.append(dir_name)
    
    return indexing_dirs, execution_dirs


def split_b3d_directories(dry_run: bool = False):
    """
    Split the b3d cache directory structure into indexing and execution directories.
    
    Current structure: {CACHE_DIR}/b3d/{video_file}/<steps-results>
    New structure: 
    - {CACHE_DIR}/b3d-{video_file_without_extension}/indexing/<indexing-steps-results>
    - {CACHE_DIR}/b3d-{video_file_without_extension}/execution/{video_file}/<execution-steps-results>
    
    Args:
        dry_run (bool): Whether to only show what would be moved without actually doing it
    """
    b3d_dir = os.path.join(CACHE_DIR, 'b3d')
    
    if not os.path.exists(b3d_dir):
        print(f"B3D directory {b3d_dir} does not exist, nothing to split.")
        return
    
    print(f"Processing B3D directory: {b3d_dir}")
    
    # Get all video files from the b3d directory
    video_files = [f for f in os.listdir(b3d_dir) 
                   if os.path.isdir(os.path.join(b3d_dir, f))]
    
    if not video_files:
        print(f"No video files found in {b3d_dir}")
        return
    
    total_moves = 0
    
    for video_file in sorted(video_files):
        video_file_path = os.path.join(b3d_dir, video_file)
        print(f"  Processing video: {video_file}")
        
        # Get video name without extension
        video_name = get_video_name_without_extension(video_file)
        
        # Get all subdirectories in the video directory
        subdirs = [d for d in os.listdir(video_file_path) 
                  if os.path.isdir(os.path.join(video_file_path, d))]
        
        if not subdirs:
            print(f"    No subdirectories found in {video_file_path}")
            continue
        
        # Categorize directories
        indexing_dirs, execution_dirs = categorize_directories(subdirs)
        
        print(f"    Indexing directories: {indexing_dirs}")
        print(f"    Execution directories: {execution_dirs}")
        
        # Create new directory structure
        new_base_dir = os.path.join(CACHE_DIR, f'b3d-{video_name}')
        indexing_dir = os.path.join(new_base_dir, 'indexing')
        execution_dir = os.path.join(new_base_dir, 'execution', video_file)
        
        # Move indexing directories
        for dir_name in indexing_dirs:
            old_path = os.path.join(video_file_path, dir_name)
            
            # Rename 'segments' to 'segment' for indexing
            new_dir_name = 'segment' if dir_name == 'segments' else dir_name
            new_path = os.path.join(indexing_dir, new_dir_name)
            
            if dry_run:
                print(f"    Would move indexing: {old_path} -> {new_path}")
            else:
                try:
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(old_path, new_path)
                    print(f"    Moved indexing: {dir_name} -> {new_path}")
                except Exception as e:
                    print(f"    Error moving indexing {dir_name}: {e}")
                    continue
            total_moves += 1
        
        # Move execution directories
        for dir_name in execution_dirs:
            old_path = os.path.join(video_file_path, dir_name)
            new_path = os.path.join(execution_dir, dir_name)
            
            if dry_run:
                print(f"    Would move execution: {old_path} -> {new_path}")
            else:
                try:
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(old_path, new_path)
                    print(f"    Moved execution: {dir_name} -> {new_path}")
                except Exception as e:
                    print(f"    Error moving execution {dir_name}: {e}")
                    continue
            total_moves += 1
        
        # Remove empty video directory if it exists and is empty
        if not dry_run:
            try:
                if os.path.exists(video_file_path) and not os.listdir(video_file_path):
                    os.rmdir(video_file_path)
                    print(f"    Removed empty directory: {video_file_path}")
            except Exception as e:
                print(f"    Error removing empty directory {video_file_path}: {e}")
    
    # Remove empty b3d directory if it exists and is empty
    if not dry_run:
        try:
            if os.path.exists(b3d_dir) and not os.listdir(b3d_dir):
                os.rmdir(b3d_dir)
                print(f"Removed empty B3D directory: {b3d_dir}")
        except Exception as e:
            print(f"Error removing empty B3D directory {b3d_dir}: {e}")
    
    print(f"  Total directories {'would be' if dry_run else ''} moved: {total_moves}")


def main(args):
    """
    Main function that orchestrates the directory splitting process.
    
    This function serves as the entry point for the script. It:
    1. Validates the b3d directory exists
    2. Splits directories for each video file into indexing and execution
    3. Provides feedback on the splitting process
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dry_run (bool): Whether to only show what would be moved
            
    Note:
        - The script reorganizes directories from:
          {CACHE_DIR}/b3d/{video_file}/<steps-results>
        - To:
          {CACHE_DIR}/b3d-{video_file_without_extension}/indexing/<indexing-steps-results>
          {CACHE_DIR}/b3d-{video_file_without_extension}/execution/{video_file}/<execution-steps-results>
        - Indexing steps: segments (renamed to segment) and training
        - Execution steps: everything else (groundtruth, relevancy, compressed_frames, etc.)
        - Use --dry-run to preview changes without making them
    """
    print("B3D Directory Splitting Script")
    print("=" * 40)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
        print()
    
    split_b3d_directories(args.dry_run)
    
    if args.dry_run:
        print("\nDry run completed. Use without --dry-run to perform actual splitting.")
    else:
        print("\nDirectory splitting completed!")


if __name__ == '__main__':
    main(parse_args())