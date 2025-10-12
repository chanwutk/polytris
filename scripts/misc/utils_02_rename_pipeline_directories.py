#!/usr/local/bin/python

import argparse
import os
import shutil

from polyis.utilities import CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description='Rename pipeline directories to add numeric prefixes for chronological ordering')
    parser.add_argument('--datasets', required=True, nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be renamed without actually doing it')
    return parser.parse_args()


def rename_directories(dataset_name: str, dry_run: bool = False):
    """
    Rename directories in the dataset's execution directory to add numeric prefixes.
    
    Args:
        dataset_name (str): Name of the dataset to process
        dry_run (bool): Whether to only show what would be renamed without actually doing it
        
    Note:
        This function also reports directories that are not moved (either because they don't exist
        or because of errors during the move operation).
    """
    dataset_execution_dir = os.path.join(CACHE_DIR, dataset_name, 'execution')
    
    if not os.path.exists(dataset_execution_dir):
        print(f"Dataset execution directory {dataset_execution_dir} does not exist, skipping...")
        return
    
    print(f"Processing dataset: {dataset_name}")
    
    # Get all video files from the dataset execution directory
    video_files = [f for f in os.listdir(dataset_execution_dir) 
                   if os.path.isdir(os.path.join(dataset_execution_dir, f))]
    
    if not video_files:
        print(f"No video files found in {dataset_execution_dir}")
        return
    
    total_renames = 0
    total_not_moved = 0
    
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_execution_dir, video_file)
        print(f"  Processing video: {video_file}")
        
        # Define the directory mappings
        directory_mappings = [
            ('groundtruth', '000_groundtruth'),
            ('relevancy', '020_relevancy'),
            ('packing', '030_compressed_frames'),
            ('packed_detections', '040_compressed_detections'),
            ('uncompressed_detections', '050_uncompressed_detections'),
            ('uncompressed_tracking', '060_uncompressed_tracks'),
            ('evaluation', '070_tracking_accuracy')
        ]
        
        # Track directories that are not moved
        not_moved_dirs = []
        
        for old_name, new_name in directory_mappings:
            old_path = os.path.join(video_file_path, old_name)
            new_path = os.path.join(video_file_path, new_name)
            
            if os.path.exists(old_path):
                if dry_run:
                    print(f"    Would rename: {old_name} -> {new_name}")
                else:
                    try:
                        shutil.move(old_path, new_path)
                        print(f"    Renamed: {old_name} -> {new_name}")
                    except Exception as e:
                        print(f"    Error renaming {old_name} -> {new_name}: {e}")
                        not_moved_dirs.append(f"{old_name} (error: {e})")
                        continue
                total_renames += 1
            else:
                print(f"    Directory {old_name} not found, skipping")
                not_moved_dirs.append(f"{old_name} (not found)")
        
        # Report directories that are not moved for this video
        if not_moved_dirs:
            print(f"    Directories not {'would be' if dry_run else ''} moved: {', '.join(not_moved_dirs)}")
            total_not_moved += len(not_moved_dirs)
    
    print(f"  Total directories {'would be' if dry_run else ''} renamed: {total_renames}")
    if total_not_moved > 0:
        print(f"  Total directories not {'would be' if dry_run else ''} moved: {total_not_moved}")


def main(args):
    """
    Main function that orchestrates the directory renaming process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset execution directories exist
    2. Renames directories in each video file's execution directory
    3. Provides feedback on the renaming process
    4. Reports directories that are not moved (missing or errors)
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - dry_run (bool): Whether to only show what would be renamed
            
    Note:
        - The script looks for directories in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/
        - Renames directories with numeric prefixes:
          * 'groundtruth' -> '000_groundtruth'
          * 'relevancy' -> '020_relevancy'
          * 'packing' -> '030_compressed_frames'
          * 'packed_detections' -> '040_compressed_detections'
          * 'uncompressed_detections' -> '050_uncompressed_detections'
          * 'uncompressed_tracking' -> '060_uncompressed_tracks'
          * 'evaluation' -> '070_tracking_accuracy'
        - Use --dry-run to preview changes without making them
        - Existing directories with the new names will cause errors
        - Reports directories that are not moved with reasons (not found, errors)
    """
    print("Pipeline Directory Renaming Script")
    print("=" * 40)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
        print()
    
    for dataset_name in args.datasets:
        rename_directories(dataset_name, args.dry_run)
        print()


if __name__ == '__main__':
    main(parse_args())
