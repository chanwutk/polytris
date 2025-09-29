#!/usr/local/bin/python

import argparse
import os
import shutil

from polyis.utilities import CACHE_DIR


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (List[str]): Dataset names to process (default: ['b3d'])
            - dry_run (bool): Whether to only show what would be renamed without actually doing it
    """
    parser = argparse.ArgumentParser(description='Rename pipeline directories to add numeric prefixes for chronological ordering')
    parser.add_argument('--datasets', required=False,
                        default=['b3d'],
                        nargs='+',
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
                        continue
                total_renames += 1
            else:
                print(f"    Directory {old_name} not found, skipping")
    
    print(f"  Total directories {'would be' if dry_run else ''} renamed: {total_renames}")


def main(args):
    """
    Main function that orchestrates the directory renaming process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset execution directories exist
    2. Renames directories in each video file's execution directory
    3. Provides feedback on the renaming process
    
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
    """
    print("Pipeline Directory Renaming Script")
    print("=" * 40)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
        print()
    
    for dataset_name in args.datasets:
        rename_directories(dataset_name, args.dry_run)
        print()
    
    if args.dry_run:
        print("Dry run completed. Use without --dry-run to perform actual renaming.")
    else:
        print("Directory renaming completed!")


if __name__ == '__main__':
    main(parse_args())
