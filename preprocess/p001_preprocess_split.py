#!/usr/local/bin/python

import argparse
import os
import shutil
import random

from polyis.utilities import DATASETS_DIR, DATASETS_TO_TEST


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into train, valid, test sets')
    parser.add_argument('-d', '--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--split', type=str, default='8:5:5',
                        help='Exact file counts in format tr:va:te (default: 8:5:5)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing splits')
    
    args = parser.parse_args()
    
    # Parse split counts
    try:
        counts = [int(x) for x in args.split.split(':')]
        if len(counts) != 3:
            raise ValueError("Split must have exactly 3 values")
        args.train_count, args.valid_count, args.test_count = counts
        
        # Validate that all counts are non-negative
        if any(count < 0 for count in counts):
            raise ValueError("All split counts must be non-negative")
        
    except ValueError as e:
        raise ValueError(f"Invalid split format '{args.split}': {e}")
    
    return args


def split_dataset_files(dataset_dir: str, train_count: int, valid_count: int, 
                       test_count: int, force: bool) -> None:
    """
    Split dataset files into train, valid, test directories.
    
    Args:
        dataset_dir: Path to the dataset directory
        train_count: Number of files for training set
        valid_count: Number of files for validation set  
        test_count: Number of files for test set
        force: Whether to force overwrite existing splits
    """
    # Set random seed for reproducible splits
    random.seed(0)
    
    # Get all video files in the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return
    
    # Sort files for consistent ordering
    video_files.sort()
    
    print(f"Found {len(video_files)} video files in {dataset_dir}")
    
    # Check if splits already exist
    split_dirs = ['train', 'valid', 'test']
    existing_splits = [d for d in split_dirs if os.path.exists(os.path.join(dataset_dir, d))]
    
    if existing_splits and not force:
        print(f"Splits already exist: {existing_splits}. Use --force to overwrite.")
        return
    
    # Remove existing split directories if force is enabled
    if force and existing_splits:
        for split_dir in existing_splits:
            split_path = os.path.join(dataset_dir, split_dir)
            print(f"Removing existing {split_dir} directory")
            shutil.rmtree(split_path)
    
    # Validate split counts against total files
    total_files = len(video_files)
    total_requested = train_count + valid_count + test_count
    
    if total_requested > total_files:
        raise ValueError(f"Requested split counts ({total_requested}) exceed total files ({total_files})")
    
    # Shuffle files randomly
    shuffled_files = video_files.copy()
    random.shuffle(shuffled_files)
    
    # Split files
    train_files = shuffled_files[:train_count]
    valid_files = shuffled_files[train_count:train_count + valid_count]
    test_files = shuffled_files[train_count + valid_count:total_requested]
    
    print(f"Split sizes: train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")
    
    # Create split directories and move files
    splits = [
        ('train', train_files),
        ('valid', valid_files), 
        ('test', test_files)
    ]
    
    for split_name, files in splits:
        if not files:
            print(f"No files assigned to {split_name} split")
            continue
            
        split_dir = os.path.join(dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"Moving {len(files)} files to {split_name}/")
        for file in files:
            src_path = os.path.join(dataset_dir, file)
            dst_path = os.path.join(split_dir, split_name[:2] + file)
            shutil.move(src_path, dst_path)
            print(f"  Moved {file} -> {split_name}/{file}")


def split_dataset(args: argparse.Namespace, dataset: str) -> None:
    """
    Process a single dataset for splitting.
    
    Args:
        args: Parsed command line arguments
        dataset: Name of the dataset to process
    """
    dataset_dir = os.path.join(DATASETS_DIR, dataset)
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    print(f"\nProcessing dataset: {dataset}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Check if dataset already has train/valid/test structure
    existing_splits = [d for d in ['train', 'valid', 'test'] 
                      if os.path.exists(os.path.join(dataset_dir, d))]
    
    if existing_splits and not args.force:
        print(f"Dataset {dataset} already has splits: {existing_splits}")
        print("Use --force to recreate splits")
        return
    
    # Check if there are video files in the root directory
    root_video_files = [f for f in os.listdir(dataset_dir) 
                       if f.endswith('.mp4') and os.path.isfile(os.path.join(dataset_dir, f))]
    
    if not root_video_files:
        print(f"No video files found in root directory of {dataset}")
        return
    
    # Split the dataset
    split_dataset_files(
        dataset_dir=dataset_dir,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        force=args.force
    )


def main(args):
    """
    Main function to split datasets into train, valid, test sets.
    
    Args:
        args: Parsed command line arguments
    """
    datasets = args.datasets
    
    print(f"Splitting datasets: {datasets}")
    print(f"Split counts: {args.split} (train={args.train_count}, valid={args.valid_count}, test={args.test_count})")
    
    for dataset in datasets:
        try:
            split_dataset(args, dataset)
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            continue
    
    print("\nDataset splitting completed!")


if __name__ == '__main__':
    main(parse_args())
