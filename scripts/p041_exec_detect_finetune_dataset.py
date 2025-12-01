#!/usr/local/bin/python
"""
Create fine-tuning datasets from compressed/packed frames.

This script creates training datasets for fine-tuning object detection models
on compressed images. It processes compressed frames, extracts annotations from
ground truth tracking results, and converts them to training formats.

Output formats supported:
- Intermediate JSONL format for inspection and debugging
- Ultralytics YOLO format
- COCO format for Detectron2 and other frameworks
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from polyis.utilities import get_config, load_tracking_results
from polyis.train.data.finetune import (
    CompressedImageAnnotation,
    load_offset_lookup,
    load_index_map,
    get_annotations_for_compressed_image,
    save_intermediate_dataset,
    load_intermediate_dataset,
    convert_to_ultralytics,
    convert_to_coco,
    convert_to_darknet,
    visualize_annotations,
    read_ultralytics_labels,
    read_coco_annotations,
    read_darknet_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create fine-tuning datasets from compressed frames')
    parser.add_argument('--dataset', type=str, nargs='*', default=None,
                        help='Dataset name(s) to process (default: all from config)')
    parser.add_argument('--tilesize', type=int, nargs='*', default=None,
                        help='Tile size(s) to process (default: all from config)')
    parser.add_argument('--tilepadding', type=str, nargs='*', default=None,
                        help='Tile padding mode(s) to process (default: all from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: /polyis-cache/{dataset}/finetune/{tilesize}_{tilepadding}/)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction for validation split (default: 0.2)')
    parser.add_argument('--format', type=str, default='both',
                        choices=['ultralytics', 'coco', 'darknet', 'all', 'both', 'intermediate-only'],
                        help='Output format: ultralytics, coco, darknet, all, both (ultralytics+coco), intermediate-only (default: both)')
    parser.add_argument('--skip-intermediate', action='store_true',
                        help='Skip intermediate dataset creation, convert existing')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization (visualization enabled by default)')
    parser.add_argument('--num-visualize', type=int, default=10,
                        help='Number of sample images to visualize (default: 10)')
    return parser.parse_args()


def create_intermediate_dataset(
    dataset: str,
    tilesize: int,
    tilepadding: str,
    config: dict,
) -> list[CompressedImageAnnotation]:
    """
    Create intermediate dataset combining all videos and classifiers.
    
    Args:
        dataset: Dataset name
        tilesize: Tile size used for compression
        tilepadding: Tile padding mode
        config: Configuration dictionary
        
    Returns:
        List of CompressedImageAnnotation entries
    """
    cache_dir = Path(config['DATA']['CACHE_DIR'])
    datasets_dir = Path(config['DATA']['DATASETS_DIR'])
    classifiers = config['EXEC']['CLASSIFIERS']
    
    entries: list[CompressedImageAnnotation] = []
    
    # Get all test videos for this dataset
    video_dir = datasets_dir / dataset / "test"
    if not video_dir.exists():
        print(f"Warning: Video directory {video_dir} does not exist, skipping...")
        return entries
    
    # Get all video files
    videos = sorted([f.name for f in video_dir.iterdir() if f.suffix == '.mp4'])
    
    for video in tqdm(videos, desc=f"Processing {dataset}"):
        # Load ground truth tracking for this video
        try:
            tracking_results = load_tracking_results(
                str(cache_dir), dataset, video, verbose=False
            )
        except FileNotFoundError:
            print(f"Warning: Tracking results not found for {dataset}/{video}, skipping...")
            continue
        
        for classifier in classifiers:
            # Construct path to compressed frames
            param_str = f"{classifier}_{tilesize}_{tilepadding}"
            compressed_dir = cache_dir / dataset / "execution" / video / "033_compressed_frames" / param_str
            
            # Skip if compressed frames don't exist
            if not compressed_dir.exists():
                continue
            
            images_dir = compressed_dir / "images"
            index_maps_dir = compressed_dir / "index_maps"
            offset_lookups_dir = compressed_dir / "offset_lookups"
            
            # Process each compressed image
            for image_file in sorted(images_dir.glob("*.jpg")):
                # Parse filename: {collage_idx:04d}_{start_frame:04d}_{end_frame:04d}.jpg
                parts = image_file.stem.split('_')
                if len(parts) != 3:
                    print(f"Warning: Unexpected filename format: {image_file.name}")
                    continue
                    
                collage_idx = int(parts[0])
                start_frame = int(parts[1])
                end_frame = int(parts[2])
                
                # Load mappings
                prefix = image_file.stem
                index_map_path = index_maps_dir / f"{prefix}.npy"
                offset_lookup_path = offset_lookups_dir / f"{prefix}.jsonl"
                
                # Skip if mapping files don't exist
                if not index_map_path.exists() or not offset_lookup_path.exists():
                    print(f"Warning: Missing mapping files for {image_file.name}")
                    continue
                
                index_map = load_index_map(index_map_path)
                offset_lookup = load_offset_lookup(offset_lookup_path)
                
                # Get image dimensions
                img = cv2.imread(str(image_file))
                if img is None:
                    print(f"Warning: Could not read image {image_file}")
                    continue
                height, width = img.shape[:2]
                
                # Get valid annotations
                annotations = get_annotations_for_compressed_image(
                    start_frame, end_frame, tracking_results,
                    offset_lookup, index_map, tilesize, width, height
                )
                
                # Create entry
                entries.append(CompressedImageAnnotation(
                    image_path=str(image_file),
                    image_width=width,
                    image_height=height,
                    dataset=dataset,
                    video=video,
                    classifier=classifier,
                    tilesize=tilesize,
                    tilepadding=tilepadding,
                    collage_idx=collage_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    annotations=annotations,
                ))
    
    return entries


def visualize_intermediate_dataset(
    entries: list[CompressedImageAnnotation],
    output_dir: Path,
    num_samples: int,
    dataset: str,
) -> None:
    """
    Visualize sample images from intermediate dataset.
    
    Args:
        entries: List of CompressedImageAnnotation entries
        output_dir: Output directory for dataset
        num_samples: Number of sample images to visualize
        dataset: Dataset name
    """
    if len(entries) == 0:
        return
    
    # Select sample images (randomly, but try to get diverse examples)
    num_samples = min(num_samples, len(entries))
    sample_indices = random.sample(range(len(entries)), num_samples)
    
    # Create visualization directory: /polyis/output/visualizations/finetune/{dataset}/{tilesize}_{tilepadding}/intermediate
    # Extract tilesize_tilepadding from output_dir name
    tilesize_tilepadding = output_dir.name
    viz_dir = Path("/polyis/output/visualizations/finetune") / dataset / tilesize_tilepadding / "intermediate"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVisualizing {num_samples} sample images from intermediate dataset...")
    
    for idx in tqdm(sample_indices, desc="Creating visualizations"):
        entry = entries[idx]
        
        # Create unique filename
        source_path = Path(entry.image_path)
        unique_name = f"{entry.dataset}_{entry.video}_{entry.classifier}_{source_path.stem}"
        output_path = viz_dir / f"{unique_name}_visualized.jpg"
        
        # Visualize
        visualize_annotations(entry.image_path, entry.annotations, output_path)


def visualize_ultralytics_format(
    output_dir: Path,
    num_samples: int,
    dataset: str,
) -> None:
    """
    Visualize sample images from Ultralytics format dataset.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of sample images to visualize
        dataset: Dataset name
    """
    ultralytics_dir = output_dir / "ultralytics"
    if not ultralytics_dir.exists():
        return
    
    # Create visualization directory: /polyis/output/visualizations/finetune/{dataset}/{tilesize}_{tilepadding}/ultralytics
    tilesize_tilepadding = output_dir.name
    viz_dir = Path("/polyis/output/visualizations/finetune") / dataset / tilesize_tilepadding / "ultralytics"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files from both train and val
    label_files = []
    for split in ["train", "val"]:
        label_dir = ultralytics_dir / "labels" / split
        if label_dir.exists():
            label_files.extend(list(label_dir.glob("*.txt")))
    
    if len(label_files) == 0:
        return
    
    # Select sample label files
    num_samples = min(num_samples, len(label_files))
    sample_labels = random.sample(label_files, num_samples)
    
    print(f"\nVisualizing {num_samples} sample images from Ultralytics format...")
    
    for label_path in tqdm(sample_labels, desc="Creating visualizations"):
        # Find corresponding image
        label_name = label_path.stem
        split = label_path.parent.name
        
        # Try to find image in train or val
        image_path = None
        for img_split in ["train", "val"]:
            potential_image = ultralytics_dir / "images" / img_split / f"{label_name}.jpg"
            if potential_image.exists() or potential_image.is_symlink():
                image_path = potential_image
                break
        
        if image_path is None:
            continue
        
        # Resolve symlink to get actual image
        if image_path.is_symlink():
            image_path = image_path.resolve()
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        
        # Read annotations
        annotations = read_ultralytics_labels(label_path, width, height)
        
        # Visualize
        output_path = viz_dir / f"{label_name}_visualized.jpg"
        visualize_annotations(image_path, annotations, output_path)


def visualize_coco_format(
    output_dir: Path,
    num_samples: int,
    dataset: str,
) -> None:
    """
    Visualize sample images from COCO format dataset.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of sample images to visualize
        dataset: Dataset name
    """
    coco_dir = output_dir / "coco"
    if not coco_dir.exists():
        return
    
    # Create visualization directory: /polyis/output/visualizations/finetune/{dataset}/{tilesize}_{tilepadding}/coco
    tilesize_tilepadding = output_dir.name
    viz_dir = Path("/polyis/output/visualizations/finetune") / dataset / tilesize_tilepadding / "coco"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get annotation files
    annotation_files = []
    for split in ["train2017", "val2017"]:
        ann_file = coco_dir / "annotations" / f"instances_{split}.json"
        if ann_file.exists():
            annotation_files.append((ann_file, split))
    
    if len(annotation_files) == 0:
        return
    
    # Load all images from annotation files
    all_images = []
    for ann_file, split in annotation_files:
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        for img_info in coco_data.get('images', []):
            all_images.append((img_info['file_name'], split, ann_file))
    
    if len(all_images) == 0:
        return
    
    # Select sample images
    num_samples = min(num_samples, len(all_images))
    sample_images = random.sample(all_images, num_samples)
    
    print(f"\nVisualizing {num_samples} sample images from COCO format...")
    
    for image_filename, split, ann_file in tqdm(sample_images, desc="Creating visualizations"):
        # Find image file
        image_path = coco_dir / split / image_filename
        if not image_path.exists() and not image_path.is_symlink():
            continue
        
        # Resolve symlink
        if image_path.is_symlink():
            image_path = image_path.resolve()
        
        # Read annotations
        annotations = read_coco_annotations(ann_file, image_filename)
        
        # Visualize
        output_path = viz_dir / f"{Path(image_filename).stem}_visualized.jpg"
        visualize_annotations(image_path, annotations, output_path)


def visualize_darknet_format(
    output_dir: Path,
    num_samples: int,
    dataset: str,
) -> None:
    """
    Visualize sample images from Darknet format dataset.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of sample images to visualize
        dataset: Dataset name
    """
    darknet_dir = output_dir / "darknet"
    if not darknet_dir.exists():
        return
    
    # Create visualization directory: /polyis/output/visualizations/finetune/{dataset}/{tilesize}_{tilepadding}/darknet
    tilesize_tilepadding = output_dir.name
    viz_dir = Path("/polyis/output/visualizations/finetune") / dataset / tilesize_tilepadding / "darknet"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_dir = darknet_dir / "labels"
    if not label_dir.exists():
        return
    
    label_files = list(label_dir.glob("*.txt"))
    if len(label_files) == 0:
        return
    
    # Select sample label files
    num_samples = min(num_samples, len(label_files))
    sample_labels = random.sample(label_files, num_samples)
    
    print(f"\nVisualizing {num_samples} sample images from Darknet format...")
    
    for label_path in tqdm(sample_labels, desc="Creating visualizations"):
        # Find corresponding image
        label_name = label_path.stem
        image_path = darknet_dir / "images" / f"{label_name}.jpg"
        
        if not image_path.exists() and not image_path.is_symlink():
            continue
        
        # Resolve symlink
        if image_path.is_symlink():
            image_path = image_path.resolve()
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        
        # Read annotations
        annotations = read_darknet_labels(label_path, width, height)
        
        # Visualize
        output_path = viz_dir / f"{label_name}_visualized.jpg"
        visualize_annotations(image_path, annotations, output_path)


def print_summary(entries: list[CompressedImageAnnotation], output_dir: Path):
    """
    Print summary statistics for the created dataset.
    
    Args:
        entries: List of CompressedImageAnnotation entries
        output_dir: Output directory path
    """
    total_images = len(entries)
    total_annotations = sum(len(e.annotations) for e in entries)
    
    # Count by classifier
    classifiers = {}
    for e in entries:
        if e.classifier not in classifiers:
            classifiers[e.classifier] = {'images': 0, 'annotations': 0}
        classifiers[e.classifier]['images'] += 1
        classifiers[e.classifier]['annotations'] += len(e.annotations)
    
    # Count by video
    videos = {}
    for e in entries:
        video_key = f"{e.dataset}/{e.video}"
        if video_key not in videos:
            videos[video_key] = {'images': 0, 'annotations': 0}
        videos[video_key]['images'] += 1
        videos[video_key]['annotations'] += len(e.annotations)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations / max(1, total_images):.2f}")
    
    print(f"\nBy classifier:")
    for clf, stats in sorted(classifiers.items()):
        print(f"  {clf}: {stats['images']} images, {stats['annotations']} annotations")
    
    print(f"\nBy video ({len(videos)} videos):")
    for video, stats in sorted(videos.items())[:5]:
        print(f"  {video}: {stats['images']} images, {stats['annotations']} annotations")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more videos")


def main():
    """Main function to create fine-tuning datasets."""
    args = parse_args()
    config = get_config()
    
    # Set random seed for reproducible visualization sampling
    random.seed(42)
    
    # Determine datasets, tilesizes, tilepaddings to process
    datasets = args.dataset or config['EXEC']['DATASETS']
    tilesizes = args.tilesize or config['EXEC']['TILE_SIZES']
    tilepaddings = args.tilepadding or config['EXEC']['TILEPADDING_MODES']
    
    print(f"Processing datasets: {datasets}")
    print(f"Tile sizes: {tilesizes}")
    print(f"Tile paddings: {tilepaddings}")
    print(f"Output format: {args.format}")
    print(f"Validation split: {args.val_split}")
    
    for dataset in datasets:
        for tilesize in tilesizes:
            for tilepadding in tilepaddings:
                print(f"\n{'='*60}")
                print(f"Processing: {dataset} / tilesize={tilesize} / tilepadding={tilepadding}")
                print(f"{'='*60}")
                
                # Create output directory
                if args.output_dir:
                    # Use provided output directory
                    output_dir = Path(args.output_dir) / f"{dataset}_{tilesize}_{tilepadding}"
                else:
                    # Use default structure: /polyis-cache/{dataset}/finetune/{tilesize}_{tilepadding}/
                    output_dir = Path(f"/polyis-cache/{dataset}/finetune/{tilesize}_{tilepadding}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Stage 1: Create or load intermediate dataset
                intermediate_path = output_dir / "intermediate.jsonl"
                
                if not args.skip_intermediate:
                    # Create intermediate dataset
                    print("Creating intermediate dataset...")
                    intermediate = create_intermediate_dataset(
                        dataset, tilesize, tilepadding, config
                    )
                    
                    if len(intermediate) == 0:
                        print(f"Warning: No data found for {dataset}/{tilesize}/{tilepadding}")
                        continue
                    
                    # Save intermediate dataset
                    save_intermediate_dataset(intermediate, intermediate_path)
                    print(f"Saved intermediate dataset to {intermediate_path}")
                else:
                    # Load existing intermediate dataset
                    if not intermediate_path.exists():
                        print(f"Error: Intermediate dataset not found at {intermediate_path}")
                        continue
                    print(f"Loading existing intermediate dataset from {intermediate_path}")
                    intermediate = load_intermediate_dataset(intermediate_path)
                
                # Visualize intermediate dataset if enabled
                if not args.no_visualize:
                    visualize_intermediate_dataset(intermediate, output_dir, args.num_visualize, dataset)
                
                # Stage 2: Convert to requested formats
                if args.format in ['ultralytics', 'both', 'all']:
                    print("\nConverting to Ultralytics format...")
                    convert_to_ultralytics(
                        intermediate,
                        output_dir / "ultralytics",
                        args.val_split
                    )
                    # Visualize Ultralytics format if enabled
                    if not args.no_visualize:
                        visualize_ultralytics_format(output_dir, args.num_visualize, dataset)
                
                if args.format in ['coco', 'both', 'all']:
                    print("\nConverting to COCO format...")
                    convert_to_coco(
                        intermediate,
                        output_dir / "coco",
                        args.val_split
                    )
                    # Visualize COCO format if enabled
                    if not args.no_visualize:
                        visualize_coco_format(output_dir, args.num_visualize, dataset)
                
                if args.format in ['darknet', 'all']:
                    print("\nConverting to Darknet format...")
                    convert_to_darknet(
                        intermediate,
                        output_dir / "darknet",
                        args.val_split
                    )
                    # Visualize Darknet format if enabled
                    if not args.no_visualize:
                        visualize_darknet_format(output_dir, args.num_visualize, dataset)
                
                # Print summary
                print_summary(intermediate, output_dir)
    
    print("\n" + "="*60)
    print("Dataset creation complete!")


if __name__ == '__main__':
    main()
