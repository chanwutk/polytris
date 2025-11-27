#!/usr/local/bin/python

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from polyis.utilities import get_config, register_tracked_detections, save_tracking_results


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Transform OTIF runtime and tracking data into our format')
    parser.add_argument('--sota-dir', type=str, default='/sota-results',
                        help='Directory containing OTIF output data')
    return parser.parse_args()


def transform_tracking_json(input_json_path: str, output_jsonl_path: str):
    """
    Transform OTIF tracking JSON format to our JSONL format.
    
    OTIF format: dict with keys like left, top, right, bottom, class, score, track_id
    Our format: {"frame_idx": XX, "tracks": [[track_id, left, top, right, bottom], ...]}
    
    Args:
        input_json_path (str): Path to OTIF tracking JSON file
        output_jsonl_path (str): Path to output JSONL file
    """
    # Read OTIF tracking JSON file
    assert os.path.exists(input_json_path), f"Input JSON file {input_json_path} does not exist"
    with open(input_json_path, 'r') as f:
        otif_data: list[list[dict] | None] = json.load(f)
    if otif_data is None:
        otif_data = []
    print(len(otif_data))
    # assert otif_data is not None, f"OTIF data is None for {input_json_path}"
    
    # Group detections by frame_idx
    # OTIF format could be a list of detections or a dict keyed by frame
    # We'll handle both cases
    frame_detections: list[dict] = []

    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    for frame_idx, detections in enumerate(otif_data):
        
        frame_detection: dict = {"frame_idx": frame_idx, "tracks": []}
        for detection in detections or []:
            # Extract bounding box and track_id
            track_id = detection['track_id']
            left = detection['left']
            top = detection['top']
            right = detection['right']
            bottom = detection['bottom']
            
            # Validate required fields
            if track_id is None or left is None or top is None or right is None or bottom is None:
                raise ValueError(f"Missing required fields in detection: {detection}")
            
            # frame_detection['tracks'].append([int(track_id), float(left), float(top), float(right), float(bottom)])
            register_tracked_detections([(float(left), float(top), float(right), float(bottom), int(track_id))],
                                        frame_idx, frame_tracks, trajectories, no_interpolate=False)
        frame_detections.append(frame_detection)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    save_tracking_results(frame_tracks, output_jsonl_path)
    # with open(output_jsonl_path, 'w') as fw:
    #     for frame_detection in frame_detections:
    #         fw.write(json.dumps(frame_detection) + '\n')


def process_dataset(sota_dir: str, dataset: str, cache_dir: str):
    """
    Process a single dataset's OTIF data.
    
    Args:
        sota_dir (str): Root directory containing SOTA results
        dataset (str): Dataset name
        cache_dir (str): Cache directory for output
    """
    print(f"Processing dataset: {dataset}")
    
    # Construct paths
    sota_dataset_dir = os.path.join(sota_dir, dataset)
    assert os.path.exists(sota_dataset_dir), f"SOTA dataset directory {sota_dataset_dir} does not exist"
    
    output_dataset_dir = os.path.join(cache_dir, 'SOTA', 'otif', dataset)
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # Process stat measurement CSV
    stat_csv_input = os.path.join(sota_dataset_dir, f'otif_{dataset}.csv')
    stat_csv_output = os.path.join(output_dataset_dir, 'stat.csv')

    assert os.path.exists(stat_csv_input), f"Stat CSV not found: {stat_csv_input}"
    print(f"  Copying stat CSV: {stat_csv_input} -> {stat_csv_output}")
    shutil.copy2(stat_csv_input, stat_csv_output)

    input_df = pd.read_csv(stat_csv_input)
    input_df.columns = input_df.columns.str.replace('Unnamed: 0', 'param_id')
    input_df = input_df[['param_id', 'detector_cfg', 'segmentation_cfg', 'tracker_cfg', 'runtime']]
    input_df.to_csv(stat_csv_output, index=False)
    
    # Process tracking outputs
    tracks_dir = os.path.join(sota_dataset_dir, f'otif_{dataset}_tracks')
    assert os.path.exists(tracks_dir), f"Tracking directory not found: {tracks_dir}"

    # Iterate through param_id directories
    for row in input_df.itertuples():
        param_id = row.param_id
    # for param_id in sorted(map(int, os.listdir(tracks_dir))):
        param_dir = os.path.join(tracks_dir, str(param_id))
        assert os.path.isdir(param_dir), f"Param directory not found: {param_dir}"
        print(f"  Processing param_id: {param_id}")
        
        # Process each video JSON file
        for video_file in os.listdir(param_dir):
            assert video_file.endswith('.json'), f"Video file not found: {video_file}"
            
            # Extract video_id from filename (remove .json extension)
            video_id = int(video_file.replace('.json', ''))
            
            # Construct video_file name (e.g., 'te01.mp4')
            video_file_name = f'te{video_id:02d}.mp4'
            
            # Create output directory structure: {dataset}/{video_file}/tracking_results/{param_id:03d}/
            output_video_dir = os.path.join(output_dataset_dir, video_file_name)
            output_tracking_results_dir = os.path.join(output_video_dir, 'tracking_results')
            output_param_dir = os.path.join(output_tracking_results_dir, f"{param_id:03d}")
            os.makedirs(output_param_dir, exist_ok=True)
            
            # Construct input and output paths
            input_json_path = os.path.join(param_dir, str(video_file))
            output_jsonl_path = os.path.join(output_param_dir, 'tracking.jsonl')
            
            print(f"    Transforming: {input_json_path} -> {output_jsonl_path}")
            transform_tracking_json(input_json_path, output_jsonl_path)


def main(args):
    """
    Main function that orchestrates the OTIF data transformation process.
    
    This function serves as the entry point for the script. It:
    1. Discovers all datasets in the SOTA directory (or uses provided list)
    2. For each dataset, transforms stat CSV and tracking JSON files
    3. Saves transformed data to the cache directory in our format
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects OTIF output data in:
          {sota_dir}/{dataset}/otif_{dataset}.csv
          {sota_dir}/{dataset}/otif_{dataset}_tracks/{param_id}/{video_id}.json
        - Transformed data is saved to:
          {CACHE_DIR}/SOTA/otif/{dataset}/stat.csv
          {CACHE_DIR}/SOTA/otif/{dataset}/{video_file}/tracking_results/{param_id:03d}/tracking.jsonl
    """
    sota_dir = args.sota_dir
    assert os.path.exists(sota_dir), f"SOTA directory {sota_dir} does not exist"
    
    # Discover datasets if not provided
    datasets = []
    for item in os.listdir(sota_dir):
        item_path = os.path.join(sota_dir, item)
        if os.path.isdir(item_path):
            if item in DATASETS:
                datasets.append(item)
    print(f"Discovered {len(datasets)} datasets: {datasets}")
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(sota_dir, dataset, CACHE_DIR)
    
    print(f"\nTransformation complete. Output saved to: {os.path.join(CACHE_DIR, 'SOTA', 'otif')}")


if __name__ == '__main__':
    main(parse_args())
