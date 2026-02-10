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

DATASETS_IN_MAP = {
    'caldot1-y05': 'caldot1_y5',
    'caldot1-y11': 'caldot1_y11',
    'caldot2-y05': 'caldot2_y5',
    'caldot2-y11': 'caldot2_y11',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Transform OTIF and LEAP runtime and tracking data into our format')
    parser.add_argument('--sota-dir', type=str, default='/sota-results',
                        help='Directory containing SOTA output data (OTIF and LEAP)')
    return parser.parse_args()


def transform_tracking_json(input_json_path: str, output_jsonl_path: str) -> int:
    """
    Transform OTIF tracking JSON format to our JSONL format.

    OTIF format: dict with keys like left, top, right, bottom, class, score, track_id
    Our format: {"frame_idx": XX, "tracks": [[track_id, left, top, right, bottom], ...]}

    Args:
        input_json_path (str): Path to OTIF tracking JSON file
        output_jsonl_path (str): Path to output JSONL file

    Returns:
        int: The sample_rate used for this video (1 or 2)
    """
    # Read OTIF tracking JSON file
    assert os.path.exists(input_json_path), f"Input JSON file {input_json_path} does not exist"
    with open(input_json_path, 'r') as f:
        otif_data: list[list[dict] | None] = json.load(f)
    if otif_data is None:
        otif_data = []
    print(len(otif_data))
    # assert otif_data is not None, f"OTIF data is None for {input_json_path}"

    # Sample frames by 2 if the last frame index exceeds 1500
    # TODO: This is a hack to adjusting sample_rate.
    last_frame_idx = (len(otif_data) - 1) if otif_data else 0
    sample_rate = 2 if last_frame_idx > 1500 else 1

    # Handle empty data early
    if not otif_data:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_jsonl_path)
        os.makedirs(output_dir, exist_ok=True)
        # Save empty tracking results
        save_tracking_results({}, output_jsonl_path)
        return sample_rate

    # Group detections by frame_idx
    # OTIF format could be a list of detections or a dict keyed by frame
    # We'll handle both cases
    frame_detections: list[dict] = []

    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    for frame_idx, detections in enumerate(otif_data):
        # Skip odd frames when sampling by 2
        if frame_idx % sample_rate != 0:
            continue

        frame_detection: dict = {"frame_idx": frame_idx // sample_rate, "tracks": []}
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
                                        frame_idx // sample_rate, frame_tracks, trajectories, no_interpolate=False)
        frame_detections.append(frame_detection)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    save_tracking_results(frame_tracks, output_jsonl_path)
    # with open(output_jsonl_path, 'w') as fw:
    #     for frame_detection in frame_detections:
    #         fw.write(json.dumps(frame_detection) + '\n')

    # Return the sample_rate used for this video
    return sample_rate


def process_tracking_outputs(tracks_dir: str, output_dataset_dir: str, param_ids: list[int]) -> dict[int, int]:
    """
    Process tracking outputs for a given tracks directory.

    Iterates through param_id directories and transforms tracking JSON files
    to the output format.

    Args:
        tracks_dir (str): Directory containing tracking results
        output_dataset_dir (str): Output directory for transformed tracking data
        param_ids: List of param_ids

    Returns:
        dict[int, int]: Dictionary mapping video_id to sample_rate used
    """
    assert os.path.exists(tracks_dir), f"Tracking directory not found: {tracks_dir}"

    # Dictionary to store video_id -> sample_rate mapping
    video_sample_rates: dict[int, int] = {}

    # Iterate through param_id directories
    for param_id in param_ids:
        param_dir = os.path.join(tracks_dir, str(param_id))
        assert os.path.isdir(param_dir), f"{param_dir} is not a directory"
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
            # Transform and capture the sample_rate used for this video
            sample_rate = transform_tracking_json(input_json_path, output_jsonl_path)
            # Store sample_rate for this video_id (sample_rate is same across all param_ids for a given video)
            video_sample_rates[video_id] = sample_rate

    return video_sample_rates


def setup_sota_dataset_paths(sota_dir: str, dataset: str, cache_dir: str, system: str) -> tuple[str, str, str, str]:
    """
    Set up common paths and directories for SOTA dataset processing.
    
    Args:
        sota_dir (str): Root directory containing SOTA results
        dataset (str): Dataset name
        cache_dir (str): Cache directory for output
        system (str): System name ('otif' or 'leap')
    
    Returns:
        tuple[str, str, str, str]: (dataset_in, sota_dataset_dir, output_dataset_dir, stat_csv_input)
    """
    dataset_in = DATASETS_IN_MAP.get(dataset, dataset)
    print(f"Processing {system.upper()} dataset: {dataset} ({dataset_in})")
    
    # Construct paths
    sota_dataset_dir = os.path.join(sota_dir, dataset_in)
    assert os.path.exists(sota_dataset_dir), f"SOTA dataset directory {sota_dataset_dir} does not exist"
    
    output_dataset_dir = os.path.join(cache_dir, 'SOTA', system, dataset)
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # Construct stat CSV paths
    stat_csv_input = os.path.join(sota_dataset_dir, f'{system}_{dataset_in}.csv')
    
    return dataset_in, sota_dataset_dir, output_dataset_dir, stat_csv_input


def process_otif_dataset(sota_dir: str, dataset: str, cache_dir: str):
    """
    Process a single dataset's OTIF data.
    
    Args:
        sota_dir (str): Root directory containing SOTA results
        dataset (str): Dataset name
        cache_dir (str): Cache directory for output
    """
    dataset_in, sota_dataset_dir, output_dataset_dir, stat_csv_input = setup_sota_dataset_paths(
        sota_dir, dataset, cache_dir, 'otif'
    )
    
    # Process tracking outputs first to get sample_rates
    tracks_dir = os.path.join(sota_dataset_dir, f'otif_{dataset_in}_tracks')

    # Load stat CSV to get param_ids
    input_df = pd.read_csv(stat_csv_input)
    input_df.columns = input_df.columns.str.replace('Unnamed: 0', 'param_id')
    param_ids = [int(row['param_id']) for _, row in input_df.iterrows()]

    # Process tracking outputs and get video sample rates
    video_sample_rates = process_tracking_outputs(tracks_dir, output_dataset_dir, param_ids)
    print(f"  Video sample rates: {video_sample_rates}")

    # Process stat measurement CSV
    stat_csv_output = os.path.join(output_dataset_dir, 'stat.csv')

    runtime_column = 'runtime' if 'runtime' in input_df.columns else 'runtime_total'
    input_df['runtime'] = input_df[runtime_column]
    input_df = input_df[['param_id', 'detector_cfg', 'segmentation_cfg', 'tracker_cfg', 'runtime']]
    input_df.to_csv(stat_csv_output, index=False)


def process_leap_dataset(sota_dir: str, dataset: str, cache_dir: str):
    """
    Process a single dataset's LEAP data.
    
    LEAP has only one parameter set (param_id == 0). The CSV has many rows,
    but we need the row where video_name == "total". The runtime is the sum
    of inference_total + detector + differencer + reid + match.
    
    Args:
        sota_dir (str): Root directory containing SOTA results
        dataset (str): Dataset name
        cache_dir (str): Cache directory for output
    """
    dataset_in, sota_dataset_dir, output_dataset_dir, stat_csv_input = setup_sota_dataset_paths(
        sota_dir, dataset, cache_dir, 'leap'
    )
    
    # Process tracking outputs first to get sample_rates
    tracks_dir = os.path.join(sota_dataset_dir, f'leap_{dataset_in}_tracks')
    video_sample_rates = process_tracking_outputs(tracks_dir, output_dataset_dir, [0])
    print(f"  Video sample rates: {video_sample_rates}")

    # Process stat measurement CSV
    stat_csv_output = os.path.join(output_dataset_dir, 'stat.csv')

    assert os.path.exists(stat_csv_input), f"Stat CSV not found: {stat_csv_input}"
    print(f"  Processing stat CSV: {stat_csv_input} -> {stat_csv_output}")

    # Read LEAP CSV file
    input_df = pd.read_csv(stat_csv_input)

    # Find the row where video_name == "total"
    total_row = input_df.query("video_name == 'total'")
    assert len(total_row) == 1, f"Expected exactly one row with video_name=='total', found {len(total_row)}"

    # Extract runtime components and sum them
    runtime_components = ['inference_total', 'decode']
    for col in runtime_components:
        assert col in input_df.columns, f"Column {col} not found in LEAP CSV"

    # Extract values from the total row (convert to dict for easier access)
    total_row['runtime'] = total_row['inference_total'] - total_row['decode']
    total_runtime = total_row['runtime'].iloc[0]

    # Create output DataFrame with single row for param_id == 0
    output_df = pd.DataFrame({
        'param_id': [0],
        'detector_cfg': [None],
        'segmentation_cfg': [None],
        'tracker_cfg': [None],
        'runtime': [total_runtime]
    })
    output_df.to_csv(stat_csv_output, index=False)
    print(f"  Calculated total runtime: {total_runtime}")


def process_dataset(sota_dir: str, dataset: str, cache_dir: str):
    """
    Process a single dataset's SOTA data (OTIF or LEAP).
    
    Automatically detects whether the dataset is OTIF or LEAP based on
    the presence of corresponding CSV files.
    
    Args:
        sota_dir (str): Root directory containing SOTA results
        dataset (str): Dataset name
        cache_dir (str): Cache directory for output
    """
    dataset_in = DATASETS_IN_MAP.get(dataset, dataset)
    sota_dataset_dir = os.path.join(sota_dir, dataset_in)
    assert os.path.exists(sota_dataset_dir), f"SOTA dataset directory {sota_dataset_dir} does not exist"
    
    # Check for OTIF or LEAP CSV files to determine dataset type
    otif_csv = os.path.join(sota_dataset_dir, f'otif_{dataset_in}.csv')
    leap_csv = os.path.join(sota_dataset_dir, f'leap_{dataset_in}.csv')
    
    if os.path.exists(otif_csv):
        process_otif_dataset(sota_dir, dataset, cache_dir)
    if os.path.exists(leap_csv):
        process_leap_dataset(sota_dir, dataset, cache_dir)


def main(args):
    """
    Main function that orchestrates the OTIF and LEAP data transformation process.
    
    This function serves as the entry point for the script. It:
    1. Discovers all datasets in the SOTA directory (or uses provided list)
    2. For each dataset, detects whether it's OTIF or LEAP format
    3. Transforms stat CSV and tracking JSON files accordingly
    4. Saves transformed data to the cache directory in our format
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects OTIF output data in:
          {sota_dir}/{dataset}/otif_{dataset}.csv
          {sota_dir}/{dataset}/otif_{dataset}_tracks/{param_id}/{video_id}.json
        - The script expects LEAP output data in:
          {sota_dir}/{dataset}/leap_{dataset}.csv (with video_name column, row with video_name=="total")
          {sota_dir}/{dataset}/leap_{dataset}_tracks/{param_id}/{video_id}.json (param_id is always 0)
        - Transformed OTIF data is saved to:
          {CACHE_DIR}/SOTA/otif/{dataset}/stat.csv
          {CACHE_DIR}/SOTA/otif/{dataset}/{video_file}/tracking_results/{param_id:03d}/tracking.jsonl
        - Transformed LEAP data is saved to:
          {CACHE_DIR}/SOTA/leap/{dataset}/stat.csv
          {CACHE_DIR}/SOTA/leap/{dataset}/{video_file}/tracking_results/000/tracking.jsonl
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
    for dataset in DATASETS:
        process_dataset(sota_dir, dataset, CACHE_DIR)
    
    print(f"\nTransformation complete. Output saved to: {os.path.join(CACHE_DIR, 'SOTA')}")


if __name__ == '__main__':
    main(parse_args())
