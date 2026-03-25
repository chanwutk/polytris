#!/usr/local/bin/python

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.manifests import build_split_video_manifest
from polyis.io import cache
from polyis.utilities import get_config, register_tracked_detections, save_tracking_results


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']

DATASETS_IN_MAP = {
    'caldot1-y05': 'caldot1-y5',
    'caldot1-y11': 'caldot1-y11',
    'caldot2-y05': 'caldot2-y5',
    'caldot2-y11': 'caldot2-y11',
    'ams-y05': 'amsterdam-y5',
    'ams-y11': 'amsterdam-y11',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Transform OTIF and LEAP runtime and tracking data into our format')
    parser.add_argument('--sota-dir', type=str, default='/sota-results',
                        help='Directory containing SOTA output data (OTIF and LEAP)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Process test videoset')
    return parser.parse_args()


def transform_tracking_json(input_json_path: str, output_jsonl_path: str, dataset: str, video: str):
    # Fail fast when the input tracking JSON is missing.
    assert os.path.exists(input_json_path), f"Input JSON file {input_json_path} does not exist"

    # Load the OTIF/LEAP tracking payload from disk.
    with open(input_json_path, 'r') as f:
        otif_data: list[list[dict] | None] | None = json.load(f)

    # Normalize null payloads to an empty list so downstream logic stays uniform.
    if otif_data is None:
        otif_data = []

    # Create the parent directory before writing transformed tracking results.
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Short-circuit empty inputs while still writing a valid empty JSONL file.
    if not otif_data:
        save_tracking_results({}, output_jsonl_path)

    # Accumulate per-frame tracks in the JSONL format expected by the rest of the pipeline.
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    # Convert each sampled OTIF/LEAP detection into our shared frame-track representation.
    for frame_idx, detections in enumerate(otif_data):

        # Register each detection in the common tracking accumulator.
        for detection in detections or []:
            # Extract the required OTIF/LEAP tracking fields from the raw payload.
            track_id = detection['track_id']
            left = detection['left']
            top = detection['top']
            right = detection['right']
            bottom = detection['bottom']

            # Validate that the raw detection contains a complete bounding box and track id.
            if track_id is None or left is None or top is None or right is None or bottom is None:
                raise ValueError(f"Missing required fields in detection: {detection}")

            # Append the converted detection to the shared frame/trajectory accumulators.
            register_tracked_detections(
                [(left, top, right, bottom, int(track_id))],
                frame_idx,
                frame_tracks,
                trajectories,
            )

    # Persist the transformed tracking results in the shared JSONL format.
    save_tracking_results(frame_tracks, output_jsonl_path)


def dataset_name_in_sota(dataset: str) -> str:
    # Map configured dataset names to the SOTA directory naming convention.
    return DATASETS_IN_MAP.get(dataset, dataset)


def extract_video_id(video_name: str) -> int:
    # Extract the numeric video id shared by filenames like te01.mp4 and 1.json.
    match = re.search(r'(\d+)', Path(video_name).stem)
    assert match is not None, f"Could not extract numeric video id from {video_name}"

    return int(match.group(1))


def build_expected_test_video_manifest(dataset: str) -> pd.DataFrame:
    # Materialize the configured test videos for the dataset from the shared dataset store.
    video_df = build_split_video_manifest(datasets=[dataset], videosets=['test']).copy()
    # Fail fast when the configured test split is empty.
    assert not video_df.empty, f"No test videos found for dataset {dataset}"

    # Derive the SOTA numeric video ids from the configured video filenames.
    video_df['video_id'] = video_df['video'].map(extract_video_id)
    # Fail fast when two configured videos collapse to the same numeric id.
    duplicate_df = video_df[video_df.duplicated(subset=['video_id'], keep=False)]
    assert duplicate_df.empty, f"Duplicate numeric video ids in test split for {dataset}:\n{duplicate_df}"

    return video_df[['dataset', 'videoset', 'video', 'video_id']].drop_duplicates().reset_index(drop=True)


def build_available_tracking_manifest(tracks_dir: str) -> pd.DataFrame:
    # Fail fast when the raw SOTA tracking directory is missing.
    assert os.path.isdir(tracks_dir), f"Tracking directory not found: {tracks_dir}"

    # Collect all raw tracking JSON files under param-id directories.
    tracking_paths = sorted(Path(tracks_dir).glob('*/*.json'))
    # Fail fast when the tracking directory tree is unexpectedly empty.
    assert tracking_paths, f"No tracking JSON files found in {tracks_dir}"

    # Materialize the discovered tracking files as a DataFrame for vectorized validation.
    tracking_df = pd.DataFrame.from_records([
        {
            'param_id': int(path.parent.name),
            'video_id': int(path.stem),
            'input_json_path': str(path),
        }
        for path in tracking_paths
    ])

    # Fail fast when duplicate param/video pairs would make the transform ambiguous.
    duplicate_df = tracking_df[tracking_df.duplicated(subset=['param_id', 'video_id'], keep=False)]
    assert duplicate_df.empty, f"Duplicate tracking files found in {tracks_dir}:\n{duplicate_df}"

    return tracking_df


def normalize_otif_stat_csv(stat_csv_input: str) -> pd.DataFrame:
    # Load the raw OTIF stat CSV emitted by the SOTA system.
    input_df = pd.read_csv(stat_csv_input).copy()
    # Rename the unnamed index column to param_id when OTIF exported it that way.
    input_df = input_df.rename(columns={'Unnamed: 0': 'param_id'})
    # Fail fast when the raw OTIF CSV does not expose param ids.
    assert 'param_id' in input_df.columns, f"param_id column not found in {stat_csv_input}"

    # Pick the runtime column used by the current OTIF export version.
    runtime_column = 'runtime' if 'runtime' in input_df.columns else 'runtime_total'
    assert runtime_column in input_df.columns, f"Runtime column not found in {stat_csv_input}"

    # Keep the native OTIF parameter columns and expose the canonical runtime column.
    stat_df = input_df.assign(
        param_id=input_df['param_id'].astype(int),
        runtime=input_df[runtime_column],
    )[['param_id', 'detector_cfg', 'segmentation_cfg', 'tracker_cfg', 'runtime']].copy()

    # Fail fast when the OTIF export contains duplicate param ids.
    duplicate_df = stat_df[stat_df.duplicated(subset=['param_id'], keep=False)]
    assert duplicate_df.empty, f"Duplicate OTIF param_id rows found in {stat_csv_input}:\n{duplicate_df}"

    return stat_df.sort_values('param_id').reset_index(drop=True)


def normalize_leap_stat_csv(stat_csv_input: str) -> pd.DataFrame:
    # Load the raw LEAP stat CSV emitted by the SOTA system.
    input_df = pd.read_csv(stat_csv_input).copy()

    # Keep only the total row because LEAP exposes one effective parameter set.
    total_df = input_df.query("video_name == 'total'").copy()
    assert len(total_df) == 1, f"Expected exactly one LEAP total row in {stat_csv_input}, found {len(total_df)}"

    # Validate the runtime columns required by the existing LEAP runtime contract.
    for column in ['inference_total', 'decode']:
        assert column in total_df.columns, f"Column {column} not found in {stat_csv_input}"

    # Preserve the historical LEAP runtime derivation while normalizing the stat schema.
    stat_df = total_df.assign(
        param_id=0,
        detector_cfg=pd.NA,
        segmentation_cfg=pd.NA,
        tracker_cfg=pd.NA,
        runtime=total_df['inference_total'] - total_df['decode'],
    )[['param_id', 'detector_cfg', 'segmentation_cfg', 'tracker_cfg', 'runtime']].copy()

    return stat_df.reset_index(drop=True)


def build_tracking_transform_manifest(system: str,
                                      dataset: str,
                                      stat_df: pd.DataFrame,
                                      tracks_dir: str) -> pd.DataFrame:
    # Materialize the configured test videos used by downstream SOTA evaluation.
    expected_video_df = build_expected_test_video_manifest(dataset)
    # Materialize the configured param ids from the normalized stat CSV.
    expected_param_df = stat_df[['param_id']].drop_duplicates().sort_values('param_id').reset_index(drop=True)
    # Cross join test videos and param ids so every configured pair is validated.
    expected_df = expected_video_df.merge(expected_param_df, how='cross')

    # Materialize the raw tracking files discovered on disk.
    available_df = build_available_tracking_manifest(tracks_dir)
    # Join the expected param/video grid to the discovered tracking paths.
    transform_df = expected_df.merge(available_df, on=['param_id', 'video_id'], how='left')

    # Fail fast when any configured param/video pair is missing a raw SOTA tracking file.
    missing_df = transform_df[transform_df['input_json_path'].isna()]
    assert missing_df.empty, (
        "Missing SOTA tracking files for configured dataset/test videos:\n"
        f"{missing_df[['dataset', 'videoset', 'video', 'param_id', 'video_id']]}"
    )

    # Warn when the raw SOTA export contains extra videos outside the configured test split.
    unexpected_df = available_df.merge(
        expected_df[['param_id', 'video_id']],
        on=['param_id', 'video_id'],
        how='left',
        indicator=True,
    )
    unexpected_df = unexpected_df[unexpected_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    if not unexpected_df.empty:
        print(
            f"  Warning: Ignoring {len(unexpected_df)} extra raw tracking files outside the configured test split "
            f"for {system.upper()} on {dataset}"
        )

    # Resolve the transformed output path for every validated param/video pair.
    transform_df['output_jsonl_path'] = [
        str(cache.sota(system, dataset, video, 'tracking_results', f'{int(param_id):03d}', 'tracking.jsonl'))
        for video, param_id in zip(transform_df['video'], transform_df['param_id'], strict=True)
    ]

    return transform_df[['dataset', 'videoset', 'video', 'video_id', 'param_id', 'input_json_path', 'output_jsonl_path']]


def transform_tracking_manifest(transform_df: pd.DataFrame):
    # Keep the manifest local so the caller receives the derived sample-rate summary.
    transform_df = transform_df.copy()
    # Convert each validated raw tracking file to the shared JSONL format.
    for row in transform_df.itertuples(index=False):
        transform_tracking_json(str(row.input_json_path), str(row.output_jsonl_path), str(row.dataset), str(row.video))


def setup_dataset_paths(sota_dir: str, dataset: str, system: str) -> tuple[str, str, str]:
    # Resolve the SOTA dataset directory name for the current configured dataset.
    dataset_in = dataset_name_in_sota(dataset)
    # Resolve the raw SOTA dataset directory on disk.
    sota_dataset_dir = os.path.join(sota_dir, dataset_in)
    # Fail fast when the configured SOTA dataset directory is missing.
    assert os.path.exists(sota_dataset_dir), f"SOTA dataset directory {sota_dataset_dir} does not exist"

    # Resolve the transformed output root for the current system/dataset pair.
    output_dataset_dir = str(cache.sota(system, dataset))
    # Ensure the transformed output root exists before writing files beneath it.
    os.makedirs(output_dataset_dir, exist_ok=True)

    return dataset_in, sota_dataset_dir, output_dataset_dir


def process_otif_dataset(sota_dir: str, dataset: str):
    # Resolve the OTIF input/output directories for the configured dataset.
    dataset_in, sota_dataset_dir, output_dataset_dir = setup_dataset_paths(sota_dir, dataset, 'otif')
    # Resolve the raw OTIF stat CSV and tracking directory.
    stat_csv_input = os.path.join(sota_dataset_dir, f'otif_{dataset_in}.csv')
    tracks_dir = os.path.join(sota_dataset_dir)  # , f'otif_{dataset_in}_tracks')
    # Normalize the OTIF stat CSV to the shared schema.
    stat_df = normalize_otif_stat_csv(stat_csv_input)
    # Build the validated transform manifest for all configured test videos and param ids.
    transform_df = build_tracking_transform_manifest('otif', dataset, stat_df, tracks_dir)
    # Run the actual file-by-file tracking conversion step.
    transform_tracking_manifest(transform_df)

    # Persist the normalized OTIF stat CSV used by downstream evaluation scripts.
    stat_csv_output = os.path.join(output_dataset_dir, 'stat.csv')
    stat_df.to_csv(stat_csv_output, index=False)


def process_leap_dataset(sota_dir: str, dataset: str):
    # Resolve the LEAP input/output directories for the configured dataset.
    dataset_in, sota_dataset_dir, output_dataset_dir = setup_dataset_paths(sota_dir, dataset, 'leap')
    # Resolve the raw LEAP stat CSV and tracking directory.
    stat_csv_input = os.path.join(sota_dataset_dir, f'leap_{dataset_in}.csv')
    tracks_dir = os.path.join(sota_dataset_dir)  # , f'leap_{dataset_in}_tracks')
    # Normalize the LEAP stat CSV to the shared schema.
    stat_df = normalize_leap_stat_csv(stat_csv_input)
    # Build the validated transform manifest for the fixed LEAP param id.
    transform_df = build_tracking_transform_manifest('leap', dataset, stat_df, tracks_dir)
    # Run the actual file-by-file tracking conversion step.
    transform_tracking_manifest(transform_df)

    # Persist the normalized LEAP stat CSV used by downstream evaluation scripts.
    stat_csv_output = os.path.join(output_dataset_dir, 'stat.csv')
    stat_df.to_csv(stat_csv_output, index=False)


def process_dataset(sota_dir: str, dataset: str):
    # Resolve the raw SOTA dataset directory so system detection stays explicit.
    dataset_in = dataset_name_in_sota(dataset)
    sota_dataset_dir = os.path.join(sota_dir, dataset_in)
    # Fail fast when the configured dataset is absent from the SOTA export root.
    assert os.path.exists(sota_dataset_dir), f"SOTA dataset directory {sota_dataset_dir} does not exist"

    # Resolve the raw OTIF and LEAP stat CSVs used to detect available systems.
    otif_csv = os.path.join(sota_dataset_dir, f'otif_{dataset_in}.csv')
    leap_csv = os.path.join(sota_dataset_dir, f'leap_{dataset_in}.csv')

    # Transform OTIF outputs when the raw OTIF export exists for the dataset.
    if os.path.exists(otif_csv):
        process_otif_dataset(sota_dir, dataset)

    # Transform LEAP outputs when the raw LEAP export exists for the dataset.
    if os.path.exists(leap_csv):
        process_leap_dataset(sota_dir, dataset)


def main(args):
    # Resolve the raw SOTA export root requested by the caller.
    sota_dir = args.sota_dir
    assert args.test, "This script only supports the test videoset"

    # Fail fast when the SOTA export root is missing.
    assert os.path.exists(sota_dir), f"SOTA directory {sota_dir} does not exist"

    # Log the configured datasets before starting the transform stage.
    print(f"Processing configured datasets: {DATASETS}")

    # Transform every configured dataset using the deterministic config-driven workflow.
    for dataset in DATASETS:
        process_dataset(sota_dir, dataset)

    # Log the destination root that now contains the transformed SOTA outputs.
    print(f"\nTransformation complete. Output saved to: {cache.root('SOTA')}")


if __name__ == '__main__':
    main(parse_args())
