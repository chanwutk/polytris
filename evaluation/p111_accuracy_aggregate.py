#!/usr/local/bin/python

import json
import os
import shutil

import pandas as pd

from polyis.io import cache
from polyis.utilities import get_config


config = get_config()
DATASETS = config['EXEC']['DATASETS']


def find_saved_results(dataset: str) -> list[tuple[str, str]]:
    """
    Find all split-aware parameter directories with saved accuracy results.

    Args:
        dataset (str): Dataset root name

    Returns:
        list[tuple[str, str]]: List of (split_dataset_name, param_str) tuples
    """
    # Construct path to evaluation directory for this dataset root.
    evaluation_dir = cache.eval(dataset, 'acc')
    assert os.path.exists(evaluation_dir), f"Evaluation directory {evaluation_dir} does not exist"

    # Construct path to raw-results directory.
    raw_dir = cache.eval(dataset, 'acc', 'raw')
    assert os.path.exists(raw_dir), f"Raw results directory {raw_dir} does not exist"

    # Collect discovered split dataset / parameter combinations.
    split_param_combinations: list[tuple[str, str]] = []

    # Iterate over split-aware dataset directories (e.g., dataset, dataset-val).
    for split_dataset_name in sorted(os.listdir(raw_dir)):
        # Build absolute path to split-dataset raw directory.
        split_dataset_dir = os.path.join(raw_dir, split_dataset_name)
        # Skip non-directory entries.
        if not os.path.isdir(split_dataset_dir):
            continue

        # Iterate over per-configuration parameter directories.
        for param_str in sorted(os.listdir(split_dataset_dir)):
            # Build absolute path to parameter directory.
            param_dir = os.path.join(split_dataset_dir, param_str)
            # Skip non-directory entries.
            if not os.path.isdir(param_dir):
                continue

            # Validate combined-result JSON exists to confirm completed evaluation.
            dataset_results_path = os.path.join(param_dir, 'DATASET.json')
            assert os.path.exists(dataset_results_path), f"Dataset results path {dataset_results_path} does not exist"

            # Store this discovered split/parameter pair.
            split_param_combinations.append((split_dataset_name, param_str))

    # Return discovered split/parameter pairs.
    return split_param_combinations


def parse_result(result: dict) -> dict:
    """
    Parse raw accuracy result dictionary into flattened format for analysis.

    Args:
        result (dict): Raw result dictionary from JSON file

    Returns:
        dict: Parsed result with flattened metrics
    """
    # Build base flattened metadata fields.
    parsed = {
        'Video': result['video'] or 'Combined',
        'Dataset': result['dataset'],
        'Videoset': result.get('videoset', 'unknown'),
        'Classifier': result['classifier'],
        'Tile_Size': result['tilesize'],
        'Tile_Padding': result['tilepadding'],
        'Sample_Rate': result.get('sample_rate', 1),  # Default for backward compatibility.
        'Tracking_Accuracy_Threshold': result.get('tracking_accuracy_threshold', None),
        'Canvas_Scale': result.get('canvas_scale', 1.0),  # Default for backward compatibility.
        'Tracker': result.get('tracker', 'unknown'),  # Default for backward compatibility.
    }

    # Extract nested metric payload.
    metrics = result['metrics']

    # Parse HOTA metric family when present.
    if 'HOTA' in metrics:
        parsed['HOTA_HOTA'] = sum(metrics['HOTA']['HOTA']) / len(metrics['HOTA']['HOTA'])
        parsed['HOTA_AssA'] = sum(metrics['HOTA']['AssA']) / len(metrics['HOTA']['AssA'])
        parsed['HOTA_DetA'] = sum(metrics['HOTA']['DetA']) / len(metrics['HOTA']['DetA'])

    # Parse CLEAR metric family when present.
    if 'CLEAR' in metrics:
        raise NotImplementedError("CLEAR metrics not implemented")

    # Parse Count metric family when present.
    if 'Count' in metrics:
        count = metrics['Count']
        # Calculate detection MAPE.
        parsed['Count_DetsMAPE'] = (abs(count['Dets'] - count['GT_Dets']) * 100 / count['GT_Dets']
                                    if count['GT_Dets'] > 0
                                    else (0 if count['Dets'] == 0 else float('inf')))
        # Calculate track-ID MAPE.
        parsed['Count_TracksMAPE'] = (abs(count['IDs'] - count['GT_IDs']) * 100 / count['GT_IDs']
                                      if count['GT_IDs'] > 0
                                      else (0 if count['IDs'] == 0 else float('inf')))

    # Return flattened result row.
    return parsed


def load_saved_results(dataset: str, combined: bool = False) -> pd.DataFrame:
    """
    Load saved accuracy results from result files.

    Args:
        dataset (str): Dataset root name
        combined (bool): Whether to load combined results (`DATASET.json`)

    Returns:
        pd.DataFrame: DataFrame of parsed evaluation results
    """
    # Find split/parameter directories with available results.
    split_param_combinations = find_saved_results(dataset)
    assert len(split_param_combinations) > 0, f"No saved results found for dataset {dataset}"

    # Initialize collected parsed rows.
    results: list[dict] = []

    # Build raw results directory path.
    raw_dir = cache.eval(dataset, 'acc', 'raw')

    # Process each discovered split/parameter directory.
    for split_dataset_name, param_str in split_param_combinations:
        # Build absolute directory path for this split/config pair.
        combination_dir = os.path.join(raw_dir, split_dataset_name, param_str)

        # Iterate over JSON files in this directory.
        for filename in sorted(os.listdir(combination_dir)):
            # Keep only JSON files matching requested combined/non-combined mode.
            if not filename.endswith('.json'):
                continue
            if (filename == 'DATASET.json') != combined:
                continue

            # Build full path to result file.
            results_path = os.path.join(combination_dir, filename)
            # Log current file being loaded.
            print(f"Loading results from {results_path}")

            # Load result JSON payload.
            with open(results_path, 'r') as f:
                result_data = json.load(f)

            # Parse and append flattened result row.
            results.append(parse_result(result_data))

    # Log number of loaded result rows.
    print(f"Loaded {len(results)} saved evaluation results")
    # Return DataFrame built from parsed rows.
    return pd.DataFrame.from_records(results)


def save_results_csv(results: pd.DataFrame, output_dir: str, combined: bool = False):
    """
    Save aggregated accuracy results to CSV file.

    Args:
        results (pd.DataFrame): Parsed accuracy results
        output_dir (str): Output directory path
        combined (bool): Whether these rows are combined split-level rows
    """
    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Build output CSV path based on combined flag.
    csv_file_path = os.path.join(output_dir, f'accuracy{'_combined' if combined else ''}.csv')
    # Write CSV file.
    results.to_csv(csv_file_path, index=False)
    # Log save location.
    print(f"Saved accuracy results to: {csv_file_path}")


def aggregate_accuracy_results(dataset: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate raw accuracy results and save CSV files.

    Args:
        dataset (str): Dataset root name
        output_dir (str): Output directory path

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (individual_rows, combined_rows)
    """
    # Log dataset being aggregated.
    print(f"Aggregating accuracy results for dataset: {dataset}")

    # Load per-video rows.
    individual_results = load_saved_results(dataset, combined=False)
    assert len(individual_results) > 0, f"No individual results found for dataset {dataset}"

    # Load combined split-level rows.
    combined_results = load_saved_results(dataset, combined=True)
    assert len(combined_results) > 0, f"No combined results found for dataset {dataset}"

    # Save individual rows CSV.
    save_results_csv(individual_results, output_dir, combined=False)
    # Save combined rows CSV.
    save_results_csv(combined_results, output_dir, combined=True)

    # Return both DataFrames.
    return individual_results, combined_results


def main():
    """
    Main function that orchestrates the accuracy result aggregation process.

    Note:
        - The script expects raw accuracy results from p110_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/raw/{dataset_or_dataset-val}/{param_str}/
          ├── DATASET.json
          ├── {video}.json
          └── LOG.txt
        - CSV files are saved to: {CACHE_DIR}/{dataset}/evaluation/070_accuracy/
    """
    # Log configured dataset roots.
    print(f"Starting accuracy result aggregation for datasets: {DATASETS}")

    # Process each configured dataset root.
    for dataset in DATASETS:
        # Log current dataset root.
        print(f"\nProcessing dataset: {dataset}")

        # Build output directory path.
        output_dir = cache.eval(dataset, 'acc')

        # Aggregate and save result CSVs.
        aggregate_accuracy_results(dataset, output_dir)

        # Log output location.
        print(f"Results saved to: {output_dir}")


def _cleanup_output_dirs():
    # This helper is unused in normal runs; keep for local cleanup workflows.
    for dataset in DATASETS:
        output_dir = cache.eval(dataset, 'acc')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    main()
