#!/usr/local/bin/python

import argparse
import json
import os

import pandas as pd

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate raw accuracy results from p070_accuracy_compute.py into CSV files')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def find_saved_results(cache_dir: str, dataset: str) -> list[tuple[str, int, str]]:
    """
    Find all classifier/tilesize/tilepadding combinations with saved accuracy results.
    
    Scans the evaluation directory to discover all classifier/tilesize/tilepadding combinations
    that have completed accuracy evaluation results available.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        list[tuple[str, int, str]]: list of (classifier, tilesize, tilepadding) tuples
    """
    # Construct path to evaluation directory for this dataset
    evaluation_dir = os.path.join(cache_dir, dataset, 'evaluation', '070_accuracy')
    assert os.path.exists(evaluation_dir), f"Evaluation directory {evaluation_dir} does not exist"
    
    # Construct path to raw results directory
    raw_dir = os.path.join(evaluation_dir, 'raw')
    assert os.path.exists(raw_dir), f"Raw results directory {raw_dir} does not exist"
    
    # Collect all classifier/tilesize/tilepadding combinations
    classifier_tile_combinations: list[tuple[str, int, str]] = []
    
    # Iterate through all classifier-tilesize-tilepadding directories
    for classifier_tilesize_tilepadding in os.listdir(raw_dir):
        # Parse classifier, tile size, and tilepadding from directory name
        parts = classifier_tilesize_tilepadding.split('_')
        assert len(parts) == 3, f"Expected format 'classifier_tilesize_tilepadding', got '{classifier_tilesize_tilepadding}'"
        classifier, tilesize, tilepadding = parts
        ts = int(tilesize)
        
        # Verify that the required DATASET.json file exists
        # This ensures the evaluation was completed successfully
        dataset_results_path = os.path.join(raw_dir, f'{classifier}_{ts}_{tilepadding}', 'DATASET.json')
        assert os.path.exists(dataset_results_path), f"Dataset results path {dataset_results_path} does not exist"
        
        # Add this combination to our list
        classifier_tile_combinations.append((classifier, ts, tilepadding))
    
    return classifier_tile_combinations


def parse_result(result: dict) -> dict:
    """
    Parse raw accuracy result dictionary into flattened format for analysis.
    
    Converts nested metric structures into flat column names (e.g., HOTA.HOTA)
    and computes derived metrics like MAPE for Count metrics.
    
    Args:
        result (dict): Raw result dictionary from JSON file
        
    Returns:
        dict: Parsed result with flattened metrics
    """
    parsed = {
        'Video': result['video'] or 'Combined',
        'Dataset': result['dataset'],
        'Classifier': result['classifier'],
        'Tile_Size': result['tilesize'],
        'Tile_Padding': result['tilepadding'],
    }

    metrics = result['metrics']
    if 'HOTA' in metrics:
        parsed['HOTA_HOTA'] = sum(metrics['HOTA']['HOTA']) / len(metrics['HOTA']['HOTA'])
        parsed['HOTA_AssA'] = sum(metrics['HOTA']['AssA']) / len(metrics['HOTA']['AssA'])
        parsed['HOTA_DetA'] = sum(metrics['HOTA']['DetA']) / len(metrics['HOTA']['DetA'])
    if 'CLEAR' in metrics:
        raise NotImplementedError("CLEAR metrics not implemented")
        # base['CLEAR.MOTA'] = sum(metrics['CLEAR']['MOTA']) / len(metrics['CLEAR']['MOTA'])
    if 'Count' in metrics:
        count = metrics['Count']
        # mean absolute percentage error
        parsed['Count_DetsMAPE'] = (abs(count['Dets'] - count['GT_Dets']) * 100 / count['GT_Dets']
                                    if count['GT_Dets'] > 0
                                    else (0 if count['Dets'] == 0 else float('inf')))
        parsed['Count_TracksMAPE'] = (abs(count['IDs'] - count['GT_IDs']) * 100 / count['GT_IDs']
                                      if count['GT_IDs'] > 0
                                      else (0 if count['IDs'] == 0 else float('inf')))
    return parsed


def load_saved_results(dataset: str, combined: bool = False) -> pd.DataFrame:
    """
    Load saved accuracy results from result files.
    
    Loads either individual video results or combined dataset results based on
    the combined parameter. Individual results are used for per-video visualizations,
    while combined results are used for dataset-level visualizations.
    
    Args:
        dataset (str): Dataset name
        combined (bool): Whether to load combined results (DATASET.json) or individual video results
        
    Returns:
        pd.DataFrame: DataFrame of parsed evaluation results
    """
    # Find all classifier/tilesize combinations with available results
    classifier_tile_combinations = find_saved_results(CACHE_DIR, dataset)
    assert len(classifier_tile_combinations) > 0, f"No saved results found for dataset {dataset}"

    # Initialize results list
    results = []
    evaluation_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
    raw_dir = os.path.join(evaluation_dir, 'raw')
    
    # Process each classifier/tilesize/tilepadding combination
    for classifier, tilesize, tilepadding in classifier_tile_combinations:
        combination_dir = os.path.join(raw_dir, f'{classifier}_{tilesize}_{tilepadding}')
        
        # Load result files based on the combined parameter
        for filename in os.listdir(combination_dir):
            # Load DATASET.json if combined=True, otherwise load individual video files
            if filename.endswith('.json') and (filename == 'DATASET.json') == combined:
                results_path = os.path.join(combination_dir, filename)
                
                print(f"Loading results from {results_path}")
                # Load and parse JSON result file
                with open(results_path, 'r') as f:
                    result_data = json.load(f)
                    results.append(parse_result(result_data))
    
    print(f"Loaded {len(results)} saved evaluation results")
    return pd.DataFrame.from_records(results)


def save_results_csv(results: pd.DataFrame, output_dir: str, combined: bool = False):
    """
    Save aggregated accuracy results to CSV file.
    
    Saves the results DataFrame to a CSV file in the specified output directory,
    using the same naming convention as the visualization script.
    
    Args:
        results (pd.DataFrame): DataFrame of parsed accuracy results
        output_dir (str): Output directory where CSV file will be saved
        combined (bool): Whether these are combined results (affects filename prefix)
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV for further analysis
    csv_file_path = os.path.join(output_dir, f'accuracy{'_combined' if combined else ''}.csv')
    results.to_csv(csv_file_path, index=False)
    print(f"Saved accuracy results to: {csv_file_path}")


def aggregate_accuracy_results(dataset: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate raw accuracy results and save CSV files.
    
    Loads both individual video results and combined dataset results from raw JSON files,
    parses them into DataFrames, and saves CSV files to the output directory.
    
    Args:
        dataset (str): Dataset name
        output_dir (str): Output directory where CSV files will be saved
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame of individual video results
            - DataFrame of combined dataset results
    """
    print(f"Aggregating accuracy results for dataset: {dataset}")
    
    # Load individual video results for per-video analysis
    individual_results = load_saved_results(dataset, combined=False)
    assert len(individual_results) > 0, f"No individual results found for dataset {dataset}"
    
    # Load combined dataset results for dataset-level analysis
    combined_results = load_saved_results(dataset, combined=True)
    assert len(combined_results) > 0, f"No combined results found for dataset {dataset}"
    
    # Save CSV files using the same format as visualization script
    save_results_csv(individual_results, output_dir, combined=False)
    save_results_csv(combined_results, output_dir, combined=True)
    
    return individual_results, combined_results


def main(args):
    """
    Main function that orchestrates the accuracy result aggregation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all classifier/tilesize/tilepadding combinations with saved accuracy results
    2. Loads raw JSON result files and parses them into DataFrames
    3. Saves aggregated results as CSV files for further analysis
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/raw/{classifier}_{tilesize}_{tilepadding}/
          ├── DATASET.json (combined results)
          ├── {video}.json (individual video results)
          └── LOG.txt (evaluation logs)
        - CSV files are saved to: {CACHE_DIR}/{dataset}/evaluation/071_accuracy_aggregate/
    """
    print(f"Starting accuracy result aggregation for datasets: {args.datasets}")
    
    # Process each dataset separately
    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Create output directory for this dataset's aggregated results
        output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
        
        # Aggregate and save results
        aggregate_accuracy_results(dataset, output_dir)
        
        print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main(parse_args())
