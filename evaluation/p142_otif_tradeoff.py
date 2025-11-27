#!/usr/local/bin/python

import argparse
import json
import os

import pandas as pd

from polyis.utilities import get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Join OTIF accuracy results to stat.csv by param_id')
    return parser.parse_args()


def extract_accuracy_metrics(result: dict) -> dict:
    """
    Extract and process accuracy metrics from evaluation result dictionary.
    
    Extracts HOTA metrics (averaging lists) and Count metrics (calculating MAPE)
    from the result dictionary returned by TrackEval evaluation.
    
    Args:
        result (dict): Result dictionary from DATASET.json file
        
    Returns:
        dict: Dictionary with extracted accuracy metrics
    """
    metrics = result['metrics']
    accuracy_metrics = {}
    
    # Extract HOTA metrics and average the lists
    # HOTA metrics come as lists of values (one per threshold), we need the average
    if 'HOTA' in metrics:
        hota_metrics = metrics['HOTA']
        # Calculate average of HOTA list values
        accuracy_metrics['HOTA_HOTA'] = sum(hota_metrics['HOTA']) / len(hota_metrics['HOTA'])
        accuracy_metrics['HOTA_AssA'] = sum(hota_metrics['AssA']) / len(hota_metrics['AssA'])
        accuracy_metrics['HOTA_DetA'] = sum(hota_metrics['DetA']) / len(hota_metrics['DetA'])
    
    # Extract Count metrics and calculate MAPE (Mean Absolute Percentage Error)
    # MAPE = |actual - predicted| / actual * 100
    if 'Count' in metrics:
        count = metrics['Count']
        # Calculate MAPE for detections
        accuracy_metrics['Count_DetsMAPE'] = (abs(count['Dets'] - count['GT_Dets']) * 100 / count['GT_Dets']
                                    if count['GT_Dets'] > 0
                                    else (0 if count['Dets'] == 0 else float('inf')))
        
        # Calculate MAPE for tracks
        accuracy_metrics['Count_TracksMAPE'] = (abs(count['IDs'] - count['GT_IDs']) * 100 / count['GT_IDs']
                                      if count['GT_IDs'] > 0
                                      else (0 if count['IDs'] == 0 else float('inf')))
    
    return accuracy_metrics


def load_accuracy_results(dataset: str) -> pd.DataFrame:
    """
    Load accuracy results for all param_ids from DATASET.json files.
    
    Scans the accuracy results directory to find all param_id directories
    and loads the combined dataset results from each DATASET.json file.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        pd.DataFrame: DataFrame with param_id and accuracy metrics columns
    """
    # Construct path to accuracy results directory
    accuracy_dir = os.path.join(CACHE_DIR, 'SOTA', 'otif', dataset, 'accuracy', 'raw')
    assert os.path.exists(accuracy_dir), f"Accuracy directory {accuracy_dir} does not exist"
    
    # Collect accuracy results for all param_ids
    accuracy_results = []
    
    # Iterate through all param_id directories
    for param_dir_name in os.listdir(accuracy_dir):
        param_dir = os.path.join(accuracy_dir, param_dir_name)
        if not os.path.isdir(param_dir):
            continue
        
        # Parse param_id from directory name (should be formatted as 000, 001, etc.)
        param_id = int(param_dir_name)
        
        # Construct path to DATASET.json file (contains combined results across all videos)
        dataset_json_path = os.path.join(param_dir, 'DATASET.json')
        assert os.path.exists(dataset_json_path), f"DATASET.json not found for param_id {param_id}"
        
        # Load and parse the accuracy result JSON file
        with open(dataset_json_path, 'r') as f:    
            result_data = json.load(f)
        
        # Extract accuracy metrics from the result
        accuracy_metrics = extract_accuracy_metrics(result_data)
        
        # Add param_id to the metrics dictionary
        accuracy_metrics['param_id'] = param_id
        
        # Add to results list
        accuracy_results.append(accuracy_metrics)
        print(f"Loaded accuracy results for param_id {param_id}")
    
    # Convert results list to DataFrame
    if not accuracy_results:
        raise ValueError(f"No accuracy results found for dataset {dataset}")
    
    return pd.DataFrame.from_records(accuracy_results)


def join_accuracy_to_stat(dataset: str):
    """
    Join accuracy results to stat.csv file by param_id.
    
    Reads the existing stat.csv file, loads accuracy results, and merges them
    together by param_id. The merged results are saved to stat_with_accuracy.csv.
    
    Args:
        dataset (str): Dataset name
    """
    # Construct path to stat.csv file
    stat_csv_path = os.path.join(CACHE_DIR, 'SOTA', 'otif', dataset, 'stat.csv')
    assert os.path.exists(stat_csv_path), f"stat.csv not found: {stat_csv_path}"
    
    # Load existing stat.csv file
    stat_df = pd.read_csv(stat_csv_path)
    print(f"Loaded stat.csv with {len(stat_df)} rows")
    
    # Validate that param_id column exists
    assert 'param_id' in stat_df.columns, f"param_id column not found in stat.csv"
    
    # Load accuracy results
    accuracy_df = load_accuracy_results(dataset)
    print(f"Loaded accuracy results for {len(accuracy_df)} param_ids")
    
    # Merge accuracy results into stat.csv by param_id
    # This performs a left join, keeping all rows from stat_df
    merged_df = stat_df.merge(accuracy_df, on='param_id', how='left')
    
    # Validate that all param_ids in stat.csv have corresponding accuracy results
    missing_param_ids = merged_df[merged_df['HOTA_HOTA'].isna()]['param_id'].tolist()
    if missing_param_ids:
        raise ValueError(f"Missing accuracy results for param_ids: {missing_param_ids}")
    
    # Save merged results to a new file with accuracy metrics
    output_csv_path = os.path.join(CACHE_DIR, 'SOTA', 'otif', dataset, 'tradeoff.csv')
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Saved merged results with accuracy metrics: {output_csv_path}")
    print(f"Added columns: {list(accuracy_df.columns)}")


def main(args):
    """
    Main function that orchestrates the accuracy-to-stat joining process.
    
    This function serves as the entry point for the script. It:
    1. Loads accuracy results from DATASET.json files for each param_id
    2. Joins accuracy metrics to the existing stat.csv file by param_id
    3. Saves the updated stat.csv file with accuracy metrics included
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p141_otif_accuracy.py in:
          {CACHE_DIR}/SOTA/otif/{dataset}/accuracy/raw/{param_id:03d}/DATASET.json
        - The stat.csv file should exist at:
          {CACHE_DIR}/SOTA/otif/{dataset}/stat.csv
        - Merged results are saved to:
          {CACHE_DIR}/SOTA/otif/{dataset}/stat_with_accuracy.csv
        - Output CSV will include additional columns:
          HOTA_HOTA, HOTA_AssA, HOTA_DetA, Count_DetsMAPE, Count_TracksMAPE
    """
    print(f"Starting accuracy-to-stat joining for datasets: {DATASETS}")
    
    # Process each dataset separately
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        
        try:
            # Join accuracy results to stat.csv for this dataset
            join_accuracy_to_stat(dataset)
            print(f"Successfully processed dataset: {dataset}")
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            raise
    
    print(f"\nAccuracy-to-stat joining complete for all datasets")


if __name__ == '__main__':
    main(parse_args())

