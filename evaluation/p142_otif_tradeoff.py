#!/usr/local/bin/python

import argparse
import json
import os

import pandas as pd

from evaluation.manifests import load_sota_stat_manifest
from polyis.io import cache
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                        help='Process test videoset')
    return parser.parse_args()

def extract_accuracy_metrics(result: dict) -> dict:
    # Resolve the nested TrackEval metric payload for the current param_id.
    metrics = result['metrics']
    accuracy_metrics: dict[str, float] = {}

    # Flatten the HOTA family into scalar summary columns when present.
    if 'HOTA' in metrics:
        accuracy_metrics['HOTA_HOTA'] = sum(metrics['HOTA']['HOTA']) / len(metrics['HOTA']['HOTA'])
        accuracy_metrics['HOTA_AssA'] = sum(metrics['HOTA']['AssA']) / len(metrics['HOTA']['AssA'])
        accuracy_metrics['HOTA_DetA'] = sum(metrics['HOTA']['DetA']) / len(metrics['HOTA']['DetA'])

    # Flatten the Count family into scalar percentage errors when present.
    if 'Count' in metrics:
        count = metrics['Count']
        accuracy_metrics['Count_DetsMAPE'] = (
            abs(count['Dets'] - count['GT_Dets']) * 100 / count['GT_Dets']
            if count['GT_Dets'] > 0
            else (0 if count['Dets'] == 0 else float('inf'))
        )
        accuracy_metrics['Count_TracksMAPE'] = (
            abs(count['IDs'] - count['GT_IDs']) * 100 / count['GT_IDs']
            if count['GT_IDs'] > 0
            else (0 if count['IDs'] == 0 else float('inf'))
        )

    return accuracy_metrics


def calculate_max_gap(data: dict) -> int:
    # Resolve the tracker threshold list from the decoded tracker_cfg payload.
    thresholds = data.get("Thresholds", [])

    # Fall back to sample_rate 1 when the payload has fewer than two thresholds.
    if len(thresholds) < 2:
        return 1

    # Compute the doubling gap used by the upstream OTIF tracker configuration.
    gap = 1
    for threshold in thresholds[1:]:
        if threshold == 1.0:
            return gap
        gap *= 2

    return gap


def load_accuracy_results(dataset: str, system: str) -> pd.DataFrame:
    # Materialize the configured SOTA stat manifest so param_id enumeration is deterministic.
    stat_df = load_sota_stat_manifest(system, dataset)

    # Resolve the expected combined accuracy JSON path for each configured param_id.
    accuracy_df = stat_df[['dataset', 'videoset', 'param_id']].drop_duplicates().copy()
    accuracy_df['result_path'] = accuracy_df['param_id'].apply(
        lambda param_id: os.path.join(cache.sota(system, dataset, 'accuracy', 'raw', 'test', f'{int(param_id):03d}'), 'DATASET.json')
    )

    # Fail fast when any configured combined accuracy JSON is missing.
    missing_df = accuracy_df[~accuracy_df['result_path'].map(os.path.exists)]
    assert missing_df.empty, f"Missing SOTA accuracy results:\n{missing_df[['param_id', 'result_path']]}"

    # Load and flatten the combined accuracy JSON for each configured param_id.
    accuracy_df['accuracy_metrics'] = accuracy_df['result_path'].apply(
        lambda result_path: extract_accuracy_metrics(load_json(result_path))
    )

    # Expand the flattened metrics dictionaries into regular DataFrame columns.
    metrics_df = pd.json_normalize(accuracy_df['accuracy_metrics'])
    accuracy_df = pd.concat([accuracy_df.drop(columns=['accuracy_metrics', 'result_path']), metrics_df], axis=1)

    return accuracy_df


def load_json(path: str) -> dict:
    # Read the JSON file through a context manager so file handles are closed promptly.
    with open(path, 'r') as f:
        return json.load(f)


def join_accuracy_to_stat(dataset: str, system: str):
    # Skip systems that have not been transformed for the current dataset.
    stat_path = cache.sota(system, dataset, 'stat.csv')
    if not os.path.exists(stat_path):
        print(f"  Warning: stat.csv not found: {stat_path}, skipping")
        return

    # Load the transformed SOTA stat manifest and the paired accuracy rows.
    stat_df = load_sota_stat_manifest(system, dataset)
    accuracy_df = load_accuracy_results(dataset, system)

    # Join the native SOTA stat columns and the flattened accuracy metrics by param_id.
    tradeoff_df = stat_df.merge(accuracy_df.drop(columns=['dataset', 'videoset']), on='param_id', how='left')
    # Fail fast when any configured param_id is missing accuracy metrics.
    missing_param_ids = tradeoff_df[tradeoff_df['HOTA_HOTA'].isna()]['param_id'].tolist()
    assert not missing_param_ids, f"Missing {system.upper()} accuracy results for param_ids: {missing_param_ids}"

    # Derive the OTIF sample_rate from tracker_cfg and default LEAP to 1.
    if 'tracker_cfg' in tradeoff_df.columns:
        tradeoff_df['sample_rate'] = tradeoff_df['tracker_cfg'].apply(
            lambda tracker_cfg: calculate_max_gap(json.loads(tracker_cfg)) if isinstance(tracker_cfg, str) else 1
        )
    else:
        tradeoff_df['sample_rate'] = 1

    # Scale runtime by the ratio of sampled to original frames saved by the transform stage.
    tradeoff_df['time'] = tradeoff_df['runtime'] * tradeoff_df['adjustment_factor']

    # Persist the native SOTA parameter columns alongside the split-aware metadata.
    output_csv_path = cache.sota(system, dataset, 'tradeoff.csv')
    tradeoff_df.to_csv(output_csv_path, index=False)
    print(f"Saved merged {system.upper()} results with accuracy metrics: {output_csv_path}")


def main(args):
    assert args.test, "This script only supports the test videoset"

    # Log the configured datasets before the SOTA tradeoff join starts.
    print(f"Starting accuracy-to-stat joining for datasets: {DATASETS}")

    # Join the transformed stat.csv and accuracy rows for each configured dataset/system pair.
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        for system in ['otif', 'leap']:
            join_accuracy_to_stat(dataset, system)


if __name__ == '__main__':
    main(parse_args())
