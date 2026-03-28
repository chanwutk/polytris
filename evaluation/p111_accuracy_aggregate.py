#!/usr/local/bin/python

import argparse
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from evaluation.manifests import build_split_variant_manifest
from polyis.io import cache
from polyis.pareto import load_pareto_params, pareto_params_exist
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def parse_result(result: dict) -> dict:
    # Start from the split-aware metadata emitted by the compute stage.
    parsed = {
        'dataset': result['dataset'],
        'videoset': result['videoset'],
        'variant': result['variant'],
        'variant_id': result['variant_id'],
        'classifier': result['classifier'],
        'tilesize': result['tilesize'],
        'sample_rate': result['sample_rate'],
        'tracking_accuracy_threshold': result['tracking_accuracy_threshold'],
        'tilepadding': result['tilepadding'],
        'canvas_scale': result['canvas_scale'],
        'tracker': result['tracker'],
    }

    # Resolve the nested TrackEval metric payload once for flattening.
    metrics = result['metrics']

    # Flatten the HOTA family into scalar summary columns when present.
    if 'HOTA' in metrics:
        parsed['HOTA_HOTA'] = sum(metrics['HOTA']['HOTA']) / len(metrics['HOTA']['HOTA'])
        parsed['HOTA_AssA'] = sum(metrics['HOTA']['AssA']) / len(metrics['HOTA']['AssA'])
        parsed['HOTA_DetA'] = sum(metrics['HOTA']['DetA']) / len(metrics['HOTA']['DetA'])

    # Flatten the Count family into scalar percentage errors when present.
    if 'Count' in metrics:
        count = metrics['Count']
        parsed['Count_DetsMAPE'] = (abs(count['Dets'] - count['GT_Dets']) * 100 / count['GT_Dets']
                                    if count['GT_Dets'] > 0
                                    else (0 if count['Dets'] == 0 else float('inf')))
        parsed['Count_TracksMAPE'] = (abs(count['IDs'] - count['GT_IDs']) * 100 / count['GT_IDs']
                                      if count['GT_IDs'] > 0
                                      else (0 if count['IDs'] == 0 else float('inf')))

    return parsed


def load_dataset_accuracy_rows(dataset: str, videoset: str) -> pd.DataFrame:
    # Build the configured split/variant manifest for the single requested videoset.
    split_variant_df = build_split_variant_manifest(datasets=[dataset], videosets=[videoset], include_naive=True)
    # Fail fast when the dataset has no configured split-level accuracy tasks.
    assert not split_variant_df.empty, f"No split-level accuracy manifest rows found for dataset {dataset}"

    # Collect the flattened split-level rows in manifest order.
    rows: list[dict] = []

    # Load each expected combined result file without scanning directories.
    for task_row in split_variant_df.to_dict('records'):
        # For test: only aggregate Pareto-optimal Polytris variants.
        if videoset == 'test' and task_row['variant'] == 'polytris':
            pareto_df = load_pareto_params(dataset)
            pareto_variant_ids = set(pareto_df['variant_id'].dropna().unique())
            if task_row['variant_id'] not in pareto_variant_ids:
                continue

        # Resolve the expected combined TrackEval output for this split-level task.
        result_path = cache.eval(dataset, 'acc', 'raw', task_row['videoset'], task_row['variant_id'], 'DATASET.json')
        # Fail fast when the configured result file is missing.
        assert os.path.exists(result_path), f"Accuracy result not found: {result_path}"

        # Load the combined result JSON payload for this split-level task.
        with open(result_path, 'r') as f:
            result = json.load(f)

        # Flatten the combined result row into the canonical CSV schema.
        rows.append(parse_result(result))

    # Materialize the canonical split-level accuracy table.
    return pd.DataFrame.from_records(rows)


def save_accuracy_csv(results: pd.DataFrame, output_dir: Path):
    # Ensure the output directory exists before writing the canonical CSV.
    os.makedirs(output_dir, exist_ok=True)
    # Resolve the canonical accuracy CSV path.
    output_path = output_dir / 'accuracy.csv'
    # Persist the split-level accuracy table without an index column.
    results.to_csv(output_path, index=False)
    # Log the save location for traceability.
    print(f"Saved accuracy results to: {output_path}")


def main(args):
    # Log the configured datasets before aggregation starts.
    print(f"Starting split-level accuracy aggregation for datasets: {DATASETS}")

    # Resolve the single videoset from the mutually exclusive CLI flags.
    videoset = 'test' if args.test else 'valid'

    # Assert Pareto params exist when processing the test split.
    if videoset == 'test':
        for dataset in DATASETS:
            assert pareto_params_exist(dataset), \
                f"Pareto params not found for {dataset}. Run p135_pareto_extract.py first."

    # Aggregate the configured split-level results for each dataset independently.
    for dataset in DATASETS:
        # Log the dataset currently being aggregated.
        print(f"\nProcessing dataset: {dataset}")
        # Load the split-level combined rows for the current dataset.
        results = load_dataset_accuracy_rows(dataset, videoset)
        # Fail fast when aggregation produced no rows.
        assert not results.empty, f"No accuracy results found for dataset {dataset}"
        # Save the canonical split-level accuracy CSV for the current dataset.
        save_accuracy_csv(results, cache.eval(dataset, 'acc'))


def _cleanup_output_dirs():
    # Keep this helper for local cleanup workflows outside the normal pipeline.
    for dataset in DATASETS:
        # Resolve the dataset-local accuracy output directory.
        output_dir = cache.eval(dataset, 'acc')
        # Remove the directory only when it exists.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    main(parse_args())
