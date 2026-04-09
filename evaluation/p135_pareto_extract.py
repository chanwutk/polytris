#!/usr/local/bin/python

"""
Extract Pareto-optimal parameter sets from valid-split tradeoff data.

For each configured dataset, loads the tradeoff CSV produced by the valid pass,
computes the Pareto front on time vs HOTA_HOTA, and saves the resulting parameter
sets to the evaluation cache for use by the test-pass pipeline scripts.
"""

import argparse

from polyis.pareto import compute_pareto_fronts_by_group, save_pareto_params
from polyis.utilities import get_config, load_tradeoff_data, split_tradeoff_variants


config = get_config()
DATASETS = config['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def main(args):
    # Resolve the single videoset from the mutually exclusive CLI flags.
    videoset = 'test' if args.test else 'valid'

    # Pareto extraction only operates on valid data; skip when test is requested.
    if videoset != 'valid':
        print("Skipping: Pareto extraction only operates on valid data.")
        return

    # Log the datasets being processed for traceability.
    print(f"Extracting Pareto params for datasets: {DATASETS}")

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")

        # Load the canonical split-level tradeoff table for this dataset.
        tradeoff_df = load_tradeoff_data(dataset)

        # Ensure the dataset column is present for downstream grouping.
        if 'dataset' not in tradeoff_df.columns:
            tradeoff_df['dataset'] = dataset

        # Split tradeoff data into Polytris and naive subsets.
        polytris_df, _ = split_tradeoff_variants(tradeoff_df)

        # Restrict to the valid videoset to avoid test-data leakage.
        polytris_df = polytris_df[polytris_df['videoset'] == 'valid'].copy()

        # Exclude the Perfect classifier since it is not a real deployable configuration.
        polytris_df = polytris_df[polytris_df['classifier'] != 'Perfect'].copy()

        if polytris_df.empty:
            print(f"  Warning: No valid Polytris rows found for {dataset}; skipping")
            continue

        # Compute the Pareto front per dataset group (minimize time, maximize HOTA).
        pareto_df = compute_pareto_fronts_by_group(
            polytris_df,
            group_cols=['dataset'],
            x_col='time',
            y_col='HOTA_HOTA',
            minx=True, miny=False,
        )

        if pareto_df.empty:
            print(f"  Warning: Pareto front is empty for {dataset}; skipping")
            continue

        # Log the number of Pareto-optimal points found.
        print(f"  Found {len(pareto_df)} Pareto-optimal rows for {dataset}")

        # Save the Pareto parameter sets to the evaluation cache.
        save_pareto_params(dataset, pareto_df)

    print("\nPareto extraction complete.")


if __name__ == '__main__':
    main(parse_args())
