#!/usr/local/bin/python

"""
Extract Pareto-optimal parameter sets from valid-split tradeoff data.

For each configured dataset, loads the tradeoff CSV produced by the valid pass,
computes Pareto fronts for each ablation condition (full system, no frame
sampling, neither sampling nor pruning), and saves the union of all
Pareto-optimal parameter sets to the evaluation cache.  The union drives
test-pass filtering so that every parameter combo needed by any ablation
curve is executed on the test split.
"""

import argparse

import pandas as pd

from evaluation.ablation import ABLATION_CONDITIONS, filter_by_ablation_condition
from polyis.pareto import PARETO_PARAM_COLS, compute_pareto_fronts_by_group, save_pareto_params
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

        # Compute a Pareto front for each ablation condition and collect the union.
        pareto_parts: list[pd.DataFrame] = []

        for condition in ABLATION_CONDITIONS:
            # Filter to the parameter subset defined by this ablation condition.
            condition_df = filter_by_ablation_condition(polytris_df, condition)

            if condition_df.empty:
                print(f"  [{condition.label}] No rows after filtering; skipping")
                continue

            # Compute the Pareto front per dataset group (minimize time, maximize HOTA).
            pareto_df = compute_pareto_fronts_by_group(
                condition_df,
                group_cols=['dataset'],
                x_col='time',
                y_col='HOTA_HOTA',
                minx=True, miny=False,
            )

            if pareto_df.empty:
                print(f"  [{condition.label}] Pareto front is empty; skipping")
                continue

            print(f"  [{condition.label}] {len(pareto_df)} Pareto-optimal rows")
            pareto_parts.append(pareto_df)

        if not pareto_parts:
            print(f"  Warning: All ablation Pareto fronts empty for {dataset}; skipping")
            continue

        # Merge all ablation fronts into one DataFrame, deduplicating by parameter columns.
        union_df = pd.concat(pareto_parts, ignore_index=True)
        available_param_cols = [c for c in PARETO_PARAM_COLS if c in union_df.columns]
        union_df = union_df.drop_duplicates(subset=available_param_cols)

        # Log the final union size.
        print(f"  Union of all ablation fronts: {len(union_df)} unique param combos")

        # Save the union Pareto parameter sets to the evaluation cache.
        save_pareto_params(dataset, union_df)

    print("\nPareto extraction complete.")


if __name__ == '__main__':
    main(parse_args())
