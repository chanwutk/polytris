#!/usr/local/bin/python

import argparse
from multiprocessing import Pool
import os
import shutil

import pandas as pd
from rich.progress import track

from polyis.io import cache
from polyis.utilities import get_config, get_video_frame_count


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
KEY_COLUMNS = ['dataset', 'videoset', 'variant', 'variant_id']


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()
PARAM_COLUMNS = [
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
]


def load_accuracy_results(dataset: str) -> pd.DataFrame:
    # Resolve the canonical split-level accuracy CSV for the current dataset.
    accuracy_path = os.path.join(cache.eval(dataset, 'acc'), 'accuracy.csv')
    # Fail fast when the configured accuracy table is missing.
    assert os.path.exists(accuracy_path), f"Accuracy results not found: {accuracy_path}"

    # Load the split-level accuracy table through pandas.
    accuracy_df = pd.read_csv(accuracy_path)
    # Fail fast when the table is unexpectedly empty.
    assert not accuracy_df.empty, f"No accuracy rows found for dataset {dataset}"

    return accuracy_df


def load_query_runtime_results(dataset: str) -> pd.DataFrame:
    # Resolve the processed query runtime summary emitted by the throughput compute stage.
    query_path = os.path.join(cache.eval(dataset, 'tp', 'measurements'), 'query_execution_overall.csv')
    # Fail fast when the throughput summary is missing.
    assert os.path.exists(query_path), f"Query execution results not found: {query_path}"

    # Load the split-aware query runtime table through pandas.
    query_df = pd.read_csv(query_path)
    # Fail fast when the table is unexpectedly empty.
    assert not query_df.empty, f"No query runtime rows found for dataset {dataset}"

    return query_df


def build_frame_count_table(query_df: pd.DataFrame) -> pd.DataFrame:
    # Resolve the unique dataset/split/video combinations that appear in the query manifest.
    frame_df = query_df[['dataset', 'videoset', 'video']].drop_duplicates().copy()

    # Add the concrete video frame count used for split-level throughput aggregation.
    frame_df['frame_count'] = frame_df.apply(
        lambda row: get_video_frame_count(str(row['dataset']), str(row['video'])),
        axis=1,
    )

    return frame_df


def aggregate_runtime_per_video(query_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the columns that define one split-level runtime configuration per video.
    runtime_key_columns = ['dataset', 'videoset', 'video', 'variant', 'variant_id']

    # Sum stage runtimes into one per-video runtime total for each split-level variant.
    per_video_df = query_df.groupby(runtime_key_columns, dropna=False)['time'].sum().reset_index()

    # Attach the per-video frame count exactly once per video/configuration pair.
    per_video_df = per_video_df.merge(build_frame_count_table(query_df), on=['dataset', 'videoset', 'video'], how='inner')

    return per_video_df


def aggregate_runtime_per_split(query_df: pd.DataFrame) -> pd.DataFrame:
    # Collapse the per-video runtime totals into one split-level runtime row per variant.
    per_video_df = aggregate_runtime_per_video(query_df)
    split_df = per_video_df.groupby(KEY_COLUMNS, dropna=False).agg({
        'time': 'sum',
        'frame_count': 'sum',
    }).reset_index()

    # Recompute throughput from the aggregated split-level runtime totals.
    split_df['throughput_fps'] = split_df['frame_count'] / split_df['time']

    return split_df


def build_tradeoff_table(accuracy_df: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate query runtimes to one split-level row per variant.
    runtime_df = aggregate_runtime_per_split(query_df)

    # Join the split-level runtime totals to the split-level accuracy table.
    tradeoff_df = accuracy_df.merge(runtime_df, on=KEY_COLUMNS, how='inner')
    # Fail fast when the join drops every configured row.
    assert not tradeoff_df.empty, "No split-level tradeoff rows found after joining accuracy and runtime data"

    # Retain the canonical column ordering for downstream visualization scripts.
    ordered_columns = [
        *KEY_COLUMNS,
        *PARAM_COLUMNS,
        'frame_count',
        'time',
        'throughput_fps',
    ]
    metric_columns = [column for column in tradeoff_df.columns if column not in ordered_columns]

    return tradeoff_df[[*ordered_columns, *metric_columns]]


def process_dataset(dataset: str):
    # Log the dataset currently being processed.
    print(f"Processing split-level tradeoff data for dataset: {dataset}")

    # Load the canonical split-level accuracy table.
    accuracy_df = load_accuracy_results(dataset)
    # Load the processed split-aware query runtime table.
    query_df = load_query_runtime_results(dataset)
    # Build the canonical split-level tradeoff table.
    tradeoff_df = build_tradeoff_table(accuracy_df, query_df)

    # Resolve the dataset-local tradeoff output directory.
    output_dir = cache.eval(dataset, 'tradeoff')
    # Persist the canonical tradeoff CSV without an index column.
    tradeoff_df.to_csv(output_dir / 'tradeoff.csv', index=False)


def main(args):
    # Log the configured datasets before tradeoff computation starts.
    print(f"Processing datasets: {DATASETS}")

    # Recreate each dataset-local output directory before writing fresh tradeoff data.
    for dataset in DATASETS:
        output_dir = cache.eval(dataset, 'tradeoff')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Execute dataset-level tradeoff computation in parallel for throughput.
    with Pool() as pool:
        _ = [*track(pool.imap(process_dataset, DATASETS), total=len(DATASETS))]


if __name__ == '__main__':
    main(parse_args())
