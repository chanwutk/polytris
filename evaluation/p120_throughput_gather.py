#!/usr/local/bin/python

import argparse
import os

import pandas as pd

from evaluation.manifests import (
    build_index_video_manifest,
    build_polytris_variant_manifest,
    build_split_video_manifest,
)
from polyis.io import cache
from polyis.pareto import load_pareto_params, pareto_params_exist
from polyis.utilities import build_param_str, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']

QUERY_COLUMNS = [
    'dataset',
    'videoset',
    'video',
    'variant',
    'variant_id',
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'relevance_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
    'stage',
    'runtime_file',
]
INDEX_COLUMNS = [
    'dataset',
    'videoset',
    'video',
    'variant',
    'variant_id',
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'relevance_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
    'stage',
    'runtime_file',
]


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def assert_runtime_paths_exist(manifest_df: pd.DataFrame, label: str):
    # Collect only the rows whose expected runtime file is missing.
    missing_df = manifest_df[~manifest_df['runtime_file'].map(os.path.exists)]
    # Fail fast with a compact preview when any configured runtime file is absent.
    assert missing_df.empty, f"Missing {label} runtime paths:\n{missing_df[['stage', 'dataset', 'videoset', 'video', 'variant_id', 'runtime_file']].head(20)}"


def assert_and_filter_polytris_runtime_paths(manifest_df: pd.DataFrame, datasets: list[str], videoset: str) -> pd.DataFrame:
    """Assert runtime files exist; for test, keep only Pareto-optimal variants."""
    if videoset == 'valid':
        # Assert that all valid-split runtime files are present.
        missing = manifest_df[~manifest_df['runtime_file'].map(os.path.exists)]
        assert missing.empty, (
            f"Missing Polytris valid runtime paths:\n"
            f"{missing[['stage', 'dataset', 'videoset', 'video', 'variant_id', 'runtime_file']].head(20)}"
        )
        return manifest_df

    # For test: filter each dataset to its own Pareto-optimal variants only.
    # Using a combined set across datasets would incorrectly include combos that are
    # Pareto for one dataset but were never run for another.
    parts = []
    for ds in datasets:
        ds_df = manifest_df[manifest_df['dataset'] == ds].copy()
        pareto_variant_ids = set(load_pareto_params(ds)['variant_id'].dropna().unique())
        parts.append(ds_df[ds_df['variant_id'].isin(pareto_variant_ids)])
    filtered_df = pd.concat(parts, ignore_index=True)

    # Warn and skip any Pareto-set test runtime files that are missing (e.g. interrupted
    # runs that produced output data but did not record a runtime file).
    missing = filtered_df[~filtered_df['runtime_file'].map(os.path.exists)]
    if not missing.empty:
        print(
            f"WARNING: Skipping {len(missing)} missing Pareto test runtime paths:\n"
            f"{missing[['stage', 'dataset', 'videoset', 'video', 'variant_id', 'runtime_file']].head(20)}"
        )
        filtered_df = filtered_df[filtered_df['runtime_file'].map(os.path.exists)].copy()
    return filtered_df


def add_default_param_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the caller keeps its original frame unchanged.
    result_df = df.copy()
    # Add the shared query/index columns that are absent for the current frame.
    for column in ['classifier', 'tilesize', 'sample_rate', 'tracking_accuracy_threshold', 'relevance_threshold', 'tilepadding', 'canvas_scale', 'tracker']:
        if column not in result_df.columns:
            result_df[column] = pd.NA
    return result_df


def build_index_construction_manifest(datasets: list[str]) -> pd.DataFrame:
    # Materialize the training videos used by the indexing stages.
    index_video_df = build_index_video_manifest(datasets=datasets)
    # Add the shared variant token for all index rows.
    index_video_df['variant'] = 'index'
    # Add the shared variant identifier for all index rows.
    index_video_df['variant_id'] = 'index'
    # Ensure the shared parameter columns exist even when a stage does not use them.
    index_video_df = add_default_param_columns(index_video_df)

    # Derive the path to the stage-011 runtime file from each train-split video.
    detect_df = index_video_df.copy()
    detect_df['stage'] = '011_tune_detect'
    detect_df['runtime_file'] = detect_df.apply(
        lambda row: cache.index(row['dataset'], 'det', f"{row['video']}.detections.jsonl"),
        axis=1,
    )

    # Cross join train videos with tile sizes for stage-012 training-data generation.
    tilesize_df = pd.DataFrame({'tilesize': CONFIG['EXEC']['TILE_SIZES']})
    create_training_df = index_video_df[['dataset', 'videoset', 'video', 'variant', 'variant_id']].merge(tilesize_df, how='cross')
    create_training_df = add_default_param_columns(create_training_df)
    create_training_df['stage'] = '012_tune_create_training_data'
    create_training_df['runtime_file'] = create_training_df.apply(
        lambda row: cache.index(
            row['dataset'],
            'training',
            'runtime',
            f"tilesize_{int(row['tilesize'])}",
            f"{row['video']}_creating_training_data.jsonl",
        ),
        axis=1,
    )

    # Cross join datasets with classifier/tile-size pairs for stage-013 training throughput.
    train_rows = []
    for dataset in datasets:
        train_rows.append({
            'dataset': dataset,
            'videoset': 'train',
            'video': 'dataset_level',
            'variant': 'index',
            'variant_id': 'index',
        })
    train_df = pd.DataFrame.from_records(train_rows)
    classifier_tile_df = pd.DataFrame.from_records([
        {'classifier': classifier, 'tilesize': tilesize}
        for classifier in CONFIG['EXEC']['CLASSIFIERS']
        if classifier != 'Perfect'
        for tilesize in CONFIG['EXEC']['TILE_SIZES']
    ])
    train_classifier_df = train_df.merge(classifier_tile_df, how='cross')
    train_classifier_df = add_default_param_columns(train_classifier_df)
    train_classifier_df['stage'] = '013_tune_train_classifier'
    train_classifier_df['runtime_file'] = train_classifier_df.apply(
        lambda row: cache.index(
            row['dataset'],
            'training',
            'results',
            f"{row['classifier']}_{int(row['tilesize'])}",
            'throughput_per_epoch.jsonl',
        ),
        axis=1,
    )

    # Combine the configured index-stage manifests into one canonical table.
    manifest_df = pd.concat([detect_df, create_training_df, train_classifier_df], ignore_index=True)
    # Validate that every configured index runtime file already exists.
    assert_runtime_paths_exist(manifest_df, 'index')

    return manifest_df[INDEX_COLUMNS]


def build_naive_query_manifest(video_df: pd.DataFrame) -> pd.DataFrame:
    # Start from the concrete query-time videos and tag them as the naive baseline.
    naive_df = video_df.copy()
    naive_df['variant'] = 'naive'
    naive_df['variant_id'] = 'naive'
    naive_df = add_default_param_columns(naive_df)

    # Build the stage-001 runtime rows for the naive detection stage.
    detect_df = naive_df.copy()
    detect_df['stage'] = '001_preprocess_groundtruth_detection'
    detect_df['runtime_file'] = detect_df.apply(
        lambda row: cache.execution(row['dataset'], row['video'], '002_naive', 'detection_runtime.jsonl'),
        axis=1,
    )

    # Build the stage-002 runtime rows for the naive tracking stage.
    track_df = naive_df.copy()
    track_df['stage'] = '002_preprocess_groundtruth_tracking'
    track_df['runtime_file'] = track_df.apply(
        lambda row: cache.execution(row['dataset'], row['video'], '002_naive', 'tracking_runtime.jsonl'),
        axis=1,
    )

    # Combine the two naive runtime stages into one canonical query manifest.
    manifest_df = pd.concat([detect_df, track_df], ignore_index=True)
    # Validate that the configured naive runtime files already exist.
    assert_runtime_paths_exist(manifest_df, 'naive query')

    return manifest_df[QUERY_COLUMNS]


def build_stage020_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Duplicate stage-020 rows across the downstream config space for simplicity.
    stage_df = polytris_df.copy()
    stage_df['stage'] = '020_exec_classify'
    stage_df['runtime_file'] = stage_df.apply(
        lambda row: cache.execution(
            row['dataset'],
            row['video'],
            '020_relevancy',
            f"{row['classifier']}_{int(row['tilesize'])}_{int(row['sample_rate'])}",
            'score',
            'runtime.jsonl',
        ),
        axis=1,
    )
    return stage_df[QUERY_COLUMNS]


def build_stage022_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only pruning-enabled rows because stage-022 is skipped without a threshold.
    stage_df = polytris_df[polytris_df['tracking_accuracy_threshold'].notna()].copy()
    # Return an empty manifest early when the config does not enable pruning.
    if stage_df.empty:
        stage_df['stage'] = pd.Series(dtype='object')
        stage_df['runtime_file'] = pd.Series(dtype='object')
        return stage_df.reindex(columns=QUERY_COLUMNS)

    stage_df['stage'] = '022_exec_prune_polyominoes'
    stage_df['runtime_file'] = [
        os.path.join(
            cache.execution(row.dataset, row.video),
            '022_pruned_polyominoes',
            build_param_str(
                classifier=row.classifier,
                tilesize=int(row.tilesize),
                sample_rate=int(row.sample_rate),
                tracker=row.tracker,
                tracking_accuracy_threshold=float(row.tracking_accuracy_threshold),
                relevance_threshold=float(row.relevance_threshold),
            ),
            'score',
            'runtime.jsonl',
        )
        for row in stage_df.itertuples(index=False)
    ]
    return stage_df[QUERY_COLUMNS]


def resolve_upstream_tracker(row: pd.Series) -> str | None:
    # Remove the tracker dimension upstream of stage-060 when pruning is disabled.
    if pd.isna(row['tracking_accuracy_threshold']):
        return None
    return str(row['tracker'])


def build_shared_execution_param(row: pd.Series) -> str:
    # Encode the shared stage-030/040/050 parameter string for this runtime row.
    return build_param_str(
        classifier=row['classifier'],
        tilesize=int(row['tilesize']),
        sample_rate=int(row['sample_rate']),
        tracking_accuracy_threshold=None if pd.isna(row['tracking_accuracy_threshold']) else float(row['tracking_accuracy_threshold']),
        relevance_threshold=float(row['relevance_threshold']),
        tilepadding=row['tilepadding'],
        canvas_scale=float(row['canvas_scale']),
        tracker=resolve_upstream_tracker(row),
    )


def build_stage030_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Build the stage-030 runtime rows from the full Polytris config space.
    stage_df = polytris_df.copy()
    stage_df['stage'] = '030_exec_compress'
    stage_df['runtime_file'] = stage_df.apply(
        lambda row: os.path.join(
            cache.execution(row['dataset'], row['video']),
            '033_compressed_frames',
            build_shared_execution_param(row),
            'runtime.jsonl',
        ),
        axis=1,
    )
    return stage_df[QUERY_COLUMNS]


def build_stage040_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Build the stage-040 runtime rows from the full Polytris config space.
    stage_df = polytris_df.copy()
    stage_df['stage'] = '040_exec_detect'
    stage_df['runtime_file'] = stage_df.apply(
        lambda row: os.path.join(
            cache.execution(row['dataset'], row['video']),
            '040_compressed_detections',
            build_shared_execution_param(row),
            'runtimes.jsonl',
        ),
        axis=1,
    )
    return stage_df[QUERY_COLUMNS]


def build_stage050_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Build the stage-050 runtime rows from the full Polytris config space.
    stage_df = polytris_df.copy()
    stage_df['stage'] = '050_exec_uncompress'
    stage_df['runtime_file'] = stage_df.apply(
        lambda row: os.path.join(
            cache.execution(row['dataset'], row['video']),
            '050_uncompressed_detections',
            build_shared_execution_param(row),
            'runtime.jsonl',
        ),
        axis=1,
    )
    return stage_df[QUERY_COLUMNS]


def build_stage060_manifest(polytris_df: pd.DataFrame) -> pd.DataFrame:
    # Build the stage-060 runtime rows from the fully expanded Polytris config space.
    stage_df = polytris_df.copy()
    stage_df['stage'] = '060_exec_track'
    stage_df['runtime_file'] = stage_df.apply(
        lambda row: os.path.join(
            cache.execution(row['dataset'], row['video']),
            '060_uncompressed_tracks',
            row['variant_id'],
            'runtimes.jsonl',
        ),
        axis=1,
    )
    return stage_df[QUERY_COLUMNS]


def build_query_execution_manifest(datasets: list[str], videoset: str) -> pd.DataFrame:
    # Materialize only the requested videoset videos to avoid unnecessary cross-join overhead.
    query_video_df = build_split_video_manifest(datasets=datasets, videosets=[videoset])
    # Materialize the fully expanded Polytris parameter grid.
    polytris_variant_df = build_polytris_variant_manifest()
    # Cross join videos and Polytris params so each downstream config gets one row.
    polytris_df = query_video_df.merge(polytris_variant_df, how='cross')

    # Build the naive manifest once from the concrete query-time videos.
    naive_df = build_naive_query_manifest(query_video_df)
    # Build the per-stage Polytris manifests from the full downstream grid.
    stage_frames = [
        build_stage020_manifest(polytris_df),
        build_stage022_manifest(polytris_df),
        build_stage030_manifest(polytris_df),
        build_stage040_manifest(polytris_df),
        build_stage050_manifest(polytris_df),
        build_stage060_manifest(polytris_df),
    ]
    polytris_stage_df = pd.concat(stage_frames, ignore_index=True)
    # Assert valid-split files exist; for test-split assert and keep only Pareto-set files.
    polytris_stage_df = assert_and_filter_polytris_runtime_paths(polytris_stage_df, DATASETS, videoset)

    # Combine the Polytris and naive query manifests into one canonical table.
    return pd.concat([naive_df, polytris_stage_df], ignore_index=True)


def print_manifest_table(manifest_df: pd.DataFrame, title: str, group_columns: list[str], write_line=print):
    # Print the section title so the text summary stays readable.
    write_line(title)
    write_line('=' * len(title))
    write_line()

    # Summarize the manifest counts by the requested grouping columns.
    summary_df = manifest_df.groupby(group_columns, dropna=False).size().reset_index(name='count')
    write_line(summary_df.to_string(index=False))


def save_manifest_tables(index_df: pd.DataFrame, query_df: pd.DataFrame):
    # Persist one manifest pair per configured dataset root.
    for dataset in DATASETS:
        # Resolve the dataset-local throughput evaluation directory.
        output_dir = cache.eval(dataset, 'tp')
        # Recreate the output directory before writing fresh manifest files.
        os.makedirs(output_dir, exist_ok=True)

        # Select the dataset-local index and query rows.
        dataset_index_df = index_df[index_df['dataset'] == dataset].copy()
        dataset_query_df = query_df[query_df['dataset'] == dataset].copy()

        # Persist the canonical CSV manifests consumed by the compute stage.
        dataset_index_df.to_csv(output_dir / 'index_construction.csv', index=False)
        dataset_query_df.to_csv(output_dir / 'query_execution.csv', index=False)

        # Persist a compact text summary for quick manual inspection.
        with open(output_dir / 'index_construction.txt', 'w') as f:
            print_manifest_table(
                dataset_index_df,
                'INDEX CONSTRUCTION MANIFEST',
                ['dataset', 'videoset', 'stage', 'classifier', 'tilesize'],
                write_line=lambda text='': f.write(text + '\n'),
            )
        with open(output_dir / 'query_execution.txt', 'w') as f:
            print_manifest_table(
                dataset_query_df,
                'QUERY EXECUTION MANIFEST',
                ['dataset', 'videoset', 'variant', 'classifier', 'tilesize',
                 'sample_rate', 'tracking_accuracy_threshold', 'relevance_threshold', 'tilepadding',
                 'canvas_scale', 'tracker', 'stage'],
                write_line=lambda text='': f.write(text + '\n'),
            )


def main(args):
    # Log the configured datasets before manifest generation starts.
    print(f"Building throughput manifests for datasets: {DATASETS}")

    # Resolve the single videoset from the mutually exclusive CLI flags.
    videoset = 'test' if args.test else 'valid'

    # Assert Pareto params exist when processing the test split.
    if videoset == 'test':
        for dataset in DATASETS:
            assert pareto_params_exist(dataset), \
                f"Pareto params not found for {dataset}. Run p135_pareto_extract.py first."

    # Materialize the deterministic index manifest for the configured datasets.
    index_df = build_index_construction_manifest(DATASETS)
    # Materialize the deterministic query manifest for the configured datasets.
    query_df = build_query_execution_manifest(DATASETS, videoset)

    # Fail fast when either manifest is unexpectedly empty.
    assert not index_df.empty, "No index construction manifest rows found"
    assert not query_df.empty, "No query execution manifest rows found"

    # Persist the canonical CSV and text summaries consumed by later stages.
    save_manifest_tables(index_df, query_df)


if __name__ == '__main__':
    main(parse_args())
