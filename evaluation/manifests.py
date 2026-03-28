#!/usr/local/bin/python

import itertools
import os

import pandas as pd

from polyis.io import cache, store
from polyis.utilities import build_param_str, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
QUERY_VIDEOSETS = CONFIG['EXEC']['VIDEO_SETS']
CLASSIFIERS = CONFIG['EXEC']['CLASSIFIERS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']
TILEPADDING_MODES = CONFIG['EXEC']['TILEPADDING_MODES']
SAMPLE_RATES = CONFIG['EXEC']['SAMPLE_RATES']
TRACKERS = CONFIG['EXEC']['TRACKERS']
CANVAS_SCALES = CONFIG['EXEC']['CANVAS_SCALE']
TRACKING_ACCURACY_THRESHOLDS = CONFIG['EXEC']['TRACKING_ACCURACY_THRESHOLDS']

# Keep the supported video extensions in one place so all manifest builders agree.
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def _cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # Return an empty DataFrame early when either side has no rows.
    if left.empty or right.empty:
        return pd.DataFrame(columns=[*left.columns.tolist(), *right.columns.tolist()])

    # Use pandas cross join so the helper stays declarative and easy to audit.
    return left.merge(right, how='cross')


def list_split_videos(dataset: str, videoset: str) -> list[str]:
    # Resolve the split directory from the shared store helper.
    videoset_dir = store.dataset(dataset, videoset)
    # Fail fast when the configured split directory is missing.
    assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

    # Filter to supported video files only and keep a deterministic ordering.
    return sorted(
        video_file
        for video_file in os.listdir(videoset_dir)
        if video_file.endswith(VIDEO_EXTENSIONS)
    )


def build_split_video_manifest(datasets: list[str] | None = None,
                               videosets: list[str] | None = None) -> pd.DataFrame:
    # Default to the configured datasets when the caller does not override them.
    datasets = DATASETS if datasets is None else datasets
    assert datasets is not None
    # Default to the configured query splits when the caller does not override them.
    videosets = QUERY_VIDEOSETS if videosets is None else videosets
    assert videosets is not None

    # Collect one manifest row per dataset/split/video triple.
    rows: list[dict] = []

    # Expand each configured dataset and split into concrete video filenames.
    for dataset, videoset in itertools.product(datasets, videosets):
        # Enumerate the concrete videos for this dataset/split pair.
        for video in list_split_videos(dataset, videoset):
            # Store the explicit split-aware manifest row.
            rows.append({
                'dataset': dataset,
                'videoset': videoset,
                'video': video,
            })

    # Materialize the manifest as a DataFrame for downstream joins and filters.
    return pd.DataFrame.from_records(rows, columns=['dataset', 'videoset', 'video'])


def build_index_video_manifest(datasets: list[str] | None = None) -> pd.DataFrame:
    # Reuse the query split helper with the fixed training split for index stages.
    return build_split_video_manifest(datasets=datasets, videosets=['train'])


def build_polytris_variant_manifest() -> pd.DataFrame:
    # Collect the fully expanded Polytris query parameter grid.
    rows: list[dict] = []

    # Expand the exact stage-060 parameter grid used by the execution pipeline.
    for classifier, tilesize, sample_rate, threshold, tilepadding, canvas_scale, tracker in itertools.product(
        CLASSIFIERS,
        TILE_SIZES,
        SAMPLE_RATES,
        TRACKING_ACCURACY_THRESHOLDS,
        TILEPADDING_MODES,
        CANVAS_SCALES,
        TRACKERS,
    ):
        # Encode the stage-060 output directory name for this configuration.
        variant_id = build_param_str(
            classifier=classifier,
            tilesize=tilesize,
            sample_rate=sample_rate,
            tracking_accuracy_threshold=threshold,
            tilepadding=tilepadding,
            canvas_scale=canvas_scale,
            tracker=tracker,
        )

        # Keep the fully explicit parameter columns alongside the encoded identifier.
        rows.append({
            'variant': 'polytris',
            'variant_id': variant_id,
            'classifier': classifier,
            'tilesize': tilesize,
            'sample_rate': sample_rate,
            'tracking_accuracy_threshold': threshold,
            'tilepadding': tilepadding,
            'canvas_scale': canvas_scale,
            'tracker': tracker,
        })

    # Return a deterministic manifest ordered by the config expansion above.
    return pd.DataFrame.from_records(rows)


def build_naive_variant_manifest() -> pd.DataFrame:
    # Represent the naive baseline with a dedicated token instead of fake Polytris params.
    return pd.DataFrame.from_records([{
        'variant': 'naive',
        'variant_id': 'naive',
        'classifier': pd.NA,
        'tilesize': pd.NA,
        'sample_rate': pd.NA,
        'tracking_accuracy_threshold': pd.NA,
        'tilepadding': pd.NA,
        'canvas_scale': pd.NA,
        'tracker': pd.NA,
    }])


def build_variant_manifest(include_naive: bool = True) -> pd.DataFrame:
    # Start from the real Polytris parameter grid.
    variant_frames = [build_polytris_variant_manifest()]

    # Append the dedicated naive baseline when the caller requests it.
    if include_naive:
        variant_frames.append(build_naive_variant_manifest())

    # Combine all variant rows into one canonical manifest.
    return pd.concat(variant_frames, ignore_index=True)


def build_query_task_manifest(datasets: list[str] | None = None,
                              videosets: list[str] | None = None,
                              include_naive: bool = True) -> pd.DataFrame:
    # Materialize the concrete dataset/split/video rows for query-time evaluation.
    videos_df = build_split_video_manifest(datasets=datasets, videosets=videosets)
    # Materialize the configured variant rows for query-time evaluation.
    variants_df = build_variant_manifest(include_naive=include_naive)

    # Cross join videos and variants so each downstream config gets a concrete row.
    return _cross_join(videos_df, variants_df)


def build_split_variant_manifest(datasets: list[str] | None = None,
                                 videosets: list[str] | None = None,
                                 include_naive: bool = True) -> pd.DataFrame:
    # Materialize the concrete dataset/split/video rows so split presence matches the query manifest builder.
    videos_df = build_split_video_manifest(datasets=datasets, videosets=videosets)
    # Collapse the concrete video rows down to one row per dataset/split.
    split_df = videos_df[['dataset', 'videoset']].drop_duplicates().reset_index(drop=True)
    # Materialize the configured variant rows for split-level evaluation.
    variants_df = build_variant_manifest(include_naive=include_naive)

    # Cross join splits and variants directly to avoid building redundant per-video rows first.
    return _cross_join(split_df, variants_df)


def load_sota_stat_manifest(system: str, dataset: str) -> pd.DataFrame:
    # Resolve the transformed SOTA stat.csv emitted by the transform stage.
    stat_path = cache.sota(system, dataset, 'stat.csv')
    # Fail fast when the transformed SOTA manifest is missing.
    assert os.path.exists(stat_path), f"SOTA stat.csv not found: {stat_path}"

    # Load the stat table through pandas so downstream code can stay vectorized.
    stat_df = pd.read_csv(stat_path)
    # Require a param_id column so later joins stay explicit.
    assert 'param_id' in stat_df.columns, f"param_id column not found in {stat_path}"

    # Normalize the param identifier to integer type for stable joins and folder names.
    stat_df = stat_df.copy()
    stat_df['param_id'] = stat_df['param_id'].astype(int)
    # Add the explicit split label used throughout SOTA evaluation.
    stat_df['videoset'] = 'test'
    # Add the base dataset so the returned manifest is self-describing.
    stat_df['dataset'] = dataset

    return stat_df


def build_sota_video_param_manifest(system: str, dataset: str) -> pd.DataFrame:
    # Materialize the transformed SOTA parameter rows from stat.csv.
    stat_df = load_sota_stat_manifest(system, dataset)
    # Materialize the concrete test videos for this dataset.
    videos_df = build_split_video_manifest(datasets=[dataset], videosets=['test'])

    # Keep only the columns needed for the param/video cartesian product.
    params_df = stat_df[['dataset', 'videoset', 'param_id']].drop_duplicates().reset_index(drop=True)
    videos_df = videos_df[['dataset', 'videoset', 'video']].drop_duplicates().reset_index(drop=True)

    # Join on dataset and split so every param_id is paired with every expected test video.
    return videos_df.merge(params_df, on=['dataset', 'videoset'], how='inner')
