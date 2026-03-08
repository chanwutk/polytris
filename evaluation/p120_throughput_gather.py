#!/usr/local/bin/python

import itertools
import os
import shutil

import pandas as pd

from polyis.io import cache, store
from polyis.utilities import build_param_str, dataset_name_for_videoset, get_config


config = get_config()
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = config['EXEC']['CLASSIFIERS']
DATASETS = config['EXEC']['DATASETS']
TILEPADDING_MODES = config['EXEC']['TILEPADDING_MODES']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']
TRACKERS = config['EXEC']['TRACKERS']
CANVAS_SCALES = config['EXEC']['CANVAS_SCALE']
TRACKING_ACCURACY_THRESHOLDS = config['EXEC']['TRACKING_ACCURACY_THRESHOLDS']


def discover_available_videos(datasets: list[str]) -> list[tuple[str, str, str, str]]:
    """
    Discover available videos from dataset split directories.

    Args:
        datasets (list[str]): Dataset roots to scan

    Returns:
        list[tuple[str, str, str, str]]:
            (dataset_root, split_dataset_name, videoset, video_file)
    """
    # Initialize collected dataset/split/video tuples.
    datasets_videos: list[tuple[str, str, str, str]] = []

    # Iterate over configured dataset roots.
    for dataset in datasets:
        # Iterate over query-time splits.
        for videoset in ['valid', 'test']:
            # Build absolute split directory path.
            videoset_dir = store.dataset(dataset, videoset)
            # Ensure split directory exists.
            assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

            # Resolve split-aware dataset name used by evaluation outputs.
            split_dataset_name = dataset_name_for_videoset(dataset, videoset)

            # Enumerate videos in this split directory.
            for video_file in sorted(os.listdir(videoset_dir)):
                # Keep supported video extensions only.
                if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                # Append dataset/split/video tuple.
                datasets_videos.append((dataset, split_dataset_name, videoset, video_file))

    # Return discovered dataset/split/video tuples.
    return datasets_videos


def gather_index_construction_data(datasets: list[str]):
    """
    Gather runtime data for index construction stages.

    Args:
        datasets (list[str]): Dataset roots to process

    Returns:
        list[dict]: Index-construction runtime-file entries
    """
    # Initialize index-construction output rows.
    index_data = []

    # Iterate dataset roots.
    for dataset in datasets:
        # Gather stage 011 runtime files.
        indexing_detect_dir = cache.index(dataset, 'det')
        assert os.path.exists(indexing_detect_dir), f"Indexing detect directory {indexing_detect_dir} does not exist"
        for video_file in os.listdir(indexing_detect_dir):
            if not video_file.endswith('.detections.jsonl'):
                continue
            video_name = video_file.replace('.detections.jsonl', '')
            detect_path = os.path.join(indexing_detect_dir, video_file)
            assert os.path.exists(detect_path), f"Detect path {detect_path} does not exist"
            index_data.append({
                'dataset': dataset,
                'video': video_name,
                'videoset': 'indexing',
                'classifier': '_NA_',
                'tilesize': 0,
                'sample_rate': 1,
                'tracking_accuracy_threshold': None,
                'tilepadding': '_NA_',
                'canvas_scale': 1.0,
                'tracker': None,
                'stage': '011_tune_detect',
                'runtime_file': detect_path,
            })

        # Gather stage 012 runtime files.
        training_runtime_dir = cache.index(dataset, 'training', 'runtime')
        assert os.path.exists(training_runtime_dir), f"Training runtime directory {training_runtime_dir} does not exist"
        for tilesize in TILE_SIZES:
            tile_dir = os.path.join(training_runtime_dir, f'tilesize_{tilesize}')
            assert os.path.exists(tile_dir), f"Training runtime directory {tile_dir} does not exist"
            for runtime_file in os.listdir(tile_dir):
                if not runtime_file.endswith('_creating_training_data.jsonl'):
                    continue
                video_name = runtime_file.replace('_creating_training_data.jsonl', '')
                training_data_path = os.path.join(tile_dir, runtime_file)
                assert os.path.exists(training_data_path), f"Training data path {training_data_path} does not exist"
                index_data.append({
                    'dataset': dataset,
                    'video': video_name,
                    'videoset': 'indexing',
                    'classifier': '_NA_',
                    'tilesize': tilesize,
                    'sample_rate': 1,
                    'tracking_accuracy_threshold': None,
                    'tilepadding': '_NA_',
                    'canvas_scale': 1.0,
                    'tracker': None,
                    'stage': '012_tune_create_training_data',
                    'runtime_file': training_data_path,
                })

        # Gather stage 013 runtime files.
        training_results_dir = cache.index(dataset, 'training', 'results')
        assert os.path.exists(training_results_dir), f"Training results directory {training_results_dir} does not exist"
        for tilesize in TILE_SIZES:
            for classifier in CLASSIFIERS:
                if classifier == 'Perfect':
                    continue
                classifier_dir = os.path.join(training_results_dir, f'{classifier}_{tilesize}')
                assert os.path.exists(classifier_dir), f"Training results directory {classifier_dir} does not exist"
                throughput_path = os.path.join(classifier_dir, 'throughput_per_epoch.jsonl')
                assert os.path.exists(throughput_path), f"Throughput path {throughput_path} does not exist"
                index_data.append({
                    'dataset': dataset,
                    'video': 'dataset_level',
                    'videoset': 'indexing',
                    'classifier': classifier,
                    'tilesize': tilesize,
                    'sample_rate': 1,
                    'tracking_accuracy_threshold': None,
                    'tilepadding': '_NA_',
                    'canvas_scale': 1.0,
                    'tracker': None,
                    'stage': '013_tune_train_classifier',
                    'runtime_file': throughput_path,
                })

    # Return index-construction rows.
    return index_data


def gather_query_execution_data(datasets_videos: list[tuple[str, str, str, str]]):
    """
    Gather runtime data for query-execution stages.

    Args:
        datasets_videos (list[tuple[str, str, str, str]]):
            (dataset_root, split_dataset_name, videoset, video_file)

    Returns:
        list[dict]: Query-execution runtime-file entries
    """
    # Initialize query-execution output rows.
    query_data = []

    # Iterate discovered dataset/split/video tuples.
    for dataset_root, split_dataset_name, videoset, video in datasets_videos:
        # Build execution root for this video.
        video_path = cache.execution(dataset_root, video)

        # Gather naive detection runtime path.
        naive_detection_path = os.path.join(video_path, '002_naive', 'detection_runtime.jsonl')
        # Gather naive tracking runtime path.
        naive_tracking_path = os.path.join(video_path, '002_naive', 'tracking_runtime.jsonl')
        # Validate naive detection runtime file exists.
        assert os.path.exists(naive_detection_path), f"Groundtruth detection path {naive_detection_path} does not exist"
        # Validate naive tracking runtime file exists.
        assert os.path.exists(naive_tracking_path), f"Groundtruth tracking path {naive_tracking_path} does not exist"

        # Append stage-001 runtime row.
        query_data.append({
            'dataset': split_dataset_name,
            'video': video,
            'videoset': videoset,
            'classifier': '_NA_',
            'tilesize': 0,
            'sample_rate': 1,
            'tracking_accuracy_threshold': None,
            'tilepadding': '_NA_',
            'canvas_scale': 1.0,
            'tracker': None,
            'stage': '001_preprocess_groundtruth_detection',
            'runtime_file': naive_detection_path,
        })

        # Append stage-002 runtime row.
        query_data.append({
            'dataset': split_dataset_name,
            'video': video,
            'videoset': videoset,
            'classifier': '_NA_',
            'tilesize': 0,
            'sample_rate': 1,
            'tracking_accuracy_threshold': None,
            'tilepadding': '_NA_',
            'canvas_scale': 1.0,
            'tracker': None,
            'stage': '002_preprocess_groundtruth_tracking',
            'runtime_file': naive_tracking_path,
        })

        # Iterate classifier-driven parameter space.
        for classifier, tilesize, sample_rate in itertools.product(CLASSIFIERS, TILE_SIZES, SAMPLE_RATES):
            # Build stage-020 parameter directory name.
            classify_param = f'{classifier}_{tilesize}_{sample_rate}'
            # Build stage-020 runtime path.
            classify_path = os.path.join(video_path, '020_relevancy', classify_param, 'score', 'runtime.jsonl')
            # Validate stage-020 runtime path exists.
            assert os.path.exists(classify_path), f"Classify path {classify_path} does not exist"

            # Iterate pruning thresholds.
            for threshold in TRACKING_ACCURACY_THRESHOLDS:
                # Iterate trackers because stage-060 always emits one directory per tracker.
                for tracker_name in TRACKERS:
                    # Resolve upstream tracker value used by stages 022-050.
                    upstream_tracker = tracker_name if threshold is not None else None

                    # Append stage-020 runtime row (duplicated per downstream configuration).
                    query_data.append({
                        'dataset': split_dataset_name,
                        'video': video,
                        'videoset': videoset,
                        'classifier': classifier,
                        'tilesize': tilesize,
                        'sample_rate': sample_rate,
                        'tracking_accuracy_threshold': threshold,
                        'tilepadding': '_NA_',
                        'canvas_scale': 1.0,
                        'tracker': tracker_name,
                        'stage': '020_exec_classify',
                        'runtime_file': classify_path,
                    })

                    # Append stage-022 runtime row only for pruning-enabled configurations.
                    if threshold is not None:
                        # Build stage-022 parameter string.
                        prune_param = build_param_str(
                            classifier=classifier,
                            tilesize=tilesize,
                            sample_rate=sample_rate,
                            tracker=tracker_name,
                            tracking_accuracy_threshold=threshold,
                        )
                        # Build stage-022 runtime path.
                        prune_path = os.path.join(video_path, '022_pruned_polyominoes', prune_param,
                                                  'score', 'runtime.jsonl')
                        # Validate stage-022 runtime path exists.
                        assert os.path.exists(prune_path), f"Prune path {prune_path} does not exist"
                        # Append stage-022 runtime row.
                        query_data.append({
                            'dataset': split_dataset_name,
                            'video': video,
                            'videoset': videoset,
                            'classifier': classifier,
                            'tilesize': tilesize,
                            'sample_rate': sample_rate,
                            'tracking_accuracy_threshold': threshold,
                            'tilepadding': '_NA_',
                            'canvas_scale': 1.0,
                            'tracker': tracker_name,
                            'stage': '022_exec_prune_polyominoes',
                            'runtime_file': prune_path,
                        })

                    # Iterate post-pruning execution dimensions.
                    for tilepadding, canvas_scale in itertools.product(TILEPADDING_MODES, CANVAS_SCALES):
                        # Build shared parameter string for stages 030/040/050.
                        compress_param = build_param_str(
                            classifier=classifier,
                            tilesize=tilesize,
                            sample_rate=sample_rate,
                            tracking_accuracy_threshold=threshold,
                            tilepadding=tilepadding,
                            canvas_scale=canvas_scale,
                            tracker=upstream_tracker,
                        )

                        # Build stage-030 runtime path.
                        compress_path = os.path.join(video_path, '033_compressed_frames', compress_param,
                                                     'runtime.jsonl')
                        # Build stage-040 runtime path.
                        detect_path = os.path.join(video_path, '040_compressed_detections', compress_param,
                                                   'runtimes.jsonl')
                        # Build stage-050 runtime path.
                        uncompress_path = os.path.join(video_path, '050_uncompressed_detections', compress_param,
                                                       'runtime.jsonl')

                        # Validate stage-030 runtime path exists.
                        assert os.path.exists(compress_path), f"Compress path {compress_path} does not exist"
                        # Validate stage-040 runtime path exists.
                        assert os.path.exists(detect_path), f"Detect path {detect_path} does not exist"
                        # Validate stage-050 runtime path exists.
                        assert os.path.exists(uncompress_path), f"Uncompress path {uncompress_path} does not exist"

                        # Append stage-030 runtime row.
                        query_data.append({
                            'dataset': split_dataset_name,
                            'video': video,
                            'videoset': videoset,
                            'classifier': classifier,
                            'tilesize': tilesize,
                            'sample_rate': sample_rate,
                            'tracking_accuracy_threshold': threshold,
                            'tilepadding': tilepadding,
                            'canvas_scale': canvas_scale,
                            'tracker': tracker_name,
                            'stage': '030_exec_compress',
                            'runtime_file': compress_path,
                        })

                        # Append stage-040 runtime row.
                        query_data.append({
                            'dataset': split_dataset_name,
                            'video': video,
                            'videoset': videoset,
                            'classifier': classifier,
                            'tilesize': tilesize,
                            'sample_rate': sample_rate,
                            'tracking_accuracy_threshold': threshold,
                            'tilepadding': tilepadding,
                            'canvas_scale': canvas_scale,
                            'tracker': tracker_name,
                            'stage': '040_exec_detect',
                            'runtime_file': detect_path,
                        })

                        # Append stage-050 runtime row.
                        query_data.append({
                            'dataset': split_dataset_name,
                            'video': video,
                            'videoset': videoset,
                            'classifier': classifier,
                            'tilesize': tilesize,
                            'sample_rate': sample_rate,
                            'tracking_accuracy_threshold': threshold,
                            'tilepadding': tilepadding,
                            'canvas_scale': canvas_scale,
                            'tracker': tracker_name,
                            'stage': '050_exec_uncompress',
                            'runtime_file': uncompress_path,
                        })

                        # Build stage-060 output parameter string (tracker always included).
                        track_param = build_param_str(
                            classifier=classifier,
                            tilesize=tilesize,
                            sample_rate=sample_rate,
                            tracking_accuracy_threshold=threshold,
                            tilepadding=tilepadding,
                            canvas_scale=canvas_scale,
                            tracker=tracker_name,
                        )
                        # Build stage-060 runtime path.
                        track_path = os.path.join(video_path, '060_uncompressed_tracks', track_param,
                                                  'runtimes.jsonl')
                        # Validate stage-060 runtime path exists.
                        assert os.path.exists(track_path), f"Track path {track_path} does not exist"

                        # Append stage-060 runtime row.
                        query_data.append({
                            'dataset': split_dataset_name,
                            'video': video,
                            'videoset': videoset,
                            'classifier': classifier,
                            'tilesize': tilesize,
                            'sample_rate': sample_rate,
                            'tracking_accuracy_threshold': threshold,
                            'tilepadding': tilepadding,
                            'canvas_scale': canvas_scale,
                            'tracker': tracker_name,
                            'stage': '060_exec_track',
                            'runtime_file': track_path,
                        })

    # Return query-execution rows.
    return query_data


def print_index_construction_table(index_data, write_line=print):
    """Print index-construction data as a table."""
    write_line("INDEX CONSTRUCTION DATASET")
    write_line("=" * 80)
    write_line("Stages: 011_tune_detect, 012_tune_create_training_data, 013_tune_train_classifier")
    write_line()

    df = pd.DataFrame.from_dict(index_data)
    sizes = df.groupby(['dataset', 'stage', 'classifier', 'tilesize', 'tilepadding']).size()
    assert isinstance(sizes, pd.Series), "sizes is not a pandas Series"
    write_line(sizes.reset_index(name='count').to_string(index=False))


def print_query_execution_table(query_data, write_line=print):
    """Print query-execution data as a table."""
    write_line("QUERY EXECUTION DATASET")
    write_line("=" * 120)
    write_line("Stages: 020_exec_classify, 022_exec_prune_polyominoes, 030_exec_compress, 040_exec_detect, 050_exec_uncompress, 060_exec_track")
    write_line()

    df = pd.DataFrame.from_dict(query_data)
    sizes = df.groupby([
        'dataset',
        'videoset',
        'classifier',
        'tilesize',
        'sample_rate',
        'tracking_accuracy_threshold',
        'tilepadding',
        'canvas_scale',
        'tracker',
    ]).size()
    assert isinstance(sizes, pd.Series), "sizes is not a pandas Series"
    write_line(sizes.reset_index(name='count').to_string(index=False))


def save_data_tables(index_data, query_data):
    """Save both data tables to per-dataset-root evaluation directories."""
    # Collect dataset roots from index-construction rows.
    datasets = {entry['dataset'] for entry in index_data}

    # Save one set of files per dataset root.
    for dataset in datasets:
        # Build output directory path.
        output_dir = cache.eval(dataset, 'tp')
        # Ensure output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        # Save index-construction rows for this dataset root.
        save_index_construction_data(index_data, dataset, output_dir)
        # Save query-execution rows for this dataset root.
        save_query_execution_data(query_data, dataset, output_dir)

        # Log output location.
        print(f"\nData tables saved to: {output_dir}")


def save_index_construction_data(index_data, dataset, output_dir):
    """Save index-construction rows in CSV and text-table formats."""
    # Filter rows by dataset root.
    dataset_data = [entry for entry in index_data if entry['dataset'] == dataset]

    # Build DataFrame from full index data.
    df = pd.DataFrame.from_dict(index_data)
    # Filter DataFrame by dataset root.
    df = df[df['dataset'] == dataset]
    # Save CSV file.
    df.to_csv(os.path.join(output_dir, 'index_construction.csv'), index=False)

    # Build path for human-readable text table.
    txt_file = os.path.join(output_dir, 'index_construction.txt')
    # Write text table output.
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_index_construction_table(dataset_data, write_line=fwrite)


def save_query_execution_data(query_data, dataset, output_dir):
    """Save query-execution rows in CSV and text-table formats."""
    # Select rows belonging to this dataset root.
    dataset_data = [entry for entry in query_data
                    if entry['dataset'] == dataset or entry['dataset'].startswith(f'{dataset}-')]

    # Build DataFrame from selected rows.
    df = pd.DataFrame.from_dict(dataset_data)
    # Save query-execution CSV file.
    df.to_csv(os.path.join(output_dir, 'query_execution.csv'), index=False)

    # Build path for human-readable query table.
    txt_file = os.path.join(output_dir, 'query_execution.txt')
    # Write text table output.
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_query_execution_table(dataset_data, write_line=fwrite)


def main():
    """Main entry point for throughput data gathering."""
    # Print configured datasets.
    print(f"Gathering throughput data for datasets: {DATASETS}")

    # Discover available videos across valid/test splits.
    datasets_videos = discover_available_videos(DATASETS)
    # Ensure at least one video is available.
    assert len(datasets_videos) > 0, "No videos found in dataset directories"

    # Gather index-construction runtime-file entries.
    index_data = gather_index_construction_data(DATASETS)
    # Gather query-execution runtime-file entries.
    query_data = gather_query_execution_data(datasets_videos)

    # Ensure index data is not empty.
    assert len(index_data) > 0, "No index construction data found"
    # Ensure query data is not empty.
    assert len(query_data) > 0, "No query execution data found"

    # Save gathered tables to per-dataset-root evaluation directories.
    save_data_tables(index_data, query_data)


if __name__ == '__main__':
    main()
