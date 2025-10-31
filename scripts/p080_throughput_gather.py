#!/usr/local/bin/python

import argparse
import os
from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATASETS_DIR, DATASETS_TO_TEST, TILE_SIZES, TILEPADDING_MODES

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Gather throughput data from pipeline stages')
    parser.add_argument('--datasets', required=False, default=DATASETS_TO_TEST, nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()

CLASSIFIERS = CLASSIFIERS_TO_TEST
EXEC_CLASSIFIERS = CLASSIFIERS_TO_TEST + ['Perfect']


def discover_available_videos(datasets: list[str]):
    """
    Discover available videos from execution directories for given datasets.
    
    Args:
        datasets (list): List of dataset names to search
        
    Returns:
        list[tuple[str, str]]: List of (dataset, video_file) tuples
    """
    datasets_videos = []
    
    for dataset in datasets:
        videoset_dir = os.path.join(DATASETS_DIR, dataset, 'test')
        assert os.path.exists(videoset_dir), \
            f"Videoset directory {videoset_dir} does not exist"

        for video_file in sorted(os.listdir(videoset_dir)):
            if not video_file.endswith('.mp4'):
                continue
            datasets_videos.append((dataset, video_file))
    
    return datasets_videos


def gather_index_construction_data(datasets):
    """
    Gather runtime data for index construction stages:
    - 011_tune_detect.py
    - 012_tune_create_training_data.py  
    - 013_tune_train_classifier.py
    
    Args:
        datasets (list): List of dataset names to process
    
    Returns list of dicts with columns: dataset, runtime_files
    Note: Index construction is done at dataset level, not per video
    """
    index_data = []
    
    for dataset in datasets:
        runtime_files = []
        dataset_path = os.path.join(CACHE_DIR, dataset)
        
        # 011_tune_detect.py - per video detections in indexing/segment/detection/
        indexing_detect_dir = os.path.join(dataset_path, 'indexing', 'segment', 'detection')
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
                'classifier': '_NA_',
                'tilesize': 0,
                'tilepadding': '_NA_',
                'stage': '011_tune_detect',
                'runtime_file': detect_path,
            })
        
        # 012_tune_create_training_data.py - per video runtime files for all tile sizes
        training_runtime_dir = os.path.join(dataset_path, 'indexing', 'training', 'runtime')
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
                    'classifier': '_NA_',
                    'tilesize': tilesize,
                    'tilepadding': '_NA_',
                    'stage': '012_tune_create_training_data',
                    'runtime_file': training_data_path,
                })
        
        # 013_tune_train_classifier.py - per classifier/tilesize training results
        training_results_dir = os.path.join(dataset_path, 'indexing', 'training', 'results')
        assert os.path.exists(training_results_dir), f"Training results directory {training_results_dir} does not exist"
        for tilesize in TILE_SIZES:
            for classifier in CLASSIFIERS:
                classifier_dir = os.path.join(training_results_dir, f'{classifier}_{tilesize}')
                assert os.path.exists(classifier_dir), f"Training results directory {classifier_dir} does not exist"

                throughput_path = os.path.join(classifier_dir, 'throughput_per_epoch.jsonl')
                assert os.path.exists(throughput_path), f"Throughput path {throughput_path} does not exist"
                index_data.append({
                    'dataset': dataset,
                    'video': 'dataset_level',
                    'classifier': classifier,
                    'tilesize': tilesize,
                    'tilepadding': '_NA_',
                    'stage': '013_tune_train_classifier',
                    'runtime_file': throughput_path,
                })
    
    return index_data


def gather_query_execution_data(datasets_videos):
    """
    Gather runtime data for query execution stages:
    - 001_preprocess_groundtruth_detection.py
    - 002_preprocess_groundtruth_tracking.py
    - 020_exec_classify.py
    - 030_exec_compress.py
    - 040_exec_detect.py
    - 050_exec_uncompress.py
    - 060_exec_track.py
    
    Args:
        datasets_videos (list): List of (dataset, video_file) tuples
    
    Returns list of dicts with columns: dataset/video, classifier, tilesize, runtime_files
    Note: Query execution is done per video
    """
    query_data = []
    
    for dataset, video in datasets_videos:
        video_path = os.path.join(CACHE_DIR, dataset, 'execution', video)
        
        # Groundtruth detection and tracking (no tilesize)
        groundtruth_detection_path = os.path.join(video_path, '000_groundtruth', 'detections.jsonl')
        groundtruth_tracking_path = os.path.join(video_path, '000_groundtruth', 'tracking_runtimes.jsonl')
        assert os.path.exists(groundtruth_detection_path), \
            f"Groundtruth detection path {groundtruth_detection_path} does not exist"
        assert os.path.exists(groundtruth_tracking_path), \
            f"Groundtruth tracking path {groundtruth_tracking_path} does not exist"
        
        query_data.append({
            'dataset': dataset,
            'video': video,
            'classifier': '_NA_',
            'tilesize': 0,
            'tilepadding': '_NA_',
            'stage': '001_preprocess_groundtruth_detection',
            'runtime_file': groundtruth_detection_path
        })
        
        query_data.append({
            'dataset': dataset,
            'video': video,
            'classifier': '_NA_',
            'tilesize': 0,
            'tilepadding': '_NA_',
            'stage': '002_preprocess_groundtruth_tracking',
            'runtime_file': groundtruth_tracking_path
        })

        # Classifier-based stages
        for classifier in EXEC_CLASSIFIERS:
            for tilesize in TILE_SIZES:
                cl_ts = f'{classifier}_{tilesize}'
                for tilepadding in TILEPADDING_MODES:
                    runtime_files = []
                    cl_ts_tp = f'{classifier}_{tilesize}_{tilepadding}'

                    # 020_exec_classify.py
                    classify_path = os.path.join(video_path, '020_relevancy', cl_ts, 'score', 'score.jsonl')
                    assert os.path.exists(classify_path), f"Classify path {classify_path} does not exist"
                    runtime_files.append(('020_exec_classify', classify_path))
                    
                    # 030_exec_compress.py
                    compress_path = os.path.join(video_path, '030_compressed_frames', cl_ts_tp, 'runtime.jsonl')
                    assert os.path.exists(compress_path), f"Compress path {compress_path} does not exist"
                    runtime_files.append(('030_exec_compress', compress_path))
                    
                    # 040_exec_detect.py
                    detect_path = os.path.join(video_path, '040_compressed_detections', cl_ts_tp, 'runtimes.jsonl')
                    assert os.path.exists(detect_path), f"Detect path {detect_path} does not exist"
                    runtime_files.append(('040_exec_detect', detect_path))

                    # # 050_exec_uncompress.py
                    # uncompress_path = os.path.join(video_path, '050_uncompressed_detections', cl_ts_tp, 'runtime.jsonl')
                    # assert os.path.exists(uncompress_path), f"Uncompress path {uncompress_path} does not exist"
                    # runtime_files.append(('050_exec_uncompress', uncompress_path))
                    
                    # 060_exec_track.py
                    track_path = os.path.join(video_path, '060_uncompressed_tracks', cl_ts_tp, 'runtimes.jsonl')
                    assert os.path.exists(track_path), f"Track path {track_path} does not exist"
                    runtime_files.append(('060_exec_track', track_path))
                    
                    query_data.extend({
                        'dataset': dataset,
                        'video': video,
                        'classifier': classifier,
                        'tilesize': tilesize,
                        'tilepadding': tilepadding,
                        'stage': stage,
                        'runtime_file': runtime_file
                    } for stage, runtime_file in runtime_files)
    
    return query_data


def print_index_construction_table(index_data, write_line=print):
    """Print index construction data as a table to stdout and optionally to a file."""
    
    write_line("INDEX CONSTRUCTION DATASET")
    write_line("=" * 80)
    write_line("Stages: 011_tune_detect, 012_tune_create_training_data, 013_tune_train_classifier")
    write_line()

    df = pd.DataFrame.from_dict(index_data)
    sizes = df.groupby(['dataset', 'stage', 'classifier', 'tilesize', 'tilepadding']).size()
    assert isinstance(sizes, pd.Series), "sizes is not a pandas DataFrame"
    write_line(sizes.reset_index(name='count').to_string(index=False))


def print_query_execution_table(query_data, write_line=print):
    """Print query execution data as a table to stdout and optionally to a file."""
    
    write_line("QUERY EXECUTION DATASET")
    write_line("=" * 120)
    write_line("Stages: 020_exec_classify, 030_exec_compress, 040_exec_detect, 050_exec_uncompress, 060_exec_track")
    write_line()

    df = pd.DataFrame.from_dict(query_data)
    sizes = df.groupby(['dataset', 'classifier', 'tilesize', 'tilepadding']).size()
    assert isinstance(sizes, pd.Series), "sizes is not a pandas DataFrame"
    write_line(sizes.reset_index(name='count').to_string(index=False))


def save_data_tables(index_data, query_data):
    """Save both data tables to cache directory."""
    # Get unique datasets
    datasets = set()
    for entry in index_data:
        dataset = entry['dataset']
        datasets.add(dataset)
    
    for dataset in datasets:
        # Create output directory
        output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save index construction data
        save_index_construction_data(index_data, dataset, output_dir)
        
        # Save query execution data
        save_query_execution_data(query_data, dataset, output_dir)
        
        print(f"\nData tables saved to: {output_dir}")


def save_index_construction_data(index_data, dataset, output_dir):
    """Save index construction data in multiple formats."""
    # Filter data for this dataset
    dataset_data = [entry for entry in index_data if entry['dataset'] == dataset]
    
    # Save as JSON
    # json_file = os.path.join(output_dir, 'index_construction.json')
    # with open(json_file, 'w') as f:
    #     json.dump(dataset_data, f, indent=2)
    df = pd.DataFrame.from_dict(index_data)
    df = df[df['dataset'] == dataset]
    df.to_csv(os.path.join(output_dir, 'index_construction.csv'), index=False)
    
    # Save as text table using the print function
    txt_file = os.path.join(output_dir, 'index_construction.txt')
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_index_construction_table(dataset_data, write_line=fwrite)


def save_query_execution_data(query_data, dataset, output_dir):
    """Save query execution data in multiple formats."""
    # Filter data for this dataset
    dataset_data = [entry for entry in query_data if entry['dataset'] == dataset]
    
    # Save as JSON
    # json_file = os.path.join(output_dir, 'query_execution.json')
    # with open(json_file, 'w') as f:
    #     json.dump(dataset_data, f, indent=2)
    df = pd.DataFrame.from_dict(query_data)
    df = df[df['dataset'] == dataset]
    df.to_csv(os.path.join(output_dir, 'query_execution.csv'), index=False)
    
    # Save as text table using the print function
    txt_file = os.path.join(output_dir, 'query_execution.txt')
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_query_execution_table(dataset_data, write_line=fwrite)


def main(args):
    """Main function to gather and print runtime data."""
    print("Gathering runtime data from all stages and configurations...")

    datasets = args.datasets
    
    # Discover available videos from execution directories
    datasets_videos = discover_available_videos(datasets)
    print(f"Found {len(datasets_videos)} dataset/video combinations")
    
    # Gather data
    index_data = gather_index_construction_data(datasets)
    query_data = gather_query_execution_data(datasets_videos)
    
    # Print tables
    print("\n1. INDEX CONSTRUCTION DATASET")
    print_index_construction_table(index_data)
    print("\n2. QUERY EXECUTION DATASET")  
    print_query_execution_table(query_data)
    
    # Save data tables
    save_data_tables(index_data, query_data)


if __name__ == '__main__':
    main(parse_args())
