
#!/usr/local/bin/python

import os
import json
from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST

DATASETS_VIDEOS = [
    # ('b3d', 'jnc00.mp4'),
    # ('b3d', 'jnc02.mp4'),
    # ('b3d', 'jnc06.mp4'),
    # ('b3d', 'jnc07.mp4'),

    ('caldot1', 'caldot1-1.mp4'),
    ('caldot1', 'caldot1-2.mp4'),
    ('caldot1', 'caldot1-3.mp4'),
    ('caldot1', 'caldot1-4.mp4'),
    ('caldot1', 'caldot1-5.mp4'),
    ('caldot1', 'caldot1-6.mp4'),
    ('caldot1', 'caldot1-7.mp4'),
    ('caldot1', 'caldot1-8.mp4'),
    ('caldot1', 'caldot1-9.mp4'),

    ('caldot2', 'caldot2-1.mp4'),
    ('caldot2', 'caldot2-2.mp4'),
    ('caldot2', 'caldot2-3.mp4'),
    ('caldot2', 'caldot2-4.mp4'),
    ('caldot2', 'caldot2-5.mp4'),
    ('caldot2', 'caldot2-6.mp4'),
    ('caldot2', 'caldot2-7.mp4'),
]

# CLASSIFIERS = ['SimpleCNN']
CLASSIFIERS = CLASSIFIERS_TO_TEST
# EXEC_CLASSIFIERS = ['SimpleCNN', 'Perfect']
EXEC_CLASSIFIERS = CLASSIFIERS_TO_TEST + ['Perfect']
TILE_SIZES = [30, 60]  #, 120]


def gather_index_construction_data():
    """
    Gather runtime data for index construction stages:
    - 011_tune_detect.py
    - 012_tune_create_training_data.py  
    - 013_tune_train_classifier.py
    
    Returns list of dicts with columns: dataset, classifier, runtime_files
    Note: Index construction is done at dataset level, not per video
    """
    index_data = []
    
    # Get unique datasets
    datasets = set()
    for dataset, video in DATASETS_VIDEOS:
        datasets.add(dataset)
    
    for dataset in datasets:
        for classifier in CLASSIFIERS:
            runtime_files = []
            dataset_path = os.path.join(CACHE_DIR, dataset)
            
            # 011_tune_detect.py - per video detections in indexing/segment/detection/
            indexing_detect_dir = os.path.join(dataset_path, 'indexing', 'segment', 'detection')
            if os.path.exists(indexing_detect_dir):
                for video_file in os.listdir(indexing_detect_dir):
                    if video_file.endswith('.detections.jsonl'):
                        video_name = video_file.replace('.detections.jsonl', '')
                        detect_path = os.path.join(indexing_detect_dir, video_file)
                        runtime_files.append(('011_tune_detect', detect_path, video_name))
            
            # 012_tune_create_training_data.py - per video runtime files for all tile sizes
            training_runtime_dir = os.path.join(dataset_path, 'indexing', 'training', 'runtime')
            if os.path.exists(training_runtime_dir):
                for tile_size in TILE_SIZES:
                    tile_dir = os.path.join(training_runtime_dir, f'tilesize_{tile_size}')
                    if os.path.exists(tile_dir):
                        for runtime_file in os.listdir(tile_dir):
                            if runtime_file.endswith('_creating_training_data.jsonl'):
                                video_name = runtime_file.replace('_creating_training_data.jsonl', '')
                                training_data_path = os.path.join(tile_dir, runtime_file)
                                runtime_files.append(('012_tune_create_training_data', training_data_path, video_name))
            
            # 013_tune_train_classifier.py - per classifier/tile_size training results
            training_results_dir = os.path.join(dataset_path, 'indexing', 'training', 'results')
            if os.path.exists(training_results_dir):
                for tile_size in TILE_SIZES:
                    classifier_dir = os.path.join(training_results_dir, f'{classifier}_{tile_size}')
                    if os.path.exists(classifier_dir):
                        train_logs_path = os.path.join(classifier_dir, 'train_losses.json')
                        test_logs_path = os.path.join(classifier_dir, 'test_losses.json')
                        
                        if os.path.exists(train_logs_path):
                            runtime_files.append(('013_tune_train_classifier', train_logs_path, 'dataset_level'))
                        if os.path.exists(test_logs_path):
                            runtime_files.append(('013_tune_train_classifier', test_logs_path, 'dataset_level'))
            
            index_data.append({
                'dataset': dataset,
                'classifier': classifier,
                'runtime_files': runtime_files
            })
    
    return index_data


def gather_query_execution_data():
    """
    Gather runtime data for query execution stages:
    - 001_preprocess_groundtruth_detection.py
    - 002_preprocess_groundtruth_tracking.py
    - 020_exec_classify.py
    - 030_exec_compress.py
    - 040_exec_detect.py
    - 050_exec_uncompress.py
    - 060_exec_track.py
    
    Returns list of dicts with columns: dataset/video, classifier, tile_size, runtime_files
    Note: Query execution is done per video
    """
    query_data = []
    
    for dataset, video in DATASETS_VIDEOS:
        video_path = os.path.join(CACHE_DIR, dataset, 'execution', video)
        
        # Groundtruth detection and tracking (no tile_size)
        groundtruth_detection_path = os.path.join(video_path, '000_groundtruth', 'detections.jsonl')
        groundtruth_tracking_path = os.path.join(video_path, '000_groundtruth', 'tracking_runtimes.jsonl')
        
        if os.path.exists(groundtruth_detection_path):
            query_data.append({
                'dataset/video': f"{dataset}/{video}",
                'classifier': 'groundtruth',
                'tile_size': 0,
                'runtime_files': [('001_preprocess_groundtruth_detection', groundtruth_detection_path)]
            })
        
        if os.path.exists(groundtruth_tracking_path):
            query_data.append({
                'dataset/video': f"{dataset}/{video}",
                'classifier': 'groundtruth',
                'tile_size': 0,
                'runtime_files': [('002_preprocess_groundtruth_tracking', groundtruth_tracking_path)]
            })

        # Classifier-based stages
        for classifier in EXEC_CLASSIFIERS:
            for tile_size in TILE_SIZES:
                runtime_files = []
                
                # 020_exec_classify.py
                classify_path = os.path.join(video_path, '020_relevancy', f'{classifier}_{tile_size}', 'score', 'score.jsonl')
                if os.path.exists(classify_path):
                    runtime_files.append(('020_exec_classify', classify_path))
                
                # 030_exec_compress.py
                compress_path = os.path.join(video_path, '030_compressed_frames', f'{classifier}_{tile_size}', 'runtime.jsonl')
                if os.path.exists(compress_path):
                    runtime_files.append(('030_exec_compress', compress_path))
                
                # 040_exec_detect.py
                detect_path = os.path.join(video_path, '040_compressed_detections', f'{classifier}_{tile_size}', 'runtimes.jsonl')
                if os.path.exists(detect_path):
                    runtime_files.append(('040_exec_detect', detect_path))
                
                # 060_exec_track.py
                track_path = os.path.join(video_path, '060_uncompressed_tracks', f'{classifier}_{tile_size}', 'runtimes.jsonl')
                if os.path.exists(track_path):
                    runtime_files.append(('060_exec_track', track_path))
                
                # Note: 050_exec_uncompress has no dedicated runtime file
                
                query_data.append({
                    'dataset/video': f"{dataset}/{video}",
                    'classifier': classifier,
                    'tile_size': tile_size,
                    'runtime_files': runtime_files
                })
    
    return query_data


def print_index_construction_table(index_data, write_line=print):
    """Print index construction data as a table to stdout and optionally to a file."""
    
    write_line("INDEX CONSTRUCTION DATASET")
    write_line("=" * 80)
    write_line("Stages: 011_tune_detect, 012_tune_create_training_data, 013_tune_train_classifier")
    write_line()
    
    # Header
    write_line(f"{'Dataset':<15} {'Classifier':<12} {'Runtime Files':<60}")
    write_line("-" * 87)
    
    for entry in index_data:
        dataset = entry['dataset']
        classifier = entry['classifier']
        
        # Create a compact representation of runtime files
        stage_files = {}
        for stage, path, video_name in entry['runtime_files']:
            if stage not in stage_files:
                stage_files[stage] = 0
            stage_files[stage] += 1
        
        runtime_summary = []
        for stage in ['011_tune_detect', '012_tune_create_training_data', '013_tune_train_classifier']:
            count = stage_files.get(stage, 0)
            if count > 0:
                runtime_summary.append(f"{stage}({count})")
        
        runtime_files_str = ", ".join(runtime_summary)
        
        write_line(f"{dataset:<15} {classifier:<12} {runtime_files_str:<60}")


def print_query_execution_table(query_data, write_line=print):
    """Print query execution data as a table to stdout and optionally to a file."""
    
    write_line("QUERY EXECUTION DATASET")
    write_line("=" * 120)
    write_line("Stages: 020_exec_classify, 030_exec_compress, 040_exec_detect, 050_exec_uncompress, 060_exec_track")
    write_line()
    
    # Header
    write_line(f"{'Dataset/Video':<20} {'Classifier':<12} {'Tile Size':<10} {'Runtime Files':<70}")
    write_line("-" * 112)
    
    for entry in query_data:
        dataset_video = entry['dataset/video']
        classifier = entry['classifier']
        tile_size = entry['tile_size']
        
        # Show all runtime files
        stage_files = []
        for stage, path in entry['runtime_files']:
            stage_files.append(stage)
        
        runtime_files_str = ", ".join(stage_files) if stage_files else "None"
        
        write_line(f"{dataset_video:<20} {classifier:<12} {tile_size:<10} {runtime_files_str:<70}")


def save_data_tables(index_data, query_data):
    """Save both data tables to cache directory."""
    # Get unique datasets
    datasets = set()
    for entry in index_data:
        dataset = entry['dataset']
        datasets.add(dataset)
    
    for dataset in datasets:
        # Create output directory
        output_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'throughput')
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
    json_file = os.path.join(output_dir, 'index_construction.json')
    with open(json_file, 'w') as f:
        json.dump(dataset_data, f, indent=2)
    
    # Save as text table using the print function
    txt_file = os.path.join(output_dir, 'index_construction.txt')
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_index_construction_table(dataset_data, write_line=fwrite)


def save_query_execution_data(query_data, dataset, output_dir):
    """Save query execution data in multiple formats."""
    # Filter data for this dataset
    dataset_data = [entry for entry in query_data if entry['dataset/video'].startswith(dataset)]
    
    # Save as JSON
    json_file = os.path.join(output_dir, 'query_execution.json')
    with open(json_file, 'w') as f:
        json.dump(dataset_data, f, indent=2)
    
    # Save as text table using the print function
    txt_file = os.path.join(output_dir, 'query_execution.txt')
    with open(txt_file, 'w') as f:
        def fwrite(text=""):
            f.write(text + "\n")
        print_query_execution_table(dataset_data, write_line=fwrite)


def main():
    """Main function to gather and print runtime data."""
    print("Gathering runtime data from all stages and configurations...")
    
    # Gather data
    index_data = gather_index_construction_data()
    query_data = gather_query_execution_data()
    
    # Print tables
    print("\n1. INDEX CONSTRUCTION DATASET")
    print_index_construction_table(index_data)
    print("\n2. QUERY EXECUTION DATASET")  
    print_query_execution_table(query_data)
    
    # Save data tables
    save_data_tables(index_data, query_data)
    
    # Summary statistics
    print("\n3. SUMMARY STATISTICS")
    print("=" * 50)
    
    # Count files by stage for index construction
    index_stage_counts = {}
    for entry in index_data:
        for stage, _, _ in entry['runtime_files']:
            index_stage_counts[stage] = index_stage_counts.get(stage, 0) + 1
    
    print("\nIndex Construction Stage Coverage:")
    for stage, count in sorted(index_stage_counts.items()):
        stage_short = stage.replace('0', '').replace('_', ' ').title()
        print(f"  {stage_short:<35}: {count:>3} files")
    
    # Count files by stage for query execution
    query_stage_counts = {}
    for entry in query_data:
        for stage, _ in entry['runtime_files']:
            query_stage_counts[stage] = query_stage_counts.get(stage, 0) + 1
    
    print("\nQuery Execution Stage Coverage:")
    for stage, count in sorted(query_stage_counts.items()):
        stage_short = stage.replace('0', '').replace('_exec_', ' ').replace('_', ' ').title()
        print(f"  {stage_short:<35}: {count:>3} files")
    
    print(f"\nTotal Entries:")
    print(f"  Index Construction                : {len(index_data):>3} entries")
    print(f"  Query Execution                   : {len(query_data):>3} entries")
    print(f"  Total Runtime Files Found        : {sum(index_stage_counts.values()) + sum(query_stage_counts.values()):>3} files")


if __name__ == '__main__':
    main()
