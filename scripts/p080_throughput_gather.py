
#!/usr/local/bin/python

import os
import json
from polyis.utilities import CACHE_DIR

DATASETS_VIDEOS = [
    ('b3d', 'jnc00.mp4'),
    ('b3d', 'jnc02.mp4'),
    ('b3d', 'jnc06.mp4'),
    ('b3d', 'jnc07.mp4'),
]

CLASSIFIERS = ['SimpleCNN']
EXEC_CLASSIFIERS = ['SimpleCNN', 'groundtruth']
TILE_SIZES = [30, 60, 120]


def gather_index_construction_data():
    """
    Gather runtime data for index construction stages:
    - 011_tune_detect.py
    - 012_tune_create_training_data.py  
    - 013_tune_train_classifier.py
    
    Returns list of dicts with columns: dataset/video, classifier, runtime_files
    """
    index_data = []
    
    for dataset, video in DATASETS_VIDEOS:
        for classifier in CLASSIFIERS:
            runtime_files = []
            video_path = os.path.join(CACHE_DIR, dataset, video)
            
            # 011_tune_detect.py - embedded in detections.jsonl (shared across all configs)
            detect_path = os.path.join(video_path, 'segments', 'detection', 'detections.jsonl')
            if os.path.exists(detect_path):
                runtime_files.append(('011_tune_detect', detect_path))
            
            # 012_tune_create_training_data.py - for all tile sizes
            for tile_size in TILE_SIZES:
                training_data_path = os.path.join(video_path, 'training', 'runtime', f'tilesize_{tile_size}', 'create_training_data.jsonl')
                if os.path.exists(training_data_path):
                    runtime_files.append(('012_tune_create_training_data', training_data_path))
            
            # 013_tune_train_classifier.py - for all tile sizes with this classifier
            for tile_size in TILE_SIZES:
                train_logs_dir = os.path.join(video_path, 'training', 'results', f'{classifier}_{tile_size}')
                train_logs_path = os.path.join(train_logs_dir, 'train_losses.json')
                test_logs_path = os.path.join(train_logs_dir, 'test_losses.json')
                
                if os.path.exists(train_logs_path):
                    runtime_files.append(('013_tune_train_classifier', train_logs_path))
                if os.path.exists(test_logs_path):
                    runtime_files.append(('013_tune_train_classifier', test_logs_path))
            
            index_data.append({
                'dataset/video': f"{dataset}/{video}",
                'classifier': classifier,
                'runtime_files': runtime_files
            })
    
    return index_data


def gather_query_execution_data():
    """
    Gather runtime data for query execution stages:
    - 020_exec_classify.py
    - 030_exec_compress.py
    - 040_exec_detect.py
    - 050_exec_uncompress.py
    - 060_exec_track.py
    
    Returns list of dicts with columns: dataset/video, classifier, tile_size, runtime_files
    """
    # Define stage configurations: (stage_name, path_template)
    stage_configs = [
        ('020_exec_classify', 'relevancy/{classifier}_{tile_size}/score/score.jsonl'),
        ('030_exec_compress', 'packing/{classifier}_{tile_size}/runtime.jsonl'),
        ('040_exec_detect', 'packed_detections/{classifier}_{tile_size}/runtimes.jsonl'),
        ('060_exec_track', 'uncompressed_tracking/{classifier}_{tile_size}/runtimes.jsonl')
        # Note: 050_exec_uncompress has no dedicated runtime file
    ]
    
    query_data = []
    
    for dataset, video in DATASETS_VIDEOS:
        query_data.append({
            'dataset/video': f"{dataset}/{video}",
            'classifier': 'groundtruth',
            'tile_size': 0,
            'runtime_files': [('001_preprocess_groundtruth_detection',
                               os.path.join(CACHE_DIR, dataset, video, 'groundtruth', 'detection.jsonl'))]
        })
        query_data.append({
            'dataset/video': f"{dataset}/{video}",
            'classifier': 'groundtruth',
            'tile_size': 0,
            'runtime_files': [('002_preprocess_groundtruth_tracking',
                               os.path.join(CACHE_DIR, dataset, video, 'groundtruth', 'tracking_runtimes.jsonl'))]
        })

        for classifier in EXEC_CLASSIFIERS:
            for tile_size in TILE_SIZES:
                runtime_files = []
                video_path = os.path.join(CACHE_DIR, dataset, video)
                
                # Check each stage configuration
                for stage_name, path_template in stage_configs:
                    path = path_template.format(classifier=classifier, tile_size=tile_size)
                    stage_path = os.path.join(video_path, path)
                    
                    if os.path.exists(stage_path):
                        runtime_files.append((stage_name, stage_path))
                
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
    write_line(f"{'Dataset/Video':<20} {'Classifier':<12} {'Runtime Files':<60}")
    write_line("-" * 92)
    
    for entry in index_data:
        dataset_video = entry['dataset/video']
        classifier = entry['classifier']
        
        # Create a compact representation of runtime files
        stage_files = {}
        for stage, path in entry['runtime_files']:
            if stage not in stage_files:
                stage_files[stage] = 0
            stage_files[stage] += 1
        
        runtime_summary = []
        for stage in ['011_tune_detect', '012_tune_create_training_data', '013_tune_train_classifier']:
            count = stage_files.get(stage, 0)
            if count > 0:
                runtime_summary.append(f"{stage}({count})")
        
        runtime_files_str = ", ".join(runtime_summary)
        
        write_line(f"{dataset_video:<20} {classifier:<12} {runtime_files_str:<60}")


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
        dataset = entry['dataset/video'].split('/')[0]
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
    dataset_data = [entry for entry in index_data if entry['dataset/video'].startswith(dataset)]
    
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
        for stage, _ in entry['runtime_files']:
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
