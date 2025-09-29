#!/usr/local/bin/python

import os
import json
import argparse
from typing import Callable, Dict, List, Tuple, Any
from collections import defaultdict

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process runtime measurement data for throughput analysis')
    parser.add_argument('--datasets', required=False,
                        default=['caldot1', 'caldot2'],
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_data_tables(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load the index construction and query execution data tables."""
    index_file = os.path.join(data_dir, 'index_construction.json')
    query_file = os.path.join(data_dir, 'query_execution.json')
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"Loaded {len(index_data)} index construction entries")
    print(f"Loaded {len(query_data)} query execution entries")
    
    return index_data, query_data


def parse_runtime_file(file_path: str, stage: str, accessor: Callable[[Dict], List[Dict]] | None = None) -> List[Dict]:
    """Parse a runtime file and extract timing data."""
    if not os.path.exists(file_path):
        return []
    
    timings = []
    ignored_ops = ['total_frame_time', 'read_frame', 'save_canvas', 'save_mapping_files']
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if accessor is not None:
                        data = accessor(data)
                    if isinstance(data, dict):
                        data = [data]
                    assert isinstance(data, list), f"Expected a list of dictionaries, got {type(data)}, {stage}, {file_path}"
                    for item in data:
                        if 'time' in item and isinstance(item['time'], (int, float)):
                            # Convert from milliseconds to seconds
                            item_copy = item.copy()
                            item_copy['time'] = item['time'] / 1000.0
                            if item_copy['op'] in ignored_ops:
                                continue
                            timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
        elif file_path.endswith('.json'):
            # JSON format - array of objects
            data = json.load(f)
            assert isinstance(data, list), f"Expected a list of objects, got {type(data)}, {stage}, {stage}"
            for entry in data:
                if accessor is not None:
                    entry = accessor(entry)
                if isinstance(entry, dict):
                    entry = [entry]

                assert isinstance(entry, list), f"Expected a list of dictionaries, got {type(entry)}, {stage}"
                for item in entry:
                    if 'time' in item and isinstance(item['time'], (int, float)):
                        # Convert from milliseconds to seconds
                        item_copy = item.copy()
                        item_copy['time'] = item['time'] / 1000.0
                        if item_copy['op'] in ignored_ops:
                            continue
                        timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
    
    return timings


def parse_index_construction_timings(index_data: List[Dict]) -> Dict[str, Any]:
    """Parse timing data for index construction stages."""
    timings = {
        '011_tune_detect': defaultdict(list),
        '012_tune_create_training_data': defaultdict(list),
        '013_tune_train_classifier': defaultdict(list)
    }
    
    stage_summaries = {
        '011_tune_detect': defaultdict(list),
        '012_tune_create_training_data': defaultdict(list),
        '013_tune_train_classifier': defaultdict(list)
    }

    accessors = {
        '011_tune_detect': lambda row: row[3],
        '012_tune_create_training_data': lambda row: row,
        '013_tune_train_classifier': lambda row: row
    }
    
    for entry in index_data:
        dataset = entry['dataset']
        classifier = entry['classifier']
        config_key = f"{dataset}_{classifier}"
        
        for stage, file_path, video_name in entry['runtime_files']:
            file_timings = parse_runtime_file(file_path, stage, accessors[stage])
            
            if not file_timings:
                continue
            
            # Aggregate individual timings by operation
            op_aggregates = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
            for timing in file_timings:
                op = timing.get('op', 'unknown')
                time_val = timing.get('time', 0)
                if isinstance(time_val, (int, float)):
                    op_aggregates[op]['total_time'] += time_val
                    op_aggregates[op]['count'] += 1
            
            # Store aggregated timings
            for op, agg_data in op_aggregates.items():
                timings[stage][config_key].append({
                    'stage': stage,
                    'op': op,
                    'time': agg_data['total_time'],
                    'count': agg_data['count']
                })
            
            # Calculate stage summary
            if stage == '011_tune_detect':
                # Detection stage - sum all operations
                total_time = sum(agg['total_time'] for agg in op_aggregates.values())
                stage_summaries[stage][config_key].append(total_time)
            
            elif stage == '012_tune_create_training_data':
                # Training data creation - group by operation and tile size
                for timing in file_timings:
                    op = timing.get('op', 'unknown')
                    tile_size = timing.get('tile_size', 'unknown')
                    time_val = timing.get('time', 0)
                    
                    op_key = f"{config_key}_{tile_size}_{op}"
                    stage_summaries[stage][op_key].append(time_val)
            
            elif stage == '013_tune_train_classifier':
                # Classifier training and testing - sum all training and testing epochs
                total_time = sum(agg['total_time'] for agg in op_aggregates.values())
                stage_summaries[stage][config_key].append(total_time)
    
    return {
        'timings': timings,
        'summaries': stage_summaries
    }


def parse_query_execution_timings(query_data: List[Dict]) -> Dict[str, Any]:
    """Parse timing data for query execution stages."""
    timings = {
        '001_preprocess_groundtruth_detection': defaultdict(list),
        '002_preprocess_groundtruth_tracking': defaultdict(list),
        '020_exec_classify': defaultdict(list),
        '030_exec_compress': defaultdict(list),
        '040_exec_detect': defaultdict(list),
        '060_exec_track': defaultdict(list)
    }
    
    stage_summaries = {
        '001_preprocess_groundtruth_detection': defaultdict(list),
        '002_preprocess_groundtruth_tracking': defaultdict(list),
        '020_exec_classify': defaultdict(list),
        '030_exec_compress': defaultdict(list),
        '040_exec_detect': defaultdict(list),
        '060_exec_track': defaultdict(list)
    }

    accessors = {
        '001_preprocess_groundtruth_detection': lambda row: row['runtime'],
        '002_preprocess_groundtruth_tracking': lambda row: row['runtime'],
        '020_exec_classify': lambda row: row['runtime'],
        '030_exec_compress': lambda row: row['runtime'],
        '040_exec_detect': lambda row: row,
        '060_exec_track': lambda row: row['runtime']
    }
    
    for entry in query_data:
        dataset_video = entry['dataset/video']
        classifier = entry['classifier']
        tile_size = entry['tile_size']
        config_key = f"{dataset_video}_{classifier}_{tile_size}"
        
        for stage, file_path in entry['runtime_files']:
            file_timings = parse_runtime_file(file_path, stage, accessors[stage])
            
            if not file_timings:
                continue
            
            # Aggregate individual timings by operation
            op_aggregates = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
            for timing in file_timings:
                op = timing.get('op', 'unknown')
                time_val = timing.get('time', 0)
                if isinstance(time_val, (int, float)):
                    op_aggregates[op]['total_time'] += time_val
                    op_aggregates[op]['count'] += 1
            
            # Store aggregated timings
            for op, agg_data in op_aggregates.items():
                timings[stage][config_key].append({
                    'stage': stage,
                    'op': op,
                    'time': agg_data['total_time'],
                    'count': agg_data['count']
                })
            
            # Calculate stage summary
            total_time = sum(agg['total_time'] for agg in op_aggregates.values())
            stage_summaries[stage][config_key].append(total_time)
    
    return {
        'timings': timings,
        'summaries': stage_summaries
    }


def extract_videos_from_data(query_timings: Dict[str, Any]) -> List[str]:
    """Extract unique video names from the query timing data."""
    videos = set()
    for stage_timings in query_timings['timings'].values():
        for config_key in stage_timings.keys():
            # config_key format: dataset/video_classifier_tilesize
            parts = config_key.split('_')
            if len(parts) >= 3:
                # Extract video name (everything before the last two underscores)
                video_part = '_'.join(parts[:-2])  # Remove classifier and tilesize
                videos.add(video_part)
    return sorted(list(videos))


def save_measurements(index_timings: Dict[str, Any], query_timings: Dict[str, Any], 
                     output_dir: str, dataset: str):
    """Save processed timing measurements to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save index construction measurements
    index_file = os.path.join(output_dir, 'index_construction_measurements.json')
    with open(index_file, 'w') as f:
        json.dump(index_timings, f, indent=2, default=str)
    print(f"Saved index construction measurements to: {index_file}")
    
    # Save query execution timings (individual timing data)
    query_timings_file = os.path.join(output_dir, 'query_execution_timings.json')
    with open(query_timings_file, 'w') as f:
        json.dump(query_timings['timings'], f, indent=2, default=str)
    print(f"Saved query execution timings to: {query_timings_file}")
    
    # Save query execution summaries (aggregated timing data)
    query_summaries_file = os.path.join(output_dir, 'query_execution_summaries.json')
    with open(query_summaries_file, 'w') as f:
        json.dump(query_timings['summaries'], f, indent=2, default=str)
    print(f"Saved query execution summaries to: {query_summaries_file}")
    
    # Save metadata
    videos = extract_videos_from_data(query_timings)
    metadata = {
        'dataset': dataset,
        'videos': videos,
        'classifiers': CLASSIFIERS_TO_TEST + ['groundtruth'],
        'tile_sizes': [30, 60],
        'index_stages': ['011_tune_detect', '012_tune_create_training_data', '013_tune_train_classifier'],
        'query_stages': ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking', 
                        '020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")
    
    print(f"Found {len(videos)} videos: {videos}")


def main():
    """Main function to process runtime measurement data."""
    args = parse_args()
    
    for dataset in args.datasets:
        print(f"Loading throughput data tables for dataset: {dataset}")
        data_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'throughput')
        print(f"Data directory: {data_dir}")
        index_data, query_data = load_data_tables(data_dir)
        
        print("Parsing index construction timing data...")
        index_timings = parse_index_construction_timings(index_data)
        
        print("Parsing query execution timing data...")
        query_timings = parse_query_execution_timings(query_data)
        
        print("Saving processed measurements...")
        measurements_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'throughput', 'measurements')
        save_measurements(index_timings, query_timings, measurements_dir, dataset)
        
        print(f"\nData processing complete for {dataset}! Measurements saved to: {measurements_dir}")


if __name__ == '__main__':
    main()