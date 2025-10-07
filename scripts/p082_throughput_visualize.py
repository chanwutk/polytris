#!/usr/local/bin/python

import os
import json
import argparse
import numpy as np
import altair as alt
import pandas as pd
from collections import defaultdict
from typing import Any

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATASETS_TO_TEST


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize runtime breakdown of training configurations')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_measurements(measurements_dir: str) -> tuple[dict, dict, dict]:
    """Load the processed measurement data."""
    index_file = os.path.join(measurements_dir, 'index_construction_measurements.json')
    query_timings_file = os.path.join(measurements_dir, 'query_execution_timings.json')
    query_summaries_file = os.path.join(measurements_dir, 'query_execution_summaries.json')
    metadata_file = os.path.join(measurements_dir, 'metadata.json')
    
    with open(index_file, 'r') as f:
        index_timings = json.load(f)
    
    with open(query_timings_file, 'r') as f:
        query_timings_data = json.load(f)
    
    with open(query_summaries_file, 'r') as f:
        query_summaries_data = json.load(f)
    
    # Reconstruct the query_timings structure
    query_timings = {
        'timings': query_timings_data,
        'summaries': query_summaries_data
    }
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded index construction measurements")
    print(f"Loaded query execution timings and summaries")
    print(f"Loaded metadata for {len(metadata['videos'])} videos")
    
    return index_timings, query_timings, metadata


def visualize_breakdown_query_execution(query_timings: dict, output_dir: str, dataset: str, 
                                        video: str, video_name: str):
    """Create query execution visualization for a specific video.
    
    Args:
        query_timings: Query timing data
        output_dir: Output directory for saving plots
        dataset: Dataset name
        video: Specific video for per-video analysis
        video_name: Display name for the video
    """
    
    # Prepare data for pipeline visualization
    stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    stage_names = ['Classify', 'Compress', 'Detect', 'Track']
    
    # Group by classifier, tile size, and tilepadding
    classifiers = CLASSIFIERS_TO_TEST
    tilesizes = [30, 60]
    
    # Check if this dataset uses the new format with tilepadding parameter
    has_tilepadding_format = False
    for stage in stages:
        for classifier in classifiers:
            for tilesize in tilesizes:
                test_config_key = f"{video}_{classifier}_{tilesize}_padded"
                if test_config_key in query_timings['timings'][stage]:
                    has_tilepadding_format = True
                    break
            if has_tilepadding_format:
                break
        if has_tilepadding_format:
            break
    
    if has_tilepadding_format:
        tilepadding_values = ['padded', 'unpadded']
    else:
        tilepadding_values = ['N/A']
    
    # Find which classifiers actually have data for this video
    available_classifiers = set()
    for stage in stages:
        for classifier in classifiers:
            for tilesize in tilesizes:
                for tilepadding in tilepadding_values:
                    if has_tilepadding_format:
                        config_key = f"{video}_{classifier}_{tilesize}_{tilepadding}"
                    else:
                        config_key = f"{video}_{classifier}_{tilesize}_N/A"
                    if config_key in query_timings['timings'][stage]:
                        available_classifiers.add(classifier)
    
    # Only process classifiers that have data
    classifiers = [c for c in classifiers if c in available_classifiers]
    
    if not classifiers:
        return  # No data available for this video
    
    # Prepare data for all charts
    chart_data = []
    
    for stage, stage_name in zip(stages, stage_names):
        # Collect data for this stage grouped by operation
        config_labels = []
        op_data = defaultdict(list)  # op_name -> list of values for each config
        
        for classifier in classifiers:
            for tilesize in tilesizes:
                for tilepadding in tilepadding_values:
                    # Per-video analysis
                    if has_tilepadding_format:
                        config_key = f"{video}_{classifier}_{tilesize}_{tilepadding}"
                        config_labels.append(f"{classifier} {tilesize} {tilepadding}")
                    else:
                        config_key = f"{video}_{classifier}_{tilesize}_N/A"
                        config_labels.append(f"{classifier} {tilesize}")
                    
                    # Get individual timings for this config to group by operation
                    config_ops = defaultdict(float)
                    if config_key in query_timings['timings'][stage]:
                        for timing in query_timings['timings'][stage][config_key]:
                            op_name = timing.get('op', 'unknown')
                            config_ops[op_name] += timing['time']
                    
                    # Ensure all operations have a value for this config (0 if not present)
                    all_ops = set()
                    for other_classifier in classifiers:
                        for other_tilesize in tilesizes:
                            for other_tilepadding in tilepadding_values:
                                if has_tilepadding_format:
                                    other_config_key = f"{video}_{other_classifier}_{other_tilesize}_{other_tilepadding}"
                                else:
                                    other_config_key = f"{video}_{other_classifier}_{other_tilesize}_N/A"
                                if other_config_key in query_timings['timings'][stage]:
                                    for timing in query_timings['timings'][stage][other_config_key]:
                                        all_ops.add(timing.get('op', 'unknown'))
                    
                    for op_name in all_ops:
                        op_data[op_name].append(config_ops.get(op_name, 0))
        
        if config_labels and op_data:
            # Calculate total values for each config to sort by bar size
            config_totals = []
            for i in range(len(config_labels)):
                total = sum(op_data[op_name][i] for op_name in op_data.keys() if i < len(op_data[op_name]))
                config_totals.append(total)
            
            # Sort configs by total value (descending)
            sorted_indices = sorted(range(len(config_labels)), key=lambda i: config_totals[i], reverse=True)
            sorted_config_labels = [config_labels[i] for i in sorted_indices]
            
            # Sort operation data to match the sorted config order
            sorted_op_data = {}
            for op_name, values in op_data.items():
                sorted_op_data[op_name] = [values[i] for i in sorted_indices]
            
            # Prepare data for this stage
            for op_name, values in sorted(sorted_op_data.items()):
                if any(v > 0 for v in values):  # Only include if there are non-zero values
                    for i, value in enumerate(values):
                        chart_data.append({
                            'Stage': stage_name,
                            'Config': sorted_config_labels[i],
                            'Operation': op_name,
                            'Runtime': value
                        })
    
    if not chart_data:
        return
    
    df = pd.DataFrame(chart_data)
    
    # Create individual charts for each stage
    charts = []
    for stage_name in stage_names:
        stage_data = df[df['Stage'] == stage_name]
        assert isinstance(stage_data, pd.DataFrame)
        if len(stage_data) > 0:
            chart = alt.Chart(stage_data).mark_bar().encode(
                x=alt.X('Runtime:Q', title='Runtime (seconds)'),
                y=alt.Y('Config:N',
                        sort=alt.SortField(field='Runtime', order='descending'),
                        axis=alt.Axis(labelExpr="split(datum.label, ' ')",
                                      labelBaseline='alphabetic', labelLineHeight=9)),
                color=alt.Color('Operation:N', legend=alt.Legend(
                    orient='bottom',
                    columns=3,
                )),
                tooltip=['Config', 'Operation', alt.Tooltip('Runtime:Q', format='.2f', title='Runtime (s)')]
            ).properties(
                title=f'{stage_name} Runtime by Operation',
                width=300,
                height=220
            )
            charts.append(chart)
    
    # Combine charts in a 2x2 grid
    if len(charts) >= 4:
        combined_chart = alt.vconcat(
            alt.hconcat(charts[0], charts[1]).resolve_scale(
                color='independent'
            ),
            alt.hconcat(charts[2], charts[3]).resolve_scale(
                color='independent'
            )
        )
    elif len(charts) >= 2:
        combined_chart = alt.hconcat(charts[0], charts[1]).resolve_scale(
            color='independent'
        )
    else:
        combined_chart = charts[0] if charts else alt.Chart().mark_text(text='No data')
    
    # Save plot with appropriate naming and directory structure
    output_dir = os.path.join(output_dir, 'per_video')
    os.makedirs(output_dir, exist_ok=True)
    suffix = video_name.replace('/', '_').replace('.', '_') if video_name else 'unknown'

    # Save the chart
    combined_chart.save(os.path.join(output_dir, f'breakdown_{suffix}.png'), scale_factor=2)


def visualize_overall_runtime(index_timings: dict, query_timings: dict, 
                              output_dir: str, video: str, video_name: str):
    """Create comparative analysis between index construction and query execution, split by tile size.
    
    Note: Index construction is per-dataset (shared across all videos), while query execution is per-video.
    
    Args:
        index_timings: Index construction timing data (per-dataset)
        query_timings: Query execution timing data (per-video)
        output_dir: Output directory for saving plots
        video: Specific video for per-video analysis
        video_name: Display name for the video
    """
    
    # Discover all tile sizes, classifiers, and tilepadding values present in the query timing data
    all_tilesizes = set()
    all_classifiers = set()
    all_tilepadding_values = set()
    for stage_name, stage_summaries in query_timings['summaries'].items():
        for config_key in stage_summaries.keys():
            # Extract classifier, tile size, and tilepadding from config key format: video_classifier_tilesize_tilepadding
            parts = config_key.split('_')
            if len(parts) >= 4:
                if config_key.startswith(video + '_'):
                    # Extract classifier, tile size, and tilepadding
                    remaining = '_'.join(parts[1:])
                    if remaining.split('_')[-1] in ['padded', 'unpadded'] and remaining.split('_')[-2].isdigit():
                        tilepadding = remaining.split('_')[-1]
                        tilesize = int(remaining.split('_')[-2])
                        classifier = '_'.join(remaining.split('_')[:-2])
                        all_tilesizes.add(tilesize)
                        all_classifiers.add(classifier)
                        all_tilepadding_values.add(tilepadding)
    
    # Sort tile sizes and classifiers for consistent ordering
    # Filter out tilesize = 0 as requested, but keep it for naive query execution
    sorted_tilesizes = sorted([ts for ts in all_tilesizes if ts != 0])
    sorted_classifiers = sorted([c for c in all_classifiers if c != 'Perfect'])
    if 'Perfect' in all_classifiers:
        sorted_classifiers.append('Perfect')
    
    # Check if tilesize=0 data exists for naive query execution
    # Note: With tilepadding parameter, we need to check for groundtruth classifier with tilesize=0
    has_naive_query_data = False
    for stage_name, stage_summaries in query_timings['summaries'].items():
        for config_key in stage_summaries.keys():
            if config_key.startswith(video + '_'):
                parts = config_key.split('_')
                if len(parts) >= 3:
                    remaining = '_'.join(parts[1:])
                    if remaining.split('_')[-1].isdigit():
                        config_tilesize = int(remaining.split('_')[-1])
                        if config_tilesize == 0:
                            has_naive_query_data = True
                            break
        if has_naive_query_data:
            break
    
    # Calculate breakdown for index construction stages by tile size
    index_stages_by_tile = {}
    for tilesize in sorted_tilesizes:
        index_stages_by_tile[tilesize] = {
            'Detection': 0,
            'Create Training Data': 0,
            'Classifier Training': 0
        }
    
    # Index construction stage breakdown by tile size
    # Note: Index construction is per-dataset, not per-video, so we don't filter by video
    for stage_name, stage_summaries in index_timings['summaries'].items():
        for k, times in stage_summaries.items():
            if times:
                stage_total = np.sum(times)
                
                # For index construction stages, check if config key has tile size
                parts = k.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    # Config key has tile size (e.g., "caldot1_SimpleCNN_30")
                    tilesize = int(parts[-1])
                    if tilesize in index_stages_by_tile:
                        if '011_tune_detect' in stage_name:
                            index_stages_by_tile[tilesize]['Detection'] += stage_total
                        elif '012_tune_create_training_data' in stage_name:
                            index_stages_by_tile[tilesize]['Create Training Data'] += stage_total
                        elif '013_tune_train_classifier' in stage_name:
                            index_stages_by_tile[tilesize]['Classifier Training'] += stage_total
                else:
                    # Config key doesn't have tile size (e.g., "caldot1_SimpleCNN")
                    # Add to all tile sizes since index construction is shared
                    for tilesize in index_stages_by_tile.keys():
                        if '011_tune_detect' in stage_name:
                            index_stages_by_tile[tilesize]['Detection'] += stage_total
                        elif '012_tune_create_training_data' in stage_name:
                            index_stages_by_tile[tilesize]['Create Training Data'] += stage_total
                        elif '013_tune_train_classifier' in stage_name:
                            index_stages_by_tile[tilesize]['Classifier Training'] += stage_total
    
    # Calculate breakdown for query execution stages per classifier, tile size, and tilepadding
    classifier_query_stages_by_tile_tilepadding = {}
    for tilesize in sorted_tilesizes:
        classifier_query_stages_by_tile_tilepadding[tilesize] = {}
        for classifier in sorted_classifiers:
            classifier_query_stages_by_tile_tilepadding[tilesize][classifier] = {}
            for tilepadding in sorted(all_tilepadding_values):
                classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding] = {
                    'Classify': 0,
                    'Compress': 0,
                    'Detect': 0,
                    'Track': 0
                }
    
    # Query execution stage breakdown per classifier, tile size, and tilepadding
    excluded_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name in excluded_stages:
            continue
        
        for tilesize in sorted_tilesizes:
            for classifier in sorted_classifiers:
                for tilepadding in sorted(all_tilepadding_values):
                    # Per-video analysis
                    stage_total = 0
                    for config_key, times in stage_summaries.items():
                        # Only include configs for this video, this classifier, this tile size, and this tilepadding
                        if not config_key.startswith(video + '_'):
                            continue
                        # Extract classifier, tile size, and tilepadding from config key
                        parts = config_key.split('_')
                        if len(parts) >= 4:
                            remaining = '_'.join(parts[1:])
                            if (remaining.split('_')[-1] in ['padded', 'unpadded'] and 
                                remaining.split('_')[-2].isdigit()):
                                config_tilepadding = remaining.split('_')[-1]
                                config_tilesize = int(remaining.split('_')[-2])
                                config_classifier = '_'.join(remaining.split('_')[:-2])
                            else:
                                config_tilepadding = None
                                config_tilesize = None
                                config_classifier = remaining
                            
                            if (config_classifier == classifier and 
                                config_tilesize == tilesize and 
                                config_tilepadding == tilepadding and times):
                                stage_total += times
                    
                    if '020_exec_classify' in stage_name:
                        classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding]['Classify'] += stage_total
                    elif '030_exec_compress' in stage_name:
                        classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding]['Compress'] += stage_total
                    elif '040_exec_detect' in stage_name:
                        classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding]['Detect'] += stage_total
                    elif '060_exec_track' in stage_name:
                        classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding]['Track'] += stage_total
    
    # Calculate breakdown for naive query execution (tilesize=0) if available
    naive_query_stages = {
        'Classify': 0,
        'Compress': 0,
        'Detect': 0,
        'Track': 0
    }
    
    if has_naive_query_data:
        # Query execution stage breakdown for tilesize=0 (naive approach)
        excluded_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
        for stage_name, stage_summaries in query_timings['summaries'].items():
            if stage_name in excluded_stages:
                continue
            
            # Per-video analysis for tilesize=0
            stage_total = 0
            for config_key, times in stage_summaries.items():
                # Only include configs for this video and tilesize=0
                if not config_key.startswith(video + '_'):
                    continue
                # Extract tile size from config key (new format: video_classifier_tilesize_tilepadding)
                parts = config_key.split('_')
                if len(parts) >= 3:
                    remaining = '_'.join(parts[1:])
                    if remaining.split('_')[-1].isdigit():
                        config_tilesize = int(remaining.split('_')[-1])
                        if config_tilesize == 0 and times:
                            stage_total += times
                    elif len(remaining.split('_')) >= 2 and remaining.split('_')[-2].isdigit():
                        # Handle case where tilepadding is present: classifier_tilesize_tilepadding
                        config_tilesize = int(remaining.split('_')[-2])
                        if config_tilesize == 0 and times:
                            stage_total += times
            
            if '020_exec_classify' in stage_name:
                naive_query_stages['Classify'] += stage_total
            elif '030_exec_compress' in stage_name:
                naive_query_stages['Compress'] += stage_total
            elif '040_exec_detect' in stage_name:
                naive_query_stages['Detect'] += stage_total
            elif '060_exec_track' in stage_name:
                naive_query_stages['Track'] += stage_total
    
    # Calculate breakdown for preprocessing stages by tile size
    preprocessing_stages_by_tile = {}
    for tilesize in sorted_tilesizes:
        preprocessing_stages_by_tile[tilesize] = {
            'Detect': 0.,
            'Track': 0.
        }
    
    # Preprocessing stage breakdown by tile size
    preprocessing_stage_names = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_timings['summaries'].items():
        if stage_name not in preprocessing_stage_names:
            continue
        
        # Preprocessing stages use tile size 0, but we need to add them to all tile sizes
        # since preprocessing is shared across all tile sizes
        for tilesize in sorted_tilesizes:
            # Per-video analysis
            stage_total = 0
            for config_key, times in stage_summaries.items():
                if not config_key.startswith(video + '_'):
                    continue
                # Extract tile size from config key (new format: video_classifier_tilesize_tilepadding)
                parts = config_key.split('_')
                if len(parts) >= 3:
                    remaining = '_'.join(parts[1:])
                    if remaining.split('_')[-1].isdigit():
                        config_tilesize = int(remaining.split('_')[-1])
                        # Preprocessing stages use tile size 0, but we add them to all tile sizes
                        if config_tilesize == 0 and times:
                            stage_total += times
                    elif len(remaining.split('_')) >= 2 and remaining.split('_')[-2].isdigit():
                        # Handle case where tilepadding is present: classifier_tilesize_tilepadding
                        config_tilesize = int(remaining.split('_')[-2])
                        # Preprocessing stages use tile size 0, but we add them to all tile sizes
                        if config_tilesize == 0 and times:
                            stage_total += times
            
            if '001_preprocess_groundtruth_detection' in stage_name:
                preprocessing_stages_by_tile[tilesize]['Detect'] += float(stage_total)
            elif '002_preprocess_groundtruth_tracking' in stage_name:
                preprocessing_stages_by_tile[tilesize]['Track'] += float(stage_total)
    
    # Create separate plots for each tile size
    for tilesize in sorted_tilesizes:
        # Index construction stack for this tile size
        index_values = list(index_stages_by_tile[tilesize].values())
        index_labels = list(index_stages_by_tile[tilesize].keys())
        
        # Get preprocessing values for this tile size
        preprocessing_values = list(preprocessing_stages_by_tile[tilesize].values())
        preprocessing_labels = list(preprocessing_stages_by_tile[tilesize].keys())
        
        # Define stage labels for query execution
        stage_labels = ['Classify', 'Compress', 'Detect', 'Track']
        
        # Prepare data for altair charts
        chart_data = []
        
        # Add index construction data (per-dataset, shared across all videos)
        for value, label in zip(index_values, index_labels):
            chart_data.append({
                'Category': 'Index Constr.',
                'Operation': f'Index: {label}',
                'Runtime': value,
                'Chart': 'With Index Construction'
            })
        
        # Add query execution data for each classifier and tilepadding combination
        for classifier in sorted_classifiers:
            for tilepadding in sorted(all_tilepadding_values):
                query_values = [classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding][label] for label in stage_labels]
                for value, label in zip(query_values, stage_labels):
                    chart_data.append({
                        'Category': f'Query Exec.\n({classifier.title()}_{tilepadding})',
                        'Operation': f'Query: {label}',
                        'Runtime': value,
                        'Chart': 'With Index Construction'
                    })
    
        # Add preprocessing data
        for value, label in zip(preprocessing_values, preprocessing_labels):
            chart_data.append({
                'Category': 'Naive',
                'Operation': f'Query: {label}',
                'Runtime': value,
                'Chart': 'With Index Construction'
            })
        
        # Create second chart data (without index construction)
        chart_data2 = []
        for classifier in sorted_classifiers:
            for tilepadding in sorted(all_tilepadding_values):
                query_values = [classifier_query_stages_by_tile_tilepadding[tilesize][classifier][tilepadding][label] for label in stage_labels]
                for value, label in zip(query_values, stage_labels):
                    chart_data2.append({
                        'Category': f'Query Exec.\n({classifier.title()}_{tilepadding})',
                        'Operation': f'Query: {label}',
                        'Runtime': value,
                        'Chart': 'Without Index Construction'
                    })
        
        # Add preprocessing data
        for value, label in zip(preprocessing_values, preprocessing_labels):
            chart_data2.append({
                'Category': 'Naive',
                'Operation': f'Query: {label}',
                'Runtime': value,
                'Chart': 'Without Index Construction'
            })
        
        # Create charts
        df1 = pd.DataFrame(chart_data)
        df2 = pd.DataFrame(chart_data2)
        
        chart1 = alt.Chart(df1).mark_bar().encode(
            x=alt.X('Runtime:Q', title='Runtime (seconds)'),
            y=alt.Y('Category:N',
                    sort=alt.SortField(field='Runtime', order='descending'),
                    axis=alt.Axis(labelExpr="split(datum.label, '\\n')",
                                  labelBaseline='alphabetic')),
            color=alt.Color('Operation:N', legend=alt.Legend(orient='top')),
            tooltip=['Category', 'Operation', alt.Tooltip('Runtime:Q', format='.2f', title='Runtime (s)')]
        ).properties(
            title='With Index Construction',
            width=400,
            height=400
        )
        
        chart2 = alt.Chart(df2).mark_bar().encode(
            x=alt.X('Runtime:Q', title='Runtime (seconds)'),
            y=alt.Y('Category:N',
                    sort=alt.SortField(field='Runtime', order='descending'),
                    axis=alt.Axis(labelExpr="split(datum.label, '\\n')",
                                  labelBaseline='alphabetic')),
            color=alt.Color('Operation:N', legend=alt.Legend(orient='top')),
            tooltip=['Category', 'Operation', alt.Tooltip('Runtime:Q', format='.2f', title='Runtime (s)')]
        ).properties(
            title='Without Index Construction',
            width=400,
            height=400
        )
        
        # Combine charts horizontally
        combined_chart = alt.hconcat(chart1, chart2, spacing=20).properties(
            title='Index Construction (Per-Dataset) vs Query Execution (Per-Video) Runtime Breakdown '
                  f'- {video_name} (Tile Size: {tilesize})'
        )
        
        # Save plot with appropriate naming and directory structure
        # Create per-video subdirectory for per-video analysis
        video_output_dir = os.path.join(output_dir, 'per_video')
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Save plot with safe filename including tile size
        safe_video_name = video_name.replace('/', '_').split('.')[0]
        file_name = f'overall_{safe_video_name}_tile{tilesize}.png'
        combined_chart.save(os.path.join(video_output_dir, file_name), 
                            scale_factor=2)


def extract_video_names(query_timings: dict) -> list[str]:
    """Extract unique video names from the query timing data."""
    videos = set()
    for stage_timings in query_timings['timings'].values():
        for config_key in stage_timings.keys():
            # config_key format: dataset/video_classifier_tilesize_tilepadding or dataset/video_classifier_tilesize
            parts = config_key.split('_')
            if len(parts) >= 3:
                # Check if the last part is padded or unpadded to determine format
                if parts[-1] in ['padded', 'unpadded']:
                    # New format: dataset/video_classifier_tilesize_tilepadding
                    # Extract video name (everything before the last three underscores)
                    video_part = '_'.join(parts[:-3])  # Remove classifier, tilesize, and tilepadding
                else:
                    # Old format: dataset/video_classifier_tilesize
                    # Extract video name (everything before the last two underscores)
                    video_part = '_'.join(parts[:-2])  # Remove classifier and tilesize
                videos.add(video_part)
    return sorted(list(videos))


def visualize_breakdown_query_execution_all(query_timings: dict, output_dir: str, dataset: str = 'b3d'):
    """Create visualizations for query execution runtime breakdown."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all videos from the data
    videos = extract_video_names(query_timings)
    print(f"Found {len(videos)} videos: {videos}")
    
    # Create per-video visualizations
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        visualize_breakdown_query_execution(query_timings, output_dir, dataset, video, video_name)


def visualize_overal_runtime_all(index_timings: dict, query_timings: dict, output_dir: str):
    """Create comparative analysis between index construction and query execution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all videos from the data
    videos = extract_video_names(query_timings)
    print(f"Creating comparative analysis for {len(videos)} videos: {videos}")

    # Create per-video comparative analyses
    for video in videos:
        video_name = video.split('/')[-1] if '/' in video else video  # Extract just the filename
        visualize_overall_runtime(index_timings, query_timings, output_dir, video, video_name)


def main():
    """Main function to create runtime breakdown visualizations."""
    args = parse_args()
    
    for dataset in args.datasets:
        print(f"Loading processed measurements for dataset: {dataset}")
        measurements_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput', 'measurements')
        print(f"Measurements directory: {measurements_dir}")
        
        assert os.path.exists(measurements_dir), f"Error: Measurements directory {measurements_dir} does not exist."
        
        index_timings, query_timings, metadata = load_measurements(measurements_dir)
        
        throughput_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '082_throughput_visualize')
        visualize_breakdown_query_execution_all(query_timings, throughput_dir, dataset)
        
        visualize_overal_runtime_all(index_timings, query_timings, throughput_dir)
        
        print(f"Visualization complete for {dataset}! Results saved to: {throughput_dir}")
        print("- Per-video visualizations saved in per_video/ subdirectory")


if __name__ == '__main__':
    main()
