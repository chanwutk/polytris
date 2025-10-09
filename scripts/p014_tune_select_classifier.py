#!/usr/local/bin/python

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import pandas as pd
import altair as alt
import numpy as np

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST


def load_training_results(dataset_dir: str, classifiers_filter: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Load training results from all classifier experiments.

    Args:
        dataset_dir: Path to the dataset directory
        classifiers_filter: Optional list of classifier names to filter by

    Returns:
        List of dictionaries containing training results for each classifier/tile_size/video combination
    """
    results = []

    for video in sorted(os.listdir(dataset_dir)):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path) and not video.endswith('.mp4'):
            continue

        training_path = os.path.join(video_path, 'training', 'results')
        if not os.path.exists(training_path):
            continue

        for result_dir in os.listdir(training_path):
            if not os.path.isdir(os.path.join(training_path, result_dir)):
                continue

            # Parse classifier and tile_size from directory name (format: classifier_tile_size)
            try:
                classifier, tile_size_str = result_dir.rsplit('_', 1)
                tile_size = int(tile_size_str)
            except ValueError:
                continue

            # Filter by classifier if specified
            if classifiers_filter is not None and classifier not in classifiers_filter:
                continue

            result_path = os.path.join(training_path, result_dir)

            # Load the JSON files
            test_losses_path = os.path.join(result_path, 'test_losses.json')
            train_losses_path = os.path.join(result_path, 'train_losses.json')
            throughput_path = os.path.join(result_path, 'throughput_per_epoch.jsonl')

            if not all(os.path.exists(p) for p in [test_losses_path, train_losses_path, throughput_path]):
                continue

            try:
                with open(test_losses_path, 'r') as f:
                    test_losses = json.load(f)
                with open(train_losses_path, 'r') as f:
                    train_losses = json.load(f)

                # Load throughput data (JSONL format)
                throughput_data = []
                with open(throughput_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            throughput_data.append(json.loads(line.strip()))

                # Extract key metrics
                num_epochs = len(test_losses)
                best_val_loss = min(epoch['loss'] for epoch in test_losses)
                best_val_accuracy = max(epoch['accuracy'] for epoch in test_losses)
                total_train_time = sum(epoch['time'] for epoch in train_losses)
                total_val_time = sum(epoch['time'] for epoch in test_losses)

                # Extract train step and inference times from throughput data
                train_step_times = []
                inference_times = []

                for epoch_throughput in throughput_data:
                    for op_data in epoch_throughput:
                        if op_data['op'] == 'train_step':
                            train_step_times.append(op_data['time'])
                        elif op_data['op'] == 'test_inference':
                            inference_times.append(op_data['time'])

                avg_train_step_time = np.mean(train_step_times) if train_step_times else 0
                avg_inference_time = np.mean(inference_times) if inference_times else 0

                results.append({
                    'video': video,
                    'classifier': classifier,
                    'tile_size': tile_size,
                    'num_epochs': num_epochs,
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_accuracy,
                    'total_train_time': total_train_time,
                    'total_val_time': total_val_time,
                    'avg_train_step_time': avg_train_step_time,
                    'avg_inference_time': avg_inference_time,
                })

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error loading results from {result_path}: {e}")
                continue

    return results


def create_scatterplots(df: pd.DataFrame, output_dir: str, title_suffix: str = ""):
    """
    Create the 5 required scatterplots using altair.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the plots to create
    plots = [
        {
            'x': 'num_epochs',
            'y': 'best_val_accuracy',
            'title': f'Number of Epochs vs Best Validation Accuracy{title_suffix}',
            'xlabel': 'Number of Epochs',
            'ylabel': 'Best Validation Accuracy',
            'filename': 'epochs_vs_val_accuracy.png'
        },
        {
            'x': 'total_train_time',
            'y': 'best_val_accuracy',
            'title': f'Total Training Time vs Validation Accuracy{title_suffix}',
            'xlabel': 'Total Training Time (ms)',
            'ylabel': 'Best Validation Accuracy',
            'filename': 'train_time_vs_val_accuracy.png'
        },
        {
            'x': 'total_val_time',
            'y': 'best_val_accuracy',
            'title': f'Total Validation Time vs Validation Accuracy{title_suffix}',
            'xlabel': 'Total Validation Time (ms)',
            'ylabel': 'Best Validation Accuracy',
            'filename': 'val_time_vs_val_accuracy.png'
        },
        {
            'x': 'avg_train_step_time',
            'y': 'best_val_accuracy',
            'title': f'Average Train Step Time vs Validation Accuracy{title_suffix}',
            'xlabel': 'Average Train Step Time (ms)',
            'ylabel': 'Best Validation Accuracy',
            'filename': 'train_step_time_vs_val_accuracy.png'
        },
        {
            'x': 'avg_inference_time',
            'y': 'best_val_accuracy',
            'title': f'Average Inference Time vs Validation Accuracy{title_suffix}',
            'xlabel': 'Average Inference Time (ms)',
            'ylabel': 'Best Validation Accuracy',
            'filename': 'inference_time_vs_val_accuracy.png'
        }
    ]

    for plot_config in plots:
        # Create scatter plot with different colors for classifiers and sizes for tile sizes
        scatter = alt.Chart(df).mark_circle().encode(
            x=alt.X(f'{plot_config["x"]}:Q', title=plot_config['xlabel']),
            y=alt.Y(f'{plot_config["y"]}:Q', title=plot_config['ylabel'], 
                   scale=alt.Scale(zero=False)),
            color='classifier:N',
            size=alt.Size('tile_size:O', scale=alt.Scale(range=[50, 200])),
            tooltip=['classifier', 'tile_size', plot_config['x'], plot_config['y']]
        ).properties(
            title=plot_config['title'],
            width=600,
            height=400
        )

        # Add lines connecting the same classifier across tile sizes
        line_data = []
        for classifier in df['classifier'].unique():
            classifier_data = df[df['classifier'] == classifier]
            assert isinstance(classifier_data, pd.DataFrame)
            classifier_data = classifier_data.sort_values('tile_size')
            if len(classifier_data) > 1:
                for i in range(len(classifier_data) - 1):
                    line_data.append({
                        'x1': classifier_data.iloc[i][plot_config['x']],
                        'y1': classifier_data.iloc[i][plot_config['y']],
                        'x2': classifier_data.iloc[i+1][plot_config['x']],
                        'y2': classifier_data.iloc[i+1][plot_config['y']],
                        'classifier': classifier
                    })
        
        if line_data:
            line_df = pd.DataFrame(line_data)
            lines = alt.Chart(line_df).mark_rule(strokeDash=[5, 5], opacity=0.5).encode(
                x='x1:Q',
                y='y1:Q',
                x2='x2:Q',
                y2='y2:Q',
                color='classifier:N'
            )
            chart = scatter + lines
        else:
            chart = scatter

        # # Add correlation coefficient
        # correlation = df[plot_config['x']].corr(df[plot_config['y']])
        # correlation_text = alt.Chart(pd.DataFrame([{'text': f'Correlation: {correlation:.3f}'}])).mark_text(
        #     align='left', baseline='top', fontSize=12
        # ).encode(
        #     text='text:N'
        # ).properties(
        #     width=600,
        #     height=400
        # )
        
        # final_chart = alt.layer(chart, correlation_text)
        final_chart = chart

        # Save the plot
        output_path = os.path.join(output_dir, plot_config['filename'])
        final_chart.save(output_path, scale_factor=2)

        print(f"Saved plot: {output_path}")


def create_average_plots(results: List[Dict[str, Any]], output_dir: str):
    """
    Create plots with average values across all videos for each classifier/tile_size combination.
    """
    # Group by classifier and tile_size, then calculate averages
    grouped_data = defaultdict(list)

    for result in results:
        key = (result['classifier'], result['tile_size'])
        grouped_data[key].append(result)

    # Calculate averages
    averaged_results = []
    for (classifier, tile_size), group_results in grouped_data.items():
        if not group_results:
            continue

        avg_result = {
            'classifier': classifier,
            'tile_size': tile_size,
            'num_epochs': np.mean([r['num_epochs'] for r in group_results]),
            'best_val_loss': np.mean([r['best_val_loss'] for r in group_results]),
            'best_val_accuracy': np.mean([r['best_val_accuracy'] for r in group_results]),
            'total_train_time': np.mean([r['total_train_time'] for r in group_results]),
            'total_val_time': np.mean([r['total_val_time'] for r in group_results]),
            'avg_train_step_time': np.mean([r['avg_train_step_time'] for r in group_results]),
            'avg_inference_time': np.mean([r['avg_inference_time'] for r in group_results]),
        }
        averaged_results.append(avg_result)

    # Convert to DataFrame and create plots
    df_avg = pd.DataFrame(averaged_results)
    create_scatterplots(df_avg, output_dir, " (Averaged Across Videos)")


def create_per_video_plots(results: List[Dict[str, Any]], output_dir: str):
    """
    Create individual plots for each video.
    """
    # Group by video
    videos = set(result['video'] for result in results)

    for video in videos:
        video_results = [r for r in results if r['video'] == video]
        if not video_results:
            continue

        # Create subdirectory for this video
        video_output_dir = os.path.join(output_dir, 'each', video)
        os.makedirs(video_output_dir, exist_ok=True)

        # Convert to DataFrame and create plots
        df_video = pd.DataFrame(video_results)
        create_scatterplots(df_video, video_output_dir, f" - {video}")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze classifier training results and create visualizations')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name')
    parser.add_argument('--classifiers', required=False, nargs='+',
                        default=CLASSIFIERS_TO_TEST,
                        choices=['SimpleCNN', 'YoloN', 'YoloS', 'YoloM', 'YoloL',
                                 'YoloX', 'ShuffleNet05', 'ShuffleNet20', 'MobileNetL',
                                 'MobileNetS', 'WideResNet50', 'WideResNet101',
                                 'ResNet152', 'ResNet101', 'ResNet18', 'EfficientNetS',
                                 'EfficientNetL'],
                        help='Specific classifiers to analyze (if not specified, all classifiers will be analyzed)')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = os.path.join(CACHE_DIR, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    print(f"Loading training results from {dataset_dir}...")
    if args.classifiers:
        print(f"Filtering for classifiers: {', '.join(args.classifiers)}")
    results = load_training_results(dataset_dir, args.classifiers)

    if not results:
        print("No training results found!")
        return

    print(f"Found {len(results)} training results")

    # Create output directory
    output_base_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'classifiers')
    os.makedirs(output_base_dir, exist_ok=True)

    # Create average plots
    print("Creating average plots...")
    create_average_plots(results, output_base_dir)

    # Create per-video plots
    print("Creating per-video plots...")
    create_per_video_plots(results, output_base_dir)

    print(f"All plots saved to {output_base_dir}")


if __name__ == '__main__':
    main()
