#!/usr/local/bin/python

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import track

from polyis.utilities import CACHE_DIR, DATA_DIR, CLASSIFIERS_TO_TEST


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and compare filter strategy tradeoffs.')
    parser.add_argument('--dataset', type=str, default='b3d', help='Dataset name to process')
    parser.add_argument('--classifiers', nargs='+', default=CLASSIFIERS_TO_TEST, help='Classifiers to test')
    parser.add_argument('--filters', nargs='+', default=['none', 'neighbor'], help='Filter strategies to compare (must match existing directory names)')
    return parser.parse_args()


def gather_data(dataset: str, classifiers: list[str], filters: list[str]) -> pd.DataFrame:
    """Gather all performance and accuracy data."""
    all_data = []
    video_files = [f for f in os.listdir(os.path.join(DATA_DIR, dataset)) if f.endswith('.mp4')]

    for video in track(sorted(video_files), description="Gathering data..."):
        for filter_type in filters:
            for classifier in classifiers:
                for tile_size in [60]:  # Assuming 60 for now
                    # Handle both implicit 'none' (no suffix) and explicit '..._none' naming conventions
                    if filter_type == 'none':
                        path_with_suffix = Path(CACHE_DIR) / dataset / video / 'relevancy' / f'{classifier}_{tile_size}_none'
                        path_without_suffix = Path(CACHE_DIR) / dataset / video / 'relevancy' / f'{classifier}_{tile_size}'
                        base_path = path_with_suffix if path_with_suffix.exists() else path_without_suffix
                    else:
                        base_path = Path(CACHE_DIR) / dataset / video / 'relevancy' / f'{classifier}_{tile_size}_{filter_type}'
                    
                    # Load accuracy metrics
                    metrics_file = base_path / 'statistics' / 'summary_metrics.json'
                    if not metrics_file.exists():
                        print(f"Warning: Metrics file not found, skipping: {metrics_file}")
                        continue
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    # Load runtime and pruned tiles data
                    score_file = base_path / 'score' / 'score.jsonl'
                    if not score_file.exists():
                        print(f"Warning: Score file not found, skipping: {score_file}")
                        continue

                    # Load wall-clock time from metadata file (if available)
                    metadata_file = base_path / 'score' / 'metadata.json'
                    wall_clock_time_seconds = None
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            wall_clock_time_seconds = metadata.get('total_wall_clock_time_seconds')

                    inference_runtime = 0
                    overall_runtime = 0
                    with open(score_file, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            for op in data.get('runtime', []):
                                if op['op'] == 'inference':
                                    inference_runtime += op['time']
                                if op['op'] == 'overall':
                                    overall_runtime += op['time']

                    # Use wall-clock time if available, otherwise fall back to summed overall runtime
                    if wall_clock_time_seconds is not None:
                        overall_runtime = wall_clock_time_seconds * 1000  # Convert to ms for consistency

                    all_data.append({
                        'video': video,
                        'classifier': classifier,
                        'filter': filter_type,
                        'tile_size': tile_size,
                        'inference_runtime_ms': inference_runtime,
                        'overall_runtime_ms': overall_runtime,
                        'avg_pruned_prop': metrics.get('avg_pruned_tiles_prop', 0.0),
                        'precision': metrics.get('overall_precision', 0.0),
                        'recall': metrics.get('overall_recall', 0.0),
                        'f1_score': metrics.get('overall_f1', 0.0),
                    })

    return pd.DataFrame(all_data)


def create_comparison_charts(df: pd.DataFrame, output_dir: Path):
    """Create horizontal bar charts for comparison."""
    if df.empty:
        print("No data to plot.")
        return

    videos = df['video'].unique()
    classifiers = df['classifier'].unique()

    for classifier in classifiers:
        clf_df = df[df['classifier'] == classifier].copy()
        if clf_df.empty:
            continue

        # Calculate speedup relative to the 'none' filter
        none_overall_runtimes = clf_df[clf_df['filter'] == 'none'].set_index('video')['overall_runtime_ms']
        if none_overall_runtimes.empty:
            print(f"Warning: No 'none' filter data for classifier {classifier}, cannot calculate speedup.")
            clf_df['speedup'] = 1.0
        else:
            # Use map to align runtimes by video
            clf_df['baseline_runtime'] = clf_df['video'].map(none_overall_runtimes)
            # Calculate speedup based on overall runtime, handling potential division by zero
            clf_df['speedup'] = clf_df.apply(
                lambda row: row['baseline_runtime'] / row['overall_runtime_ms'] if row['overall_runtime_ms'] > 0 else 0,
                axis=1
            )

        # Calculate inference speedup relative to the 'none' filter
        none_inference_runtimes = clf_df[clf_df['filter'] == 'none'].set_index('video')['inference_runtime_ms']
        if none_inference_runtimes.empty:
            print(f"Warning: No 'none' filter data for classifier {classifier}, cannot calculate inference speedup.")
            clf_df['inference_speedup'] = 1.0
        else:
            clf_df['baseline_inference_runtime'] = clf_df['video'].map(none_inference_runtimes)
            clf_df['inference_speedup'] = clf_df.apply(
                lambda row: row['baseline_inference_runtime'] / row['inference_runtime_ms'] if row['inference_runtime_ms'] > 0 else 0,
                axis=1
            )

        # Plotting
        n_videos = len(videos)
        n_metrics = 8  # Added Overhead Runtime
        fig, axes = plt.subplots(n_videos, n_metrics, figsize=(28, 4 * n_videos), sharey='row')
        if n_videos == 1:
            axes = np.array([axes])

        fig.suptitle(f'Filter Performance Comparison for Classifier: {classifier}', fontsize=16, y=1.0)

        for i, video in enumerate(videos):
            video_df = clf_df[clf_df['video'] == video]
            if video_df.empty:
                for j in range(n_metrics):
                    axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center')
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                axes[i, 0].set_ylabel(video.replace('.mp4', ''), rotation=0, size='large', ha='right', va='center')
                continue

            filters_present = video_df['filter'].unique()
            y_pos = np.arange(len(filters_present))

            # 1. Overall Runtime (Wall-Clock)
            ax = axes[i, 0]
            runtimes = video_df['overall_runtime_ms'] / 1000  # to seconds
            bars = ax.barh(y_pos, runtimes, align='center')
            ax.bar_label(bars, fmt='%.2fs', padding=3)
            ax.set_yticks(y_pos, labels=video_df['filter'])
            ax.invert_yaxis()
            ax.set_xlabel('Wall-Clock Runtime (s)')
            if i == 0: ax.set_title('Wall-Clock Runtime')
            ax.set_ylabel(video.replace('.mp4', ''), rotation=0, size='large', ha='right', va='center')
            ax.margins(x=0.1)

            # 2. Inference Runtime
            ax = axes[i, 1]
            runtimes = video_df['inference_runtime_ms'] / 1000  # to seconds
            bars = ax.barh(y_pos, runtimes, align='center')
            ax.bar_label(bars, fmt='%.2fs', padding=3)
            ax.set_xlabel('Inference Runtime (s)')
            if i == 0: ax.set_title('Inference Runtime')
            ax.margins(x=0.1)

            # 3. Overhead Runtime
            ax = axes[i, 2]
            overhead_runtime = (video_df['overall_runtime_ms'] - video_df['inference_runtime_ms']) / 1000  # to seconds
            bars = ax.barh(y_pos, overhead_runtime, align='center')
            ax.bar_label(bars, fmt='%.2fs', padding=3)
            ax.set_xlabel('Overhead Runtime (s)')
            if i == 0: ax.set_title('Overhead Runtime')
            ax.margins(x=0.1)

            # 4. Speedup (relative to 'none')
            ax = axes[i, 3]
            speedups = video_df['speedup']
            colors = ['red' if s < 1 else 'green' for s in speedups]
            bars = ax.barh(y_pos, speedups, align='center', color=colors)
            ax.bar_label(bars, fmt='%.2fx', padding=3)
            ax.axvline(1, color='grey', linestyle='--')
            ax.set_xlabel('Speedup (x)')
            if i == 0: ax.set_title('Wall-Clock Speedup vs. No Filter')
            ax.margins(x=0.1)

            # 5. Inference Speedup (relative to 'none')
            ax = axes[i, 4]
            speedups = video_df['inference_speedup']
            colors = ['red' if s < 1 else 'green' for s in speedups]
            bars = ax.barh(y_pos, speedups, align='center', color=colors)
            ax.bar_label(bars, fmt='%.2fx', padding=3)
            ax.axvline(1, color='grey', linestyle='--')
            ax.set_xlabel('Speedup (x)')
            if i == 0: ax.set_title('Inference Speedup vs. No Filter')
            ax.margins(x=0.1)

            # 6. Pruned Tiles
            ax = axes[i, 5]
            pruned = video_df['avg_pruned_prop'] * 100
            bars = ax.barh(y_pos, pruned, align='center')
            ax.bar_label(bars, fmt='%.1f%%', padding=3)
            ax.set_xlabel('Avg. Pruned Tiles (%)')
            ax.set_xlim(0, 100)
            if i == 0: ax.set_title('Tiles Pruned')
            ax.margins(x=0.1)

            # 7. F1-Score
            ax = axes[i, 6]
            f1s = video_df['f1_score']
            ax.barh(y_pos, f1s, align='center')
            ax.set_xlabel('F1-Score')
            ax.set_xlim(0, 1)
            if i == 0: ax.set_title('F1-Score')
            ax.margins(x=0.1)

            # 8. Precision & Recall
            ax = axes[i, 7]
            precision = video_df['precision']
            recall = video_df['recall']
            p_bars = ax.barh(y_pos - 0.2, precision, 0.4, align='center', label='Precision')
            r_bars = ax.barh(y_pos + 0.2, recall, 0.4, align='center', label='Recall')
            ax.bar_label(p_bars, fmt='%.2f', padding=3)
            ax.bar_label(r_bars, fmt='%.2f', padding=3)
            ax.set_xlabel('Score')
            ax.set_xlim(0, 1)
            if i == 0: ax.legend()
            if i == 0: ax.set_title('Precision & Recall')
            ax.margins(x=0.1)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = output_dir / f'filter_comparison_{classifier}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved comparison chart to: {plot_path}")


def main():
    """Main function to run experiments and generate analysis."""
    args = parse_args()
    
    print("--- Analyzing results from existing data ---")
    df = gather_data(args.dataset, args.classifiers, args.filters)

    if df.empty:
        print("Could not gather any data. Exiting.")
        return

    # Create output directory for summary plots
    output_dir = Path(CACHE_DIR) / 'summary' / args.dataset / 'filters'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    df.to_csv(output_dir / 'filter_comparison_data.csv', index=False)
    print(f"Saved raw data to {output_dir / 'filter_comparison_data.csv'}")

    # Create plots
    create_comparison_charts(df, output_dir)

    print("\n🎉 Analysis complete!")


if __name__ == '__main__':
    main()