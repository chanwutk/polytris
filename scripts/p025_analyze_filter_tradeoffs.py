#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
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
    parser.add_argument('--filters', nargs='+', default=['none', 'neighbor'], help='Filter strategies to compare')
    parser.add_argument('--skip-run', action='store_true', help='Skip running experiments and only generate plots from existing data')
    return parser.parse_args()


def run_script(script_name: str, args: list[str]):
    """Run a script as a subprocess."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=Path(__file__).parents[1])
        print(f"✅ {script_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed:")
        print(f"   Error code: {e.returncode}")
        print(f"   stdout (last 10 lines):\n{''.join(e.stdout.strip().splitlines(True)[-10:])}")
        print(f"   stderr (last 10 lines):\n{''.join(e.stderr.strip().splitlines(True)[-10:])}")
        return False


def gather_data(dataset: str, classifiers: list[str], filters: list[str]) -> pd.DataFrame:
    """Gather all performance and accuracy data."""
    all_data = []
    video_files = [f for f in os.listdir(os.path.join(DATA_DIR, dataset)) if f.endswith('.mp4')]

    for video in track(sorted(video_files), description="Gathering data..."):
        for filter_type in filters:
            for classifier in classifiers:
                for tile_size in [60]:  # Assuming 60 for now
                    base_path = Path(CACHE_DIR) / dataset / video / 'relevancy' / f'{classifier}_{tile_size}_{filter_type}'
                    
                    # Load runtime and pruned tiles data
                    score_file = base_path / 'score' / 'score.jsonl'
                    if not score_file.exists():
                        print(f"Warning: Score file not found, skipping: {score_file}")
                        continue
                    
                    total_runtime = 0
                    pruned_props = []
                    with open(score_file, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            total_runtime += sum(op['time'] for op in data.get('runtime', []))
                            pruned_props.append(data.get('pruned_tiles_prop', 0.0))
                    
                    # Load accuracy metrics
                    metrics_file = base_path / 'statistics' / 'summary_metrics.json'
                    if not metrics_file.exists():
                        print(f"Warning: Metrics file not found, skipping: {metrics_file}")
                        continue

                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    all_data.append({
                        'video': video,
                        'classifier': classifier,
                        'filter': filter_type,
                        'tile_size': tile_size,
                        'total_runtime_ms': total_runtime,
                        'avg_pruned_prop': np.mean(pruned_props) if pruned_props else 0.0,
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
        none_runtimes = clf_df[clf_df['filter'] == 'none'].set_index('video')['total_runtime_ms']
        if none_runtimes.empty:
            print(f"Warning: No 'none' filter data for classifier {classifier}, cannot calculate speedup.")
            clf_df['speedup'] = 1.0
        else:
            # Use map to align runtimes by video
            clf_df['baseline_runtime'] = clf_df['video'].map(none_runtimes)
            # Calculate speedup, handling potential division by zero
            clf_df['speedup'] = clf_df.apply(
                lambda row: row['baseline_runtime'] / row['total_runtime_ms'] if row['total_runtime_ms'] > 0 else 0,
                axis=1
            )

        # Plotting
        n_videos = len(videos)
        n_metrics = 5  # Runtime, Pruned, Precision, Recall, F1
        fig, axes = plt.subplots(n_videos, n_metrics, figsize=(20, 4 * n_videos), sharey='row')
        if n_videos == 1:
            axes = np.array([axes])

        fig.suptitle(f'Filter Performance Comparison for Classifier: {classifier}', fontsize=16, y=1.02)

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

            # 1. Total Runtime
            ax = axes[i, 0]
            runtimes = video_df['total_runtime_ms'] / 1000  # to seconds
            ax.barh(y_pos, runtimes, align='center')
            ax.set_yticks(y_pos, labels=video_df['filter'])
            ax.invert_yaxis()
            ax.set_xlabel('Total Runtime (s)')
            if i == 0: ax.set_title('Runtime')
            ax.set_ylabel(video.replace('.mp4', ''), rotation=0, size='large', ha='right', va='center')

            # 2. Speedup (relative to 'none')
            ax = axes[i, 1]
            speedups = video_df['speedup']
            colors = ['red' if s < 1 else 'green' for s in speedups]
            ax.barh(y_pos, speedups, align='center', color=colors)
            ax.axvline(1, color='grey', linestyle='--')
            ax.set_xlabel('Speedup (x)')
            if i == 0: ax.set_title('Speedup vs. No Filter')

            # 3. Pruned Tiles
            ax = axes[i, 2]
            pruned = video_df['avg_pruned_prop'] * 100
            ax.barh(y_pos, pruned, align='center')
            ax.set_xlabel('Avg. Pruned Tiles (%)')
            ax.set_xlim(0, 100)
            if i == 0: ax.set_title('Tiles Pruned')

            # 4. F1-Score
            ax = axes[i, 3]
            f1s = video_df['f1_score']
            ax.barh(y_pos, f1s, align='center')
            ax.set_xlabel('F1-Score')
            ax.set_xlim(0, 1)
            if i == 0: ax.set_title('F1-Score')

            # 5. Precision & Recall
            ax = axes[i, 4]
            precision = video_df['precision']
            recall = video_df['recall']
            ax.barh(y_pos - 0.2, precision, 0.4, align='center', label='Precision')
            ax.barh(y_pos + 0.2, recall, 0.4, align='center', label='Recall')
            ax.set_xlabel('Score')
            ax.set_xlim(0, 1)
            ax.legend()
            if i == 0: ax.set_title('Precision & Recall')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = output_dir / f'filter_comparison_{classifier}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved comparison chart to: {plot_path}")


def main():
    """Main function to run experiments and generate analysis."""
    args = parse_args()

    if not args.skip_run:
        # Run experiments for each filter
        for filter_type in track(args.filters, description="Running experiments..."):
            # Clear previous results for this filter run to ensure a clean state
            for video_file in [f for f in os.listdir(os.path.join(DATA_DIR, args.dataset)) if f.endswith('.mp4')]:
                for classifier in args.classifiers:
                    filter_dir = Path(CACHE_DIR) / args.dataset / video_file / 'relevancy' / f'{classifier}_60_{filter_type}'
                    if filter_dir.exists():
                        print(f"Clearing previous results in {filter_dir}")
                        shutil.rmtree(filter_dir)

            print(f"\n--- Running for filter: {filter_type} ---")
            # Run classification
            run_script('p020_exec_classify.py', [
                '--dataset', args.dataset,
                '--classifiers', *args.classifiers,
                '--filter', filter_type,
            ])
            # Run accuracy analysis
            run_script('p022_exec_classify_visualize.py', [
                '--dataset', args.dataset,
                '--filter', filter_type,
            ])

    # --- Analysis ---
    print("\n--- Analyzing results ---")
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