#!/usr/local/bin/python

from functools import partial
import os
import argparse
import shutil
import altair as alt
import pandas as pd
import multiprocessing as mp

from rich.progress import track

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize runtime breakdown of training configurations')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear the 082_throughput_visualize directory before creating visualizations')
    return parser.parse_args()


def load_measurements(measurements_dir: str):
    """Load the processed measurement data."""
    index_overall_file = os.path.join(measurements_dir, 'index_construction_overall.csv')
    # index_per_op_file = os.path.join(measurements_dir, 'index_construction_per_op.csv')
    query_overall_file = os.path.join(measurements_dir, 'query_execution_overall.csv')
    query_per_op_file = os.path.join(measurements_dir, 'query_execution_per_op.csv')
    
    index_overall = pd.read_csv(index_overall_file)
    # index_per_op = pd.read_csv(index_per_op_file)
    query_overall = pd.read_csv(query_overall_file)
    query_per_op = pd.read_csv(query_per_op_file)
    
    print(f"Loaded index construction measurements")
    print(f"Loaded query execution timings and summaries")

    return index_overall, query_overall, query_per_op


def visualize_breakdown_query_execution(query_per_op: pd.DataFrame, output_dir: str, video: str | None = None):
    """Create query execution visualization for a specific video or all videos.

    Args:
        query_per_op: Query per-operation timing DataFrame
        output_dir: Output directory for saving plots
        video: Specific video for per-video analysis, or None for all videos
    """
    # Filter data for specific video or use all videos
    if video is None:
        df = query_per_op.copy()
        output_dir = os.path.join(output_dir)
        output_name = 'breakdown.png'
    else:
        df = query_per_op[query_per_op['video'] == video].copy()
        output_dir = os.path.join(output_dir, 'per_video')
        output_name = f'breakdown_{video}.png'
    os.makedirs(output_dir, exist_ok=True)

    # Define stage mapping for display names
    stage_mapping = {
        '020_exec_classify': 'Classify',
        '030_exec_compress': 'Compress',
        '040_exec_detect': 'Detect',
        '050_exec_uncompress': 'Uncompress',
        '060_exec_track': 'Track'
    }

    # Filter relevant stages
    df = df[df['stage'].isin(stage_mapping.keys())]

    # Transform data: create Config labels and map stage names
    df['Config'] = df['classifier'].str[:5] + ' ' + df['tilesize'].astype(str) + ' ' + df['tilepadding'].str[:4]
    df['Stage'] = df['stage'].map(stage_mapping)

    # Aggregate data by grouping and summing times
    df = df.groupby(['Stage', 'Config', 'op']).agg({'time': 'sum'}).reset_index()

    # Calculate total runtime per config within each stage for sorting
    config_totals = df.groupby(['Stage', 'Config'])['time'].sum().reset_index()
    config_totals = config_totals.rename(columns={'time': 'TotalRuntime'})
    df = df.merge(config_totals, on=['Stage', 'Config'])

    # Create individual bar charts for each stage
    charts = []
    for stage_name in stage_mapping.values():
        stage_df = df[df['Stage'] == stage_name]
        if len(stage_df) > 0:
            chart = alt.Chart(stage_df).mark_bar().encode(
                x=alt.X('time:Q', title='Runtime (seconds)'),
                y=alt.Y('Config:N', sort=alt.SortField(field='TotalRuntime', order='descending')),
                color=alt.Color('op:N', legend=alt.Legend(
                    title='Operation',
                    orient='bottom',
                    columns=3,
                )),
                tooltip=['Config', 'op', alt.Tooltip('time:Q', format='.2f', title='Runtime (s)')]
            ).properties(
                title=f'{stage_name} Runtime by Operation',
                width=300,
                height=220
            )
            charts.append(chart)

    # Combine charts in a 2x2 grid
    combined_chart = alt.vconcat(
        alt.hconcat(charts[0], charts[1], charts[2]).resolve_scale(
            color='independent'
        ),
        alt.hconcat(charts[3], charts[4]).resolve_scale(
            color='independent'
        )
    )

    # Save combined chart
    combined_chart.save(os.path.join(output_dir, output_name), scale_factor=2)


def visualize_overall_runtime(index_overall: pd.DataFrame, query_overall: pd.DataFrame,
                              output_dir: str, video: str | None):
    """Create comparative analysis between index construction and query execution.

    Note: Index construction is per-dataset (shared across all videos), while query execution is per-video.

    Args:
        index_overall: Index construction overall timing DataFrame (per-dataset)
        query_overall: Query execution overall timing DataFrame (per-video)
        output_dir: Output directory for saving plots
        video: Specific video for per-video analysis, or None for all videos
    """

    # Filter query data for specific video or use all videos
    if video is None:
        video_query_data = query_overall.copy()
        video_output_dir = os.path.join(output_dir)
        file_name = f'overall.png'
    else:
        video_query_data = query_overall[query_overall['video'] == video].copy()
        video_output_dir = os.path.join(output_dir, 'per_video')
        file_name = f'overall_{video}.png'

    os.makedirs(video_output_dir, exist_ok=True)

    naive_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']

    # Define stage mapping for display names
    stage_mapping = {
        '001_preprocess_groundtruth_detection': 'Detection',
        '002_preprocess_groundtruth_tracking': 'Tracking',
        '011_tune_detect': 'Detection',
        '012_tune_create_training_data': 'Create Training Data',
        '013_tune_train_classifier': 'Classifier Training',
        '020_exec_classify': 'Classification',
        '030_exec_compress': 'Compression',
        '040_exec_detect': 'Detection',
        '050_exec_uncompress': 'Uncompression',
        '060_exec_track': 'Tracking'
    }

    # Transform index construction data: aggregate by stage and add labels
    index_df = index_overall.groupby('stage').agg({'time': 'sum'}).reset_index()
    index_df['Category'] = 'Index Constr.'
    index_df['Operation'] = index_df['stage'].map(stage_mapping)

    # Transform execution data: filter, create category labels, map operations, and aggregate
    exec_df = video_query_data[~video_query_data['stage'].isin(naive_stages)].copy()
    exec_df['Category'] = exec_df['classifier'].str[:4] + ' ' + exec_df['tilesize'].astype(str) + ' ' + exec_df['tilepadding'].str[:4]
    exec_df['Operation'] = exec_df['stage'].map(stage_mapping)
    exec_df = exec_df.groupby(['Category', 'Operation']).agg({'time': 'sum'}).reset_index()

    # Transform naive baseline data: filter naive stages and aggregate
    naive_df = video_query_data[video_query_data['stage'].isin(naive_stages)]
    naive_df = naive_df.groupby('stage').agg({'time': 'sum'}).reset_index()
    naive_df['Category'] = 'Naive'
    naive_df['Operation'] = naive_df['stage'].map(stage_mapping)

    # Combine data for two comparison views
    df1 = pd.concat((exec_df, index_df, naive_df), ignore_index=True)  # With index construction
    df2 = pd.concat((exec_df, naive_df), ignore_index=True)  # Without index construction

    # Helper function to create stacked bar chart
    def make_chart(df: pd.DataFrame, title: str, height: int) -> alt.Chart:
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('time:Q', title='Runtime (seconds)'),
            y=alt.Y('Category:N', sort=alt.SortField(field='time', order='descending')),
            color=alt.Color('Operation:N', legend=alt.Legend(orient='top')),
            tooltip=['Category', 'Operation', alt.Tooltip('time:Q', format='.2f', title='Runtime (s)')]
        ).properties(
            title=title,
            width=800,
            height=height
        )

    # Create two charts for comparison
    chart1 = make_chart(df1, 'With Index Construction', 200)
    chart2 = make_chart(df2, 'Without Index Construction', 400)

    # Combine charts vertically and save
    combined_chart = alt.vconcat(chart1, chart2, spacing=20).properties(
        title=f'Index Construction (Per-Dataset) vs Query Execution (Per-Video) Runtime Breakdown - {video}'
    )
    combined_chart.save(os.path.join(video_output_dir, file_name), scale_factor=2)


def extract_video_names(query_overall: pd.DataFrame) -> list[str]:
    """Extract unique video names from the query overall DataFrame."""
    videos = query_overall['video'].unique()
    return sorted(list(videos))


def visualize_breakdown_query_execution_all(query_overall: pd.DataFrame, query_per_op: pd.DataFrame, output_dir: str):
    """Create visualizations for query execution runtime breakdown."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract all videos from the data
    videos = query_overall['video'].unique()
    print(f"Found {len(videos)} videos: {videos}")

    # Create per-video visualizations
    visualize_breakdown_query_execution(query_per_op, output_dir)
    for video in videos:
        visualize_breakdown_query_execution(query_per_op, output_dir, video)


def visualize_overal_runtime_all(index_overall: pd.DataFrame, query_overall: pd.DataFrame, output_dir: str):
    """Create comparative analysis between index construction and query execution."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract all videos from the data
    videos = extract_video_names(query_overall)
    print(f"Creating comparative analysis for {len(videos)} videos: {videos}")

    # Create per-video comparative analyses
    visualize_overall_runtime(index_overall, query_overall, output_dir, None)
    for video in videos:
        visualize_overall_runtime(index_overall, query_overall, output_dir, video)


def main():
    """Main function to create runtime breakdown visualizations."""
    args = parse_args()

    # Clear the 082_throughput_visualize directory if --clear flag is set
    if args.clear:
        for dataset in args.datasets:
            output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '082_throughput_visualize')
            if os.path.exists(output_dir):
                print(f"Clearing directory: {output_dir}")
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for dataset in args.datasets:
        print(f"Loading processed measurements for dataset: {dataset}")
        measurements_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput', 'measurements')
        print(f"Measurements directory: {measurements_dir}")

        assert os.path.exists(measurements_dir), f"Error: Measurements directory {measurements_dir} does not exist."

        index_overall, query_overall, query_per_op = load_measurements(measurements_dir)

        throughput_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '082_throughput_visualize')

        os.makedirs(throughput_dir, exist_ok=True)
        videos = extract_video_names(query_overall)
        for video in videos + [None]:
            tasks.append(partial(visualize_breakdown_query_execution, query_per_op, throughput_dir, video))
            tasks.append(partial(visualize_overall_runtime, index_overall, query_overall, throughput_dir, video))
    
    processes = []
    for task in tasks:
        p = mp.Process(target=task)
        p.start()
        processes.append(p)
    
    for p in track(processes, total=len(processes)):
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()
