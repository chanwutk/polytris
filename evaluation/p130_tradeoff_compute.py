#!/usr/local/bin/python

from functools import partial
import os
import shutil
from multiprocessing import Pool

from rich.progress import track
import pandas as pd

from polyis.utilities import METRICS, get_video_frame_count, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']


def load_accuracy_results(dataset: str):
    """
    Load saved accuracy results from individual video result files and combined dataset results.
    
    Loads both individual video results and combined dataset results from the new evaluation 
    directory structure created by p070_accuracy_compute.py.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        DataFrame: Individual video evaluation results
        DataFrame: Combined dataset evaluation results
    """
    results_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')

    # Load individual video results using the shared function (raw, unparsed results)
    individual_results = pd.read_csv(os.path.join(results_dir, 'accuracy.csv'))
    assert len(individual_results) > 0, f"No individual results found for dataset {dataset}"

    # Load combined dataset results using the shared function (raw, unparsed results)
    combined_results = pd.read_csv(os.path.join(results_dir, 'accuracy_combined.csv'))
    assert len(combined_results) > 0, f"No combined results found for dataset {dataset}"
    
    return individual_results, combined_results


def load_throughput_results(dataset: str) -> pd.DataFrame:
    """
    Load throughput results from the measurements directory.

    Args:
        dataset (str): Dataset name

    Returns:
        pd.DataFrame: Query execution overall timing DataFrame
        dict: Metadata
    """
    measurements_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput', 'measurements')
    assert os.path.exists(measurements_dir), \
        f"Throughput measurements directory {measurements_dir} does not exist"

    # Load CSV files
    query_overall_file = os.path.join(measurements_dir, 'query_execution_overall.csv')
    assert os.path.exists(query_overall_file), \
        f"Query execution overall file {query_overall_file} does not exist"
    query_overall = pd.read_csv(query_overall_file)

    print(f"Loaded throughput data with {len(query_overall)} query execution records")
    return query_overall


def calculate_naive_runtime(video_name: str, query_overall: pd.DataFrame, dataset: str) -> float:
    """
    Calculate naive runtime for a specific video.

    This matches the naive calculation in p082_throughput_visualize.py,
    which sums the preprocessing stages (detection and tracking).

    Args:
        video_name: Name of the video
        query_overall: Query execution overall timing DataFrame
        dataset: Dataset name

    Returns:
        float: Naive runtime in seconds
    """
    # Add preprocessing time (naive approach)
    preprocessing_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']

    # Filter for this video and preprocessing stages with tilesize 0
    naive_data = query_overall[
        (query_overall['video'] == video_name) &
        (query_overall['stage'].isin(preprocessing_stages)) &
        (query_overall['tilesize'] == 0)
    ]

    naive_runtime = naive_data['time'].sum()

    return naive_runtime


NAIVE_STAGES = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']


def prepare_accuracy(accuracy: pd.DataFrame, dataset: str) -> pd.DataFrame:
    accuracy = accuracy.copy()

    # Rename columns to match expected output format
    accuracy = accuracy.rename(columns={
        'Dataset': 'dataset',
        'Video': 'video',
        'Classifier': 'classifier',
        'Tile_Size': 'tilesize',
        'Tile_Padding': 'tilepadding',
    })
    
    return accuracy


def prepare_throughput(throughput: pd.DataFrame) -> pd.DataFrame:
    df = throughput.copy()

    datasets = df['dataset']
    assert isinstance(datasets, pd.Series), \
        f"datasets should be a Series, got {type(datasets)}"
    assert len(datasets.unique()) == 1, \
        f"Expected only one dataset, got {datasets.unique()}"

    # Group by video, classifier, tilesize, tilepadding and sum the times
    cols = ['video', 'classifier', 'tilesize', 'tilepadding']
    df = df.groupby(cols)['time'].sum().reset_index()

    return df


def match_accuracy_throughput_data(
    accuracy: pd.DataFrame,
    throughput: pd.DataFrame,
    accuracy_combined: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match accuracy and throughput data by video/classifier/tilesize combination.

    Args:
        accuracy_results: DataFrame of individual video accuracy evaluation results
        throughput_overall: DataFrame of throughput measurement data
        combined_results: DataFrame of combined dataset accuracy results

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Individual video data points DataFrame
            - Dataset-wide aggregated data points DataFrame using actual combined accuracy scores
    """
    print(f"Processing {len(accuracy)} accuracy results for matching...")
    accuracy = prepare_accuracy(accuracy, dataset)
    accuracy_combined = prepare_accuracy(accuracy_combined, dataset)

    is_naive = throughput['stage'].isin(NAIVE_STAGES)
    throughput_ = throughput[~is_naive]
    naive_throughput_ = throughput[is_naive].copy()
    assert isinstance(throughput_, pd.DataFrame)
    throughput = prepare_throughput(throughput_)

    # naive_throughput_ = throughput[is_naive].copy()
    naive_throughput_['classifier'] = 'Groundtruth'
    naive_throughput_['tilesize'] = 0
    naive_throughput_['tilepadding'] = 'Groundtruth'
    assert isinstance(naive_throughput_, pd.DataFrame)
    naive_throughput = prepare_throughput(naive_throughput_)

    throughput = pd.concat([throughput, naive_throughput], ignore_index=True, axis=0)
    # print(throughput)
    print(accuracy)

    # Merge accuracy data with runtime data
    assert len(throughput) == len(accuracy), \
        f"Expected {len(accuracy)} runtime data points, got {len(throughput)}"
    join_cols = ['video', 'classifier', 'tilesize', 'tilepadding']
    tradeoff = accuracy.merge(throughput, on=join_cols, how='inner')
    assert len(tradeoff) == len(accuracy), \
        f"Expected {len(accuracy)} tradeoff data points, got {len(tradeoff)}"

    # Calculate throughput (frames per second)
    count_frames = partial(get_video_frame_count, dataset)
    tradeoff['frame_count'] = tradeoff['video'].map(count_frames)
    tradeoff['throughput_fps'] = tradeoff['frame_count'] / tradeoff['time']

    # Aggregate runtime and frame counts by classifier/tilesize/tilepadding
    gb_cols = ['classifier', 'tilesize', 'tilepadding']
    throughput_combined = tradeoff.groupby(gb_cols).agg({
        'frame_count': 'sum',
        'time': 'sum'
    }).reset_index()

    # Merge with combined accuracy scores
    assert len(accuracy_combined) == len(throughput_combined), \
        f"Expected {len(accuracy_combined)} combined throughput data points, got {len(throughput_combined)}"
    join_cols = ['classifier', 'tilesize', 'tilepadding']
    tradeoff_combined = accuracy_combined.merge(throughput_combined, on=join_cols, how='inner')
    assert len(tradeoff_combined) == len(throughput_combined), \
        f"Expected {len(throughput_combined)} combined tradeoff data points, got {len(tradeoff_combined)}"

    # Calculate combined throughput
    tradeoff_combined['throughput_fps'] = tradeoff_combined['frame_count'] / tradeoff_combined['time']
    tradeoff_combined['video'] = 'dataset_level'

    # Print combined accuracy scores for verification
    for _, row in tradeoff_combined.iterrows():
        print(f"Using actual combined accuracy scores for {row['classifier']}_{row['tilesize']}_{row['tilepadding']}: " \
              f"HOTA={row['HOTA_HOTA']:.3f}, Count_DetsMAPE={row['Count_DetsMAPE']:.3f}")

    print(f"Created {len(tradeoff_combined)} combined tradeoff data points")
    return tradeoff, tradeoff_combined


def prepare_naive_throughput(throughput: pd.DataFrame, dataset: str):
    is_naive = throughput['stage'].isin(NAIVE_STAGES)
    naive = throughput[is_naive]
    assert isinstance(naive, pd.DataFrame)
    naive = prepare_throughput(naive)

    count_frames = partial(get_video_frame_count, dataset)
    naive['frame_count'] = naive['video'].map(count_frames)

    naive = naive.groupby(['video']).agg({
        'time': 'sum',
        'frame_count': 'sum',
    }).reset_index()
    naive['throughput_fps'] = naive['frame_count'] / naive['time']

    combined_naive = naive.agg({
        'time': 'sum',
        'frame_count': 'sum',
    }).to_frame().T.reset_index(drop=True)
    combined_naive['throughput_fps'] = combined_naive['frame_count'] / combined_naive['time']
    assert len(combined_naive) == 1, \
        f"Expected 1 combined naive throughput data point, got {len(combined_naive)}"

    return naive, combined_naive


def process_dataset(dataset: str):
    """
    Process a single dataset for accuracy-query execution runtime tradeoff computation.
    
    This function loads accuracy and throughput results for a single dataset, matches them,
    and computes tradeoff data showing the relationship between accuracy and query execution runtime.
    
    Args:
        dataset: Dataset name to process
    """
    print(f"Starting accuracy-query execution runtime tradeoff computation for: {dataset}")
    
    # Load accuracy results (both individual and combined)
    print(f"Loading accuracy results for {dataset}...")
    accuracy, accuracy_combined = load_accuracy_results(dataset)
    
    assert len(accuracy) > 0, \
        f"No accuracy results found for {dataset}. " \
        "Please run p070_accuracy_compute.py first."
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Load throughput results
    print(f"Loading throughput results for {dataset}...")
    throughput = load_throughput_results(dataset)

    assert len(throughput) > 0, \
        f"No throughput results found for {dataset}. Please run " \
        "p080_throughput_gather.py and p081_throughput_compute.py first."
    
    # Match accuracy and throughput data
    print(f"Matching accuracy and throughput data for {dataset}...")
    tradeoff, tradeoff_combined = match_accuracy_throughput_data(accuracy, throughput,
                                                                 accuracy_combined, dataset)
    naive, naive_combined = prepare_naive_throughput(throughput, dataset)
    
    assert len(tradeoff) > 0, \
        f"No matching data points found between accuracy and throughput results for {dataset}."
    assert len(tradeoff_combined) > 0, \
        f"No combined matched data points found between accuracy and throughput results for {dataset}."

    # Save tradeoff data
    output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '090_tradeoff')
    tradeoff.to_csv(os.path.join(output_dir, f'tradeoff.csv'), index=False)
    tradeoff_combined.to_csv(os.path.join(output_dir, f'tradeoff_combined.csv'), index=False)
    naive.to_csv(os.path.join(output_dir, f'naive.csv'), index=False)
    naive_combined.to_csv(os.path.join(output_dir, f'naive_combined.csv'), index=False)


def main():
    """
    Main function that orchestrates the accuracy-throughput tradeoff computation.
    
    This function serves as the entry point for the script. It:
    1. Loads accuracy results from p070_accuracy_compute.py
    2. Loads throughput results from p081_throughput_compute.py
    3. Gets video frame counts using OpenCV
    4. Matches the data by video/classifier/tilesize combination
    5. Computes tradeoff data showing accuracy vs query execution runtime relationships
    6. Computes tradeoff data showing accuracy vs throughput (frames/second) relationships
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/raw/{classifier}_{tilesize}_{tilepadding}/
        - The script expects throughput results from p081_throughput_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/080_throughput/measurements/query_execution_summaries.json
        - Results are saved to: {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/
        - Video files are expected in {DATA_DIR}/{dataset}/{video_name}.mp4 (or other extensions)
        - Only query execution runtime is used (index construction time is ignored)
        - Metrics are automatically detected from the accuracy results
    """
    print(f"Processing datasets: {DATASETS}")

    for dataset in DATASETS:
        output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '090_tradeoff')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleared existing 090_tradeoff directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Process datasets in parallel with progress tracking
    with Pool() as pool:
        ires = pool.imap(process_dataset, DATASETS)
        # ires = map(process_dataset, args.datasets)
        
        # Process datasets in parallel using imap with rich track
        _ = [*track(ires, total=len(DATASETS))]


if __name__ == '__main__':
    main()
