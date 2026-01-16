#!/usr/local/bin/python

import argparse
import json
import os
import shutil

import numpy as np
import altair as alt
import pandas as pd

from polyis.utilities import get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def visualize_compared_accuracy_bar(results: pd.DataFrame, score_field: str, xlabel: str, output_path: str):
    """
    Create a comparison plot for tracking accuracy scores by video, tile size, tilepadding, sample rate, and tracker.
    
    Creates a faceted bar chart showing accuracy scores for different classifiers
    across videos, tile sizes, tilepadding values, sample rates, and trackers. Each facet shows one video-tilesize-tilepadding-sample_rate-tracker combination,
    with bars representing different classifiers sorted by performance.
    
    Args:
        results: DataFrame of evaluation results
        score_field: Field name for scores ('HOTA_HOTA', 'HOTA_AssA', 'HOTA_DetA', 'Count_DetsMAPE', 'Count_TracksMAPE')
        xlabel: Label for x-axis
        output_path: Path to save the plot
    """

    df = results.copy()
    df['Score'] = df[score_field]
    # Create label that includes classifier, tile padding, sample rate, and tracker
    # Handle missing columns for backward compatibility
    if 'Sample_Rate' not in df.columns:
        df['Sample_Rate'] = 1
    if 'Tracker' not in df.columns:
        df['Tracker'] = 'unknown'
    
    df['Classifier_Tile_Padding'] = df['Classifier'].str.slice(0, 3) + '_' + df['Tile_Padding'].str.slice(0, 3) + '_SR' + df['Sample_Rate'].astype(str) + '_' + df['Tracker'].str.slice(0, 6)

    df['Tile_Padding'] = df['Tile_Padding'].apply(lambda x: {'connected': 'padded (+)', 'none': 'none'}.get(x, 'Naive'))
    
    # Create combined identifier for sample rate and tracker if needed for faceting
    # Check if we need additional faceting for sample rate and tracker
    has_multiple_sample_rates = df['Sample_Rate'].nunique() > 1
    has_multiple_trackers = df['Tracker'].nunique() > 1
    if has_multiple_sample_rates and has_multiple_trackers:
        # Create a combined identifier for sample rate and tracker to use as row facet
        df['SR_Tracker'] = df['Sample_Rate'].astype(str) + '_' + df['Tracker']
    
    # Create horizontal bar chart with text labels inside bars
    # Main bars showing the scores
    if score_field.startswith('Count_'):
        x_encoding = alt.X('Score:Q', title=xlabel)
    else:
        x_encoding = alt.X('Score:Q', title=xlabel, scale=alt.Scale(domain=[0, 1]))

    bars = alt.Chart(df).mark_bar().encode(
        x=x_encoding,
        # yOffset=alt.YOffset('Dilate:N'),
        color=alt.Color('Tile_Padding:N', title='Tile Padding'),
        tooltip=['Video', 'Tile_Size', 'Tile_Padding', 'Classifier', 'Sample_Rate', 'Tracker', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(
        width=200,
        height=120
    )
    
    # Add score text labels inside the bars (white text)
    text = alt.Chart(df).mark_text(
        align='right',
        baseline='middle',
        dx=-3,  # Small offset from the right edge of the bar
        color='white'
    ).transform_calculate(text='parseInt(datum.Score * 100)').encode(
        x=alt.X('Score:Q'),
        text=alt.Text('text:N'),
        # yOffset=alt.YOffset('Dilate:N')
    )

    # Add classifier name labels on the left side of bars
    labels = alt.Chart(df).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontWeight='bold',
        color='black'
    ).transform_calculate(Score2='datum.Score * 0.0001').encode(
        x=alt.X('Score2:Q'),
        text=alt.Text('Classifier_Tile_Padding:N'),
        # Use white text for high scores, black for low scores
        color=alt.condition(alt.datum.Score > 0.1, alt.value('white'), alt.value('black')),
        # yOffset=alt.YOffset('Dilate:N')
    )
    
    # Layer the charts (bars + labels + text) and apply faceting
    # Facet by video (columns), with sample rate and tracker included in labels and tooltips
    # Apply faceting - use row for sample rate or tracker if multiple values exist
    # Keep video as column facet, and add row facet for additional dimensions
    base_chart = (bars + labels + text).encode(
        y=alt.Y('Classifier_Tile_Padding:N', sort='-x', axis=alt.Axis(labels=False, ticks=False, title=None)),
    ).resolve_scale(y='independent')
    
    if has_multiple_sample_rates and has_multiple_trackers:
        # Facet by video (columns) and combined sample rate/tracker (rows)
        chart = base_chart.facet(
            column=alt.Column('Video:N', title=None),
            row=alt.Row('SR_Tracker:N', title='Sample Rate / Tracker'),
        ).resolve_scale(y='independent').properties(padding=0)
    elif has_multiple_sample_rates:
        # Facet by video (columns) and sample rate (rows)
        chart = base_chart.facet(
            column=alt.Column('Video:N', title=None),
            row=alt.Row('Sample_Rate:O', title='Sample Rate'),
        ).resolve_scale(y='independent').properties(padding=0)
    elif has_multiple_trackers:
        # Facet by video (columns) and tracker (rows)
        chart = base_chart.facet(
            column=alt.Column('Video:N', title=None),
            row=alt.Row('Tracker:N', title='Tracker'),
        ).resolve_scale(y='independent').properties(padding=0)
    else:
        # Only facet by video (original behavior)
        chart = base_chart.facet(
            column=alt.Column('Video:N', title=None),
        ).resolve_scale(y='independent').properties(padding=0)
    
    # Save the chart as PNG with high resolution
    chart.save(output_path, scale_factor=2)


def visualize_tracking_accuracy(results: pd.DataFrame, output_dir: str, combined: bool = False):
    """
    Create visualizations for tracking accuracy results using Altair.
    
    Processes evaluation results and creates comparison charts showing
    accuracy scores across different classifiers, videos, tile sizes, sample rates, and trackers.
    
    Args:
        results (pd.DataFrame): DataFrame of evaluation results
        output_dir (str): Output directory for visualizations
        combined (bool): Whether these are combined dataset results or individual video results
    """
    print("Creating visualizations...")
    # Set prefix for output files based on whether these are combined results
    prefix = "combined_" if combined else ""

    metrics = [
        ('HOTA_HOTA', 'HOTA Score'),
        ('HOTA_AssA', 'AssA Score'),
        ('HOTA_DetA', 'DetA Score'),
        ('Count_DetsMAPE', 'Dets MAPE (%)'),
        ('Count_TracksMAPE', 'Tracks MAPE (%)'),
    ]
    
    # Create comparison plots for HOTA, AssA, and MOTA scores
    # Each plot shows classifiers ranked by performance for each video-tilesize-tilepadding-sample_rate-tracker combination
    for metric_name, metric_label in metrics:
        visualize_compared_accuracy_bar(results, metric_name, metric_label,
                                        os.path.join(output_dir, f'{prefix}{metric_name}.png'))


def main():
    """
    Main function that orchestrates the tracking accuracy visualization process.
    
    This function serves as the entry point for the script. It:
    1. Finds all classifier/tilesize/tilepadding/sample_rate/tracker combinations with saved accuracy results
    2. Loads CSV files generated by p111_accuracy_aggregate.py
    3. Creates visualizations comparing accuracy across videos, classifiers, tile sizes, tilepadding values, sample rates, and trackers
    4. Generates summary reports and charts for each dataset
    
    Note:
        - The script expects accuracy results CSV files from p111_accuracy_aggregate.py in:
          {CACHE_DIR}/{dataset}/evaluation/070_accuracy/accuracy.csv and accuracy_combined.csv
        - Multiple metrics are visualized: HOTA, AssA, DetA, Count MAPE
        - Visualizations are saved to: {CACHE_DIR}/{dataset}/evaluation/072_accuracy_visualize/
    """
    print(f"Starting tracking accuracy visualization for datasets: {DATASETS}")
    
    # Process each dataset separately to create independent visualizations
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        results_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_accuracy')
        
        dataset_results = pd.read_csv(os.path.join(results_dir, 'accuracy_combined.csv'))
        combined_results = pd.read_csv(os.path.join(results_dir, 'accuracy_combined.csv'))
        assert len(dataset_results) > 0, f"No results found for dataset {dataset}"
        assert len(combined_results) > 0, f"No combined results found for dataset {dataset}"
        
        # Create output directory for this dataset's visualizations
        output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '072_accuracy_visualize')
        
        # Clean and recreate output directory to ensure fresh results
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        visualize_tracking_accuracy(dataset_results, output_dir, combined=False)
        visualize_tracking_accuracy(combined_results, output_dir, combined=True)
        
        print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
