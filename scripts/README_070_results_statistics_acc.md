# Tracking Accuracy Evaluation Script

This script (`070_results_statistics_acc.py`) evaluates the accuracy of tracking results from `060_exec_track.py` using TrackEval's B3D evaluation methods.

## Overview

The script performs the following functions:

1. **Finds tracking results**: Automatically locates all video files with tracking results for the specified dataset and tile size(s)
2. **Runs accuracy evaluation**: Uses TrackEval to compute multiple accuracy metrics:
   - **HOTA**: Higher Order Tracking Accuracy (primary metric)
   - **CLEAR**: Multiple Object Tracking Accuracy (MOTA)
   - **Identity**: Identity F1 Score (IDF1)
3. **Creates summary reports**: Generates detailed text summaries and JSON results
4. **Optional visualizations**: Can create plots using matplotlib if available

## Prerequisites

- Tracking results from `060_exec_track.py` must exist in the cache directory
- Groundtruth data must be available in the expected format
- TrackEval module must be available in the modules directory
- For visualizations: matplotlib and pandas (optional)

## Installation

### Required Dependencies
```bash
# Core dependencies (already included in project)
pip install numpy

# For visualizations (optional)
pip install matplotlib pandas
```

## Usage

### Basic Usage

```bash
# Evaluate all videos with all tile sizes using default metrics
python scripts/070_results_statistics_acc.py

# Evaluate specific dataset
python scripts/070_results_statistics_acc.py --dataset b3d

# Evaluate specific tile size only
python scripts/070_results_statistics_acc.py --tile_size 64
```

### Advanced Usage

```bash
# Custom metrics and output directory
python scripts/070_results_statistics_acc.py \
    --metrics "HOTA,CLEAR" \
    --output_dir "custom_output" \
    --num_cores 16

# Enable visualization plots (requires matplotlib/pandas)
python scripts/070_results_statistics_acc.py --create_plots

# Sequential processing (disable parallel)
python scripts/070_results_statistics_acc.py --parallel False
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `b3d` | Dataset name to process |
| `--tile_size` | str | `all` | Tile size(s) to evaluate (`64`, `128`, or `all`) |
| `--metrics` | str | `HOTA,CLEAR,Identity` | Comma-separated list of metrics to evaluate |
| `--output_dir` | str | `pipeline-stages/track-accuracy-results` | Output directory for results |
| `--parallel` | bool | `True` | Whether to use parallel processing |
| `--num_cores` | int | `8` | Number of parallel cores to use |
| `--create_plots` | bool | `False` | Whether to create visualization plots |

## Input Data Structure

The script expects the following directory structure:

```
/polyis-cache/
└── {dataset}/
    └── {video_file}/
        ├── uncompressed_tracking/
        │   └── proxy_{tile_size}/
        │       └── tracking.jsonl          # Tracking results from 060_exec_track.py
        └── groundtruth/
            └── tracking.jsonl            # Groundtruth annotations
```

## Output Files

The script generates several output files:

- `detailed_results.json`: Complete evaluation results in JSON format
- `accuracy_summary.txt`: Human-readable summary of all results
- `accuracy_results.csv`: CSV file with results (if visualization libraries available)
- Various PNG plots (if `--create_plots` is enabled and libraries available):
  - `accuracy_comparison.png`: Bar charts and heatmap of all metrics
  - `tile_size_comparison.png`: Box plots comparing different tile sizes
  - `metric_correlation.png`: Scatter plots showing metric relationships
  - `summary_statistics.png`: Formatted table of summary statistics

## Visualization Features

When `--create_plots` is enabled, the script creates comprehensive matplotlib visualizations:

### 1. Accuracy Comparison Dashboard
- **HOTA Scores**: Bar chart with value labels
- **MOTA Scores**: Bar chart with value labels  
- **IDF1 Scores**: Bar chart with value labels
- **Score Heatmap**: Color-coded matrix showing all scores

### 2. Tile Size Analysis
- Box plots comparing performance across different tile sizes
- Only generated when multiple tile sizes are evaluated

### 3. Metric Correlation Analysis
- Scatter plots showing relationships between metrics
- Correlation coefficients displayed on each plot
- HOTA vs MOTA, HOTA vs IDF1, MOTA vs IDF1

### 4. Summary Statistics Table
- Formatted table with mean, std, min, max, median
- Color-coded header and alternating rows
- Professional appearance suitable for reports

## Metrics Explained

### HOTA (Higher Order Tracking Accuracy)
- **Range**: 0.0 to 1.0 (higher is better)
- **Description**: Primary metric that balances detection, association, and localization accuracy
- **Key values**: `HOTA(0)` is the main summary score

### MOTA (Multiple Object Tracking Accuracy)
- **Range**: 0.0 to 1.0 (higher is better)
- **Description**: Measures overall tracking performance including false positives, false negatives, and ID switches
- **Formula**: `MOTA = 1 - (FN + FP + IDSW) / GT`

### IDF1 (Identity F1 Score)
- **Range**: 0.0 to 1.0 (higher is better)
- **Description**: Measures identity preservation across frames
- **Formula**: Harmonic mean of precision and recall for identity matching

## Example Output

```
Starting tracking accuracy evaluation for dataset: b3d
Tile size(s): all
Metrics: HOTA,CLEAR,Identity
Output directory: pipeline-stages/track-accuracy-results
Evaluating metrics: ['HOTA', 'CLEAR', 'Identity']

Found tracking results: jnc00.mp4 with tile size 64
Found tracking results: jnc02.mp4 with tile size 64
Found tracking results: jnc06.mp4 with tile size 64
Found tracking results: jnc07.mp4 with tile size 64

Found 4 video-tile size combinations to evaluate
Using sequential processing

Evaluating jnc00.mp4 with tile size 64
Evaluating jnc02.mp4 with tile size 64
Evaluating jnc06.mp4 with tile size 64
Evaluating jnc07.mp4 with tile size 64

Evaluation completed:
  Successful evaluations: 4
  Failed evaluations: 0

Creating accuracy summary...
Summary saved to pipeline-stages/track-accuracy-results/accuracy_summary.txt

Summary Statistics:
  HOTA: Mean=0.8234, Std=0.0456
  CLEAR: Mean=0.7891, Std=0.0678
  Identity: Mean=0.8123, Std=0.0523

Results saved to: pipeline-stages/track-accuracy-results
```

## Troubleshooting

### Common Issues

1. **"No tracking results found"**
   - Ensure `060_exec_track.py` has been run successfully
   - Check that the cache directory path is correct
   - Verify that tracking results exist in the expected locations

2. **"Error evaluating video"**
   - Check that groundtruth data exists and is in the correct format
   - Verify that tracking results are valid JSONL files
   - Ensure TrackEval module is properly installed

3. **"Visualization libraries not available"**
   - Install required packages: `pip install matplotlib pandas`
   - Or use the script without the `--create_plots` flag
   - The script will still generate text summaries and JSON results

### Performance Tips

- Use parallel processing for large datasets (default: enabled)
- Adjust `--num_cores` based on your system capabilities
- For very large datasets, consider processing tile sizes separately

## Integration with Pipeline

This script is designed to work seamlessly with the existing pipeline:

1. **After detection**: Run `060_exec_track.py` to generate tracking results
2. **Evaluate accuracy**: Run this script to measure tracking performance
3. **Analyze results**: Review the generated summaries and visualizations
4. **Iterate**: Use results to improve tracking parameters or detection methods

## Extending the Script

The script is modular and can be easily extended:

- Add new metrics by importing them from TrackEval
- Implement custom visualization functions using matplotlib
- Add support for additional dataset formats
- Integrate with other evaluation frameworks

## Technical Notes

- **Matplotlib Integration**: Uses pure matplotlib for all visualizations (no seaborn dependency)
- **Error Handling**: Gracefully handles missing visualization libraries
- **Memory Management**: Properly closes matplotlib figures to prevent memory leaks
- **High-Quality Output**: Generates 300 DPI PNG files suitable for publications
- **Responsive Design**: Automatically adjusts plot sizes and layouts
