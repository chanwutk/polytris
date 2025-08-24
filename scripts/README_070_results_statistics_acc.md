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
4. **Creates visualizations**: Automatically generates plots using matplotlib

## Prerequisites

- Tracking results from `060_exec_track.py` must exist in the cache directory
- Groundtruth data must be available in the expected format
- TrackEval module must be available in the modules directory
- Required packages: matplotlib and pandas

## Installation

### Required Dependencies
```bash
# Core dependencies (already included in project)
pip install numpy matplotlib pandas

# TrackEval module should be in /polyis/modules/TrackEval
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
# Custom metrics
python scripts/070_results_statistics_acc.py \
    --metrics "HOTA,CLEAR" \
    --parallel

# Sequential processing (disable parallel)
python scripts/070_results_statistics_acc.py --parallel False
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `b3d` | Dataset name to process |
| `--tile_size` | str | `all` | Tile size(s) to evaluate (`64`, `128`, or `all`) |
| `--metrics` | str | `HOTA,CLEAR,Identity` | Comma-separated list of metrics to evaluate |
| `--parallel` | bool | `False` | Whether to use parallel processing (default: False) |
| `--num_cores` | int | `8` | Number of parallel cores to use (when parallel is enabled) |

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

## Output Structure

The script generates results in multiple locations:

### Per-Video Results
```
/polyis-cache/{dataset}/{video_file}/results/proxy_{tile_size}/accuracy/
└── detailed_results.json                  # Individual video evaluation results
```

### Summary Results
```
/polyis-cache/{dataset}/results/accuracy/
├── detailed_results.json                  # Complete evaluation results in JSON format
├── accuracy_summary.txt                   # Human-readable summary of all results
├── accuracy_results.csv                   # CSV file with results
├── accuracy_comparison.png               # Bar charts of all metrics
└── tile_size_comparison.png              # Box plots comparing tile sizes (if multiple)
```

## Output Files

The script generates several output files:

- `detailed_results.json`: Complete evaluation results in JSON format
- `accuracy_summary.txt`: Human-readable summary of all results
- `accuracy_results.csv`: CSV file with results for visualization
- PNG plots (automatically generated):
  - `accuracy_comparison.png`: Bar charts of HOTA, MOTA, and IDF1 scores
  - `tile_size_comparison.png`: Box plots comparing different tile sizes (only if multiple tile sizes exist)

## Visualization Features

The script automatically creates matplotlib visualizations:

### 1. Accuracy Comparison Dashboard
- **HOTA Scores**: Bar chart with value labels on each bar
- **MOTA Scores**: Bar chart with value labels on each bar  
- **IDF1 Scores**: Bar chart with value labels on each bar
- All charts include grid lines and proper axis labels

### 2. Tile Size Analysis
- Box plots comparing performance across different tile sizes
- Only generated when multiple tile sizes are evaluated
- Color-coded boxes for better visualization

### 3. Data Export
- CSV file with all results for further analysis
- JSON files with detailed evaluation data

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

save results to /polyis-cache/b3d/jnc00.mp4/results/proxy_64/accuracy
save results to /polyis-cache/b3d/jnc02.mp4/results/proxy_64/accuracy
save results to /polyis-cache/b3d/jnc06.mp4/results/proxy_64/accuracy
save results to /polyis-cache/b3d/jnc07.mp4/results/proxy_64/accuracy

Evaluation completed:
  Successful evaluations: 4
  Failed evaluations: 0

Creating accuracy summary...
Summary saved to /polyis-cache/b3d/results/accuracy/accuracy_summary.txt

Summary Statistics:
  HOTA: Mean=0.8234, Std=0.0456
  CLEAR: Mean=0.7891, Std=0.0678
  Identity: Mean=0.8123, Std=0.0523

Creating matplotlib visualizations...
Matplotlib visualizations saved to /polyis-cache/b3d/results/accuracy
Generated files:
  - accuracy_comparison.png
  - accuracy_results.csv

Results saved to: /polyis-cache/b3d/results/accuracy
```

## Troubleshooting

### Common Issues

1. **"No tracking results found"**
   - Ensure `060_exec_track.py` has been run successfully
   - Check that the cache directory path is correct (`/polyis-cache`)
   - Verify that tracking results exist in the expected locations

2. **"Error evaluating video"**
   - Check that groundtruth data exists and is in the correct format
   - Verify that tracking results are valid JSONL files
   - Ensure TrackEval module is properly installed in `/polyis/modules/TrackEval`

3. **"Module not found" errors**
   - Ensure matplotlib and pandas are installed: `pip install matplotlib pandas`
   - Verify TrackEval module is in the correct location

### Performance Tips

- Parallel processing is disabled by default (`--parallel` flag required)
- The script automatically uses all available CPU cores when parallel processing is enabled
- For very large datasets, consider processing tile sizes separately

## Integration with Pipeline

This script is designed to work seamlessly with the existing pipeline:

1. **After detection**: Run `060_exec_track.py` to generate tracking results
2. **Evaluate accuracy**: Run this script to measure tracking performance
3. **Analyze results**: Review the generated summaries and visualizations
4. **Iterate**: Use results to improve tracking parameters or detection methods

## Technical Notes

- **Matplotlib Integration**: Uses pure matplotlib for all visualizations (no seaborn dependency)
- **Automatic Visualization**: Creates plots automatically without requiring additional flags
- **Memory Management**: Properly closes matplotlib figures to prevent memory leaks
- **High-Quality Output**: Generates 300 DPI PNG files suitable for publications
- **Fixed Tile Sizes**: Currently only supports tile size 64 (hardcoded in `TILE_SIZES = [64]`)
- **Output Structure**: Results are saved both per-video and in a centralized summary location
- **Error Handling**: Gracefully handles evaluation failures and continues processing other videos

## Extending the Script

The script is modular and can be easily extended:

- Add new tile sizes by modifying the `TILE_SIZES` constant
- Add new metrics by importing them from TrackEval
- Implement custom visualization functions using matplotlib
- Add support for additional dataset formats
- Integrate with other evaluation frameworks
