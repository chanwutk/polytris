# Tracking Speed Performance Analysis Script

This script (`071_results_statistics_speed.py`) analyzes the speed and performance characteristics of tracking results from `060_exec_track.py` using the runtime data that was collected during execution.

## Overview

The script performs the following functions:

1. **Finds runtime data**: Automatically locates all video files with tracking results and runtime performance data
2. **Analyzes performance metrics**: Computes comprehensive speed statistics including:
   - **FPS**: Frames per second processing rate
   - **Frame Time**: Individual frame processing time
   - **Step Timing**: Breakdown of processing steps (detection conversion, tracker update, result processing)
   - **Cumulative Performance**: Total processing time and throughput analysis
3. **Creates performance summaries**: Generates detailed text summaries and JSON results
4. **Optional visualizations**: Creates comprehensive matplotlib plots showing performance trends

## Prerequisites

- Tracking results from `060_exec_track.py` must exist in the cache directory
- Runtime data files (`runtimes.jsonl`) must be available from the tracking execution
- TrackEval module must be available in the modules directory
- For visualizations: matplotlib and pandas

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
# Analyze all videos with all tile sizes
python scripts/071_results_statistics_speed.py

# Analyze specific dataset
python scripts/071_results_statistics_speed.py --dataset b3d

# Analyze specific tile size only
python scripts/071_results_statistics_speed.py --tile_size 64
```

### Advanced Usage

```bash
# Custom output directory and parallel processing
python scripts/071_results_statistics_speed.py \
    --output_dir "custom_speed_results" \
    --num_cores 16

# Enable visualization plots
python scripts/071_results_statistics_speed.py --create_plots

# Sequential processing (disable parallel)
python scripts/071_results_statistics_speed.py --parallel False
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `b3d` | Dataset name to process |
| `--tile_size` | str | `all` | Tile size(s) to evaluate (`64`, `128`, or `all`) |
| `--output_dir` | str | `pipeline-stages/track-speed-results` | Output directory for results |
| `--parallel` | bool | `True` | Whether to use parallel processing |
| `--num_cores` | int | `8` | Number of parallel cores to use |
| `--create_plots` | bool | `False` | Whether to create visualization plots |

## Input Data Structure

The script expects the following directory structure:

```
/polyis-cache/
└── {dataset}/
    └── {video_file}/
        └── uncompressed_tracking/
            └── proxy_{tile_size}/
                ├── tracking.jsonl          # Tracking results from 060_exec_track.py
                └── runtimes.jsonl        # Runtime performance data from 060_exec_track.py
```

### Runtime Data Format

The `runtimes.jsonl` file contains performance data for each frame:

```json
{
  "frame_idx": 0,
  "step_times": {
    "convert_detections": 0.0012,
    "tracker_update": 0.0089,
    "process_results": 0.0023,
    "total_frame_time": 0.0124
  },
  "num_detections": 15,
  "num_tracks": 12
}
```

## Output Files

The script generates several output files:

- `detailed_speed_results.json`: Complete performance analysis results in JSON format
- `speed_summary.txt`: Human-readable summary of all performance metrics
- `speed_results.csv`: CSV file with performance data (if visualization libraries available)
- Various PNG plots (if `--create_plots` is enabled):
  - `speed_overview.png`: Performance dashboard with FPS, frame time, and total time
  - `detailed_timing_[video]_tile[size].png`: Detailed timing analysis for each video
  - `tile_size_speed_comparison.png`: Performance comparison across tile sizes

## Performance Metrics

### Primary Metrics

1. **FPS (Frames Per Second)**
   - **Description**: Processing throughput in frames per second
   - **Calculation**: `1.0 / average_frame_time`
   - **Higher is better**: Indicates faster processing

2. **Frame Processing Time**
   - **Description**: Time to process each individual frame
   - **Units**: Seconds per frame
   - **Lower is better**: Indicates faster frame processing

3. **Total Processing Time**
   - **Description**: Total time to process the entire video
   - **Units**: Seconds
   - **Lower is better**: Indicates faster overall processing

### Step-by-Step Analysis

The script analyzes performance at each processing step:

1. **Detection Conversion**: Time to convert detection results to numpy arrays
2. **Tracker Update**: Time for the tracking algorithm to process detections
3. **Result Processing**: Time to process and format tracking results
4. **Total Frame Time**: Sum of all step times

### Statistical Analysis

For each metric, the script computes:
- **Mean**: Average performance across all frames
- **Standard Deviation**: Variability in performance
- **Min/Max**: Best and worst case performance
- **Percentiles**: 25th, 50th, 75th, 90th, 95th, 99th percentiles

## Visualization Features

When `--create_plots` is enabled, the script creates comprehensive matplotlib visualizations:

### 1. Speed Overview Dashboard
- **FPS Comparison**: Bar chart showing processing rate for each video
- **Frame Time Analysis**: Bar chart showing average frame processing time
- **Total Time Overview**: Bar chart showing total processing time per video
- **Performance Heatmap**: Color-coded matrix showing all performance metrics

### 2. Detailed Timing Analysis (Per Video)
- **Frame Processing Time Over Time**: Line plot showing performance variation
- **Step-by-Step Breakdown**: Box plots comparing different processing steps
- **Cumulative Processing Time**: Line plot showing total time accumulation
- **FPS Over Time**: Rolling average FPS showing performance trends

### 3. Tile Size Comparison
- Box plots comparing performance across different tile sizes
- Only generated when multiple tile sizes are evaluated

## Example Output

```
Starting tracking speed performance analysis for dataset: b3d
Tile size(s): all
Output directory: pipeline-stages/track-speed-results

Found tracking results with runtime data: jnc00.mp4 with tile size 64
Found tracking results with runtime data: jnc02.mp4 with tile size 64
Found tracking results with runtime data: jnc06.mp4 with tile size 64
Found tracking results with runtime data: jnc07.mp4 with tile size 64

Found 4 video-tile size combinations to analyze
Using sequential processing

Analyzing runtime performance for jnc00.mp4 with tile size 64
Analyzing runtime performance for jnc02.mp4 with tile size 64
Analyzing runtime performance for jnc06.mp4 with tile size 64
Analyzing runtime performance for jnc07.mp4 with tile size 64

Analysis completed:
  Successful analyses: 4
  Failed analyses: 0

Creating speed performance summary...
Summary saved to pipeline-stages/track-speed-results/speed_summary.txt

Summary Statistics:
  FPS: Mean=45.23, Std=12.45
  Frame Time: Mean=0.0221s, Std=0.0056s
  Total Time: Mean=12.34s, Std=3.21s

Results saved to: pipeline-stages/track-speed-results
```

## Performance Analysis Use Cases

### 1. **Algorithm Optimization**
- Identify performance bottlenecks in specific processing steps
- Compare performance across different tile sizes
- Measure the impact of parameter changes on speed

### 2. **System Requirements**
- Determine real-time processing capabilities
- Estimate processing time for new videos
- Plan hardware requirements for deployment

### 3. **Quality vs. Speed Trade-offs**
- Analyze the relationship between accuracy and speed
- Find optimal parameters for specific use cases
- Balance performance requirements with accuracy needs

### 4. **Performance Monitoring**
- Track performance over time
- Identify performance degradation
- Monitor system health during long-running operations

## Troubleshooting

### Common Issues

1. **"No tracking results with runtime data found"**
   - Ensure `060_exec_track.py` has been run successfully
   - Check that runtime data collection was enabled
   - Verify that `runtimes.jsonl` files exist

2. **"Error analyzing runtime performance"**
   - Check that runtime data files are valid JSONL
   - Verify the expected data structure
   - Ensure all required fields are present

3. **"Visualization libraries not available"**
   - Install required packages: `pip install matplotlib pandas`
   - Or use the script without the `--create_plots` flag
   - The script will still generate text summaries and JSON results

### Performance Tips

- Use parallel processing for large datasets (default: enabled)
- Adjust `--num_cores` based on your system capabilities
- For very large datasets, consider processing tile sizes separately
- Enable visualizations only when needed to save processing time

## Integration with Pipeline

This script is designed to work seamlessly with the existing pipeline:

1. **After tracking**: Run `060_exec_track.py` to generate tracking results and runtime data
2. **Analyze performance**: Run this script to measure speed characteristics
3. **Review results**: Analyze performance summaries and visualizations
4. **Optimize**: Use insights to improve tracking parameters or system configuration

## Extending the Script

The script is modular and can be easily extended:

- Add new performance metrics by extending the analysis functions
- Implement custom visualization functions using matplotlib
- Add support for additional performance data formats
- Integrate with performance monitoring systems

## Technical Notes

- **Data Source**: Reads runtime data generated during tracking execution
- **Performance Metrics**: Computes both high-level (FPS) and detailed (step-by-step) metrics
- **Statistical Analysis**: Provides comprehensive statistical summaries including percentiles
- **Visualization**: Uses matplotlib for high-quality, publication-ready plots
- **Memory Management**: Efficiently processes large runtime datasets
- **Parallel Processing**: Supports both sequential and parallel analysis modes

## Performance Benchmarking

The script is ideal for benchmarking tracking algorithms:

- **Baseline Comparison**: Compare performance across different algorithms
- **Parameter Tuning**: Measure the impact of parameter changes
- **Hardware Testing**: Evaluate performance on different hardware configurations
- **Scalability Analysis**: Understand performance characteristics as dataset size increases
