# Runtime Analysis and Visualization Script

This script (`071_results_statistics_speed.py`) provides comprehensive runtime analysis and visualization for the object tracking pipeline.

## Overview

The script analyzes runtime performance across all pipeline stages:
1. **Classification** - Tile relevance classification
2. **Compression** - Video tile packing and compression
3. **Detection** - Object detection on packed images
4. **Tracking** - Object tracking with SORT algorithm

## Features

### Runtime Analysis
- **Per-frame timing** for each pipeline stage
- **Step-by-step breakdown** of compression and tracking operations
- **Statistical analysis** including mean, std, min, max runtime
- **Runtime distribution comparison** across stages

### Visualizations
- **Runtime Analysis Plots**: Overall performance comparison
- **Compression Details**: Detailed step-by-step compression analysis
- **Tracking Details**: Detailed step-by-step tracking analysis
- **Tracking Results**: Track count, duration, size, and trajectory analysis

### Output
- High-resolution PNG plots (300 DPI)
- JSON summary statistics
- Organized output directory structure

## Usage

### Basic Usage
```bash
python scripts/071_results_statistics_speed.py
```

### Advanced Usage
```bash
python scripts/071_results_statistics_speed.py \
    --dataset b3d \
    --tile_size 64 \
    --output_dir ./my_analysis
```

### Command Line Arguments
- `--dataset`: Dataset name (default: 'b3d')
- `--tile_size`: Tile size to analyze ('64', '128', or 'all')
- `--output_dir`: Output directory for visualizations (default: './runtime_analysis')

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements_runtime_analysis.txt
```

## Data Sources

The script automatically discovers and loads runtime data from:
- `{CACHE_DIR}/{dataset}/{video}/relevancy/score/proxy_{tile_size}/score.jsonl` - Classification runtime
- `{CACHE_DIR}/{dataset}/{video}/packing/proxy_{tile_size}/runtime.jsonl` - Compression runtime
- `{CACHE_DIR}/{dataset}/{video}/packed_detections/proxy_{tile_size}/runtimes.jsonl` - Detection runtime
- `{CACHE_DIR}/{dataset}/{video}/uncompressed_tracking/proxy_{tile_size}/runtimes.jsonl` - Tracking runtime
- `{CACHE_DIR}/{dataset}/{video}/uncompressed_tracking/proxy_{tile_size}/tracks.jsonl` - Tracking results

## Output Structure

```
{output_dir}/
├── {dataset}/
│   └── {video_file}/
│       └── tile_{tile_size}/
│           ├── {video_file}_tile{tile_size}_runtime_analysis.png
│           ├── {video_file}_tile{tile_size}_compression_details.png
│           ├── {video_file}_tile{tile_size}_tracking_details.png
│           ├── {video_file}_tile{tile_size}_tracking_analysis.png
│           └── {video_file}_tile{tile_size}_summary.json
```

## Runtime Measurements

### Classification Stage
- `runtime`: Total inference time per frame

### Compression Stage
- `read_frame`: Frame reading time
- `get_classifications`: Classification loading time
- `create_bitmap`: Bitmap creation time
- `group_tiles`: Tile grouping time
- `sort_polyominoes`: Polyomino sorting time
- `pack_append`: Packing time
- `save_canvas`: Canvas saving time
- `update_mapping`: Mapping update time
- `total_frame_time`: Total frame processing time

### Detection Stage
- `read_time`: Image reading time
- `detect_time`: Object detection time

### Tracking Stage
- `convert_detections`: Detection conversion time
- `tracker_update`: SORT tracker update time
- `process_results`: Result processing time
- `total_frame_time`: Total frame processing time

## Example Output

The script generates comprehensive visualizations showing:
- Runtime trends over frames
- Performance bottlenecks identification
- Step-by-step timing analysis
- Tracking quality metrics
- Statistical summaries

## Notes

- Runtime data is automatically collected by the pipeline scripts
- The script handles missing data gracefully
- All timing is in seconds
- High-resolution plots are suitable for publications
- Summary statistics are saved in machine-readable JSON format
