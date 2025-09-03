# Object Detector Speed Experiments

This directory contains experiments to test the inference speed of three popular object detectors across different video resolutions.

## Models Tested

1. **YOLOv3** - Using ultralytics implementation
2. **YOLOv5s** - Using ultralytics implementation  
3. **RetinaNet** - Using detectron2 implementation

## Video Resolutions Tested

- **480p**: 854x480 pixels
- **720p**: 1280x720 pixels
- **1080p**: 1920x1080 pixels

## Files

- `test_yolov3_speed.py` - YOLOv3 speed testing script
- `test_yolov5_speed.py` - YOLOv5 speed testing script
- `test_retinanet_speed.py` - RetinaNet speed testing script
- `run_all_experiments.py` - Master script to run all experiments
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For detectron2 (RetinaNet), you may need to follow the official installation guide:
   - https://detectron2.readthedocs.io/en/latest/tutorials/install.html

## Usage

### Run All Experiments

To run all three detector experiments and compare results:

```bash
python run_all_experiments.py --video /path/to/your/video.mp4
```

Options:
- `--video`: Path to input video file (required)
- `--output-dir`: Output directory for results (default: `./detection_experiments/results`)
- `--visualize`: Create visualization videos with bounding boxes
- `--skip-experiments`: Skip running experiments and only analyze existing results

### Run Individual Experiments

To run a specific detector experiment:

```bash
# YOLOv3
python test_yolov3_speed.py --video /path/to/your/video.mp4

# YOLOv5
python test_yolov5_speed.py --video /path/to/your/video.mp4

# RetinaNet
python test_retinanet_speed.py --video /path/to/your/video.mp4
```

Each script supports the same options as the master script.

## What Gets Measured

The experiments measure **only the inference time** of each model, excluding:
- Video decoding time
- Frame resizing time
- Video writing time
- Other preprocessing/postprocessing overhead

## Output

Each experiment generates:

1. **Speed Results**: JSON files with detailed timing data for each resolution
2. **Visualization Videos**: MP4 files with bounding boxes drawn on detected objects (if `--visualize` is used)
3. **Comparison Plots**: PNG charts comparing performance across models and resolutions
4. **Summary Reports**: Markdown and CSV files with comprehensive results

## Example Output Structure

```
detection_experiments/results/
├── yolov3_480p_speed_results.json
├── yolov3_720p_speed_results.json
├── yolov3_1080p_speed_results.json
├── yolov5_480p_speed_results.json
├── yolov5_720p_speed_results.json
├── yolov5_1080p_speed_results.json
├── retinanet_480p_speed_results.json
├── retinanet_720p_speed_results.json
├── retinanet_1080p_speed_results.json
├── yolov3_all_resolutions_results.json
├── yolov5_all_resolutions_results.json
├── retinanet_all_resolutions_results.json
├── speed_comparison.png
├── speed_comparison.csv
└── experiment_summary.md
```

## Performance Metrics

For each model and resolution, the following metrics are calculated:
- **Mean inference time**: Average time across all test frames
- **Standard deviation**: Consistency of performance
- **Minimum time**: Best case performance
- **Maximum time**: Worst case performance

## Notes

- Tests sample approximately 100 frames per resolution to balance speed and accuracy
- Each frame is tested with 10 inference runs to get stable timing measurements
- GPU acceleration is used when available
- Results are saved in JSON format for further analysis
- The master script automatically generates comparison visualizations and reports

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce the number of inference runs per frame in the test functions
2. **Model download issues**: Ensure internet connection for downloading pre-trained weights
3. **Detectron2 import errors**: Follow the official detectron2 installation guide

### Performance Tips

- Use GPU acceleration when possible
- Close other applications to free up GPU memory
- For very long videos, consider using a shorter sample for testing
- Adjust the `sample_interval` in the scripts to test more or fewer frames

## Example Results

After running the experiments, you'll get output like:

```
FINAL COMPARISON
==================================================
480p: 15.23 ± 2.45 ms
720p: 28.67 ± 3.12 ms
1080p: 52.89 ± 4.78 ms
```

This shows the average inference time and standard deviation for each resolution across all models.
