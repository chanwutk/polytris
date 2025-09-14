#!/usr/bin/env python3
"""
Master Script to Run All Object Detector Speed Experiments

This script runs all three object detector experiments:
- YOLOv3
- YOLOv5 
- RetinaNet

And compares their performance across different video resolutions.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Model configurations
MODELS = {
    'yolov3': {
        'script': 'test_yolov3_speed.py',
        'name': 'YOLOv3',
        'color': 'blue'
    },
    'yolov5': {
        'script': 'test_yolov5_speed.py',
        'name': 'YOLOv5s',
        'color': 'green'
    },
    'retinanet': {
        'script': 'test_retinanet_speed.py',
        'name': 'RetinaNet',
        'color': 'red'
    }
}

RESOLUTIONS = ['480p', '720p', '1080p']

def run_experiment(model_key, video_path, output_dir, visualize=False):
    """Run a single experiment for a specific model"""
    model_config = MODELS[model_key]
    script_path = Path(__file__).parent / model_config['script']
    
    print(f"\n{'='*60}")
    print(f"Running {model_config['name']} experiment")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, str(script_path),
        '--video', video_path,
        '--output-dir', output_dir
    ]
    
    if visualize:
        cmd.append('--visualize')
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {model_config['name']} experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_config['name']} experiment failed:")
        print(f"Error code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def load_results(output_dir):
    """Load results from all experiments"""
    results = {}
    
    for model_key in MODELS.keys():
        results[model_key] = {}
        for resolution in RESOLUTIONS:
            results_file = Path(output_dir) / f"{model_key}_{resolution}_speed_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results[model_key][resolution] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load results for {model_key} {resolution}: {e}")
                    results[model_key][resolution] = None
            else:
                results[model_key][resolution] = None
    
    return results

def create_comparison_plot(results, output_dir):
    """Create comparison plots of the results"""
    print("\nCreating comparison plots...")
    
    # Prepare data for plotting
    models = list(MODELS.keys())
    resolutions = RESOLUTIONS
    
    # Extract mean inference times
    data = {}
    for model_key in models:
        data[model_key] = []
        for resolution in resolutions:
            if (results.get(model_key, {}).get(resolution) and 
                'overall_stats' in results[model_key][resolution]):
                mean_time = results[model_key][resolution]['overall_stats']['mean_inference_time']
                data[model_key].append(mean_time)
            else:
                data[model_key].append(0)  # No data available
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(resolutions))
    width = 0.25
    
    for i, (model_key, times) in enumerate(data.items()):
        model_config = MODELS[model_key]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, times, width, 
                     label=model_config['name'], 
                     color=model_config['color'],
                     alpha=0.8)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            if time > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{time:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Video Resolution')
    ax.set_ylabel('Mean Inference Time (ms)')
    ax.set_title('Object Detector Speed Comparison Across Resolutions')
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / 'speed_comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    # Create detailed comparison table
    create_comparison_table(results, output_dir)
    
    plt.show()

def create_comparison_table(results, output_dir):
    """Create a detailed comparison table"""
    print("Creating comparison table...")
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Resolution', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)']
    
    for model_key in MODELS.keys():
        model_config = MODELS[model_key]
        for resolution in RESOLUTIONS:
            if (results.get(model_key, {}).get(resolution) and 
                'overall_stats' in results[model_key][resolution]):
                stats = results[model_key][resolution]['overall_stats']
                row = [
                    model_config['name'],
                    resolution,
                    f"{stats['mean_inference_time']:.2f}",
                    f"{stats['std_inference_time']:.2f}",
                    f"{stats['min_inference_time']:.2f}",
                    f"{stats['max_inference_time']:.2f}"
                ]
                table_data.append(row)
    
    # Save as CSV
    csv_path = Path(output_dir) / 'speed_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table_data:
            f.write(','.join(row) + '\n')
    
    print(f"Comparison table saved to: {csv_path}")
    
    # Print table to console
    print("\n" + "="*80)
    print("SPEED COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Resolution':<10} {'Mean (ms)':<10} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-"*80)
    
    for row in table_data:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

def generate_summary_report(results, output_dir):
    """Generate a comprehensive summary report"""
    print("\nGenerating summary report...")
    
    report_path = Path(output_dir) / 'experiment_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Object Detector Speed Experiment Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the speed performance of three object detectors:\n")
        f.write("- YOLOv3\n")
        f.write("- YOLOv5s\n")
        f.write("- RetinaNet\n\n")
        
        f.write("## Test Conditions\n\n")
        f.write("- **Video Resolutions**: 480p (854x480), 720p (1280x720), 1080p (1920x1080)\n")
        f.write("- **Metric**: Mean inference time in milliseconds\n")
        f.write("- **Test Method**: 10 inference runs per frame, ~100 frames per resolution\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Model | 480p (ms) | 720p (ms) | 1080p (ms) |\n")
        f.write("|-------|------------|------------|------------|\n")
        
        for model_key in MODELS.keys():
            model_config = MODELS[model_key]
            row = [model_config['name']]
            
            for resolution in RESOLUTIONS:
                if (results.get(model_key, {}).get(resolution) and 
                    'overall_stats' in results[model_key][resolution]):
                    mean_time = results[model_key][resolution]['overall_stats']['mean_inference_time']
                    row.append(f"{mean_time:.2f}")
                else:
                    row.append("N/A")
            
            f.write(f"| {' | '.join(row)} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find fastest model for each resolution
        for resolution in RESOLUTIONS:
            fastest_model = None
            fastest_time = float('inf')
            
            for model_key in MODELS.keys():
                if (results.get(model_key, {}).get(resolution) and 
                    'overall_stats' in results[model_key][resolution]):
                    mean_time = results[model_key][resolution]['overall_stats']['mean_inference_time']
                    if mean_time < fastest_time:
                        fastest_time = mean_time
                        fastest_model = MODELS[model_key]['name']
            
            if fastest_model:
                f.write(f"- **{resolution}**: {fastest_model} was fastest at {fastest_time:.2f} ms\n")
        
        f.write("\n## Notes\n\n")
        f.write("- All times are in milliseconds\n")
        f.write("- Results exclude video decoding and resizing overhead\n")
        f.write("- Tests performed with GPU acceleration when available\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Run all object detector speed experiments')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output-dir', default='./detection_experiments/results', 
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization videos with bounding boxes')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments and only analyze existing results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run experiments if not skipped
    if not args.skip_experiments:
        print("Starting object detector speed experiments...")
        
        # Run each experiment
        for model_key in MODELS.keys():
            success = run_experiment(model_key, args.video, output_dir, args.visualize)
            if not success:
                print(f"Warning: {MODELS[model_key]['name']} experiment failed")
    
    # Load and analyze results
    print("\nLoading experiment results...")
    results = load_results(output_dir)
    
    # Check if we have any results
    has_results = any(
        any(results.get(model_key, {}).get(resolution) for resolution in RESOLUTIONS)
        for model_key in MODELS.keys()
    )
    
    if not has_results:
        print("No results found. Please run the experiments first.")
        return
    
    # Create visualizations and reports
    create_comparison_plot(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print(f"\n🎉 All experiments completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
