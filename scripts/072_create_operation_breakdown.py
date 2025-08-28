#!/usr/local/bin/python

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict, List

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create bar chart visualization breaking down runtime of each operation')
    parser.add_argument('--results_file', type=str, 
                        default='pipeline-stages/all-speed-results/detailed_pipeline_results.json',
                        help='Path to the detailed pipeline results JSON file')
    parser.add_argument('--output_dir', type=str, 
                        default='pipeline-stages/operation-breakdown',
                        help='Output directory for visualizations')
    parser.add_argument('--show_percentages', action='store_true',
                        help='Show percentages in addition to absolute times')
    return parser.parse_args()


def extract_operation_data(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract operation-level timing data from pipeline results.
    
    Returns:
        Dict mapping stage names to operation names to lists of times
    """
    operation_data = {}
    
    for result in results:
        if not result.get('success', False) or 'stages' not in result:
            continue
            
        video_name = result['video_name']
        
        for stage_name, stage_data in result['stages'].items():
            if not stage_data.get('success', False):
                continue
                
            if stage_name not in operation_data:
                operation_data[stage_name] = {}
            
            # Extract operation-specific timings based on stage
            if stage_name == 'tune_detect':
                # Detection stage has read and detect operations
                operations = {
                    'read': stage_data.get('read_total', 0),
                    'detect': stage_data.get('detect_total', 0)
                }
            elif stage_name == 'classify':
                # Classification stage has transform and inference operations
                operations = {
                    'transform': stage_data.get('transform_total', 0),
                    'inference': stage_data.get('inference_total', 0)
                }
            elif stage_name == 'train_classifier':
                # Training stage has train and test operations
                operations = {
                    'train': stage_data.get('total_train_time', 0),
                    'test': stage_data.get('total_test_time', 0)
                }
            else:
                # For other stages, use total time as single operation
                operations = {
                    'total': stage_data.get('total_time', 0)
                }
            
            # Add operations to the data structure
            for op_name, op_time in operations.items():
                if op_name not in operation_data[stage_name]:
                    operation_data[stage_name][op_name] = []
                operation_data[stage_name][op_name].append(op_time)
    
    return operation_data


def create_operation_breakdown_chart(operation_data: Dict[str, Dict[str, List[float]]], 
                                   output_dir: str, show_percentages: bool = False):
    """Create comprehensive operation breakdown visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall operation breakdown across all stages
    create_overall_breakdown(operation_data, output_dir, show_percentages)
    
    # 2. Stage-by-stage detailed breakdown
    create_stage_breakdowns(operation_data, output_dir, show_percentages)
    
    # 3. Operation comparison across videos
    create_operation_comparison(operation_data, output_dir)
    
    # 4. Create summary data
    create_operation_summary(operation_data, output_dir)


def create_overall_breakdown(operation_data: Dict[str, Dict[str, List[float]]], 
                           output_dir: str, show_percentages: bool):
    """Create overall operation breakdown chart."""
    
    # Aggregate all operations across stages
    all_operations = {}
    total_pipeline_time = 0
    
    for stage_name, stage_ops in operation_data.items():
        for op_name, op_times in stage_ops.items():
            full_op_name = f"{stage_name}_{op_name}"
            op_total = sum(op_times)
            all_operations[full_op_name] = op_total
            total_pipeline_time += op_total
    
    # Sort operations by total time
    sorted_operations = sorted(all_operations.items(), key=lambda x: x[1], reverse=True)
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Absolute times
    op_names = [op[0] for op in sorted_operations]
    op_times = [op[1] / 1000 for op in sorted_operations]  # Convert to seconds
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(op_names)))
    bars1 = ax1.bar(range(len(op_names)), op_times, color=colors, alpha=0.8)
    
    ax1.set_title('Total Runtime by Operation (Absolute)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Operation', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_xticks(range(len(op_names)))
    ax1.set_xticklabels(op_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, time in zip(bars1, op_times):
        height = bar.get_height()
        if height > max(op_times) * 0.01:  # Only label significant bars
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(op_times)*0.01,
                    f'{height:.0f}s', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Percentage breakdown
    if show_percentages and total_pipeline_time > 0:
        op_percentages = [(op[1] / total_pipeline_time) * 100 for op in sorted_operations]
        
        bars2 = ax2.bar(range(len(op_names)), op_percentages, color=colors, alpha=0.8)
        
        ax2.set_title('Runtime by Operation (Percentage)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Operation', fontsize=12)
        ax2.set_ylabel('Percentage of Total Time (%)', fontsize=12)
        ax2.set_xticks(range(len(op_names)))
        ax2.set_xticklabels(op_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add percentage labels
        for bar, pct in zip(bars2, op_percentages):
            height = bar.get_height()
            if height > 1:  # Only label operations > 1%
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Enable --show_percentages\nfor percentage view', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Percentage View (Not Enabled)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_operation_breakdown.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_stage_breakdowns(operation_data: Dict[str, Dict[str, List[float]]], 
                          output_dir: str, show_percentages: bool):
    """Create individual breakdown charts for each stage."""
    
    num_stages = len(operation_data)
    if num_stages == 0:
        return
    
    # Calculate subplot grid
    cols = min(3, num_stages)
    rows = (num_stages + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_stages == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = list(axes)
    elif rows > 1 and cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    fig.suptitle('Operation Breakdown by Pipeline Stage', fontsize=16, fontweight='bold')
    
    for idx, (stage_name, stage_ops) in enumerate(operation_data.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Calculate statistics for this stage
        op_names = list(stage_ops.keys())
        op_means = [np.mean(times) / 1000 for times in stage_ops.values()]  # Convert to seconds
        op_stds = [np.std(times) / 1000 for times in stage_ops.values()]
        
        # Create bar chart with error bars
        colors = plt.cm.Set2(np.linspace(0, 1, len(op_names)))
        bars = ax.bar(range(len(op_names)), op_means, yerr=op_stds, 
                     color=colors, alpha=0.8, capsize=5)
        
        ax.set_title(f'{stage_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Operation', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_xticks(range(len(op_names)))
        ax.set_xticklabels(op_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, mean_time, std_time in zip(bars, op_means, op_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_time + max(op_means)*0.02,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for idx in range(num_stages, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage_operation_breakdown.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_operation_comparison(operation_data: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Create comparison charts showing operation performance across videos."""
    
    # Create a comprehensive comparison chart
    all_data = []
    
    for stage_name, stage_ops in operation_data.items():
        for op_name, op_times in stage_ops.items():
            for i, time_val in enumerate(op_times):
                all_data.append({
                    'stage': stage_name,
                    'operation': op_name,
                    'video_index': i,
                    'time_seconds': time_val / 1000,
                    'full_operation': f"{stage_name}_{op_name}"
                })
    
    if not all_data:
        return
    
    df = pd.DataFrame(all_data)
    
    # Save detailed data to CSV
    df.to_csv(os.path.join(output_dir, 'operation_breakdown_data.csv'), index=False)
    
    # Create violin plot showing distribution of operation times
    fig, ax = plt.subplots(figsize=(15, 8))
    
    unique_operations = df['full_operation'].unique()
    positions = range(len(unique_operations))
    
    violin_data = [df[df['full_operation'] == op]['time_seconds'].values for op in unique_operations]
    
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    
    # Color the violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_operations)))
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_title('Operation Time Distribution Across Videos', fontsize=16, fontweight='bold')
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_operations, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'operation_time_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_operation_summary(operation_data: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Create summary statistics and save to text file."""
    
    summary_file = os.path.join(output_dir, 'operation_breakdown_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Operation Runtime Breakdown Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_pipeline_time = 0
        stage_totals = {}
        
        # Calculate totals
        for stage_name, stage_ops in operation_data.items():
            stage_total = 0
            for op_name, op_times in stage_ops.items():
                op_total = sum(op_times)
                stage_total += op_total
                total_pipeline_time += op_total
            stage_totals[stage_name] = stage_total
        
        f.write(f"Total Pipeline Time: {total_pipeline_time/1000:.2f} seconds\n\n")
        
        # Stage-by-stage breakdown
        for stage_name, stage_ops in operation_data.items():
            f.write(f"{stage_name.upper().replace('_', ' ')}:\n")
            f.write("-" * 30 + "\n")
            
            stage_total = stage_totals[stage_name]
            stage_pct = (stage_total / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
            f.write(f"  Stage Total: {stage_total/1000:.2f}s ({stage_pct:.1f}% of pipeline)\n")
            
            for op_name, op_times in stage_ops.items():
                op_total = sum(op_times)
                op_mean = np.mean(op_times)
                op_std = np.std(op_times)
                op_pct = (op_total / stage_total) * 100 if stage_total > 0 else 0
                
                f.write(f"    {op_name}: {op_total/1000:.2f}s total, {op_mean/1000:.3f}s avg ± {op_std/1000:.3f}s ({op_pct:.1f}% of stage)\n")
            f.write("\n")
        
        # Overall operation ranking
        f.write("Operations Ranked by Total Time:\n")
        f.write("-" * 35 + "\n")
        
        all_operations = []
        for stage_name, stage_ops in operation_data.items():
            for op_name, op_times in stage_ops.items():
                all_operations.append((f"{stage_name}_{op_name}", sum(op_times)))
        
        all_operations.sort(key=lambda x: x[1], reverse=True)
        
        for i, (op_full_name, op_total) in enumerate(all_operations, 1):
            op_pct = (op_total / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
            f.write(f"  {i:2d}. {op_full_name}: {op_total/1000:.2f}s ({op_pct:.1f}%)\n")
    
    print(f"Operation breakdown summary saved to: {summary_file}")


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Loading pipeline results from: {args.results_file}")
    
    # Load the results
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found")
        return
    
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} video results")
    
    # Extract operation data
    operation_data = extract_operation_data(results)
    
    if not operation_data:
        print("No operation data found in results")
        return
    
    print(f"Found operation data for stages: {list(operation_data.keys())}")
    
    # Create visualizations
    print(f"Creating operation breakdown visualizations in: {args.output_dir}")
    create_operation_breakdown_chart(operation_data, args.output_dir, args.show_percentages)
    
    print("Operation breakdown visualization complete!")
    print(f"Generated files:")
    print(f"  - overall_operation_breakdown.png")
    print(f"  - stage_operation_breakdown.png") 
    print(f"  - operation_time_distribution.png")
    print(f"  - operation_breakdown_data.csv")
    print(f"  - operation_breakdown_summary.txt")


if __name__ == '__main__':
    main()
