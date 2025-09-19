#!/usr/local/bin/python

import argparse
import subprocess
import sys
import os
import time
from typing import List, Dict, Optional

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """Parse command line arguments for the script runner."""
    parser = argparse.ArgumentParser(description='Run all scripts in the polyis pipeline')
    parser.add_argument('--dataset', type=str, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing steps (000-002)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip tuning steps (010-015)')
    parser.add_argument('--skip-execution', action='store_true',
                        help='Skip execution steps (020-060)')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip analysis steps (070-080)')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization steps (003, 022, 031, 081, 090)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing where available')
    parser.add_argument('--parallel-visualization', action='store_true',
                        help='Run visualization scripts in parallel (faster but uses more resources)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be executed without running')
    parser.add_argument('--start-from', type=str,
                        help='Start from a specific script (e.g., "020_exec_classify")')
    parser.add_argument('--stop-at', type=str,
                        help='Stop at a specific script (e.g., "060_exec_track")')
    return parser.parse_args()


class ScriptRunner:
    """Manages execution of the polyis pipeline scripts."""
    
    def __init__(self, args):
        self.args = args
        self.root_dir = '/polyis/'
        self.failed_scripts = []
        self.completed_scripts = []
        
        # Define the pipeline stages and their scripts
        self.pipeline_stages = {
            'preprocessing': [
                ('p000_preprocess_dataset.py', 'Preprocess raw video dataset'),
                ('p001_preprocess_groundtruth_detection.py', 'Preprocess groundtruth detection data'),
                ('p002_preprocess_groundtruth_tracking.py', 'Preprocess groundtruth tracking data'),
                ('p004_preprocess_train_detectors.py', 'Preprocess training data for detectors'),
            ],
            'tuning': [
                ('p010_tune_segment_videos.py', 'Tune video segmentation parameters'),
                ('p011_tune_detect.py', 'Tune object detection parameters'),
                ('p012_tune_create_training_data.py', 'Create training data for classifiers'),
                ('p013_tune_train_classifier.py', 'Train and tune classifiers'),
                ('p014_tune_select_classifier.py', 'Select best classifier configuration'),
                ('p015_tune_regulate_tracking.py', 'Tune tracking regulation parameters'),
            ],
            'execution': [
                ('p020_exec_classify.py', 'Execute classification on video segments'),
                ('p021_exec_classify_correct.py', 'Correct classification results'),
                ('p023_exec_classify_render.py', 'Render classification results'),
                ('p024_exec_classify_tradeoff.py', 'Analyze classification tradeoffs'),
                ('p030_exec_compress.py', 'Compress relevant video segments'),
                ('p040_exec_detect.py', 'Execute object detection on compressed segments'),
                ('p050_exec_uncompress.py', 'Uncompress detection results'),
                ('p060_exec_track.py', 'Execute object tracking'),
            ],
            'analysis': [
                ('p070_accuracy_compute.py', 'Compute tracking accuracy statistics'),
                ('p080_throughput_gather.py', 'Gather throughput performance data'),
                ('p081_throughput_compute.py', 'Compute throughput performance metrics'),
            ],
            'visualization': [
                ('p003_preprocess_groundtruth_visualize.py', 'Visualize groundtruth data'),
                ('p022_exec_classify_visualize.py', 'Visualize classification results'),
                ('p031_exec_compress_visualize.py', 'Visualize compression results'),
                ('p071_accuracy_visualize.py', 'Visualize accuracy analysis'),
                ('p082_throughput_visualize.py', 'Visualize throughput analysis'),
                ('p090_results_track_visualize.py', 'Visualize tracking results'),
            ]
        }
    
    def get_script_path(self, script_name: str) -> str:
        """Get full path to a script."""
        return os.path.join(self.root_dir, script_name)
    
    def script_exists(self, script_name: str) -> bool:
        """Check if a script file exists."""
        return os.path.exists(self.get_script_path(script_name))
    
    def should_skip_stage(self, stage: str) -> bool:
        """Check if a stage should be skipped based on arguments."""
        skip_flags = {
            'preprocessing': self.args.skip_preprocessing,
            'tuning': self.args.skip_tuning,
            'execution': self.args.skip_execution,
            'analysis': self.args.skip_analysis,
            'visualization': self.args.skip_visualization,
        }
        return skip_flags.get(stage, False)
    
    def should_run_script(self, script_name: str) -> bool:
        """Check if a script should be run based on start/stop criteria."""
        if self.args.start_from:
            # Extract script number for comparison
            start_num = self.args.start_from.split('_')[0]
            script_num = script_name.split('_')[0]
            if script_num < start_num:
                return False
        
        if self.args.stop_at:
            # Extract script number for comparison
            stop_num = self.args.stop_at.split('_')[0]
            script_num = script_name.split('_')[0]
            if script_num > stop_num:
                return False
        
        return True
    
    def build_command(self, script_name: str) -> List[str]:
        """Build command to execute a script with appropriate arguments."""
        script_path = self.get_script_path(script_name)
        cmd = ['python', script_path, '--dataset', self.args.dataset]
        
        # # Add parallel flag for scripts that support it
        # parallel_scripts = [
        #     '070_results_statistics_acc.py',
        # ]
        
        # if self.args.parallel and script_name in parallel_scripts:
        #     cmd.append('--parallel')
        
        return cmd
    
    def run_script(self, script_name: str, description: str) -> bool:
        script_name = 'scripts/' + script_name

        """Run a single script and return success status."""
        if not self.script_exists(script_name):
            print(f"⚠️  Script not found: {script_name}")
            return False
        
        if not self.should_run_script(script_name):
            print(f"⏭️  Skipping {script_name} (outside start/stop range)")
            return True
        
        cmd = self.build_command(script_name)
        
        print(f"\n{'='*60}")
        print(f"🚀 Running: {script_name}")
        print(f"📝 Description: {description}")
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        if self.args.dry_run:
            print("🔍 DRY RUN: Command would be executed")
            return True
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Success: {script_name} completed in {elapsed:.2f}s")
            
            # Print stdout if there's useful information
            if result.stdout.strip():
                print(f"📄 Output preview (last 10 lines):")
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"   {line}")
            
            self.completed_scripts.append((script_name, elapsed))
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: {script_name} failed after {elapsed:.2f}s")
            print(f"💥 Error code: {e.returncode}")
            
            # Print stderr for debugging
            if e.stderr:
                print(f"🚨 Error output (last 10 lines):")
                lines = e.stderr.strip().split('\n')
                for line in lines[-10:]:
                    print(f"   {line}")
            
            self.failed_scripts.append((script_name, e.returncode, elapsed))
            return False
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"💥 Unexpected error in {script_name}: {e}")
            self.failed_scripts.append((script_name, -1, elapsed))
            return False
    
    def run_stage(self, stage_name: str, scripts: List[tuple]) -> bool:
        """Run all scripts in a stage."""
        if self.should_skip_stage(stage_name):
            print(f"\n⏭️  Skipping {stage_name.upper()} stage")
            return True
        
        print(f"\n🎯 Starting {stage_name.upper()} stage")
        print(f"📊 Scripts to run: {len(scripts)}")
        
        # Special handling for visualization stage - can run in parallel
        if stage_name == 'visualization' and self.args.parallel_visualization:
            return self.run_visualization_stage_parallel(scripts)
        
        # Regular sequential execution for other stages
        stage_success = True
        for script_name, description in scripts:
            success = self.run_script(script_name, description)
            if not success:
                print(f"⚠️  Stage {stage_name} failed at {script_name}")
                stage_success = False
                break
        
        if stage_success:
            print(f"✅ {stage_name.upper()} stage completed successfully")
        else:
            print(f"⚠️  {stage_name.upper()} stage completed with some failures")
        
        return stage_success
    
    def run_visualization_stage_parallel(self, scripts: List[tuple]) -> bool:
        """Run visualization scripts in parallel using multiprocessing."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        print(f"🔄 Running {len(scripts)} visualization scripts in parallel")
        
        # Filter scripts that exist and should be run
        valid_scripts = []
        for script_name, description in scripts:
            if self.script_exists(script_name) and self.should_run_script(script_name):
                valid_scripts.append((script_name, description))
            elif not self.script_exists(script_name):
                print(f"⚠️  Script not found: {script_name}")
            elif not self.should_run_script(script_name):
                print(f"⏭️  Skipping {script_name} (outside start/stop range)")
        
        if not valid_scripts:
            print("📭 No valid visualization scripts to run")
            return True
        
        if self.args.dry_run:
            print("🔍 DRY RUN: Would run these scripts in parallel:")
            for script_name, description in valid_scripts:
                cmd = self.build_command(script_name)
                print(f"   💻 {script_name}: {' '.join(cmd)}")
            return True
        
        # Use ProcessPoolExecutor for better control and error handling
        max_workers = min(len(valid_scripts), mp.cpu_count())
        print(f"👥 Using {max_workers} parallel workers")
        
        stage_success = True
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scripts
            future_to_script = {}
            for script_name, description in valid_scripts:
                cmd = self.build_command(script_name)
                future = executor.submit(self._run_script_subprocess, script_name, description, cmd)
                future_to_script[future] = (script_name, description)
            
            # Collect results as they complete
            for future in as_completed(future_to_script):
                script_name, description = future_to_script[future]
                try:
                    success, elapsed = future.result()
                    if success:
                        print(f"✅ Parallel success: {script_name} completed in {elapsed:.2f}s")
                        self.completed_scripts.append((script_name, elapsed))
                    else:
                        print(f"❌ Parallel failure: {script_name} failed after {elapsed:.2f}s")
                        stage_success = False
                except Exception as e:
                    print(f"💥 Parallel error in {script_name}: {e}")
                    self.failed_scripts.append((script_name, -1, 0))
                    stage_success = False
        
        total_elapsed = time.time() - start_time
        if stage_success:
            print(f"✅ VISUALIZATION stage completed successfully in {total_elapsed:.2f}s")
        else:
            print(f"⚠️  VISUALIZATION stage completed with some failures in {total_elapsed:.2f}s")
        
        return stage_success
    
    def _run_script_subprocess(self, script_name: str, description: str, cmd: List[str]) -> tuple:
        """Helper method to run a script in a subprocess (for parallel execution)."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                check=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_time
            return True, elapsed
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.failed_scripts.append((script_name, e.returncode, elapsed))
            return False, elapsed
    
    def print_summary(self):
        """Print execution summary."""
        print(f"\n{'='*80}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        total_time = sum(elapsed for _, elapsed in self.completed_scripts)
        print(f"✅ Completed scripts: {len(self.completed_scripts)}")
        print(f"❌ Failed scripts: {len(self.failed_scripts)}")
        print(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f}m)")
        
        if self.completed_scripts:
            print(f"\n✅ SUCCESSFUL SCRIPTS:")
            for script_name, elapsed in self.completed_scripts:
                print(f"   {script_name:<40} {elapsed:>8.2f}s")
        
        if self.failed_scripts:
            print(f"\n❌ FAILED SCRIPTS:")
            for script_name, error_code, elapsed in self.failed_scripts:
                print(f"   {script_name:<40} (code: {error_code:>3}) {elapsed:>8.2f}s")
        
        print(f"\n{'='*80}")
    
    def run_all(self):
        """Run all pipeline stages."""
        print(f"🚀 Starting polyis pipeline execution")
        print(f"📁 Dataset: {self.args.dataset}")
        print(f"🔧 Parallel processing: {'enabled' if self.args.parallel else 'disabled'}")
        print(f"🎨 Parallel visualization: {'enabled' if self.args.parallel_visualization else 'disabled'}")
        print(f"🔍 Dry run: {'yes' if self.args.dry_run else 'no'}")
        
        if self.args.start_from:
            print(f"▶️  Starting from: {self.args.start_from}")
        if self.args.stop_at:
            print(f"⏹️  Stopping at: {self.args.stop_at}")
        
        overall_start = time.time()
        
        # Run each stage in order
        for stage_name, scripts in self.pipeline_stages.items():
            self.run_stage(stage_name, scripts)
        
        overall_elapsed = time.time() - overall_start
        print(f"\n🏁 Pipeline execution completed in {overall_elapsed:.2f}s ({overall_elapsed/60:.1f}m)")
        
        self.print_summary()
        
        # Return exit code based on failures
        return 0 if not self.failed_scripts else 1


def main():
    """Main function."""
    args = parse_args()
    runner = ScriptRunner(args)
    exit_code = runner.run_all()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
