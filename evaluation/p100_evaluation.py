#!/usr/local/bin/python

import argparse
import os
import subprocess
import sys
from typing import List


def parse_args():
    """Parse command line arguments for evaluation script orchestration."""
    parser = argparse.ArgumentParser(
        description='Execute evaluation scripts (p111-p200) in the evaluation directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluation scripts including compute scripts
  python p100_evaluation.py

  # Skip all compute scripts
  python p100_evaluation.py --no_compute

  # Skip specific compute scripts (accuracy and throughput)
  python p100_evaluation.py --no_compute accuracy throughput

  # Skip only tradeoff compute scripts
  python p100_evaluation.py --no_compute tradeoff
        """
    )
    parser.add_argument(
        '--no_compute',
        nargs='*',
        metavar='TYPE',
        help='Skip compute scripts. If specified without arguments, skip all *_compute.py files. '
             'If arguments provided (e.g., accuracy, throughput, tradeoff), skip *_{TYPE}_compute.py files.'
    )
    return parser.parse_args()


def get_evaluation_scripts() -> List[str]:
    """
    Get all evaluation scripts from p111 to p200 in the evaluation directory.

    Returns:
        List[str]: Sorted list of script filenames in range [p111, p200]
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List all Python files in the evaluation directory
    all_files = [f for f in os.listdir(script_dir) if f.endswith('.py')]

    # Filter files in the range p111-p200
    evaluation_scripts = []
    for filename in all_files:
        # Check if filename matches pattern pXXX_*.py where XXX is a number
        if filename.startswith('p') and '_' in filename:
            try:
                # Extract the number part (e.g., "111" from "p111_accuracy_aggregate.py")
                number_str = filename.split('_')[0][1:]  # Remove 'p' prefix
                number = int(number_str)

                # Include files in range [111, 200]
                if 110 <= number <= 200:
                    evaluation_scripts.append(filename)
            except (ValueError, IndexError):
                # Skip files that don't match the expected pattern
                continue

    # Sort by number
    evaluation_scripts.sort(key=lambda f: int(f.split('_')[0][1:]))

    return evaluation_scripts


def should_skip_script(filename: str, no_compute_args: List[str]) -> bool:
    """
    Determine if a script should be skipped based on --no_compute arguments.

    Args:
        filename (str): Script filename (e.g., "p111_accuracy_aggregate.py")
        no_compute_args (List[str]): List of compute types to skip (empty list means skip all compute)

    Returns:
        bool: True if script should be skipped, False otherwise
    """
    # Check if this is a compute script
    if not '_compute.py' in filename:
        return False

    # If no_compute_args is empty list, skip all compute scripts
    if no_compute_args == []:
        return True

    # Otherwise, check if any of the specified types match
    # Extract the type from filename (e.g., "accuracy" from "p110_accuracy_compute.py")
    for compute_type in no_compute_args:
        if f'_{compute_type}_compute.py' in filename:
            return True

    return False


def execute_script(script_path: str) -> int:
    """
    Execute a Python script and return its exit code.

    Args:
        script_path (str): Absolute path to the script to execute

    Returns:
        int: Exit code from the script execution
    """
    print(f"\n{'='*80}")
    print(f"Executing: {os.path.basename(script_path)}")
    print(f"{'='*80}")

    try:
        # Execute the script using subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            check=False  # Don't raise exception on non-zero exit
        )

        if result.returncode != 0:
            print(f"WARNING: Script {os.path.basename(script_path)} exited with code {result.returncode}")
        else:
            print(f"SUCCESS: Script {os.path.basename(script_path)} completed successfully")

        return result.returncode

    except Exception as e:
        print(f"ERROR: Failed to execute {os.path.basename(script_path)}: {e}")
        return 1


def main(args):
    """
    Main function that orchestrates the execution of evaluation scripts.

    This function:
    1. Discovers all evaluation scripts in range p111-p200
    2. Filters scripts based on --no_compute arguments
    3. Executes scripts sequentially in numerical order
    4. Reports execution status and failures

    Args:
        args (argparse.Namespace): Parsed command line arguments
    """
    print("Starting evaluation pipeline execution")

    # Get all evaluation scripts
    scripts = get_evaluation_scripts()

    if not scripts:
        print("ERROR: No evaluation scripts found in range p111-p200")
        sys.exit(1)

    print(f"Found {len(scripts)} evaluation scripts")

    # Determine which scripts to skip based on --no_compute
    scripts_to_execute = []
    scripts_to_skip = []

    if args.no_compute is not None:
        # --no_compute was specified (either with or without arguments)
        for script in scripts:
            if should_skip_script(script, args.no_compute):
                scripts_to_skip.append(script)
            else:
                scripts_to_execute.append(script)

        if scripts_to_skip:
            print(f"\nSkipping {len(scripts_to_skip)} compute script(s):")
            for script in scripts_to_skip:
                print(f"  - {script}")
    else:
        # No --no_compute specified, execute all scripts
        scripts_to_execute = scripts

    if not scripts_to_execute:
        print("\nNo scripts to execute after filtering")
        return

    print(f"\nExecuting {len(scripts_to_execute)} script(s):")
    for script in scripts_to_execute:
        print(f"  - {script}")

    # Execute scripts in order
    script_dir = os.path.dirname(os.path.abspath(__file__))
    failed_scripts = []

    for script_name in scripts_to_execute:
        script_path = os.path.join(script_dir, script_name)
        exit_code = execute_script(script_path)

        if exit_code != 0:
            failed_scripts.append((script_name, exit_code))

    # Report summary
    print(f"\n{'='*80}")
    print("EVALUATION PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total scripts: {len(scripts)}")
    print(f"Executed: {len(scripts_to_execute)}")
    print(f"Skipped: {len(scripts_to_skip)}")
    print(f"Succeeded: {len(scripts_to_execute) - len(failed_scripts)}")
    print(f"Failed: {len(failed_scripts)}")

    if failed_scripts:
        print("\nFailed scripts:")
        for script_name, exit_code in failed_scripts:
            print(f"  - {script_name} (exit code: {exit_code})")
        sys.exit(1)
    else:
        print("\nAll scripts executed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main(parse_args())
