#!/usr/local/bin/python

import argparse
import os
import subprocess
import sys
from typing import List

# Scripts that are unconditionally excluded from the test evaluation pipeline.
ALWAYS_SKIP_NUMBERS = {131, 132, 135}

# Range of script numbers disabled when --no-sota is passed.
SOTA_RANGE = range(140, 151)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the test evaluation pipeline."
    )
    parser.add_argument(
        "--no-sota",
        action="store_true",
        default=False,
        help="Skip SOTA recomputation scripts (p140–p150).",
    )
    return parser.parse_args()


def get_evaluation_scripts() -> List[str]:
    """
    Get all evaluation scripts from p110 to p201 in the evaluation directory.

    Returns:
        List[str]: Sorted list of script filenames in range [p110, p201]
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List all Python files in the evaluation directory
    all_files = [f for f in os.listdir(script_dir) if f.endswith('.py')]

    # Filter files in the range p110-p201
    evaluation_scripts = []
    for filename in all_files:
        # Check if filename matches pattern pXXX_*.py where XXX is a number
        if filename.startswith('p') and '_' in filename:
            try:
                # Extract the number part (e.g., "111" from "p111_accuracy_aggregate.py")
                number_str = filename.split('_')[0][1:]  # Remove 'p' prefix
                number = int(number_str)

                # Include files in range [110, 201], skipping permanently excluded numbers.
                if 110 <= number <= 201 and number not in ALWAYS_SKIP_NUMBERS:
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
    # Print a section header so each script boundary is easy to spot in the log.
    print(f"\n{'='*80}")
    print(f"Executing: {os.path.basename(script_path)}")
    print(f"{'='*80}")

    try:
        # Execute the script using subprocess, forwarding any extra CLI flags.
        result = subprocess.run(
            [sys.executable, script_path, '--test'],
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


def main():
    args = parse_args()

    print("Starting evaluation pipeline execution")

    # Discover all evaluation scripts in the configured range.
    scripts = get_evaluation_scripts()

    # Drop SOTA scripts (p140–p150) when --no-sota is requested.
    if args.no_sota:
        scripts = [
            f for f in scripts
            if int(f.split('_')[0][1:]) not in SOTA_RANGE
        ]

    if not scripts:
        print("ERROR: No evaluation scripts found in range p111-p201")
        sys.exit(1)

    print(f"Found {len(scripts)} evaluation scripts")

    if args.no_sota:
        print("NOTE: SOTA scripts (p140–p150) are disabled via --no-sota")

    print(f"\nExecuting {len(scripts)} script(s):")
    for script in scripts:
        print(f"  - {script}")

    # Execute scripts in order.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    failed_scripts = []

    for script_name in scripts:
        script_path = os.path.join(script_dir, script_name)
        exit_code = execute_script(script_path)

        if exit_code != 0:
            failed_scripts.append((script_name, exit_code))

    # Report summary
    print(f"\n{'='*80}")
    print("EVALUATION PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total scripts: {len(scripts)}")
    print(f"Executed: {len(scripts)}")
    print(f"Succeeded: {len(scripts) - len(failed_scripts)}")
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
    main()
