#!/usr/local/bin/python

import os
import subprocess
import sys
from typing import List


def get_evaluation_scripts() -> List[str]:
    """
    Get all evaluation scripts from p110 to p135 in the evaluation directory.

    Returns:
        List[str]: Sorted list of script filenames in range [p110, p135]
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List all Python files in the evaluation directory
    all_files = [f for f in os.listdir(script_dir) if f.endswith('.py')]

    # Filter files in the range p110-p135
    evaluation_scripts = []
    for filename in all_files:
        # Check if filename matches pattern pXXX_*.py where XXX is a number
        if filename.startswith('p') and '_' in filename:
            try:
                # Extract the number part (e.g., "111" from "p111_accuracy_aggregate.py")
                number_str = filename.split('_')[0][1:]  # Remove 'p' prefix
                number = int(number_str)

                # Include files in range [111, 135]
                if 110 <= number <= 135:
                    if 'visualize' not in filename:
                        evaluation_scripts.append(filename)
            except (ValueError, IndexError):
                # Skip files that don't match the expected pattern
                continue

    # Sort by number
    evaluation_scripts.sort(key=lambda f: int(f.split('_')[0][1:]))

    return evaluation_scripts


def execute_script(script_path: str) -> int:
    # Print a section header so each script boundary is easy to spot in the log.
    print(f"\n{'='*80}")
    print(f"Executing: {os.path.basename(script_path)}")
    print(f"{'='*80}")

    try:
        # Execute the script using subprocess, forwarding any extra CLI flags.
        result = subprocess.run(
            [sys.executable, script_path, '--valid'],
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
    print("Starting evaluation pipeline execution")

    # Discover all evaluation scripts in the configured range.
    scripts = get_evaluation_scripts()

    if not scripts:
        print("ERROR: No evaluation scripts found in range p111-p201")
        sys.exit(1)

    print(f"Found {len(scripts)} evaluation scripts")

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
