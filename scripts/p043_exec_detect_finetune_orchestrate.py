#!/usr/local/bin/python
"""
Orchestrate fine-tuning training across all dataset/tilesize/tilepadding combinations.

This script creates a tmux session with multiple windows (one per GPU) and distributes
training tasks across them. Each window runs p042_exec_detect_finetune.py for assigned
combinations, with multiple tasks chained using &&.

Usage:
    ./run scripts/p043_exec_detect_finetune_orchestrate.py [--epochs N] [--batch N] [--device DEVICE]
"""

import argparse
import random
import string
import subprocess
import sys
from pathlib import Path

import torch

from polyis.utilities import get_config

# Import discovery function from p042
# Add scripts directory to path for import
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


CONFIG = get_config()
CACHE_DIR = Path(CONFIG['DATA']['CACHE_DIR'])
SCRIPT_DIR = Path(__file__).parent
P042_SCRIPT = SCRIPT_DIR / "p042_exec_detect_finetune.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Orchestrate fine-tuning training across all datasets')
    parser.add_argument('--epochs', type=int, default=13,
                        help='Training epochs (default: 13)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (1 for auto, default: 1)')
    parser.add_argument('--devices', type=str, required=True,
                        help='Comma-separated device IDs to use (e.g., "0,1,2"). Overrides auto-assignment.')
    parser.add_argument('--session-name', type=str, default=None,
                        help='Tmux session name (default: random)')
    return parser.parse_args()


def get_tasks_from_config(config: dict, cache_dir: Path) -> list[tuple[str, int, str]]:
    """
    Get training tasks from config file, verifying datasets exist.
    
    Args:
        config: Configuration dictionary
        cache_dir: Cache directory path
        
    Returns:
        List of (dataset, tilesize, tilepadding) tuples that exist
    """
    tasks = []
    
    datasets = config['EXEC']['DATASETS']
    tile_sizes = config['EXEC']['TILE_SIZES']
    tilepadding_modes = config['EXEC']['TILEPADDING_MODES']
    
    for dataset in datasets:
        for tilesize in tile_sizes:
            for tilepadding in tilepadding_modes:
                # Check if the dataset directory exists
                dataset_dir = cache_dir / dataset / "finetune" / f"{tilesize}_{tilepadding}"
                if dataset_dir.exists():
                    tasks.append((dataset, tilesize, tilepadding))
                else:
                    print(f"Warning: Dataset directory not found: {dataset_dir}, skipping...")
    
    return tasks


def generate_session_name() -> str:
    """Generate a random tmux session name."""
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"finetune_{random_suffix}"


def get_num_gpus() -> int:
    """Get the number of available GPUs."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using 1 GPU (CPU mode)")
        return 1
    return torch.cuda.device_count()


def run_tmux_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a tmux command.
    
    Args:
        cmd: Command list (will be prefixed with 'tmux')
        check: Whether to check return code
        
    Returns:
        CompletedProcess result
    """
    full_cmd = ['tmux'] + cmd
    return subprocess.run(full_cmd, check=check, capture_output=True, text=True)


def create_tmux_session(session_name: str, num_windows: int) -> None:
    """
    Create a tmux session with N windows.
    
    Args:
        session_name: Name of the tmux session
        num_windows: Number of windows to create
    """
    # Create new session with first window (detached)
    run_tmux_command(['new-session', '-d', '-s', session_name])
    
    # Create additional windows
    for i in range(1, num_windows):
        run_tmux_command(['new-window', '-t', session_name])
    
    print(f"Created tmux session '{session_name}' with {num_windows} windows")


def build_training_command(
    dataset: str,
    tilesize: int,
    tilepadding: str,
    gpu_id: int,
    epochs: int,
    batch: int,
) -> str:
    """
    Build a command string to run p042_exec_detect_finetune.py.
    
    Args:
        dataset: Dataset name
        tilesize: Tile size
        tilepadding: Tile padding mode
        gpu_id: GPU ID to use
        epochs: Number of training epochs
        batch: Batch size
        
    Returns:
        Command string
    """
    
    # Use absolute path to ensure it works from any directory
    script_path = P042_SCRIPT.resolve()
    
    cmd_parts = [
        'python',
        str(script_path),
        '--dataset', dataset,
        '--tilesize', str(tilesize),
        '--tilepadding', tilepadding,
        '--epochs', str(epochs),
        '--batch', str(batch),
        '--device', str(gpu_id),
    ]
    
    return ' '.join(cmd_parts)


def distribute_tasks(
    tasks: list[tuple[str, int, str]],
    num_gpus: int
) -> list[list[tuple[str, int, str]]]:
    """
    Distribute tasks across GPUs using round-robin.
    
    Args:
        tasks: List of (dataset, tilesize, tilepadding) tuples
        num_gpus: Number of GPUs/windows
        
    Returns:
        List of task lists, one per GPU
    """
    gpu_tasks: list[list[tuple[str, int, str]]] = [[] for _ in range(num_gpus)]
    
    for i, task in enumerate(tasks):
        gpu_id = i % num_gpus
        gpu_tasks[gpu_id].append(task)
    
    return gpu_tasks


def send_command_to_window(session_name: str, window_id: int, command: str) -> None:
    """
    Send a command to a specific tmux window.
    
    Args:
        session_name: Tmux session name
        window_id: Window index (0-based)
        command: Command to execute
    """
    # Use send-keys to send the command
    run_tmux_command([
        'send-keys', '-t', f'{session_name}:{window_id}',
        command,
        'Enter'
    ])


def main():
    """Main function to orchestrate training."""
    args = parse_args()
    
    # Get training tasks from config
    print("Getting training tasks from config...")
    print(f"  Datasets: {CONFIG['EXEC']['DATASETS']}")
    print(f"  Tile sizes: {CONFIG['EXEC']['TILE_SIZES']}")
    print(f"  Tile padding modes: {CONFIG['EXEC']['TILEPADDING_MODES']}")
    
    all_tasks = get_tasks_from_config(CONFIG, CACHE_DIR)
    
    if not all_tasks:
        print("No valid fine-tuning datasets found. Please run p041_exec_detect_finetune_dataset.py first.")
        return
    
    print(f"\nFound {len(all_tasks)} valid dataset combinations:")
    for dataset, tilesize, tilepadding in all_tasks:
        print(f"  {dataset} / tilesize={tilesize} / tilepadding={tilepadding}")
    
    # Get number of GPUs
    num_gpus = get_num_gpus()
    print(f"\nAvailable GPUs: {num_gpus}")

    all_devices: list[int] = list(map(int, args.devices.split(',')))
    assert len(all_devices) <= num_gpus, f"Number of devices {len(all_devices)} does not match number of GPUs {num_gpus}"

    for device in all_devices:
        print(f"Device {device}")
        assert device < num_gpus, f"Device {device} is out of range"
    
    # Generate session name
    session_name = args.session_name or generate_session_name()
    print(f"Tmux session name: {session_name}")
    
    # Create tmux session
    create_tmux_session(session_name, num_gpus)
    
    # Distribute tasks across GPUs
    gpu_tasks = distribute_tasks(all_tasks, len(all_devices))
    
    print(f"\nTask distribution:")
    for device_id, tasks in enumerate(gpu_tasks):
        print(f"  Device {all_devices[device_id]}: {len(tasks)} tasks")
    
    # Build and send commands to each window
    print(f"\nSending commands to tmux windows...")
    for device_id, tasks in enumerate(gpu_tasks):
        if not tasks:
            continue
        
        # Build command chain with &&
        commands = []
        for dataset, tilesize, tilepadding in tasks:
            cmd = build_training_command(
                dataset, tilesize, tilepadding,
                all_devices[device_id], args.epochs, args.batch
            )
            commands.append(cmd)
        
        # Chain commands with &&
        full_command = ' && '.join(commands)
        
        # Send to window
        send_command_to_window(session_name, all_devices[device_id], full_command)
        print(f"  Window {device_id} (Device {all_devices[device_id]}): {len(tasks)} tasks queued")
    
    print(f"\n✓ All commands sent to tmux session '{session_name}'")
    print(f"\nTo attach to the session, run:")
    print(f"  tmux attach -t {session_name}")
    print(f"\nTo list windows, run:")
    print(f"  tmux list-windows -t {session_name}")
    print(f"\nTo kill the session, run:")
    print(f"  tmux kill-session -t {session_name}")


if __name__ == '__main__':
    main()

