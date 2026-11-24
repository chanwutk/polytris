"""TrackEval scoring shared by experiments/evaluation/p110 and profiling's evaluate stage, so both score tracks identically."""

import multiprocessing as mp
import os
import sys
import warnings

# TrackEval is vendored as a submodule, not installed; p110 adds the same path.
sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import CLEAR, HOTA, Identity

from tetris.trackeval.dataset import Dataset


def build_metric_objects(metrics_list: list[str]) -> list[HOTA | CLEAR | Identity]:
    # Collect the requested TrackEval metric instances in a deterministic order.
    metrics: list[HOTA | CLEAR | Identity] = []

    # Expand each requested metric name into the corresponding TrackEval class.
    for metric_name in metrics_list:
        # Add the HOTA metric when requested.
        if metric_name == 'HOTA':
            metrics.append(HOTA())
        # Add the CLEAR metric when requested.
        elif metric_name == 'CLEAR':
            metrics.append(CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}))
        # Add the Identity metric when requested.
        elif metric_name == 'Identity':
            metrics.append(Identity({'THRESHOLD': 0.5}))

    return metrics


def run_trackeval(dataset_config: dict, metrics: list, output_dir: str, n_sequences: int) -> dict:
    """Run TrackEval on one tetris Dataset with p110's evaluator settings; return the per-sequence results.

    The result maps each sequence (and 'COMBINED_SEQ') to TrackEval's per-class metric dicts.  Shared by
    p110 and profiling's evaluate stage so both score tracks identically.
    """
    # Build the TrackEval evaluator configuration with simple per-task execution.
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': min(mp.cpu_count(), n_sequences),
        'BREAK_ON_ERROR': True,
        'LOG_ON_ERROR': os.path.join(output_dir, 'LOG.txt'),
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False,
    }

    # Ensure the destination directory exists before TrackEval writes results.
    os.makedirs(output_dir, exist_ok=True)

    # Construct the split-aware TrackEval dataset wrapper.
    eval_dataset = Dataset(dataset_config)
    # Construct the TrackEval evaluator instance.
    evaluator = trackeval.Evaluator(eval_config)

    # Suppress noisy TrackEval warnings so failures stay visible.
    with warnings.catch_warnings():
        # Silence TrackEval warning output for deterministic logs.
        warnings.filterwarnings("ignore")
        # Run TrackEval for the configured dataset and metrics.
        results = evaluator.evaluate([eval_dataset], metrics)

    # Validate the TrackEval response structure before extracting results.
    assert results and len(results) == 2, results
    # Validate the TrackEval execution status.
    assert results[1]['Dataset']['sort'] == 'Success', f"Evaluation failed: {results[1]}"

    # Extract the per-sequence tracker results from the nested TrackEval payload.
    return results[0].get('Dataset', {}).get('sort', {})
