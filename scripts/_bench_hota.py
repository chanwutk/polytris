#!/usr/local/bin/python
"""Compare HOTA score between scripts/p060 output and execution/main.py output.

Reads tracking.jsonl from each path (the user is expected to have backed up
the scripts/ output to tracking.scripts.jsonl before re-running execution).

Usage:
  python scripts/_bench_hota.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import HOTA

from polyis.io import cache
from polyis.trackeval.dataset import Dataset


COMBO_PARAM_STR = 'ShuffleNet05_60_16_040_r050_bl_s100_sortcython'
DATASET = 'caldot2-y05'
VIDEOSET = 'valid'


def _videos() -> list[str]:
    videoset_dir = f'/polyis-data/datasets/{DATASET}/{VIDEOSET}'
    return sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )


def run_hota_for_variant(label: str, tracking_filename: str) -> dict:
    """Run TrackEval HOTA on tracking output stored as ``tracking_filename``.

    The ground-truth file is ``002_naive/tracking.jsonl`` (the valid-split
    pseudo-groundtruth used by evaluation/p110).
    """
    videos = _videos()
    track_execution_dir = str(cache.execution(DATASET))
    input_track = os.path.join('060_uncompressed_tracks', COMBO_PARAM_STR, tracking_filename)
    input_gt = os.path.join('002_naive', 'tracking.jsonl')

    # Validate inputs exist for every video.
    for v in videos:
        for rel in (input_track, input_gt):
            p = os.path.join(track_execution_dir, v, rel)
            if not os.path.exists(p):
                print(f"  MISSING {p}")
                return {}

    dataset_config = {
        'output_fol': '/tmp/_bench_hota',
        'output_sub_fol': f'{VIDEOSET}_{COMBO_PARAM_STR}_{label}',
        'input_gt': input_gt,
        'input_track': input_track,
        'skip': 1,
        'tracker': COMBO_PARAM_STR,
        'seq_list': videos,
        'input_dir': track_execution_dir,
        'input_gt_dir': track_execution_dir,
        'input_track_dir': track_execution_dir,
    }
    eval_config = {
        'USE_PARALLEL': False,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False,
    }
    os.makedirs(dataset_config['output_fol'], exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        evaluator = trackeval.Evaluator(eval_config)
        ds = Dataset(dataset_config)
        results = evaluator.evaluate([ds], [HOTA()])

    # COMBINED_SEQ holds the aggregated score across all videos.
    combined = results[0]['Dataset']['sort']['COMBINED_SEQ']['vehicle']['HOTA']
    return {
        'HOTA': float(combined['HOTA'].mean()),
        'DetA': float(combined['DetA'].mean()),
        'AssA': float(combined['AssA'].mean()),
        'LocA': float(combined['LocA'].mean()),
    }


def main():
    print(f"Combo:   {COMBO_PARAM_STR}")
    print(f"Dataset: {DATASET}/{VIDEOSET}")
    print(f"Videos:  {len(_videos())} videos")
    print()

    print("Computing HOTA for scripts/ output (tracking.scripts.jsonl)...")
    scripts_scores = run_hota_for_variant('scripts', 'tracking.scripts.jsonl')

    print("Computing HOTA for execution/ output (tracking.jsonl)...")
    execution_scores = run_hota_for_variant('execution', 'tracking.jsonl')

    print()
    print(f"{'metric':<8} {'scripts/':>10} {'execution/':>12} {'diff':>10}")
    print('-' * 44)
    for key in ['HOTA', 'DetA', 'AssA', 'LocA']:
        s = scripts_scores.get(key, float('nan'))
        e = execution_scores.get(key, float('nan'))
        diff = e - s
        print(f"{key:<8} {s:>10.4f} {e:>12.4f} {diff:>+10.4f}")


if __name__ == '__main__':
    main()
