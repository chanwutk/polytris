"""Shared builders for the profiling tests: a sweep document over synthetic artifacts, videos, and references."""

import json
import os
from pathlib import Path

import numpy as np

from execution.config import MISTRACK_TOLERANCES

GRID_H, GRID_W = 2, 3
SYSTEM = {'queue_maxsize': 2, 'stages': {
    'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpu': 0}, 'prune': {'procs': 1}, 'pack': {'procs': 2},
    'detect': {'procs': 1, 'gpu': 0}, 'track': {'procs': 2}}}


def rate_table(per_tolerance: dict[float, int]) -> np.ndarray:
    """A max-rate table whose every tile has the given max sampling distance per tolerance (default 1)."""
    table = np.ones((GRID_H, GRID_W, len(MISTRACK_TOLERANCES)), dtype=np.int64)
    for tol, rate in per_tolerance.items():
        table[:, :, MISTRACK_TOLERANCES.index(tol)] = rate
    return table


def sweep_doc(tmp_path: Path, tables: dict[str, np.ndarray] | None = None, **params) -> dict:
    """A valid sweep document with every artifact file present under tmp_path/art."""
    art = tmp_path / 'art'
    art.mkdir(exist_ok=True)
    for name in ('w.pth', 'bench.jsonl', 'det.pt'):
        (art / name).write_text('')
    np.save(art / 'never.npy', np.ones((GRID_H, GRID_W), dtype=np.uint8))
    trackers = params.get('trackers', ['sortcython', 'bytetrackcython'])
    # Gaps at s=1: 8, 6, 2 (all distinct); at s=4: 2, 1, 1, so 0.8 duplicates 0.6 there.
    tables = tables or {t: rate_table({0.4: 8, 0.6: 6, 0.8: 2}) for t in trackers}
    for t, table in tables.items():
        np.save(art / f'{t}.npy', table)
    reference = tmp_path / 'reference'
    reference.mkdir(exist_ok=True)
    doc_params = {'classifier': 'C', 'tile_size': 60, 'canvas_format': 'jpeg', 'sample_rates': [1, 4],
                  'relevance_thresholds': [0.5], 'mistrack_tolerances': [None, 0.4, 0.6, 0.8],
                  'tile_paddings': ['none'], 'canvas_scales': [1.0], 'trackers': trackers}
    doc_params.update(params)
    uses_prune = any(t is not None for t in doc_params['mistrack_tolerances'])
    artifacts = {'classifier_weights': str(art / 'w.pth'), 'compilation_benchmark': str(art / 'bench.jsonl'),
                 'never_relevant_bitmap': str(art / 'never.npy'),
                 'detector': {'type': 'ultralytics', 'model_path': str(art / 'det.pt')}}
    if uses_prune:
        artifacts['max_rate_tables'] = {t: str(art / f'{t}.npy') for t in doc_params['trackers']}
    return {'dataset': 'ds', 'videoset': 'valid', 'output_root': str(tmp_path / 'out'), 'params': doc_params,
            'artifacts': artifacts, 'evaluation': {'reference_dir': str(reference),
                                                   'reference_file': '002_naive/tracking.jsonl'},
            'system': json.loads(json.dumps(SYSTEM))}


def videos_with_references(tmp_path: Path, names=('va00.mp4', 'va01.mp4')) -> str:
    """A split directory of (empty) video files, each with its reference tracks under tmp_path/reference."""
    videos = tmp_path / 'videos'
    videos.mkdir(exist_ok=True)
    for name in names:
        (videos / name).write_text('')
        ref = tmp_path / 'reference' / name / '002_naive'
        ref.mkdir(parents=True, exist_ok=True)
        (ref / 'tracking.jsonl').write_text('')
    return str(videos)


class FakeRunner:
    """Runner stand-in: writes execution's outputs (tracks and summary.json) without running anything.

    fail holds configuration ids (their run directory names) that exit with code 3 and write no summary.
    """

    def __init__(self, fail=(), runtime_s=2.0):
        self.calls: list[dict] = []
        self.fail, self.runtime_s = set(fail), runtime_s

    def __call__(self, config_path, videos_dir, out_dir, save_root, log_dir) -> int:
        run_id = os.path.basename(log_dir)
        self.calls.append({'id': run_id, 'config': config_path, 'videos': videos_dir, 'out': out_dir, 'save': save_root})
        Path(log_dir, 'stderr.log').write_text(f'stderr of {run_id}\n')
        if run_id in self.fail:
            return 3
        os.makedirs(out_dir)
        stems = sorted(os.path.splitext(v)[0] for v in os.listdir(videos_dir))
        for stem in stems:
            Path(out_dir, f'{stem}.jsonl').write_text('')
        Path(out_dir, 'summary.json').write_text(json.dumps({
            'videos': len(stems), 'runtime_s': self.runtime_s, 'frames': 10 * len(stems), 'saved': save_root is not None,
            'video_frames': {s: 10 for s in stems}}))
        return 0
