"""Integration smoke test: full pipeline on one real video.

Invokes ``execution/main.py`` as a subprocess against a small real dataset
combo and asserts that the expected tracking output file is created and
parses cleanly.  Requires the dataset's training artifacts to be present on
the test machine; tests skip otherwise.

This is intentionally a black-box test: we don't mock anything internal.
The smoke run is the cheapest proof that all stages compose end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from polyis.io import cache


PIPELINE_COMBO = {
    'dataset': 'caldot2-y05',
    'videoset': 'valid',
    'classifier': 'ShuffleNet05',
    'tile_size': '60',
    'sample_rate': '16',
    'tilepadding': 'none',
    'canvas_scale': '1',
    'tracker': 'sortcython',
    'tracking_accuracy_threshold': 'null',
    'relevance_threshold': '0.5',
}


def _required_artifacts_present() -> bool:
    """Skip the smoke test when training/index artifacts are missing."""
    paths = [
        cache.index(PIPELINE_COMBO['dataset'], 'training', 'results',
                    f"{PIPELINE_COMBO['classifier']}_{PIPELINE_COMBO['tile_size']}",
                    'model_best.pth'),
        cache.index(PIPELINE_COMBO['dataset'], 'never-relevant',
                    f"{PIPELINE_COMBO['tile_size']}_all.npy"),
    ]
    return all(os.path.exists(p) for p in paths)


@pytest.mark.skipif(
    not _required_artifacts_present(),
    reason='Dataset training artifacts not available on this machine',
)
def test_pipeline_smoke_run(tmp_path):
    """Pipeline runs end-to-end on 1 video and produces a valid tracking.jsonl."""
    cmd = [
        sys.executable, '-m', 'execution.main',
        '--dataset', PIPELINE_COMBO['dataset'],
        '--videoset', PIPELINE_COMBO['videoset'],
        '--classifier', PIPELINE_COMBO['classifier'],
        '--tile-size', PIPELINE_COMBO['tile_size'],
        '--sample-rate', PIPELINE_COMBO['sample_rate'],
        '--tilepadding', PIPELINE_COMBO['tilepadding'],
        '--canvas-scale', PIPELINE_COMBO['canvas_scale'],
        '--tracker', PIPELINE_COMBO['tracker'],
        '--tracking-accuracy-threshold', PIPELINE_COMBO['tracking_accuracy_threshold'],
        '--relevance-threshold', PIPELINE_COMBO['relevance_threshold'],
        '--classify-gpu', '0',
        '--detect-gpu', '0',
        '--prune-workers', '1',
        '--compress-workers', '1',
        '--max-videos-in-flight', '1',
        '--no-warmup',
        '--max-videos', '1',
    ]

    # 5-minute hard wall-clock cap (smoke run should be well under that).
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"Pipeline failed: returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Locate the output tracking.jsonl using the same param_str the pipeline
    # builds internally.  Re-derive it here for the assertion.
    from polyis.utilities import build_param_str
    param_str = build_param_str(
        classifier=PIPELINE_COMBO['classifier'],
        tilesize=int(PIPELINE_COMBO['tile_size']),
        sample_rate=int(PIPELINE_COMBO['sample_rate']),
        tilepadding=PIPELINE_COMBO['tilepadding'],
        canvas_scale=float(PIPELINE_COMBO['canvas_scale']),
        tracker=PIPELINE_COMBO['tracker'],
        tracking_accuracy_threshold=None,
        relevance_threshold=float(PIPELINE_COMBO['relevance_threshold']),
    )
    # Use the first video alphabetically (matches the pipeline's sort order).
    from polyis.io import store
    videoset_dir = store.dataset(PIPELINE_COMBO['dataset'], PIPELINE_COMBO['videoset'])
    videos = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )
    assert len(videos) >= 1
    first_video = videos[0]

    tracking_path = cache.exec(
        PIPELINE_COMBO['dataset'], 'ucomp-tracks', first_video,
        param_str, 'tracking.jsonl',
    )
    assert os.path.exists(tracking_path), \
        f"Expected tracking output at {tracking_path}"

    # Parse every line; verify schema.
    with open(tracking_path) as f:
        lines = f.readlines()
    assert len(lines) > 0, "tracking.jsonl is empty"
    for line in lines:
        entry = json.loads(line)
        assert 'frame_idx' in entry
        assert 'tracks' in entry
        assert isinstance(entry['tracks'], list)
        for track in entry['tracks']:
            # [track_id, x1, y1, x2, y2] minimum.
            assert len(track) >= 5
