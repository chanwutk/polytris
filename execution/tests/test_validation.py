"""Validation test: execution/ output is semantically equivalent to scripts/p060.

Runs the same parameter combo through both paths (scripts/p020..p060 and
execution/main.py), then compares the resulting tracking.jsonl files on
three numeric metrics: total frames, total track entries, and unique track
ID count.  Asserts that the two paths agree within tolerances.

Requires a previously-run scripts/p060 output saved as ``tracking.scripts.jsonl``
(produced by the test fixture).  Skips when not available.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from polyis.io import cache
from polyis.utilities import build_param_str


# Use a combo that scripts/p060 has historically produced output for in the
# project's cache.  The test fixture copies that output to .scripts.jsonl
# before re-running execution/ on the same path.
VALIDATION_COMBO = {
    'dataset': 'caldot2-y05',
    'videoset': 'valid',
    'classifier': 'ShuffleNet05',
    'tile_size': 60,
    'sample_rate': 16,
    'tilepadding': 'bl',
    'canvas_scale': 1.0,
    'tracker': 'sortcython',
    'tracking_accuracy_threshold': 0.4,
    'relevance_threshold': 0.5,
}

# Maximum acceptable absolute fractional difference in track entry counts.
TRACK_COUNT_TOLERANCE = 0.25

# Maximum acceptable absolute fractional difference in unique track ID count.
ID_COUNT_TOLERANCE = 0.20

# Videos validated.  These match the pipeline's sort order on first 3 videos.
VALIDATION_VIDEOS = ['va00.mp4', 'va01.mp4', 'va02.mp4']


def _param_str() -> str:
    return build_param_str(
        classifier=VALIDATION_COMBO['classifier'],
        tilesize=VALIDATION_COMBO['tile_size'],
        sample_rate=VALIDATION_COMBO['sample_rate'],
        tilepadding=VALIDATION_COMBO['tilepadding'],
        canvas_scale=VALIDATION_COMBO['canvas_scale'],
        tracker=VALIDATION_COMBO['tracker'],
        tracking_accuracy_threshold=VALIDATION_COMBO['tracking_accuracy_threshold'],
        relevance_threshold=VALIDATION_COMBO['relevance_threshold'],
    )


def _load_tracks(path: str) -> dict:
    """Return {frame_idx: list[track]} for the given tracking.jsonl path."""
    out: dict = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            out[entry['frame_idx']] = entry['tracks']
    return out


@pytest.mark.parametrize('video', VALIDATION_VIDEOS)
def test_execution_matches_scripts_per_video(video: str):
    """Per-video: execution/ output agrees with scripts/p060 within tolerance."""
    param_str = _param_str()
    base = cache.exec(
        VALIDATION_COMBO['dataset'], 'ucomp-tracks', video, param_str,
    )
    scripts_path = os.path.join(str(base), 'tracking.scripts.jsonl')
    exec_path = os.path.join(str(base), 'tracking.jsonl')

    if not os.path.exists(scripts_path):
        pytest.skip(f"No scripts/p060 baseline at {scripts_path}")
    if not os.path.exists(exec_path):
        pytest.skip(f"No execution/ output at {exec_path}")

    s = _load_tracks(scripts_path)
    e = _load_tracks(exec_path)

    # Same number of frames (video length is identical, this is exact).
    assert len(s) == len(e), \
        f"frame count diff: scripts={len(s)} execution={len(e)}"

    # Unique track ID counts agree within tolerance.  We don't require the
    # ID values to match because the tracker is sensitive to detection
    # ordering and may assign different IDs to the same physical objects
    # on different runs.  What matters is that approximately the same set
    # of objects is tracked.
    s_ids = {t[0] for tracks in s.values() for t in tracks}
    e_ids = {t[0] for tracks in e.values() for t in tracks}
    if len(s_ids) > 0:
        id_diff = abs(len(s_ids) - len(e_ids)) / len(s_ids)
        assert id_diff <= ID_COUNT_TOLERANCE, (
            f"unique track ID count fractional diff {id_diff:.3f} > tolerance "
            f"{ID_COUNT_TOLERANCE} for {video} "
            f"(scripts={len(s_ids)}, execution={len(e_ids)})"
        )

    # Track entry counts within tolerance (interpolation can vary slightly).
    s_total = sum(len(v) for v in s.values())
    e_total = sum(len(v) for v in e.values())
    if s_total > 0:
        diff = abs(s_total - e_total) / s_total
        assert diff <= TRACK_COUNT_TOLERANCE, (
            f"track count fractional diff {diff:.3f} > tolerance "
            f"{TRACK_COUNT_TOLERANCE} for {video} "
            f"(scripts={s_total}, execution={e_total})"
        )
