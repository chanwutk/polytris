#!/usr/local/bin/python
"""Install human-verified tracking labels as a dataset's test ground truth (003_groundtruth).

The source holds <source>/<dataset>/<video>/tracking.jsonl for every test video of each dataset, in the
evaluation's format: one {"frame_idx": i, "tracks": [[id, x1, y1, x2, y2], ...]} line per clip frame, i = 0, 1, ...
Every file is validated before anything is written. Per video, the first install moves the automatic ground
truth to 003_groundtruth.auto/; a reinstall replaces only the installed files. The installed directory holds
tracking.jsonl, a detection.jsonl derived from it (score 1.0, the same 15 FPS frame index), and SOURCE.json
(source commit and sha256 of every file), which also stops p003 from overwriting it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from polyis.groundtruth import MARKER, backup_dir, groundtruth_dir, is_installed
from polyis.io import store
from polyis.utilities import get_video_frame_count


# Name of the staging directory a video's new ground truth is written into before it is swapped in.
STAGING_NAME = '.003_groundtruth.staging'
# Files an install writes, besides MARKER.
LABEL_FILES = ('tracking.jsonl', 'detection.jsonl')


def parse_args():
    parser = argparse.ArgumentParser(description='Install human-verified tracking labels as test ground truth')
    parser.add_argument('--source', default='/polyis-cache/scratch/human-gt')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--datasets', nargs='+', default=['jnc0', 'jnc2', 'jnc6', 'jnc7'])
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def sha256(path: Path) -> str:
    # Hash a file's bytes for provenance.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tracking(path: Path, frame_count: int) -> list[list[list[float]]]:
    """Per-frame tracks of a source file, rejecting anything the scorer would misread."""
    # Collect one track list per frame, in file order.
    frames: list[list[list[float]]] = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            # Parse the frame record.
            row = json.loads(line)
            # Frames must be contiguous from zero so frame_idx equals the clip frame.
            if row.get('frame_idx') != len(frames):
                raise ValueError(f'{path}:{line_number}: frame_idx {row.get("frame_idx")}, expected {len(frames)}')
            # Track identities seen in this frame.
            seen: set[int] = set()
            for track in row['tracks']:
                # Each track is [id, x1, y1, x2, y2] with an integer, unique, non-negative id.
                if (len(track) != 5 or not isinstance(track[0], int) or isinstance(track[0], bool)
                        or track[0] < 0 or track[0] in seen):
                    raise ValueError(f'{path}:{line_number}: bad or duplicate track id in {track}')
                # Boxes must be finite with positive width and height.
                if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in track[1:])
                        or track[1] >= track[3] or track[2] >= track[4]):
                    raise ValueError(f'{path}:{line_number}: invalid box {track}')
                seen.add(track[0])
            frames.append(row['tracks'])
    # One line per clip frame, no more and no fewer.
    if len(frames) != frame_count:
        raise ValueError(f'{path}: {len(frames)} frames, but the clip has {frame_count}')
    return frames


def plan_dataset(source: Path, dataset: str) -> list[tuple[str, Path, list[list[list[float]]]]]:
    """Validated (video, source file, frames) for every test video of a dataset."""
    # The dataset's test videos, as the pipeline sees them.
    videos = sorted(f for f in os.listdir(store.dataset(dataset, 'test')) if f.endswith('.mp4'))
    # The videos the source provides labels for.
    provided = sorted(p.name for p in (source / dataset).iterdir()) if (source / dataset).is_dir() else []
    # The two sets must match exactly, so no test video keeps automatic labels by accident.
    if provided != videos:
        raise ValueError(f'{dataset}: source has {provided}, test split has {videos}')
    # Validate every file against its clip's frame count.
    return [(video, source / dataset / video / 'tracking.jsonl',
             read_tracking(source / dataset / video / 'tracking.jsonl', get_video_frame_count(dataset, video)))
            for video in videos]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # Write one JSON object per line.
    with path.open('w') as target:
        for row in rows:
            target.write(json.dumps(row) + '\n')


def install_video(dataset: str, video: str, source_file: Path, frames: list[list[list[float]]],
                  source_commit: str) -> str:
    """Swap a video's ground truth for the validated labels; returns what happened."""
    target = groundtruth_dir(dataset, video)
    backup = backup_dir(dataset, video)
    staging = target.with_name(STAGING_NAME)
    # Decide from the current state before touching anything.
    if is_installed(dataset, video):
        # A reinstall replaces our own files; the automatic backup must already exist.
        if not backup.is_dir():
            raise RuntimeError(f'{target} is installed but {backup} is missing')
        action = 'reinstalled'
    elif target.is_dir():
        # A first install needs a free backup slot; otherwise the state is ambiguous.
        if backup.exists():
            raise RuntimeError(f'{target} has no {MARKER} but {backup} exists; resolve by hand')
        action = 'installed'
    elif backup.is_dir():
        # An interrupted install moved the automatic labels but did not finish; resume it.
        action = 'resumed'
    else:
        raise RuntimeError(f'{target} does not exist; expected automatic ground truth to back up')
    # Build the complete new directory beside the target.
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    write_jsonl(staging / 'tracking.jsonl', [{'frame_idx': i, 'tracks': tracks} for i, tracks in enumerate(frames)])
    write_jsonl(staging / 'detection.jsonl', [{'frame_idx': i, 'detections': [[*t[1:5], 1.0] for t in tracks]}
                                              for i, tracks in enumerate(frames)])
    # Record provenance last, so a staging directory without it is never mistaken for an install.
    (staging / MARKER).write_text(json.dumps({
        'source_commit': source_commit,
        'source_file': str(source_file),
        'source_sha256': sha256(source_file),
        'installed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'files': {name: sha256(staging / name) for name in LABEL_FILES},
    }, indent=2) + '\n')
    # Swap: keep the automatic labels on first install, drop our previous install otherwise.
    if action == 'installed':
        target.rename(backup)
    elif action == 'reinstalled':
        shutil.rmtree(target)
    staging.rename(target)
    return action


def main():
    args = parse_args()
    source = Path(args.source)
    # Validate every dataset before installing anything.
    plans = {dataset: plan_dataset(source, dataset) for dataset in args.datasets}
    for dataset, plan in plans.items():
        for video, source_file, frames in plan:
            # Report the plan without writing when asked.
            if args.dry_run:
                print(f'{dataset}/{video}: {len(frames)} frames valid (dry run)')
                continue
            action = install_video(dataset, video, source_file, frames, args.source_commit)
            print(f'{dataset}/{video}: {action}, {len(frames)} frames')


if __name__ == '__main__':
    main()
