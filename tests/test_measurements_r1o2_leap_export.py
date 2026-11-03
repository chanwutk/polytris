import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from evaluation.p140_otif_transform import unwrap_leap_frames


SPEC = importlib.util.spec_from_file_location(
    'leap_export', Path(__file__).resolve().parent.parent / 'measurements-gpu/r1o2_leap_otif_rerun_export.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)

# A compact exporter fixture retains the original frame indexing and pattern-to-interval alignment.
SOURCE = '''import json
import os
import numpy as np
def reconstruct_tracks_json(results_file, tracks_file, output_dir, video_name):
    resolved_tuple = json.load(open(results_file))
    tracks = json.load(open(tracks_file))
    max_frame = max((value[0][1] for value in resolved_tuple.values()), default=0)
    frame_detections = [[] for _ in range(max_frame)]
    for obj_id, value in resolved_tuple.items():
        start_frame, end_frame = value[0]
        pattern = np.asarray(tracks[value[1]], dtype=float)
        selected_pattern = np.linspace(pattern[0], pattern[-1], end_frame - start_frame)
        for idx, bbox in enumerate(selected_pattern):
            frame_idx = start_frame + idx
            if any(d["track_id"] == int(obj_id) for d in frame_detections[frame_idx]):
                continue
            frame_detections[frame_idx].append(dict(track_id=int(obj_id), left=float(bbox[0]),
                top=float(bbox[1]), right=float(bbox[2]), bottom=float(bbox[3]), score=1.0, **{'class': 'car'}))
    os.makedirs(output_dir, exist_ok=True)
    json.dump(frame_detections, open(os.path.join(output_dir, video_name+'.json'), 'w'))
raise AssertionError('The exporter module entrypoint must never execute during loading')
'''


def inputs(tmp_path, objects):
    # Separate paths also exercise the wrapper's real JSON/file interface.
    source = tmp_path / 'exporter.py'
    source.write_text(SOURCE)
    parsed = tmp_path / 'parsed'
    parsed.mkdir()
    (parsed / '36_results.json').write_text(json.dumps(objects))
    (parsed / '36_tracks.json').write_text(json.dumps([[[0, 0, 10, 10], [7, 7, 17, 17]]]))
    return source, parsed


def reconstruct(source, parsed, tmp_path, clipped):
    EXPORT.load_exporter(str(source), clip_negative_starts=clipped)(
        str(parsed / '36_results.json'), str(parsed / '36_tracks.json'), str(tmp_path / 'out'), '36')
    return json.loads((tmp_path / 'out/36.json').read_text())


def test_start_before_negative_index_range_is_clipped_without_realigning_pattern(tmp_path):
    source, parsed = inputs(tmp_path, {'1': [[-5, 3], 0, 2]})
    with pytest.raises(IndexError):
        reconstruct(source, parsed, tmp_path, False)
    frames = reconstruct(source, parsed, tmp_path, True)
    assert len(frames) == 3
    assert [frame[0]['left'] for frame in frames] == [5.0, 6.0, 7.0]
    assert all(frame[0]['track_id'] == 1 for frame in frames)


def test_clipping_matches_legacy_unwrapping_and_keeps_distinct_overlapping_ids(tmp_path):
    source, parsed = inputs(tmp_path, {'1': [[-2, 5], 0, 2], '2': [[0, 4], 0, 2], '3': [[6, 8], 0, 2]})
    legacy = reconstruct(source, parsed, tmp_path, False)
    corrected, stats = unwrap_leap_frames(legacy, 8)
    clipped = reconstruct(source, parsed, tmp_path, True)
    assert clipped == corrected
    assert stats['wrapped_objects'] == [1]
    assert {box['track_id'] for box in clipped[0]} == {1, 2}
    assert {box['track_id'] for box in clipped[7]} == {3}


def test_nonnegative_intervals_are_unchanged(tmp_path):
    source, parsed = inputs(tmp_path, {'1': [[0, 5], 0, 2], '2': [[2, 8], 0, 2]})
    assert reconstruct(source, parsed, tmp_path, True) == reconstruct(source, parsed, tmp_path, False)


def test_cli_records_opt_in_policy_and_original_exporter_hash(tmp_path, monkeypatch):
    source, parsed = inputs(tmp_path, {'1': [[-5, 3], 0, 2]})
    out = tmp_path / 'export.jsonl.gz'
    monkeypatch.setattr(sys, 'argv', ['export', '--exporter', str(source), '--parsed-dir', str(parsed),
                                    '--out', str(out), '--clip-negative-starts'])
    EXPORT.main()
    with gzip.open(out, 'rt') as f:
        row = json.loads(f.readline())
    assert row['negative_frame_policy'] == 'drop_before_zero'
    assert row['exporter_sha256'] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert [frame[0]['left'] for frame in row['frames']] == [5.0, 6.0, 7.0]
