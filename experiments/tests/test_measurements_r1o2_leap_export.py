import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from experiments.evaluation.p140_otif_transform import unwrap_leap_frames


import numpy as np


# The exporter wrapper the LEAP runs call, and the exporter LEAP's scored outputs were written with.
EVAL = Path(__file__).resolve().parents[2] / 'baselines/tetris-leap/experiments/eval'
SPEC = importlib.util.spec_from_file_location('leap_export', EVAL / 'export_tracks.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
RECEIVED = EVAL / 'export-leap-outputs.py'

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


# Patterns of 1, 3, 5, and 9 boxes along a diagonal, each box 10 pixels wide.
PATTERNS = [[[0, 0, 10, 10]]] + [[[3 * i, i, 3 * i + 10, i + 10] for i in range(n)] for n in (3, 5, 9)]


def parsed_clip(tmp_path, objects, patterns=PATTERNS):
    """A parsed_results directory with one clip (36) of `objects` and `patterns`."""
    parsed = tmp_path / 'parsed'
    parsed.mkdir(exist_ok=True)
    (parsed / '36_results.json').write_text(json.dumps(objects))
    (parsed / '36_tracks.json').write_text(json.dumps(patterns))
    return parsed


def export(function, parsed, tmp_path, name):
    """Run an exporter function on clip 36 of `parsed` and return its frame list."""
    out = tmp_path / name
    function(str(parsed / '36_results.json'), str(parsed / '36_tracks.json'), str(out), '36')
    return json.loads((out / '36.json').read_text())


def received(clipped=False):
    return EXPORT.load_exporter(str(RECEIVED), clip_negative_starts=clipped)


def resampled(clipped=False):
    return lambda *paths: EXPORT.resample_tracks_json(*paths, clip_negative_starts=clipped)


def boxes(frames, track_id):
    """(frame, [left, top, right, bottom]) of `track_id`, in frame order."""
    return [(i, [d['left'], d['top'], d['right'], d['bottom']]) for i, frame in enumerate(frames)
            for d in frame if d['track_id'] == track_id]


def assert_same_frames(actual, expected):
    """Equal frame lists, with box coordinates equal up to floating-point rounding."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert [(d['track_id'], d['class'], d['score']) for d in got] == [(d['track_id'], d['class'], d['score']) for d in want]
        for g, w in zip(got, want):
            np.testing.assert_allclose([g[k] for k in ('left', 'top', 'right', 'bottom')],
                                       [w[k] for k in ('left', 'top', 'right', 'bottom')], rtol=0, atol=1e-9)


def test_shorter_equal_and_single_box_patterns_match_the_received_exporter(tmp_path):
    # Intervals as long as or longer than each pattern: the received exporter resamples or repeats these.
    objects = {'1': [[0, 7], 0, 2], '2': [[2, 5], 1, 2], '3': [[1, 12], 1, 2], '4': [[0, 5], 2, 2],
               '5': [[3, 20], 3, 2], '6': [[4, 13], 3, 2], '7': [[0, 1], 2, 2]}
    parsed = parsed_clip(tmp_path, objects)
    assert_same_frames(export(resampled(), parsed, tmp_path, 'new'), export(received(), parsed, tmp_path, 'old'))


def test_longer_pattern_is_resampled_from_first_to_last_box(tmp_path):
    # A 9-box pattern over a 3-frame interval: the received exporter keeps boxes 0-2, resampling keeps 0, 4, and 8.
    parsed = parsed_clip(tmp_path, {'1': [[5, 8], 3, 2]})
    old = boxes(export(received(), parsed, tmp_path, 'old'), 1)
    new = boxes(export(resampled(), parsed, tmp_path, 'new'), 1)
    assert old == [(5 + i, [3.0 * i, float(i), 3.0 * i + 10, i + 10.0]) for i in range(3)]
    assert new == [(5 + j, [3.0 * i, float(i), 3.0 * i + 10, i + 10.0]) for j, i in enumerate((0, 4, 8))]


def test_longer_pattern_interpolates_between_boxes(tmp_path):
    # 9 boxes over 5 frames lands on positions 0, 2, 4, 6, 8; over 4 frames on 0, 8/3, 16/3, 8.
    parsed = parsed_clip(tmp_path, {'1': [[0, 4], 3, 2]})
    lefts = [box[0] for _, box in boxes(export(resampled(), parsed, tmp_path, 'new'), 1)]
    np.testing.assert_allclose(lefts, [0, 8, 16, 24], rtol=0, atol=1e-9)


def test_one_frame_interval_takes_the_first_box(tmp_path):
    # A one-frame interval has a single position, the pattern's first box, as in the received exporter.
    parsed = parsed_clip(tmp_path, {'1': [[4, 5], 3, 2]})
    assert boxes(export(resampled(), parsed, tmp_path, 'new'), 1) == [(4, [0.0, 0.0, 10.0, 10.0])]


@pytest.mark.parametrize('interval', [[3, 3], [6, 3]])
def test_empty_or_inverted_interval_places_no_boxes(tmp_path, interval):
    parsed = parsed_clip(tmp_path, {'1': [interval, 3, 2], '2': [[0, 7], 1, 2]})
    frames = export(resampled(), parsed, tmp_path, 'new')
    assert boxes(frames, 1) == []
    assert len(frames) == 7


@pytest.mark.parametrize('index', [4, -1])
def test_pattern_index_outside_the_library_raises(tmp_path, index):
    parsed = parsed_clip(tmp_path, {'1': [[0, 3], index, 2]})
    with pytest.raises(ValueError, match='pattern index'):
        export(resampled(), parsed, tmp_path, 'new')


@pytest.mark.parametrize('pattern', [[], [[]], [[1, 2, 3]]])
def test_pattern_without_boxes_raises(tmp_path, pattern):
    parsed = parsed_clip(tmp_path, {'1': [[0, 3], 0, 2]}, [pattern])
    with pytest.raises(ValueError, match='motion pattern'):
        export(resampled(), parsed, tmp_path, 'new')


def test_repeated_id_on_a_frame_keeps_its_first_box(tmp_path):
    # Keys '1' and '01' are distinct objects in the JSON but the same integer ID in the export.
    parsed = parsed_clip(tmp_path, {'1': [[0, 3], 1, 2], '01': [[1, 4], 1, 2]})
    frames = export(resampled(), parsed, tmp_path, 'new')
    assert boxes(frames, 1) == [(0, [0.0, 0.0, 10.0, 10.0]), (1, [3.0, 1.0, 13.0, 11.0]),
                                (2, [6.0, 2.0, 16.0, 12.0]), (3, [6.0, 2.0, 16.0, 12.0])]
    assert_same_frames(frames, export(received(), parsed, tmp_path, 'old'))


def test_negative_starts_match_the_received_exporter_with_and_without_clipping(tmp_path):
    # A start two frames before the video: unclipped, those frames index from the end; clipped, they are dropped.
    parsed = parsed_clip(tmp_path, {'1': [[-2, 5], 1, 2], '2': [[0, 8], 2, 2]})
    for clipped in (False, True):
        assert_same_frames(export(resampled(clipped), parsed, tmp_path, f'new{clipped}'),
                           export(received(clipped), parsed, tmp_path, f'old{clipped}'))


def test_start_before_the_frame_list_raises_unless_clipped(tmp_path):
    parsed = parsed_clip(tmp_path, {'1': [[-5, 3], 1, 2]})
    with pytest.raises(IndexError):
        export(resampled(), parsed, tmp_path, 'new')
    # Clipped, the frames before zero are dropped without moving the pattern: frames 0-2 are positions 10/7, 12/7, 2.
    lefts = [box[0] for _, box in boxes(export(resampled(True), parsed, tmp_path, 'clipped'), 1)]
    np.testing.assert_allclose(lefts, [30 / 7, 36 / 7, 6], rtol=0, atol=1e-9)


def run_cli(tmp_path, monkeypatch, parsed, *flags):
    out = tmp_path / f'export{len(flags)}.jsonl.gz'
    monkeypatch.setattr(sys, 'argv', ['export', '--exporter', str(RECEIVED), '--parsed-dir', str(parsed),
                                      '--out', str(out), *flags])
    EXPORT.main()
    with gzip.open(out, 'rt') as f:
        return json.loads(f.readline())


def test_cli_default_uses_the_received_exporter_and_flag_records_resampling(tmp_path, monkeypatch):
    parsed = parsed_clip(tmp_path, {'1': [[5, 8], 3, 2], '2': [[0, 7], 1, 2]})
    sha = hashlib.sha256(RECEIVED.read_bytes()).hexdigest()
    default = run_cli(tmp_path, monkeypatch, parsed)
    assert default['exporter_sha256'] == sha and 'pattern_policy' not in default
    assert default['frames'] == export(received(), parsed, tmp_path, 'old')
    flagged = run_cli(tmp_path, monkeypatch, parsed, '--resample-patterns', '--clip-negative-starts')
    assert flagged['pattern_policy'] == 'resample_to_interval'
    assert flagged['negative_frame_policy'] == 'drop_before_zero'
    assert flagged['frames'] == export(resampled(True), parsed, tmp_path, 'new')
