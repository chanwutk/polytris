"""Scoring-window padding in polyis.trackeval.dataset: frames after the shorter file's last row."""
import json
import os

import numpy as np
import pytest

from polyis.trackeval.dataset import Dataset, pad_scoring_window_from_env

SEQ = 'te01.mp4'
BOX = [10.0, 10.0, 50.0, 50.0]


def write(path, frames):
    # frames: {frame_idx: [track_id, ...]}; every track uses the same box.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for idx in sorted(frames):
            f.write(json.dumps({'frame_idx': idx, 'tracks': [[t, *BOX] for t in frames[idx]]}) + '\n')


def dataset(tmp_path, gt, track, pad):
    write(tmp_path / SEQ / 'gt.jsonl', gt)
    write(tmp_path / SEQ / 'track.jsonl', track)
    config = {'seq_list': [SEQ], 'input_dir': str(tmp_path), 'input_gt': 'gt.jsonl', 'input_track': 'track.jsonl',
              'skip': 1, 'tracker': 'cfg', 'output_fol': str(tmp_path / 'out')}
    if pad is not None:
        config['pad_scoring_window'] = pad
    return Dataset(config)


def scored(raw):
    # Ground-truth and tracker boxes over the frames TrackEval actually pairs.
    n = len(raw['similarity_scores'])
    return sum(len(x) for x in raw['gt_ids'][:n]), sum(len(x) for x in raw['tracker_ids'][:n]), n


# Ground truth runs 6 frames; the system's output stops after frame 2.
GT = {i: [1] for i in range(6)}
SHORT = {0: [7], 1: [7], 2: [7]}


def test_default_pads_the_scoring_window(tmp_path, monkeypatch):
    monkeypatch.delenv('POLYIS_PAD_SCORING_WINDOW', raising=False)
    raw = dataset(tmp_path, GT, SHORT, None).get_raw_seq_data('cfg', SEQ)
    assert scored(raw) == (6, 3, 6)


def test_env_zero_restores_trackeval_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv('POLYIS_PAD_SCORING_WINDOW', '0')
    raw = dataset(tmp_path, GT, SHORT, None).get_raw_seq_data('cfg', SEQ)
    # Only the 3 frames both files cover are scored; ground truth on frames 3-5 is never counted.
    assert scored(raw) == (3, 3, 3)


def test_padding_scores_ground_truth_after_the_last_output_row(tmp_path):
    raw = dataset(tmp_path, GT, SHORT, True).get_raw_seq_data('cfg', SEQ)
    assert raw['num_timesteps'] == 6
    assert scored(raw) == (6, 3, 6)
    # The padded frames hold no output, so their ground truth can only be missed.
    assert all(len(t) == 0 for t in raw['tracker_ids'][3:])


def test_padding_scores_output_after_the_last_ground_truth_row(tmp_path):
    raw = dataset(tmp_path, {0: [1], 1: [1]}, {i: [7] for i in range(5)}, True).get_raw_seq_data('cfg', SEQ)
    assert raw['num_timesteps'] == 5
    assert scored(raw) == (2, 5, 5)


def test_padding_changes_nothing_when_files_have_equal_length(tmp_path):
    a = dataset(tmp_path, GT, {i: [7] for i in range(6)}, False).get_raw_seq_data('cfg', SEQ)
    b = dataset(tmp_path, GT, {i: [7] for i in range(6)}, True).get_raw_seq_data('cfg', SEQ)
    assert scored(a) == scored(b) == (6, 6, 6)
    assert all(np.array_equal(x, y) for x, y in zip(a['similarity_scores'], b['similarity_scores']))


def test_padding_lowers_hota_only_by_the_unscored_tail(tmp_path):
    # A perfect tracker on frames 0-2 scores HOTA 1 unpadded and is penalized for frames 3-5 when padded.
    from modules.TrackEval.trackeval.metrics import HOTA
    results = {}
    for pad in (False, True):
        ds = dataset(tmp_path / str(pad), GT, SHORT, pad)
        raw = ds.get_raw_seq_data('cfg', SEQ)
        data = ds.get_preprocessed_seq_data(raw, 'vehicle')
        results[pad] = float(np.mean(HOTA().eval_sequence(data)['HOTA']))
    assert results[False] == pytest.approx(1.0)
    assert results[True] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize('value, expected', [('0', False), ('1', True)])
def test_env_switch(monkeypatch, value, expected):
    monkeypatch.setenv('POLYIS_PAD_SCORING_WINDOW', value)
    assert pad_scoring_window_from_env() is expected


def test_env_switch_rejects_other_values(monkeypatch):
    monkeypatch.setenv('POLYIS_PAD_SCORING_WINDOW', 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        pad_scoring_window_from_env()
