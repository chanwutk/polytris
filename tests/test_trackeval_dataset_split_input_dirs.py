import json
import os

from polyis.trackeval.dataset import Dataset


def write_tracking_jsonl(path: str, track_id: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(json.dumps({
            'frame_idx': 0,
            'tracks': [[track_id, 1.0, 2.0, 3.0, 4.0, 0, 1.0]],
        }) + '\n')


def build_dataset_config(base_input_dir: str, seq_name: str) -> dict:
    return {
        'tracker_list': ['sort'],
        'seq_list': [seq_name],
        'output_fol': os.path.join(base_input_dir, 'output'),
        'output_sub_fol': 'test',
        'input_dir': base_input_dir,
        'input_gt': os.path.join('003_groundtruth', 'tracking.jsonl'),
        'input_track': os.path.join('060_uncompressed_tracks', 'cfg', 'tracking.jsonl'),
        'skip': 1,
        'tracker': 'cfg',
    }


def test_dataset_supports_split_gt_and_tracker_input_dirs(tmp_path):
    seq = 'te01.mp4'
    gt_root = tmp_path / 'gt'
    track_root = tmp_path / 'track'

    gt_file = gt_root / seq / '003_groundtruth' / 'tracking.jsonl'
    track_file = track_root / seq / '060_uncompressed_tracks' / 'cfg' / 'tracking.jsonl'
    write_tracking_jsonl(str(gt_file), track_id=11)
    write_tracking_jsonl(str(track_file), track_id=22)

    config = build_dataset_config(str(track_root), seq)
    config['input_gt_dir'] = str(gt_root)
    config['input_track_dir'] = str(track_root)

    dataset = Dataset(config)
    raw_gt = dataset._load_raw_file('cfg', seq, is_gt=True)
    raw_track = dataset._load_raw_file('cfg', seq, is_gt=False)

    assert int(raw_gt['gt_ids'][0][0]) == 11
    assert int(raw_track['tracker_ids'][0][0]) == 22


def test_dataset_defaults_to_input_dir_when_split_dirs_not_provided(tmp_path):
    seq = 'te01.mp4'
    input_root = tmp_path / 'single'

    gt_file = input_root / seq / '003_groundtruth' / 'tracking.jsonl'
    track_file = input_root / seq / '060_uncompressed_tracks' / 'cfg' / 'tracking.jsonl'
    write_tracking_jsonl(str(gt_file), track_id=31)
    write_tracking_jsonl(str(track_file), track_id=41)

    config = build_dataset_config(str(input_root), seq)
    dataset = Dataset(config)
    raw_gt = dataset._load_raw_file('cfg', seq, is_gt=True)
    raw_track = dataset._load_raw_file('cfg', seq, is_gt=False)

    assert int(raw_gt['gt_ids'][0][0]) == 31
    assert int(raw_track['tracker_ids'][0][0]) == 41
