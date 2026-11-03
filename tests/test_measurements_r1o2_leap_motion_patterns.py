import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('leap_motion_patterns', Path(__file__).resolve().parent.parent /
                                             'measurements-gpu/r1o2_leap_motion_patterns.py')
motion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motion)


def label(root, clip, contents, name=None):
    path = root / 'amsterdam-y5' / clip / 'train' / (name or f'label_{clip}.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def test_ids_are_unique_across_clips_and_stable_within_clip(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'normalized'
    row = b'12,1,1.25,2.50,30.00,40.00,0.91,2.00,-1,-1,-1\r\n'
    first = label(source, '0', row + row.replace(b'12,', b'13,', 1))
    second = label(source, '1', row)
    empty = label(source, '2', b'')
    # Multiple label files from one clip must share that clip's identity namespace.
    extra = label(source, '0', row, 'label_extra.txt')
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (first, second, empty, extra)}
    metadata = motion.normalize_training_labels(source, 'amsterdam-y5', target)
    assert metadata['identities'] == 2 and metadata['rows'] == 4
    outputs = [target / p.relative_to(source) for p in (first, second, extra)]
    assert outputs[0].read_bytes() == first.read_bytes()
    assert outputs[1].read_bytes() == row.replace(b'12,1,', b'12,2,', 1)
    assert outputs[2].read_bytes() == row
    assert (target / empty.relative_to(source)).read_bytes() == b''
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert motion.normalize_training_labels(source, 'amsterdam-y5', target) == metadata


@pytest.mark.parametrize('target', ['source', 'source/child', '.'])
def test_refuses_overlapping_source_and_destination(tmp_path, target):
    source = tmp_path / 'source'
    label(source, '0', b'0,1,0,0,5,5,1,2\n')
    destination = source if target == 'source' else source / 'child' if target == 'source/child' else tmp_path
    # The actual dataset directories must overlap, not only their containing roots.
    if target == 'source/child':
        destination = source / 'amsterdam-y5' / 'child'
    elif target == '.':
        destination = tmp_path
        source = tmp_path / 'amsterdam-y5' / 'nested'
        label(source, '0', b'0,1,0,0,5,5,1,2\n')
    with pytest.raises(ValueError, match='separate'):
        motion.normalize_training_labels(source, 'amsterdam-y5', destination)


def test_missing_or_malformed_training_labels_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        motion.normalize_training_labels(tmp_path / 'source', 'amsterdam-y5', tmp_path / 'normalized')
    label(tmp_path / 'source', '0', b'bad,row\n')
    with pytest.raises(ValueError, match='Malformed label'):
        motion.normalize_training_labels(tmp_path / 'source', 'amsterdam-y5', tmp_path / 'normalized')


def test_original_clustering_arguments_and_empty_pattern_guard(tmp_path, monkeypatch):
    source, target, output = tmp_path / 'source', tmp_path / 'normalized', tmp_path / 'patterns'
    label(source, '0', b'0,1,0,0,5,5,1,2\n')
    args = SimpleNamespace(video_name='amsterdam-y5', label_dir=source, normalized_label_dir=target, output_dir=output)
    monkeypatch.setattr(motion, 'parse_args', lambda: args)
    calls = []

    def cluster(command, check):
        calls.append(command)
        assert check is True
        patterns = output / 'amsterdam-y5' / 'amsterdam-y5_0_0_tracks_clustered.npy'
        patterns.parent.mkdir(parents=True)
        np.save(patterns, np.array([], dtype=object))

    monkeypatch.setattr(motion.subprocess, 'run', cluster)
    with pytest.raises(RuntimeError, match='empty motion-pattern'):
        motion.main()
    assert calls[0][1:] == ['track_cluster_updated.py', '--video_name', 'amsterdam-y5', '--label_dir', str(target),
                             '--output_dir', str(output), '--k', '0', '--cluster_min_num', '3']
    assert (target / 'normalization.json').is_file()
