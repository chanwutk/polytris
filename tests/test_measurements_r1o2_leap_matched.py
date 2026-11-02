import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent


def load(name, filename):
    # Load host-side scripts without requiring measurements-gpu to be a Python package; paths with a '/' are relative
    # to the tetris-leap submodule's experiments/, whose scripts the harness runs.
    base = ROOT / 'baselines' / 'tetris-leap' / 'experiments' if '/' in filename else ROOT / 'measurements-gpu'
    spec = importlib.util.spec_from_file_location(name, base / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmasked_option_uses_leaps_inverted_mask_switch():
    import argparse
    runner = load('leap_inference_runner', 'run/run_leap.py')
    args = SimpleNamespace(config='cfg.yaml',weight='detector.pt',reid_weight='reid',
                           video_dir='test',work_dir='run',motion_dir='patterns',retinanet=False)
    assert '--use_mask' not in runner.leap_argv(args)
    args.no_mask = True
    argv = runner.leap_argv(args)
    assert argv.count('--use_mask') == 1 and '--motion_dir' in argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mask',action='store_false')
    assert parser.parse_args([]).use_mask and not parser.parse_args(['--use_mask']).use_mask


@pytest.mark.parametrize('dataset,model,full_name', [
    ('caldot1-y05', 'caldot1', 'caldot1'),
    ('caldot2-y05', 'caldot2', 'caldot2'),
    ('ams-y05', 'amsterdam', 'amsterdam-y5'),
])
def test_matched_pipeline_uses_dataset_checkpoint_and_fresh_preparation(dataset, model, full_name):
    driver = load('matched_driver', 'r1o2_leap_matched.py')
    config, weight = driver.configure(dataset)
    assert f'/{model}/weights/yolov5x6/weights/best.pt' in weight
    assert driver.BASE.RAW != driver.ORIGINAL_RAW
    assert driver.BASE.SCRATCH != driver.ORIGINAL_SCRATCH
    assert config['labels'] == full_name and config['workspace_motion']
    for split in ('train', 'test'):
        command, output = driver.BASE.leap_prep_command(dataset, f'labels_{split}')
        assert f'{driver.ADAPTER} label' in command
        assert f'--yolo_weight {weight}' in command
        assert f'/labels/{full_name}/{split}' in command
        assert output.endswith(f'/labels/{full_name}')
    train, output = driver.BASE.leap_prep_command(dataset, 'reid_train')
    assert output.endswith(f'/reid/{full_name}') and output in train
    cluster, output = driver.BASE.leap_prep_command(dataset, 'motion_patterns')
    assert f'--video-name {full_name}' in cluster
    assert 'prep/motion_patterns.py' in cluster and '--normalized-label-dir' in cluster


def test_missing_preparation_artifact_stops_queue(tmp_path):
    driver = load('matched_driver', 'r1o2_leap_matched.py')
    with pytest.raises(AssertionError, match='Missing trained'):
        driver.check_stage('caldot1-y05', 'reid_train', tmp_path)
    (tmp_path / 'model_final.pth').touch()
    driver.check_stage('caldot1-y05', 'reid_train', tmp_path)


def test_empty_motion_library_stops_before_extraction(tmp_path):
    driver = load('matched_driver', 'r1o2_leap_matched.py')
    path = tmp_path / 'example_tracks_clustered.npy'
    np.save(path, np.empty(0, dtype=object))
    with pytest.raises(AssertionError, match='Empty motion-pattern library'):
        driver.check_stage('ams-y05', 'motion_patterns', tmp_path)
    np.save(path, np.array([[[0, 0, 10, 10]]], dtype=object))
    driver.check_stage('ams-y05', 'motion_patterns', tmp_path)


@pytest.mark.parametrize('exit_code,contaminated,changed,expected', [
    (0, False, False, False), (1, False, False, True),
    (0, True, False, True), (0, False, True, True),
])
def test_resume_keeps_successful_work_but_reruns_failed_or_stale_extraction(
        tmp_path, exit_code, contaminated, changed, expected):
    driver = load('matched_driver', 'r1o2_leap_matched.py')
    launch = tmp_path / 'launch.json'
    assert driver.needs_extraction(launch, False)
    launch.write_text(json.dumps(dict(exit=exit_code, contaminated=contaminated)))
    assert driver.needs_extraction(launch, changed) == expected


def test_detector_preserves_tetris_boxes_scores_and_only_maps_class(monkeypatch):
    adapter = load('matched_adapter', 'run/matched_detector.py')
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    calls = []
    boxes = torch.tensor([[1.25, 2.5, 20.75, 30.5]])
    scores = torch.tensor([0.875])

    class Model:
        names = {0: 'car'}

        def fuse(self):
            calls.append('fuse')

        def to(self, device):
            calls.append(device)

        def eval(self):
            calls.append('eval')

        def __call__(self, actual, **kwargs):
            assert actual is image and kwargs == {'verbose': False}
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, conf=scores))]

    monkeypatch.setattr(adapter, 'YOLO', lambda path, verbose: Model())
    detector = adapter.MatchedDetector('dataset.pt', device='cpu')
    actual_image, actual = detector.detect(image)
    assert actual_image is image
    assert actual == [[1.25, 2.5, 20.75, 30.5, 0.875, 2]]
    assert calls == ['fuse', 'cpu', 'eval']
    # An unrelated checkpoint must fail instead of silently relabeling another class as a vehicle.
    Model.names = {0: 'person'}
    with pytest.raises(AssertionError, match='Unexpected classes'):
        adapter.MatchedDetector('wrong.pt', device='cpu')
