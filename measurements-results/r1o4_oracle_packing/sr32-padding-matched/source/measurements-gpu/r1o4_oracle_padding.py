"""Same-frame padding control for the annotation-derived packing experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import cv2
import torch

from measurements.common import DATASET_LABELS
from polyis.io import cache
from polyis.models import detector as detectors
from polyis.utilities import build_param_str


spec = importlib.util.spec_from_file_location('oracle_packing', Path(__file__).with_name('r1o4_oracle_packing.py'))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['caldot1-y05', 'caldot2-y05', 'ams-y05'], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert Path('/scratch/polyis-cache').samefile(Path(cache.CACHE_DIR))
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    root = Path('/polyis/measurements-results/r1o4_oracle_packing')
    baseline_path = root / 'sr32' / f'{args.dataset}.json'
    baseline = json.loads(baseline_path.read_text())
    prior = json.loads(Path('/polyis/measurements-results/r1o4_packing_detection_accuracy/operating_points.json').read_text())
    padding = prior['datasets'][DATASET_LABELS[args.dataset]]['tilepadding']
    assert padding in ('bl', 'square', 'tr')
    stride = baseline['method']['stride']
    cfg = build_param_str(classifier=oracle.CLASSIFIER, tilesize=oracle.TILE_SIZE, sample_rate=stride,
                          tilepadding=padding, canvas_scale=1.0, tracker=None,
                          tracking_accuracy_threshold=None, relevance_threshold=0.5)
    model = detectors.get_detector(args.dataset, 0, batch_size=4)
    weights = Path(detectors.get_detector_info(args.dataset)['model_path'])
    assert oracle.sha256(weights) == baseline['method']['weights_sha256']
    videos = sorted(baseline['videos'])
    references = {}
    for index, video in enumerate(videos, 1):
        reference = cache.exec(args.dataset, 'naive', video, f'oracle_detection_sr{stride}.jsonl')
        references[video] = oracle.sha256(reference)
        done = cache.exec(args.dataset, 'ucomp-dets', video, cfg, 'padding_done.json')
        if not done.exists():
            oracle.packing.compress(args.dataset, 'test', video, oracle.CLASSIFIER, oracle.TILE_SIZE,
                                   stride, padding, 1.0, None, None, 0.5, oracle.packing.PackMode.Best_Fit)
            oracle.detection.detect_objects(args.dataset, video, oracle.CLASSIFIER, oracle.TILE_SIZE,
                                           stride, padding, 1.0, None, None, 0.5, 4, model)
            oracle.unpacking.unpack(args.dataset, 'test', video, oracle.CLASSIFIER, oracle.TILE_SIZE,
                                    stride, padding, 1.0, None, None, 0.5)
            oracle.save_json(done, dict(padding=padding, reference_sha256=references[video],
                                       weights_sha256=baseline['method']['weights_sha256']))
        else:
            assert json.loads(done.read_text()) == dict(padding=padding, reference_sha256=references[video],
                                                       weights_sha256=baseline['method']['weights_sha256'])
        print(f'{baseline["label"]} padding={padding} {index}/{len(videos)} {video}', flush=True)
    detectors.delete(model)
    scores = oracle.score_dataset(args.dataset, videos, cfg, stride, padding=padding)
    assert scores['all_sampled_frames']['reference'] == baseline['all_sampled_frames']['reference']
    assert scores['sampled_frames'] == baseline['sampled_frames']
    method = dict(baseline['method'], tile_padding=padding)
    result = dict(dataset=args.dataset, label=baseline['label'], method=method,
                  baseline=str(baseline_path), reference_sha256=references,
                  provisional=False, videos=baseline['videos'], **scores,
                  environment=dict(gpu=os.environ.get('BOMBE_GPU'), run=os.environ.get('BOMBE_RUN')),
                  source_sha256=oracle.sha256(__file__))
    oracle.save_json(root / 'sr32-padding-matched' / f'{args.dataset}.json', result)
    print(json.dumps(dict(label=result['label'], padding=padding,
                          all_sampled_frames=result['all_sampled_frames']), indent=2), flush=True)


if __name__ == '__main__':
    main()
