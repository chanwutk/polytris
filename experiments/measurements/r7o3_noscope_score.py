"""Score a NoScope replay with the §7.3 frame-selection evaluator, re-costing its detector at the batched rate.

The replay charges each detector call the rate in the run's manifest. Every system in §7.3 is charged the reference
detector's batched per-image time, as Tetris and OTIF batch detection, which Seiden's manifest for the same split
records (experiments/measurements/r7o3_seiden_prepare.py). Each record is re-costed from its saved detector frames and measured
frame-differencing and CNN time; the replay's unbatched charge and FPS are kept alongside.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.evaluation.manifests import list_split_videos
from experiments.measurements import r7o3_seiden_score as seiden


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=seiden.DATASET_LABELS)
    parser.add_argument('run', type=Path)
    return parser.parse_args()


def recost(result: dict, detector_ms: float) -> dict:
    """Re-cost every record's detector calls at detector_ms per frame, keeping the replay's unbatched values."""
    # The run-level rate the replay charged.
    result['detector_ms_per_frame_unbatched'] = result['detector_ms_per_frame']
    result['detector_ms_per_frame'] = detector_ms
    for record in result['records']:
        # Keep the replay's charge, then charge the same detector calls at the batched rate.
        record['detector_seconds_unbatched'] = record['detector_seconds']
        record['fps_unbatched'] = record['fps']
        record['detector_seconds'] = record['detector_frames'] * detector_ms / 1000
        # FPS over every frame of the split: measured cascade time plus the detector charge.
        record['fps'] = result['frames'] / (record['difference_seconds'] + record['cnn_seconds'] + record['detector_seconds'])
    return result


def main():
    args = parse_args()
    # Reuse the evaluator, frame order, and ground-truth hash check from the Seiden experiment.
    evaluator = seiden.load_evaluator()
    labels, clips = seiden.ground_truth(evaluator, args.dataset, list_split_videos(args.dataset, 'test'))
    reference = json.loads((seiden.REFERENCE_RUN / f'{args.dataset}.json').read_text())
    expected = reference['datasets'][seiden.DATASET_LABELS[args.dataset]]['ground_truth_count_sha256']
    digest = hashlib.sha256(np.asarray(labels, dtype='<i8').tobytes()).hexdigest()
    assert digest == expected, f'{args.dataset}: ground truth differs from §7.3'

    # Verify predictions cover exactly the same frames before invoking LEAP's evaluator.
    result = json.loads((args.run / 'run.json').read_text())
    assert [tuple(clip) for clip in result['clips']] == clips
    with np.load(args.run / 'predictions.npz') as arrays:
        selected = arrays['selected']
    assert selected.dtype == np.bool_
    assert selected.shape == (len(result['records']), len(labels))
    # The batched detector rate for the same split, from Seiden's manifest; its clips must be the replay's clips.
    manifest = json.loads((seiden.SEIDEN_INPUTS / args.dataset / 'manifest.json').read_text())
    assert [(c['video'], c['frames']) for c in manifest['clips']] == clips, f'{args.dataset}: manifest clips differ'
    assert manifest['frames'] == result['frames'], f'{args.dataset}: manifest frames differ'
    # A run scored before keeps its original replay charge; re-costing always starts from run.json.
    assert 'detector_ms_per_frame_unbatched' not in result, f'{args.run}: run.json is already re-costed'
    recost(result, manifest['detector_ms_per_frame'])
    run = seiden.make_run(evaluator, labels)
    for record, prediction in zip(result['records'], selected, strict=True):
        record['selection'] = seiden.selection(run, prediction)
    result['ground_truth_count_sha256'] = digest
    result['dataset'] = args.dataset
    result['evaluator'] = 'LEAP selection_query_1'
    (args.run / 'scores.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
