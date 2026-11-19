"""Score a NoScope replay with the §7.3 frame-selection evaluator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.manifests import list_split_videos
from measurements import r7o3_seiden_score as seiden


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=seiden.DATASET_LABELS)
    parser.add_argument('run', type=Path)
    return parser.parse_args()


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
    run = seiden.make_run(evaluator, labels)
    for record, prediction in zip(result['records'], selected, strict=True):
        record['selection'] = seiden.selection(run, prediction)
    result['ground_truth_count_sha256'] = digest
    result['dataset'] = args.dataset
    result['evaluator'] = 'LEAP selection_query_1'
    (args.run / 'scores.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
