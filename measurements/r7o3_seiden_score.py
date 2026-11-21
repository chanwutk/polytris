#!/usr/local/bin/python
"""R7.O3: score Seiden's per-frame outputs with LEAP's original evaluator, on the §7.3 ground truth.

Inputs are the runs of seiden/tetris_comparison/run.py (copied to measurements-results/r7o3_seiden/<dataset>/), one
json per query, budget fraction, and seed, holding Seiden's propagated per-frame y_pred: one Seiden run per test clip,
stitched in manifest.json's clip order.
The ground truth is rebuilt exactly as scripts/score_leap_queries.py builds it (LEAP preprocess, pseudo_label, over
the 003_groundtruth boxes) and checked against that run's saved hash, so Seiden is scored on the same counts as
Tetris, OTIF, and LEAP in §7.3.

Seiden propagates labels from its anchors, so y_pred is continuous. LEAP's selection_query_1 needs integer counts
(a prediction in (0, 1) is neither >= 1 nor 0), so each answer is scored as:
  presence  selection on y_pred >= 0.5 (Seiden's propagated probability that a car is present)
  count     aggregation on y_pred rounded to the nearest integer (`acc`) and unrounded (`acc_raw`);
            top-k on y_pred unrounded (Seiden's own ranking, `topk`) and rounded (`topk_rounded`)
Each record carries the run's cost (detector frames, Seiden's time, FPS) from the run's json.

Run inside ace's polyis container:
  python measurements/r7o3_seiden_score.py --datasets caldot1-y05
Writes measurements-results/r7o3_seiden_score/<dataset>.json.
"""
import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np

from evaluation.manifests import list_split_videos
from measurements import r1o2_leap_native_queries as native
from measurements.common import DATASET_LABELS, RESULTS_ROOT, results_dir, write_json


# LEAP's evaluator and the §7.3 scoring run whose ground-truth hashes Seiden's ground truth must match.
EVALUATOR = RESULTS_ROOT / 'r1o2_leap_original_evaluator' / 'evaluate.py'
REFERENCE_RUN = RESULTS_ROOT / 'r1o2_leap_original_evaluator'
# Seiden's runs, copied back from ace's Seiden data directory.
SEIDEN_RUNS = RESULTS_ROOT / 'r7o3_seiden'
# Seiden's manifests (clip order) written by r7o3_seiden_prepare.py.
SEIDEN_INPUTS = Path('/polyis-cache/seiden/video_data')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), required=True)
    return parser.parse_args()


def load_evaluator():
    """LEAP's unmodified `evaluation` class, without its GPU/UI imports (as score_leap_queries.py loads it)."""
    # Parse the evaluator source and keep only the evaluation class.
    source = EVALUATOR.read_bytes()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'evaluation')
    namespace = dict(np=np, Counter=Counter, settings=SimpleNamespace(coco_names={'car': 2}))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(EVALUATOR), 'exec'), namespace)
    return namespace['evaluation']


def ground_truth(evaluator, dataset: str, videos: list[str]) -> list[int]:
    """Per-frame ground-truth counts over the concatenated test clips, as score_leap_queries.py computes them."""
    con = duckdb.connect()
    con.execute('SET enable_progress_bar=false')
    # Clip lengths in teNN order.
    native.load_clips(con, dataset, videos)
    # Every ground-truth box, grouped by clip and frame.
    gt_files, external = native.ground_truth_files(dataset, videos, None)
    assert not external
    con.execute('CREATE TABLE gt_files(video VARCHAR, filename VARCHAR)')
    con.executemany('INSERT INTO gt_files VALUES (?,?)', gt_files)
    labels = con.execute(f"""
        SELECT video, frame_idx, list(box ORDER BY box[1])
        FROM {native.TRACK_BOXES} JOIN gt_files USING(filename)
        GROUP BY video, frame_idx ORDER BY video, frame_idx
    """, [[f for _, f in gt_files]]).fetchall()
    by_video = {video: {} for video in videos}
    for video, frame, boxes in labels:
        by_video[video][frame] = [box + ['car'] for box in boxes]
    # LEAP's preprocess per clip, concatenated in clip order.
    gt = []
    clips = con.execute('SELECT video, n FROM clips ORDER BY clip').fetchall()
    for video, n in clips:
        obj = evaluator('pseudo_label', 'car', dict(start_frame=0, end_frame=n))
        obj.preprocess({frame: by_video[video].get(frame, []) for frame in range(n)}, {})
        gt.extend(obj.gt_result)
    con.close()
    return gt, clips


def make_run(evaluator, gt: list[int]):
    """A function that scores one prediction vector with a fresh evaluator over the pooled frames."""
    def run(pred: np.ndarray):
        obj = evaluator('pseudo_label', 'car', dict(start_frame=0, end_frame=len(gt)))
        obj.gt_result, obj.pred_result = gt, pred.tolist()
        return obj
    return run


def selection(run, selected: np.ndarray) -> dict:
    """LEAP's selection_query_1 on a 0/1 vector of selected frames."""
    f1, recall, precision, accuracy = run(selected.astype(int)).selection_query_1()
    return dict(f1=f1, recall=recall, precision=precision, accuracy=accuracy)


def count_queries(run, count_pred: np.ndarray) -> dict:
    """LEAP's aggregation_query_1 and top_k_query_1 on propagated counts, rounded and raw."""
    rounded = np.rint(count_pred).astype(int)
    mae, acc = run(rounded).aggregation_query_1()
    mae_raw, acc_raw = run(count_pred).aggregation_query_1()
    return dict(aggregation=dict(mae=mae, acc=acc, mape=1 - acc, mae_raw=mae_raw, acc_raw=acc_raw, mape_raw=1 - acc_raw),
                topk=run(count_pred).top_k_query_1(), topk_rounded=run(rounded).top_k_query_1())


def main():
    args = parse_args()
    evaluator = load_evaluator()
    out = results_dir(__file__)
    for dataset in args.datasets:
        label = DATASET_LABELS[dataset]
        gt, clips = ground_truth(evaluator, dataset, list_split_videos(dataset, 'test'))
        # Same ground truth as §7.3's scoring run.
        reference = json.loads((REFERENCE_RUN / f'{dataset}.json').read_text())['datasets'][label]
        digest = hashlib.sha256(np.asarray(gt, dtype='<i8').tobytes()).hexdigest()
        assert digest == reference['ground_truth_count_sha256'], f'{dataset}: ground truth differs from §7.3'
        # Seiden's frames follow its manifest's clip order, which must be the evaluator's clip order.
        manifest = json.loads((SEIDEN_INPUTS / dataset / 'manifest.json').read_text())
        assert [(c['video'], c['frames']) for c in manifest['clips']] == clips, f'{dataset}: clip order differs'
        run = make_run(evaluator, gt)
        records = []
        # One record per run (query, index budget, seed).
        for path in sorted((SEIDEN_RUNS / dataset).glob('*_f*_s*.json')):
            result = json.loads(path.read_text())
            y_pred = np.asarray(result.pop('y_pred'))
            assert len(y_pred) == len(gt), f'{path.name}: frame count differs'
            if result['query'] == 'count':
                metrics = count_queries(run, y_pred)
            else:
                metrics = dict(selection=selection(run, y_pred >= 0.5))
            records.append(dict(result, **metrics))
            print(f"{label} {path.stem}: {json.dumps(records[-1])}", flush=True)
        write_json(out, f'{dataset}.json', dict(dataset=dataset, frames=len(gt), evaluator=str(EVALUATOR),
                                                ground_truth_count_sha256=digest, records=records))


if __name__ == '__main__':
    main()
