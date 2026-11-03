#!/usr/local/bin/python
"""R9.O2: the work p030 does outside its timers in the validation sweep, re-run without other workers.

p030 times its compute and canvas writes, but not two steps of compress(): loading the unit's classification results
(load_classification_results or load_pruned_classification_results) and decoding the whole video (every frame is read,
only sampled frames are kept). Its process time from file timestamps covers both, plus any time spent waiting on the
sweep's other 19 workers. This script re-runs exactly those two steps, serially on --cpus (the sweep's 20 workers on 80
CPUs had about 4 each), so their sum is p030's untimed work without the waiting:

  decode  depends only on the video, so it is timed once per validation video (median of --repeats) and charged to
          every packing unit of that video
  load    timed on a random sample (--load-sample, seed 0) of the dataset's packing units, its mean charged to every unit

Units are the distinct p030 runtime files of the validation sweep (p120's stage-030 manifest), on the videos whose every
unit exists, as in r9o2_preparation_cost.py's coverage. Directory setup and runtime.jsonl writes are not re-run.
Writes measurements-results/r9o2_packing_untimed/packing_untimed.json.
"""
import argparse
import os
import random
import statistics
import time

import cv2
import pandas as pd

from evaluation.manifests import build_polytris_variant_manifest, build_split_video_manifest
from evaluation.p120_throughput_gather import build_stage030_manifest, resolve_upstream_tracker
from measurements.common import DATASET_LABELS, host_info, results_dir, write_json
from polyis.io import store
from polyis.utilities import load_classification_results, load_pruned_classification_results


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: p030 untimed work (decode, load) without other workers')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS), choices=list(DATASET_LABELS))
    parser.add_argument('--cpus', default='5,14,16,17')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--load-sample', type=int, default=300)
    return parser.parse_args()


def packing_units(dataset: str):
    """The validation sweep's distinct p030 units on complete videos, and the split's video count."""
    # Every validation video crossed with every sweep configuration, as p120 lists them.
    videos = build_split_video_manifest(datasets=[dataset], videosets=['valid'])
    configs = videos.merge(build_polytris_variant_manifest(), how='cross')
    # p030's runtime file for each row; configurations differing only downstream share one unit.
    configs['runtime_file'] = build_stage030_manifest(configs)['runtime_file']
    units = configs.drop_duplicates('runtime_file').copy()
    # Keep the videos whose every unit was run, as the preparation cost's coverage does.
    units['exists'] = units['runtime_file'].map(os.path.exists)
    complete = units.groupby('video')['exists'].all()
    return units[units['video'].isin(complete[complete].index)], len(complete)


def decode_seconds(video_path: str) -> float:
    """Wall seconds of p030's whole-video decode loop (open, read every frame, release)."""
    # Start the clock before opening, since p030 opens the capture for this pass.
    start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), video_path
    # Read every frame as p030 does; keeping the sampled ones only retains references to frames already decoded.
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frames):
        if not cap.read()[0]:
            break
    cap.release()
    return time.perf_counter() - start


def load_seconds(row) -> float:
    """Wall seconds of p030's classification-results load for one unit."""
    # Pruned units read p022's output; unpruned units read p020's.
    start = time.perf_counter()
    if not pd.isna(row['tracking_accuracy_threshold']):
        load_pruned_classification_results(row['dataset'], row['video'], int(row['tilesize']), row['classifier'],
                                           resolve_upstream_tracker(row), float(row['tracking_accuracy_threshold']),
                                           float(row['relevance_threshold']), int(row['sample_rate']))
    else:
        load_classification_results(row['dataset'], row['video'], int(row['tilesize']), row['classifier'],
                                    int(row['sample_rate']))
    return time.perf_counter() - start


def main():
    args = parse_args()
    # Pin to the requested CPUs so the run neither waits on nor disturbs other work.
    cpus = {int(c) for c in args.cpus.split(',')}
    os.sched_setaffinity(0, cpus)
    out = dict(method=dict(cpus=sorted(cpus), repeats=args.repeats, load_sample=args.load_sample, seed=0,
                           opencv=cv2.__version__, host=host_info()), datasets={})
    for dataset in args.datasets:
        units, split_videos = packing_units(dataset)
        per_video = units.groupby('video').size()
        # Decode each complete video --repeats times; the median drops the first, cold read.
        decode = {}
        for video in per_video.index:
            runs = [decode_seconds(store.dataset(dataset, 'valid', video)) for _ in range(args.repeats)]
            decode[video] = dict(median_s=statistics.median(runs), runs_s=runs, units=int(per_video[video]))
        decode_total = sum(d['median_s'] * d['units'] for d in decode.values())
        # Time the load on a random sample of units.
        sample = units.iloc[sorted(random.Random(0).sample(range(len(units)), min(args.load_sample, len(units))))]
        loads = [load_seconds(row) for _, row in sample.iterrows()]
        load_total = statistics.mean(loads) * len(units)
        out['datasets'][dataset] = dict(
            label=DATASET_LABELS[dataset], units=len(units), complete_videos=len(per_video), split_videos=split_videos,
            decode=decode, decode_total_s=decode_total,
            load=dict(sample=len(loads), mean_s=statistics.mean(loads), median_s=statistics.median(loads),
                      max_s=max(loads)),
            load_total_s=load_total, untimed_total_s=decode_total + load_total)
        print(f'{dataset}: {len(units)} units on {len(per_video)} videos; decode {decode_total / 3600:.2f} h, '
              f'load {load_total / 3600:.2f} h', flush=True)
    write_json(results_dir(__file__), 'packing_untimed.json', out)


if __name__ == '__main__':
    main()
