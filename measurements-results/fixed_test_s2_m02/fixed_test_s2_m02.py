"""Resume the fixed s=2, Tr=.5, M=.2, no-padding SORT test experiment."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import cv2
import duckdb
import torch

from evaluation.manifests import list_split_videos
from evaluation.p121_throughput_compute import QUERY_DATA_ACCESSORS, parse_runtime_file, filter_excluded_runtime_ops
from measurements.common import metric_summary, trackeval_rerun
from polyis.io import cache, store
from polyis.models import detector as detectors
from scripts import p022_exec_prune_polyominoes as pruning
from scripts import p030_exec_compress as packing
from scripts import p040_exec_detect as detection
from scripts import p050_exec_uncompress as unpacking
from scripts import p060_exec_track as tracking


ROOT = Path('/polyis/measurements-results/fixed_test_s2_m02')
CFG = 'ShuffleNet05_60_2_080_r050_none_s100_sortcython'
PRUNE = 'ShuffleNet05_60_2_080_r050_sortcython'
STAGES = [
    ('020_exec_classify', 'relevancy', 'ShuffleNet05_60_2', 'score/runtime.jsonl'),
    ('022_exec_prune_polyominoes', 'pruned-polyominoes', PRUNE, 'score/runtime.jsonl'),
    ('030_exec_compress', 'comp-frames', CFG, 'runtime.jsonl'),
    ('040_exec_detect', 'comp-dets', CFG, 'runtimes.jsonl'),
    ('050_exec_uncompress', 'ucomp-dets', CFG, 'runtime.jsonl'),
    ('060_exec_track', 'ucomp-tracks', CFG, 'runtimes.jsonl'),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    return parser.parse_args()


def save(path, data):
    # Replace checkpoints atomically so interrupted jobs cannot look complete.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    args = parse_args()
    # Refuse the canonical cache: individual stages replace their own output directories.
    assert Path('/scratch/polyis-cache').samefile(Path(cache.CACHE_DIR))
    info = json.loads((ROOT / 'inputs.json').read_text())[args.dataset]
    videos = list_split_videos(args.dataset, 'test')
    assert sorted(videos) == sorted(info['videos'])
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = None
    frame_count = 0
    runtime_rows = []
    for index, video in enumerate(videos, 1):
        # Reuse cached classification, and resume only stages that completed successfully.
        def run(stage, action, outputs):
            directory = cache.exec(args.dataset, stage, video, PRUNE if stage == 'pruned-polyominoes' else CFG)
            marker = directory / 'fixed_test_done.json'
            if marker.exists():
                assert json.loads(marker.read_text()) == dict(config=CFG, video=video)
            else:
                action()
                for output in outputs:
                    assert (directory / output).exists(), directory / output
                save(marker, dict(config=CFG, video=video))

        def prune():
            # Serialize license use while GPU work on other datasets continues independently.
            with (ROOT / 'ilp.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                pruning.process_video(args.dataset, 'test', video, 'ShuffleNet05', 60, 2,
                                      'sortcython', .8, .5, None)

        run('pruned-polyominoes', prune, ['score/score.jsonl', 'score/runtime.jsonl'])
        run('comp-frames', lambda: packing.compress(args.dataset, 'test', video, 'ShuffleNet05', 60, 2,
            'none', 1., 'sortcython', .8, .5, packing.PackMode.Best_Fit), ['images', 'runtime.jsonl'])
        if model is None:
            # Load and warm up once per dataset, outside instrumented extraction timing.
            model = detectors.get_detector(args.dataset, 0, batch_size=4)
            images = sorted(cache.exec(args.dataset, 'comp-frames', video, CFG, 'images').glob('*.jpg'))[:4]
            if images:
                with torch.no_grad():
                    detectors.detect_batch([cv2.imread(str(path)) for path in images], model)
                    torch.cuda.synchronize()
        run('comp-dets', lambda: detection.detect_objects(args.dataset, video, 'ShuffleNet05', 60, 2,
            'none', 1., 'sortcython', .8, .5, 4, model), ['detections.jsonl', 'runtimes.jsonl'])
        run('ucomp-dets', lambda: unpacking.unpack(args.dataset, 'test', video, 'ShuffleNet05', 60, 2,
            'none', 1., 'sortcython', .8, .5), ['detections.jsonl', 'runtime.jsonl'])
        run('ucomp-tracks', lambda: tracking.track(args.dataset, video, 'ShuffleNet05', 60, 2,
            'none', 1., 'sortcython', .8, .5, False), ['tracking.jsonl', 'runtimes.jsonl'])
        # Use precisely the existing Figure 8 timing parser and exclusion policy.
        for stage, folder, param, filename in STAGES:
            path = cache.exec(args.dataset, folder, video, param, filename)
            timings = filter_excluded_runtime_ops(parse_runtime_file(str(path), stage, QUERY_DATA_ACCESSORS[stage]), stage)
            runtime_rows.extend(dict(row, video=video) for row in timings.to_dict('records'))
        capture = cv2.VideoCapture(str(store.dataset(args.dataset, 'test', video)))
        assert capture.isOpened()
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        assert frames > 0
        frame_count += frames
        save(ROOT / 'progress' / f'{args.dataset}.json', dict(completed=index, total=len(videos), video=video))
        print(f'COMPLETE {args.dataset} {index}/{len(videos)} {video}', flush=True)
    detectors.delete(model)
    # Aggregate counted operations without changing the pipeline's timing boundaries.
    con = duckdb.connect()
    con.execute('CREATE TABLE runtime(stage VARCHAR, time DOUBLE, op VARCHAR, video VARCHAR)')
    con.executemany('INSERT INTO runtime VALUES (?, ?, ?, ?)',
                    [(r['stage'], r['time'], r['op'], r['video']) for r in runtime_rows])
    seconds = con.execute('SELECT sum(time) FROM runtime').fetchone()[0]
    breakdown = dict(con.execute('SELECT stage, sum(time) FROM runtime GROUP BY stage').fetchall())
    assert seconds > 0 and abs(sum(breakdown.values()) - seconds) < 1e-6
    metrics = metric_summary(trackeval_rerun(str(cache.execution(info['ground_truth_dataset'])),
        '003_groundtruth/tracking.jsonl', str(cache.execution(args.dataset)),
        f'060_uncompressed_tracks/{CFG}/tracking.jsonl', videos, str(ROOT / 'trackeval' / args.dataset)))
    hota = metrics['HOTA']['HOTA']
    weights = Path(detectors.get_detector_info(args.dataset).get('model_path') or model.cfg.MODEL.WEIGHTS)
    with weights.open('rb') as stream:
        weights_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(dataset=args.dataset, config=CFG, clips=len(videos), frames=frame_count,
        seconds=seconds, fps=frame_count / seconds, hota=hota,
        speedup=(frame_count / seconds) / info['reference_fps'],
        relative_hota_loss=1 - hota / info['reference_hota'], reference=info,
        seconds_by_stage=breakdown, metrics=metrics, weights_sha256=weights_sha256,
        timing_note='Cached classification plus newly measured downstream operations; Figure 8 exclusions.',
        provisional_groundtruth=args.dataset.startswith('jnc'))
    save(ROOT / 'results' / f'{args.dataset}.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
