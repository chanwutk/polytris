"""Resume fixed Tr=.5, M=.2, SORT test experiments using cached classification."""
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sample-rate', type=int, choices=[1, 2], default=2)
    parser.add_argument('--padding', choices=['none', 'square'], default='none')
    return parser.parse_args()


def save(path, data):
    # Replace checkpoints atomically so interrupted jobs cannot look complete.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    args = parse_args()
    sample_rate, padding = args.sample_rate, args.padding
    name = 'fixed_test_s2_m02' if (sample_rate, padding) == (2, 'none') else f'fixed_test_s{sample_rate}_{padding}_m02'
    root = Path('/polyis/measurements-results') / name
    cfg = f'ShuffleNet05_60_{sample_rate}_080_r050_{padding}_s100_sortcython'
    prune_param = f'ShuffleNet05_60_{sample_rate}_080_r050_sortcython'
    stages = [
        ('020_exec_classify', 'relevancy', f'ShuffleNet05_60_{sample_rate}', 'score/runtime.jsonl'),
        ('022_exec_prune_polyominoes', 'pruned-polyominoes', prune_param, 'score/runtime.jsonl'),
        ('030_exec_compress', 'comp-frames', cfg, 'runtime.jsonl'),
        ('040_exec_detect', 'comp-dets', cfg, 'runtimes.jsonl'),
        ('050_exec_uncompress', 'ucomp-dets', cfg, 'runtime.jsonl'),
        ('060_exec_track', 'ucomp-tracks', cfg, 'runtimes.jsonl'),
    ]
    # Refuse the canonical cache: individual stages replace their own output directories.
    assert Path('/scratch/polyis-cache').samefile(Path(cache.CACHE_DIR))
    info = json.loads((root / 'inputs.json').read_text())[args.dataset]
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
            directory = cache.exec(args.dataset, stage, video, prune_param if stage == 'pruned-polyominoes' else cfg)
            marker = directory / 'fixed_test_done.json'
            if marker.exists():
                assert json.loads(marker.read_text()) == dict(config=cfg, video=video)
            else:
                # The input audit validates cached pruning against the exact classification frame sequence.
                if not (stage == 'pruned-polyominoes' and video in info.get('reuse_pruning', [])):
                    action()
                for output in outputs:
                    assert (directory / output).exists(), directory / output
                save(marker, dict(config=cfg, video=video))
            for output in outputs:
                assert (directory / output).exists(), directory / output

        def prune():
            # Serialize license use while GPU work on other datasets continues independently.
            with (root / 'ilp.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                pruning.process_video(args.dataset, 'test', video, 'ShuffleNet05', 60, sample_rate,
                                      'sortcython', .8, .5, None)

        run('pruned-polyominoes', prune, ['score/score.jsonl', 'score/runtime.jsonl'])
        run('comp-frames', lambda: packing.compress(args.dataset, 'test', video, 'ShuffleNet05', 60, sample_rate,
            padding, 1., 'sortcython', .8, .5, packing.PackMode.Best_Fit), ['images', 'runtime.jsonl'])
        if model is None:
            # Load and warm up once per dataset, outside instrumented extraction timing.
            model = detectors.get_detector(args.dataset, 0, batch_size=4)
            images = sorted(cache.exec(args.dataset, 'comp-frames', video, cfg, 'images').glob('*.jpg'))[:4]
            if images:
                with torch.no_grad():
                    detectors.detect_batch([cv2.imread(str(path)) for path in images], model)
                    torch.cuda.synchronize()
        run('comp-dets', lambda: detection.detect_objects(args.dataset, video, 'ShuffleNet05', 60, sample_rate,
            padding, 1., 'sortcython', .8, .5, 4, model), ['detections.jsonl', 'runtimes.jsonl'])
        run('ucomp-dets', lambda: unpacking.unpack(args.dataset, 'test', video, 'ShuffleNet05', 60, sample_rate,
            padding, 1., 'sortcython', .8, .5), ['detections.jsonl', 'runtime.jsonl'])
        run('ucomp-tracks', lambda: tracking.track(args.dataset, video, 'ShuffleNet05', 60, sample_rate,
            padding, 1., 'sortcython', .8, .5, False), ['tracking.jsonl', 'runtimes.jsonl'])
        # Use precisely the existing Figure 8 timing parser and exclusion policy.
        for stage, folder, param, filename in stages:
            path = cache.exec(args.dataset, folder, video, param, filename)
            timings = filter_excluded_runtime_ops(parse_runtime_file(str(path), stage, QUERY_DATA_ACCESSORS[stage]), stage)
            runtime_rows.extend(dict(row, video=video) for row in timings.to_dict('records'))
        capture = cv2.VideoCapture(str(store.dataset(args.dataset, 'test', video)))
        assert capture.isOpened()
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        assert frames > 0
        frame_count += frames
        save(root / 'progress' / f'{args.dataset}.json', dict(completed=index, total=len(videos), video=video))
        print(f'COMPLETE {args.dataset} {index}/{len(videos)} {video}', flush=True)
    detectors.delete(model)
    # Aggregate counted operations without changing the pipeline's timing boundaries.
    con = duckdb.connect()
    con.execute('CREATE TABLE runtime(stage VARCHAR, time DOUBLE, op VARCHAR, video VARCHAR)')
    # A bulk JSON scan avoids hundreds of thousands of individual database inserts.
    timing_path = root / 'timings' / f'{args.dataset}.json'
    save(timing_path, runtime_rows)
    con.execute('INSERT INTO runtime SELECT stage, time, op, video FROM read_json_auto(?)', [str(timing_path)])
    seconds = con.execute('SELECT sum(time) FROM runtime').fetchone()[0]
    breakdown = dict(con.execute('SELECT stage, sum(time) FROM runtime GROUP BY stage').fetchall())
    assert seconds > 0 and abs(sum(breakdown.values()) - seconds) < 1e-6
    metrics = metric_summary(trackeval_rerun(str(cache.execution(info['ground_truth_dataset'])),
        '003_groundtruth/tracking.jsonl', str(cache.execution(args.dataset)),
        f'060_uncompressed_tracks/{cfg}/tracking.jsonl', videos, str(root / 'trackeval' / args.dataset)))
    hota = metrics['HOTA']['HOTA']
    weights = Path(detectors.get_detector_info(args.dataset).get('model_path') or model.cfg.MODEL.WEIGHTS)
    with weights.open('rb') as stream:
        weights_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(dataset=args.dataset, config=cfg, clips=len(videos), frames=frame_count,
        seconds=seconds, fps=frame_count / seconds, hota=hota,
        speedup=(frame_count / seconds) / info['reference_fps'],
        relative_hota_loss=1 - hota / info['reference_hota'], reference=info,
        seconds_by_stage=breakdown, metrics=metrics, weights_sha256=weights_sha256,
        timing_note='Cached classification plus newly measured downstream operations; Figure 8 exclusions.',
        provisional_groundtruth=args.dataset.startswith('jnc'))
    save(root / 'results' / f'{args.dataset}.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
