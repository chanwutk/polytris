#!/usr/local/bin/python
"""Time preparation-only overheads using saved inputs; all outputs stay in scratch.

Run each startup profile in a fresh process on one idle GPU. The persistent profile loads
one detector and warms the reference and representative packed-input shapes once each.
p015 uses the existing benchmark and a private compiler cache, without retraining.
p130/p135 use validation tables copied into scratch, never the canonical evaluation cache.
"""
import argparse
import json
import os
import shutil
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step', required=True, choices=['train', 'reference', 'sweep', 'persistent', 'classifier', 'tables'])
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--scratch', required=True)
    return parser.parse_args()


def startup(dataset, profile, inputs):
    import cv2
    import torch
    from polyis.io import store
    from polyis.models import detector

    # Decode outside the timer, matching the chosen compute-only boundary.
    video = sorted(store.dataset(dataset, 'valid').glob('*.mp4'))[0]
    capture = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(32):
        ok, frame = capture.read()
        assert ok, f'Cannot read warm-up frame from {video}'
        frames.append(frame)
    capture.release()
    packed_paths = sorted(inputs.glob('packed_*.jpg'))
    packed = [cv2.imread(str(path)) for path in packed_paths]
    assert packed and all(frame is not None for frame in packed), inputs

    # Each profile starts with a fresh CUDA context and uses the pipeline's constructor batch size.
    begin = time.perf_counter()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    cuda_s = time.perf_counter() - begin
    begin = time.perf_counter()
    batch_size = {'train': 16, 'reference': 1, 'sweep': 4, 'persistent': 4}[profile]
    model = detector.get_detector(dataset, 0, batch_size=batch_size, num_images=0)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - begin

    # p011 has no separate warm-up; its first inference is already charged in the saved detect times.
    begin = time.perf_counter()
    if profile in ('reference', 'persistent'):
        for frame in frames:
            detector.detect(frame, model)
            torch.cuda.synchronize()
    if profile in ('sweep', 'persistent'):
        detector.detect_batch(packed[:4], model)
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - begin if profile != 'train' else 0.0
    return dict(cuda_init_s=cuda_s, load_s=load_s, warmup_s=warmup_s,
                seconds=cuda_s + load_s + warmup_s, reference_video=str(video),
                packed_images=[str(path) for path in packed_paths],
                packed_shapes=[list(frame.shape) for frame in packed], batch_size=batch_size,
                reference_warmup_frames=32 if profile in ('reference', 'persistent') else 0,
                packed_warmup_frames=min(4, len(packed)) if profile in ('sweep', 'persistent') else 0)


def classifier(dataset, inputs, work, metadata):
    # A private, initially absent cache includes compilation work without disturbing existing compiled models.
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(work / 'inductor')
    os.environ['TRITON_CACHE_DIR'] = str(work / 'triton')
    from scripts.p015_tune_optimize_classifier import benchmark_classifier, get_video_resolution
    from polyis.io import cache

    # Reuse the saved trained model; the existing benchmark writes only to our scratch cache.
    point = metadata['operating_point']
    target = cache.index(dataset, 'training', 'results', f"{point['classifier']}_{point['tilesize']}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs / 'model_best.pth', target / 'model_best.pth')
    width, height = get_video_resolution(dataset)
    begin = time.perf_counter()
    benchmark_classifier([dataset], width, height, point['classifier'], point['tilesize'], 128, 0, None)
    seconds = time.perf_counter() - begin
    # Explicitly reap compiler workers before Python 3.13 tears down their pipes at interpreter exit.
    from torch._inductor.async_compile import shutdown_compile_workers
    shutdown_compile_workers()
    # Keep the benchmark's method-level failures visible rather than treating them as successful optimizations.
    methods = [json.loads(line) for line in (target / 'model_compilation.jsonl').read_text().splitlines()]
    return dict(seconds=seconds, iterations=128, resolution=[width, height], methods=methods,
                classifier=point['classifier'], tilesize=point['tilesize'], compiler_cache='fresh private cache')


def tables(dataset, inputs):
    from evaluation import p130_tradeoff_compute as tradeoff, p135_pareto_extract as pareto
    from polyis.io import cache

    # Only validation rows are staged, since test-set reporting is outside preparation.
    for source, target in [('accuracy.csv', cache.eval(dataset, 'acc', 'accuracy.csv')),
                           ('query_execution_overall.csv', cache.eval(dataset, 'tp', 'measurements',
                                                                    'query_execution_overall.csv'))]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inputs / source, target)
    cache.eval(dataset, 'tradeoff').mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    tradeoff.process_dataset(dataset)
    tradeoff_s = time.perf_counter() - begin
    begin = time.perf_counter()
    pareto.main(SimpleNamespace(test=False))
    pareto_s = time.perf_counter() - begin
    # Compare the recomputed validation frontier with the saved one without changing selection inputs.
    import duckdb
    con = duckdb.connect()
    expected = str(inputs / 'pareto_params.csv')
    actual = str(cache.eval(dataset, 'pareto-params', 'pareto_params.csv'))
    columns = ', '.join(pareto.PARETO_PARAM_COLS)
    differences = con.execute(f'''
        SELECT count(*) FROM (
            (SELECT {columns} FROM read_csv_auto(?) EXCEPT SELECT {columns} FROM read_csv_auto(?))
            UNION ALL
            (SELECT {columns} FROM read_csv_auto(?) EXCEPT SELECT {columns} FROM read_csv_auto(?))
        )
    ''', [expected, actual, actual, expected]).fetchone()[0]
    # The validation re-timing may cover fewer clips (CalDoT2); a cost probe never replaces saved selections.
    return dict(seconds=tradeoff_s + pareto_s, tradeoff_s=tradeoff_s, pareto_s=pareto_s,
                validation_only=True, source_selections_preserved=True,
                recomputed_frontier_configuration_differences=differences)


def main():
    args = parse_args()
    inputs = Path(args.inputs).resolve() / args.dataset
    scratch = Path(args.scratch).resolve()
    # Refuse protected data/cache roots and reused step directories, including a stale compiler cache.
    config = yaml.safe_load(Path(os.environ['CONFIG']).read_text())
    for root in ('/polyis-data', '/polyis-cache', '/otif-dataset', '/sota-results', config['DATA']['CACHE_DIR']):
        protected = Path(root).resolve()
        assert scratch != protected and protected not in scratch.parents, scratch
    work = scratch / args.dataset / args.step
    work.mkdir(parents=True, exist_ok=False)
    config['DATA']['CACHE_DIR'] = str(work / 'cache')
    config['EXEC']['DATASETS'] = [args.dataset]
    config_path = work / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    os.environ['CONFIG'] = str(config_path)
    metadata = json.loads((inputs / 'metadata.json').read_text())
    # Imports that read CONFIG occur only after pointing it at the isolated cache.
    if args.step == 'classifier':
        result = classifier(args.dataset, inputs, work, metadata)
    elif args.step == 'tables':
        result = tables(args.dataset, inputs)
        result['input_reconstruction'] = metadata.get('table_input_reconstruction')
    else:
        result = startup(args.dataset, args.step, inputs)
    result.update(dataset=args.dataset, step=args.step, host=socket.gethostname(),
                  physical_gpu=os.environ.get('BOMBE_GPU'), launcher_run=os.environ.get('BOMBE_RUN'),
                  cpu_set=os.environ.get('BOMBE_CPUS'), measured_at=time.time(), input_files=metadata['input_files'])
    # One immutable step result and an append-only collection consumed by the accounting measurement.
    (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    with (scratch / 'timings.jsonl').open('a') as output:
        output.write(json.dumps(result) + '\n')
    print(f"{args.dataset} {args.step}: {result['seconds']:.3f} seconds", flush=True)


if __name__ == '__main__':
    main()
