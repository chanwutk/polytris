"""
Benchmark execution2 against the p020 -> p060 scripts on the same videos and GPUs, without touching any
existing results.

Runs inside a throwaway polyis container whose /polyis-cache is a scratch directory and whose real
cache is mounted read-only at /polyis-cache-ro, so neither the scripts (which clear their output
directories) nor ProgressBar's logs can overwrite old results.  Launch from the ace host:

  RUN=/work/cwkt/projects/tetris/measurements-results/execution2-benchmark/caldot1-y05-s4
  mkdir -p $RUN/cache
  docker run --rm --gpus '"device=2,3"' --shm-size 64g --network host -w /polyis \
    -v /work/cwkt/projects/tetris:/polyis -v /data/cwkt/data/polyis-data:/polyis-data:ro \
    -v /work/cwkt/data/otif-dataset:/otif-dataset:ro -v ~/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
    -v /data/cwkt/data/polyis-cache:/polyis-cache-ro:ro -v $RUN/cache:/polyis-cache -v $RUN:/bench \
    polyis-polyis bash -c 'pip install -q duckdb && python -m execution.tools.benchmark \
      --dataset caldot1-y05 --config execution/configs/caldot1-y05-s4.yaml --bench /bench'

What it measures (wall clock with perf_counter, after reading every video once to warm the page cache):
  scripts      each of p020, (p022), p030, p040, p050, p060 run as `python <script> --test`, one after another,
               narrowed to this dataset and configuration (a one-row pareto_params.csv); the sum of their
               wall clocks is the scripts' end-to-end time, including each stage's process start and model load.
  execution2   `python -m execution.main` as configured (classify and detect on separate GPUs), and again with
               detect moved onto classify's GPU (one GPU, like the scripts use): the process's wall clock, and
               main's own runtime_s (first to last result, so startup and model load are excluded).
  outputs      every execution2 output compared with the scripts' fresh tracking.jsonl from this same run.
Writes <bench>/result.json and per-run logs; the scratch cache stays under <bench>/cache for inspection.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from execution.config import load_config
from execution.tools.compare_scripts import compare_track_lines
from polyis.utilities import build_param_str

REAL_CACHE = Path('/polyis-cache-ro')
SCRATCH_CACHE = Path('/polyis-cache')
DATASETS = Path('/polyis-data/datasets')
CHECKOUT = Path('/polyis')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--bench', required=True)
    parser.add_argument('--videoset', default='test')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--skip-scripts', action='store_true')
    return parser.parse_args()


def copy_inputs(dataset: str, use_prune: bool, tile_size: int, classifier: str, tracker: str) -> list[str]:
    """Copy the indexing inputs the scripts read from the read-only cache into the scratch cache."""
    rels = [f'{dataset}/indexing/training/results/{classifier}_{tile_size}/model_best.pth',
            f'{dataset}/indexing/training/results/{classifier}_{tile_size}/model_compilation.jsonl',
            f'{dataset}/indexing/always_relevant/{tile_size}_all.npy']
    if use_prune:
        rels.append(f'{dataset}/indexing/track_rate/{tracker}_{tile_size}/max_rate_table.npy')
    for rel in rels:
        (SCRATCH_CACHE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REAL_CACHE / rel, SCRATCH_CACHE / rel)
    return rels


def write_pareto_row(dataset: str, cfg) -> dict:
    """The scripts' test-pass filter: only this configuration's row, copied verbatim from the real file."""
    src = REAL_CACHE / dataset / 'evaluation' / '095_pareto_params' / 'pareto_params.csv'
    p = cfg.params
    with open(src) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['videoset'] == 'valid' and r['classifier'] == cfg.artifacts.classifier.name
                and float(r['tilesize']) == p.tile_size and float(r['sample_rate']) == p.sample_rate
                and (r['tracking_accuracy_threshold'] == '' if p.mistrack_tolerance is None
                     else r['tracking_accuracy_threshold'] != '' and float(r['tracking_accuracy_threshold']) == p.mistrack_tolerance)
                and float(r['relevance_threshold']) == p.relevance_threshold and r['tilepadding'] == p.tile_padding
                and float(r['canvas_scale']) == p.canvas_scale and r['tracker'] == p.tracker]
        fieldnames = reader.fieldnames
    if len(rows) != 1:
        raise SystemExit(f'expected exactly one Pareto row for this configuration in {src}, found {len(rows)}')
    dst = SCRATCH_CACHE / dataset / 'evaluation' / '095_pareto_params' / 'pareto_params.csv'
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(rows[0])
    return rows[0]


def write_global_config(bench: Path, dataset: str, videos_dir: Path) -> Path:
    """configs/global.yaml narrowed to this dataset, with the scratch cache and the run's videos directory."""
    config = yaml.safe_load((CHECKOUT / 'configs' / 'global.yaml').read_text())
    config['EXEC']['DATASETS'] = [dataset]
    config['DATA']['CACHE_DIR'] = f'{SCRATCH_CACHE}/'
    config['DATA']['DATASETS_DIR'] = str(videos_dir.parent.parent)
    path = bench / 'global.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def timed(cmd: list[str], env: dict, log: Path) -> dict:
    """Run cmd to completion; its wall clock, exit code, and stdout's last line."""
    t0 = time.perf_counter()
    with open(log, 'w') as f:
        proc = subprocess.run(cmd, cwd=CHECKOUT, env=env, stdout=subprocess.PIPE, stderr=f, text=True)
    seconds = time.perf_counter() - t0
    log.with_suffix('.stdout').write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f'{" ".join(cmd)} failed with exit code {proc.returncode}; see {log}')
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    return {'cmd': cmd, 'seconds': seconds, 'last_line': lines[-1] if lines else ''}


def single_gpu_config(cfg_path: str, bench: Path) -> Path:
    """The same execution2 config with detect moved onto classify's GPU."""
    data = yaml.safe_load(Path(cfg_path).read_text())
    data['system']['stages']['detect']['gpu'] = data['system']['stages']['classify']['gpu']
    path = bench / 'execution2-single-gpu.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def main() -> None:
    args = parse_args()
    bench = Path(args.bench)
    if os.path.realpath(SCRATCH_CACHE) == os.path.realpath(REAL_CACHE) or not REAL_CACHE.exists():
        raise SystemExit('the real cache must be mounted read-only at /polyis-cache-ro, separate from /polyis-cache')
    # The config's artifact paths point into /polyis-cache, which is the scratch cache here: copy the
    # inputs first (from the raw YAML), then validate the config against the copies.
    raw = yaml.safe_load(Path(args.config).read_text())
    rp = raw['params']
    inputs = copy_inputs(args.dataset, rp['mistrack_tolerance'] is not None, rp['tile_size'],
                         raw['artifacts']['classifier']['name'], rp['tracker'])
    cfg = load_config(args.config)
    p = cfg.params

    # Videos: symlinks to the read-only dataset, optionally the first --limit in sorted order.
    all_videos = sorted(f for f in os.listdir(DATASETS / args.dataset / args.videoset) if f.endswith('.mp4'))
    videos = all_videos[:args.limit]
    videos_dir = bench / 'datasets' / args.dataset / args.videoset
    videos_dir.mkdir(parents=True, exist_ok=True)
    for v in videos:
        (videos_dir / v).symlink_to(DATASETS / args.dataset / args.videoset / v)
    pareto_row = write_pareto_row(args.dataset, cfg)
    global_cfg = write_global_config(bench, args.dataset, videos_dir)
    env = dict(os.environ, CONFIG=str(global_cfg))

    # Warm the page cache so neither side pays first-read disk I/O.
    for v in videos:
        (videos_dir / v).read_bytes()

    result: dict = {'dataset': args.dataset, 'config': args.config, 'videos': videos, 'inputs': inputs,
                    'pareto_row': pareto_row, 'loadavg_start': os.getloadavg(), 'scripts': [], 'execution2': {}}
    # Scripts, one stage after another, as the pipeline runs them.
    scripts = ['scripts/p020_exec_classify.py', *(['scripts/p022_exec_prune_polyominoes.py'] if cfg.use_prune else []),
               'scripts/p030_exec_compress.py', 'scripts/p040_exec_detect.py', 'scripts/p050_exec_uncompress.py',
               'scripts/p060_exec_track.py']
    for script in [] if args.skip_scripts else scripts:
        rec = timed([sys.executable, script, '--test'], env, bench / f'{Path(script).stem}.log')
        result['scripts'].append({'script': script, 'seconds': rec['seconds']})
        print(f'{script}: {rec["seconds"]:.1f} s', flush=True)
    result['scripts_total_s'] = sum(r['seconds'] for r in result['scripts']) if result['scripts'] else None

    # Reference tracks: the scripts' fresh outputs from this run, or with --skip-scripts their cached outputs
    # (read-only mount), at the configuration's p060 directory.
    reference_cache = REAL_CACHE if args.skip_scripts else SCRATCH_CACHE
    tracks_dir = build_param_str(classifier=cfg.artifacts.classifier.name, tilesize=p.tile_size, sample_rate=p.sample_rate,
                                 tilepadding=p.tile_padding, canvas_scale=p.canvas_scale, tracker=p.tracker,
                                 tracking_accuracy_threshold=p.mistrack_tolerance,
                                 relevance_threshold=p.relevance_threshold)
    # execution2 as configured, and on one GPU unless classify and detect already share one.
    variants = [('configured', Path(args.config))]
    if cfg.system.stages['detect'].gpu != cfg.system.stages['classify'].gpu:
        variants.append(('single_gpu', single_gpu_config(args.config, bench)))
    for name, cfg_path in variants:
        out = bench / f'execution2-out-{name}'
        rec = timed([sys.executable, '-m', 'execution.main', '--config', str(cfg_path), '--videos', str(videos_dir),
                     '--out', str(out)], env, bench / f'execution2-{name}.log')
        summary = json.loads(rec['last_line'])
        checks = {v: compare_track_lines(
            (out / f'{Path(v).stem}.jsonl').read_text().splitlines(),
            (reference_cache / args.dataset / 'execution' / v / '060_uncompressed_tracks' / tracks_dir / 'tracking.jsonl')
            .read_text().splitlines()) for v in videos}
        result['execution2'][name] = {
            'process_s': rec['seconds'], 'runtime_s': summary['runtime_s'], 'frames': summary['frames'],
            'fps': summary['frames'] / summary['runtime_s'], 'process_fps': summary['frames'] / rec['seconds'],
            'reference': 'cached script outputs' if args.skip_scripts else 'fresh script outputs',
            'outputs_identical': sum(c['equal'] for c in checks.values()),
            'outputs_same_up_to_id_renaming': sum(c['same_up_to_id_renaming'] for c in checks.values()),
            'max_abs_box_diff': max(c['max_abs_box_diff_same_ids'] for c in checks.values()),
            'videos': len(videos)}
        print(name, json.dumps(result['execution2'][name]), flush=True)
    result['loadavg_end'] = os.getloadavg()
    (bench / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
