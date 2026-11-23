"""
Scripts-vs-pipelines equivalence on a scratch cache (both execution2 and profiling against freshly run scripts).

Two sources of nondeterminism are removed first: p040 reads canvases in name order (scripts/p040_exec_detect.py),
and every ILP gets a raised solve limit on both sides, with pairs whose solve still ran into the limit discarded.

  scripts     p020 -> p060 on the test videos with configs/equivalence.yaml (the small grid), p022 at
              --time-limit, its time-limit outputs linked where p030 reads pruned results
  execution2  execution2.tools.compare_scripts for every configuration of the grid, spread over GPUs
  summary     per-stage totals of the execution2 reports (profiling.tools.compare_scripts writes its own)

Run inside the polyis container, with /polyis-cache a scratch cache (never the real one):

    python -m execution2.tools.scripts_equivalence scripts --dataset caldot1-y05 --videos va00.mp4 va01.mp4 va02.mp4
    python -m execution2.tools.scripts_equivalence execution2 --dataset caldot1-y05 --videos ... --out-dir OUT --gpus 0 1
    python -m execution2.tools.scripts_equivalence summary --out-dir OUT
"""

import argparse
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from polyis.io import cache

SCRIPTS = ['p020_exec_classify', 'p022_exec_prune_polyominoes', 'p030_exec_compress', 'p040_exec_detect',
           'p050_exec_uncompress', 'p060_exec_track']


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('scripts')
    s.add_argument('--dataset', required=True)
    s.add_argument('--videos', nargs='+', required=True)
    s.add_argument('--time-limit-s', type=float, default=5.0)
    e = sub.add_parser('execution2')
    e.add_argument('--dataset', required=True)
    e.add_argument('--videos', nargs='+', required=True)
    e.add_argument('--out-dir', required=True)
    e.add_argument('--gpus', type=int, nargs='+', required=True)
    e.add_argument('--time-limit-s', type=float, default=5.0)
    m = sub.add_parser('summary')
    m.add_argument('--out-dir', required=True)
    return parser.parse_args()


def link_time_limited_prunes(dataset: str, videos: list[str], time_limit_s: float) -> int:
    """Point each 022 pruned result p030 reads at p022's --time-limit output; returns the links made."""
    made = 0
    for video in videos:
        tl_root = Path(cache.exec(dataset, 'pruned-polyominoes-tl', video))
        if not tl_root.exists():
            continue
        dst_root = Path(cache.exec(dataset, 'pruned-polyominoes', video))
        dst_root.mkdir(parents=True, exist_ok=True)
        for param_dir in sorted(tl_root.iterdir()):
            src = param_dir / f'tl{time_limit_s}'
            # p022 names the subdirectory after the limit as given (tl5.0 for --time-limit 5).
            if not src.is_dir():
                raise FileNotFoundError(f'{src} missing: p022 did not write this time limit')
            dst = dst_root / param_dir.name
            # A previous untimed result would be read instead of the timed one: refuse rather than guess.
            if dst.exists() and not dst.is_symlink():
                raise FileExistsError(f'{dst} exists and is not a link to a time-limited result')
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src, dst_root))
            made += 1
    return made


def run_scripts(dataset: str, videos: list[str], time_limit_s: float) -> None:
    """The scripts on the small grid, restricted to one dataset and these validation videos."""
    env = {**os.environ, 'CONFIG': 'equivalence.yaml', 'POLYIS_DATASETS': dataset}
    for script in SCRIPTS:
        cmd = ['python', f'scripts/{script}.py', '--valid', '--videos', *videos]
        if script == 'p022_exec_prune_polyominoes':
            cmd += ['--time-limit', str(time_limit_s)]
        print('>>>', ' '.join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        # p030 must read the time-limited pruned results.
        if script == 'p022_exec_prune_polyominoes':
            print(f'linked {link_time_limited_prunes(dataset, videos, time_limit_s)} time-limited prune results',
                  flush=True)


def grid_branches() -> list[dict]:
    """Every configuration of the small grid (unpruned ones once per tracker), as aggregate branches."""
    from profiling.tools.compare_scripts import SMALL_GRID as g
    out = []
    for s, t_r, tol, pad, scale, tracker in itertools.product(g['sample_rates'], g['relevance_thresholds'],
                                                              g['mistrack_tolerances'], g['tile_paddings'],
                                                              g['canvas_scales'], g['trackers']):
        out.append({'classifier': 'ShuffleNet05', 'tile_size': 60, 'sample_rate': s, 'relevance_threshold': t_r,
                    'mistrack_tolerance': tol, 'tile_padding': pad, 'canvas_scale': scale, 'tracker': tracker})
    return out


def config_name(b: dict) -> str:
    tol = 'none' if b['mistrack_tolerance'] is None else f"{int(b['mistrack_tolerance'] * 100):03d}"
    return (f"s{b['sample_rate']}_r{int(b['relevance_threshold'] * 100):03d}_t{tol}_{b['tile_padding']}"
            f"_c{int(b['canvas_scale'] * 100):03d}_{b['tracker']}")


def run_execution2(dataset: str, videos: list[str], out_dir: str, gpus: list[int], time_limit_s: float) -> None:
    """execution2's compare tool for every configuration, one configuration per free GPU at a time."""
    from profiling.aggregate import execution2_config
    from profiling.config import load_sweep_config
    sweep = load_sweep_config(f'profiling/configs/{dataset}.yaml')
    out = Path(out_dir) / dataset
    (out / 'configs').mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in grid_branches():
        cfg = execution2_config(sweep, b)
        cfg['params']['prune_time_limit_s'] = time_limit_s
        path = out / 'configs' / f'{config_name(b)}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(path)
    free = list(gpus)

    def one(path: Path) -> int:
        gpu = free.pop()
        try:
            # The config names the physical GPU, which each pipeline worker pins; the tool's own models see it as cuda:0.
            cfg = yaml.safe_load(path.read_text())
            for stage in ('classify', 'detect'):
                cfg['system']['stages'][stage]['gpu'] = gpu
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            report = out / f'{path.stem}.json'
            log = out / f'{path.stem}.log'
            cmd = ['python', '-m', 'execution2.tools.compare_scripts', '--config', str(path), '--dataset', dataset,
                   '--videoset', 'valid', '--videos', *videos, '--time-limit-s', str(time_limit_s),
                   '--cache-time-limit-s', str(time_limit_s), '--out', str(report)]
            with open(log, 'w') as f:
                code = subprocess.run(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                      stdout=f, stderr=subprocess.STDOUT).returncode
            print(f'{path.stem}: exit {code}', flush=True)
            return code
        finally:
            free.append(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        codes = list(pool.map(one, jobs))
    if any(codes):
        raise SystemExit(f'{sum(c != 0 for c in codes)} of {len(codes)} configurations failed; see the .log files')


def summarize(out_dir: str) -> dict:
    """Per dataset and stage: compared vs exactly equal, over every execution2 report."""
    summary = {}
    for ds_dir in sorted(p for p in Path(out_dir).iterdir() if p.is_dir()):
        t = {k: {'compared': 0, 'equal': 0} for k in ('classify', 'prune', 'pack', 'detect_raw_in_order',
                                                       'detect_unpacked_in_order', 'track', 'chained_tracks')}
        failures = []
        for report_path in sorted(ds_dir.glob('*.json')):
            r = json.loads(report_path.read_text())
            for video, v in r['isolated'].items():
                checks = {'classify': v['classify']['equal'],
                          'pack': v['pack']['index_maps_equal'] and v['pack']['offset_lookups_equal']
                          and v['pack']['canvases_equal'],
                          'detect_raw_in_order': v['detect_in_script_order'].get('p040_raw_equal', False),
                          'detect_unpacked_in_order': v['detect_in_script_order'].get('p050_equal_in_order', False),
                          'track': v['track']['equal']}
                if 'prune' in v and v['prune']['ilp']['compared']:
                    checks['prune'] = v['prune']['equal']
                for k, ok in checks.items():
                    t[k]['compared'] += 1
                    t[k]['equal'] += bool(ok)
                    if not ok:
                        failures.append(f'{report_path.stem} {video} {k}')
            if r['chained']['compared']:
                for video, c in r['chained']['videos'].items():
                    t['chained_tracks']['compared'] += 1
                    t['chained_tracks']['equal'] += bool(c['equal'])
                    if not c['equal']:
                        failures.append(f'{report_path.stem} {video} chained_tracks')
        summary[ds_dir.name] = {'totals': t, 'failures': failures}
    Path(out_dir, 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main() -> None:
    args = parse_args()
    if args.command == 'scripts':
        run_scripts(args.dataset, args.videos, args.time_limit_s)
    elif args.command == 'execution2':
        run_execution2(args.dataset, args.videos, args.out_dir, args.gpus, args.time_limit_s)
    else:
        print(json.dumps(summarize(args.out_dir), indent=1))


if __name__ == '__main__':
    main()
