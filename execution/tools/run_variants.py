"""
Run selected Tetris configurations through execution and record what the evaluation stages read.

  config  DATASET VARIANT_ID           print the execution config (YAML) for one configuration of the grid
  run     DATASET VIDEOSET SELECTION   run every configuration in SELECTION (reproduce/selection.py's CSV) over the
                                       split's videos, one at a time on one GPU; writes each video's tracks where p110
                                       reads them (060_uncompressed_tracks/<variant_id>/tracking.jsonl) and the run's
                                       runtime to 080_throughput/execution/<videoset>/<variant_id>.json
  record  DATASET VIDEOSET             add those runtimes to 080_throughput/measurements/query_execution_overall.csv
                                       (written by p121), which p130 joins with accuracy into 090_tradeoff

Throughput is execution's own runtime: from the first result to the last, after a warm-up copy of the last video
(execution/main.py), so process start-up and model loading are excluded.  p130 sums a configuration's per-video
times into one split total; each video gets the share of the runtime its frame count is of the split's, so the
total is exactly execution's runtime.

    python -m execution.tools.run_variants config jnc0 ShuffleNet05_60_4_r025_none_s100_sortcython > jnc0.yaml
    python -m execution.tools.run_variants run jnc0 test selection.csv
    python -m execution.tools.run_variants record jnc0 test
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from execution.config import DETECTOR_FIELDS, parse_config
from tetris.io import cache, store
from tetris.utilities import build_param_str

ROOT = Path(__file__).resolve().parents[2]
CACHE = '/polyis-cache'
RETINANET_CONFIG = '/polyis/configs/retinanet.json'
# reproduce/selection.py's columns (tetris.pareto.PARETO_PARAM_COLS).
PARAM_COLUMNS = ['classifier', 'tilesize', 'sample_rate', 'tracking_accuracy_threshold', 'relevance_threshold',
                 'tilepadding', 'canvas_scale', 'tracker']


def parse_args():
    parser = argparse.ArgumentParser(description='Run selected configurations through execution.')
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('config')
    c.add_argument('dataset')
    c.add_argument('variant_id')
    r = commands.add_parser('run')
    r.add_argument('dataset')
    r.add_argument('videoset', choices=['valid', 'test'])
    r.add_argument('selection', type=Path)
    rec = commands.add_parser('record')
    rec.add_argument('dataset')
    rec.add_argument('videoset', choices=['valid', 'test'])
    return parser.parse_args()


def variant_id(p: dict) -> str:
    """The configuration's name: its stage-060 directory and the evaluation's variant_id."""
    return build_param_str(classifier=p['classifier'], tilesize=p['tilesize'], sample_rate=p['sample_rate'],
                           tracking_accuracy_threshold=p['tracking_accuracy_threshold'],
                           relevance_threshold=p['relevance_threshold'], tilepadding=p['tilepadding'],
                           canvas_scale=p['canvas_scale'], tracker=p['tracker'])


def grid_params(dataset: str, vid: str) -> dict:
    """A configuration of the sweep grid by its variant_id (the evaluation's manifest)."""
    from evaluation.manifests import build_polytris_variant_manifest
    manifest = build_polytris_variant_manifest()
    rows = manifest[manifest['variant_id'] == vid]
    if rows.empty:
        raise SystemExit(f'{vid} is not a configuration of the sweep grid (configs/global.yaml)')
    return parse_row(rows.iloc[0].to_dict())


def parse_row(row: dict) -> dict:
    """Typed parameters from a selection CSV row or manifest row (missing tolerance = no pruning)."""
    tol = row['tracking_accuracy_threshold']
    tol = None if tol is None or str(tol).strip() in ('', 'nan', 'None') else float(tol)
    return {'classifier': str(row['classifier']), 'tilesize': int(row['tilesize']),
            'sample_rate': int(row['sample_rate']), 'tracking_accuracy_threshold': tol,
            'relevance_threshold': float(row['relevance_threshold']), 'tilepadding': str(row['tilepadding']),
            'canvas_scale': float(row['canvas_scale']), 'tracker': str(row['tracker'])}


def detector_spec(dataset: str) -> dict:
    """The dataset's detector from configs/detectors.yaml, with only the fields execution takes."""
    detectors = yaml.safe_load((ROOT / 'configs' / 'detectors.yaml').read_text())
    entry = detectors['dataset_detector_mapping'][detectors['dataset_name_mapping'][dataset]]
    kind = entry['detector']
    if kind == 'retina':
        return {'type': 'retina', 'config_path': RETINANET_CONFIG}
    return {'type': kind, **{k: entry[k] for k in DETECTOR_FIELDS[kind]}}


def build_config(dataset: str, p: dict) -> dict:
    """execution's config for one configuration: the dataset's trained artifacts, JPEG canvases (as the scripts
    write them), and one GPU shared by the classifier and the detector."""
    index = f'{CACHE}/{dataset}/indexing'
    model = f'{index}/training/results/{p["classifier"]}_{p["tilesize"]}'
    artifacts = {
        'classifier': {'name': p['classifier'], 'weights': f'{model}/model_best.pth',
                       'compilation_benchmark': f'{model}/model_compilation.jsonl'},
        'never_relevant_bitmap': f'{index}/always_relevant/{p["tilesize"]}_all.npy',
        'detector': detector_spec(dataset),
    }
    stages = {'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpu': 0}, 'pack': {'procs': 2},
              'detect': {'procs': 1, 'gpu': 0}, 'track': {'procs': 2}}
    # Pruning needs the tracker's per-tile sampling limits and exactly one solver process.
    if p['tracking_accuracy_threshold'] is not None:
        artifacts['max_rate_table'] = f'{index}/track_rate/{p["tracker"]}_{p["tilesize"]}/max_rate_table.npy'
        stages['prune'] = {'procs': 1}
    params = {'sample_rate': p['sample_rate'], 'relevance_threshold': p['relevance_threshold'],
              'mistrack_tolerance': p['tracking_accuracy_threshold'], 'tile_padding': p['tilepadding'],
              'tracker': p['tracker'], 'tile_size': p['tilesize'], 'canvas_scale': p['canvas_scale'],
              'canvas_format': 'jpeg'}
    return {'params': params, 'artifacts': artifacts, 'system': {'queue_maxsize': 2, 'stages': stages}}


def runtime_path(dataset: str, videoset: str, vid: str) -> Path:
    """Where run writes one configuration's runtime record."""
    return Path(cache.eval(dataset, 'tp', 'execution', videoset, f'{vid}.json'))


def run_selection(dataset: str, videoset: str, selection: Path) -> None:
    """Every selected configuration through execution over the split's videos, one after another."""
    from execution import main as e2
    with open(selection, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'{selection} selects no configurations')
    videos_dir = str(store.dataset(dataset, videoset))
    for i, row in enumerate(rows, 1):
        p = parse_row(row)
        vid = variant_id(p)
        # The selection's own id must name the same configuration as its parameters.
        if row.get('variant_id') and row['variant_id'] != vid:
            raise SystemExit(f'{selection}: variant_id {row["variant_id"]} does not match its parameters ({vid})')
        cfg = parse_config(build_config(dataset, p))
        print(f'[{i}/{len(rows)}] {dataset} {videoset} {vid}', flush=True)
        with tempfile.TemporaryDirectory(prefix='execution-tracks-') as out:
            summary = e2.execute(cfg, videos_dir, out)
            per_video = {}
            for video in e2.list_videos(videos_dir):
                name = os.path.basename(video)
                # p110 reads each configuration's tracks from the scripts' stage-060 location.
                dst = Path(cache.exec(dataset, 'ucomp-tracks', name, vid, 'tracking.jsonl'))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(os.path.join(out, Path(name).stem + '.jsonl'), dst)
                per_video[name] = e2.probe(video).frame_count
        record = {'dataset': dataset, 'videoset': videoset, 'variant_id': vid, 'params': p,
                  'runtime_s': summary['runtime_s'], 'frames': summary['frames'], 'video_frames': per_video}
        path = runtime_path(dataset, videoset, vid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + '\n')
        print(f'    {summary["frames"]} frames in {summary["runtime_s"]:.1f} s = '
              f'{summary["frames"] / summary["runtime_s"]:.1f} fps', flush=True)


def runtime_rows(record: dict) -> list[dict]:
    """One query_execution_overall row per video: its frame share of the configuration's runtime."""
    total_frames = sum(record['video_frames'].values())
    p = record['params']
    return [{'stage': 'execution', 'dataset': record['dataset'], 'videoset': record['videoset'], 'video': video,
             'variant': 'polytris', 'variant_id': record['variant_id'], 'classifier': p['classifier'],
             'tilesize': p['tilesize'], 'tilepadding': p['tilepadding'], 'sample_rate': p['sample_rate'],
             'tracking_accuracy_threshold': p['tracking_accuracy_threshold'], 'canvas_scale': p['canvas_scale'],
             'tracker': p['tracker'], 'time': record['runtime_s'] * frames / total_frames}
            for video, frames in sorted(record['video_frames'].items())]


def record_runtimes(dataset: str, videoset: str) -> int:
    """Add every recorded execution runtime of the split to p121's query table; returns the rows added."""
    records = sorted(Path(cache.eval(dataset, 'tp', 'execution', videoset)).glob('*.json'))
    if not records:
        raise SystemExit(f'no execution runtimes for {dataset} {videoset}: run `run` first')
    table = Path(cache.eval(dataset, 'tp', 'measurements', 'query_execution_overall.csv'))
    if not table.exists():
        raise SystemExit(f'{table} is missing: run evaluation/p121_throughput_compute.py first')
    with open(table, newline='') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        kept = [r for r in reader]
    records = [json.loads(path.read_text()) for path in records]
    ids = {r['variant_id'] for r in records}
    # A recorded configuration's time is execution's alone: drop every earlier row of it for this split, whether
    # from a previous record or from the scripts' runtime logs of an older run (p130 would add them up).
    kept = [r for r in kept if not (r.get('videoset') == videoset and r.get('variant_id') in ids)]
    added = [row for rec in records for row in runtime_rows(rec)]
    missing = set(added[0]) - set(columns)
    if missing:
        raise SystemExit(f'{table} lacks the columns {sorted(missing)}')
    with open(table, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows({k: ('' if v is None else v) for k, v in row.items()} for row in added)
    return len(added)


def main() -> None:
    args = parse_args()
    if args.command == 'config':
        print(yaml.safe_dump(build_config(args.dataset, grid_params(args.dataset, args.variant_id)), sort_keys=False),
              end='')
    elif args.command == 'run':
        run_selection(args.dataset, args.videoset, args.selection)
    else:
        print(f'added {record_runtimes(args.dataset, args.videoset)} execution runtime rows')


if __name__ == '__main__':
    main()
