"""Rescore the canonical LEAP results (measurements-results/leap_canonical.json) with the padded scoring window.

The canonical LEAP HOTA for CalDoT1, CalDoT2 and Amsterdam was scored by the ad hoc rerun scripts with TrackEval's
truncated window. This script rescores the saved raw/linear/pattern reconstructions of each canonical result:
  1. with pad_scoring_window=False, and checks that it reproduces the recorded HOTA exactly (same inputs);
  2. with the canonical (padded) window, and writes those values: each source comparison's version section gets the
     new 'hota' (the old one kept as 'hota_unpadded'), best reconstruction and best HOTA, and leap_canonical.json gets
     the best variant's HOTA and the comparison file's new sha256.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from evaluation.manifests import list_split_videos
from measurements.common import DATASET_LABELS, metric_summary, output_path, trackeval_rerun

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'measurements-results' / 'leap_canonical.json'
VARIANTS = ('raw', 'linear', 'pattern')
KEYS = ('HOTA', 'DetA', 'AssA', 'LocA')


def parse_args():
    parser = argparse.ArgumentParser(description='Rescore canonical LEAP HOTA with the padded scoring window')
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def tasks_for(dataset: str, tracking_root: Path, pad: bool) -> list[dict]:
    """One TrackEval task per reconstruction, configured as r1o2_leap_otif_rerun.write_variants configures it."""
    videos = sorted(list_split_videos(dataset, 'test'))
    gt_dir = str(Path(output_path(dataset, 'ground truth', '', videos[0])).parents[2])
    base = tracking_root.parent
    return [dict(dataset=dataset, variant=v, pad=pad, videos=videos, gt_dir=gt_dir,
                 gt_file=str(Path('003_groundtruth', 'tracking.jsonl')), track_dir=str(base / v), track_file='tracking.jsonl',
                 output_dir=str(base / 'trackeval-rescore' / ('padded' if pad else 'unpadded') / v)) for v in VARIANTS]


def run(task: dict) -> dict:
    # HOTA only, as p141 and the canonical scorers run it.
    metrics = trackeval_rerun(task['gt_dir'], task['gt_file'], task['track_dir'], task['track_file'], task['videos'],
                              task['output_dir'], ('HOTA',), pad_scoring_window=task['pad'])
    hota = metric_summary(metrics)['HOTA']
    return dict(dataset=task['dataset'], variant=task['variant'], pad=task['pad'], hota={k: hota[k] for k in KEYS})


def updated_section(section: dict, padded: dict[str, dict]) -> dict:
    """A comparison version section with padded HOTA, keeping the truncated-window values."""
    new = dict(section)
    new['hota_unpadded'] = section['hota']
    new['hota'] = padded
    # The best reconstruction is chosen as the canonical scorers chose it, between linear and pattern.
    new['best_reconstruction'] = max(('linear', 'pattern'), key=lambda v: padded[v]['HOTA'])
    new['best_hota'] = padded[new['best_reconstruction']]['HOTA']
    return new


def main():
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    selected = manifest['datasets']
    tasks = [t for d, s in selected.items() for pad in (False, True) for t in tasks_for(d, ROOT / s['tracking_root'], pad)]
    with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
        results = pool.map(run, tasks)
    scored = {(r['dataset'], r['pad'], r['variant']): r['hota'] for r in results}
    comparisons: dict[Path, dict] = {}
    for dataset, entry in selected.items():
        path = ROOT / entry['source_comparison']
        comparison = comparisons.setdefault(path, json.loads(path.read_text()))
        section = comparison['datasets'][DATASET_LABELS[dataset]]['versions'][entry.get('result_key', 'regions_v2')]
        # 1. The truncated window must reproduce the recorded values exactly.
        for v in VARIANTS:
            for k in KEYS:
                assert abs(scored[(dataset, False, v)][k] - section['hota'][v][k]) < 1e-12, (dataset, v, k)
        # 2. Padded values replace them.
        padded = {v: scored[(dataset, True, v)] for v in VARIANTS}
        new = updated_section(section, padded)
        print(f"{DATASET_LABELS[dataset]}: best {section.get('best_reconstruction')} {section['hota'][section.get('best_reconstruction', 'linear')]['HOTA']:.4f}"
              f" -> {new['best_reconstruction']} {new['best_hota']:.4f}")
        comparison['datasets'][DATASET_LABELS[dataset]]['versions'][entry.get('result_key', 'regions_v2')] = new
        entry['hota'] = padded[new['best_reconstruction']]
        entry['best_reconstruction'] = new['best_reconstruction']
    if args.dry_run:
        return
    for path, comparison in comparisons.items():
        comparison.setdefault('method', {})['hota_scoring_window'] = 'padded to the longer file (2026-11-03); truncated values kept as hota_unpadded'
        path.write_text(json.dumps(comparison, indent=2) + '\n')
    for dataset, entry in selected.items():
        entry['source_sha256'] = hashlib.sha256((ROOT / entry['source_comparison']).read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {MANIFEST} and {len(comparisons)} comparison file(s)')


if __name__ == '__main__':
    main()
