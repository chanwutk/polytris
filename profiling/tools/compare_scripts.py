"""
Check a profiling sweep against the scripts' cached validation outputs (profiling/PLAN.md 14A, step 7).

Runs profiling.main on a small grid over a few validation videos into a scratch output_root, then compares
every saved result with the scripts' cache (read only):

  020 scores           exact (classify runs once on every frame; p020 ran once per sample rate)
  022 pruned grids     exact, on (config, video) pairs where both solves were optimal: ours (status 2, run with
                       a raised time limit) and p022's (logged solve time below its 0.1 s limit); kept tiles too
  033 index maps,      exact
      offset lookups
  050 detections       per frame, as box sets: identical frames, box counts, largest coordinate difference
                       (bit-identity needs p040's canvas order, which compare_scripts in execution2 checks)
  060 tracks           equal up to track-id renaming
  HOTA per video       ours vs p110's run_trackeval on the cached 060 tracks, same reference

    python -m profiling.tools.compare_scripts --config profiling/configs/caldot1-y05.yaml \
        --videos va00.mp4 va01.mp4 --output-root /polyis-profiling-check --out /polyis-profiling-check/compare.json
"""

import argparse
import copy
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml

from execution2.tools.compare_scripts import cached_solve_ms, compare_detections, compare_track_lines
from profiling import main as pmain
from profiling.config import parse_sweep_config
from polyis.io import store

# The small grid: sample rates x T_r x (no pruning + 2 tolerances x 2 trackers) x 2 paddings.
SMALL_GRID = {'sample_rates': [1, 4], 'relevance_thresholds': [0.25, 0.5], 'mistrack_tolerances': [None, 0.4, 0.8],
              'tile_paddings': ['none', 'tr'], 'canvas_scales': [1.0], 'trackers': ['sortcython', 'bytetrackcython']}
# A solve that ran at least this share of its limit is treated as stopped by it (neither side logs a status
# for p022's solves, only their time), so its (config, video) pair is not compared.
LIMIT_SHARE = 0.95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--cache', default='/polyis-cache')
    parser.add_argument('--time-limit-s', type=float, default=10.0)
    parser.add_argument('--cache-time-limit-s', type=float, default=0.1)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def score_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path).read().splitlines() if l.strip()]


def grids(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    return [r['idx'] for r in rows], np.stack([np.frombuffer(bytes.fromhex(r['classification_hex']), np.uint8)
                                               .reshape(r['classification_size']) for r in rows])


def detection_rows(path: str) -> list[tuple[int, list]]:
    """(frame_idx, bboxes) per line of a detections.jsonl, in file order."""
    return [(r['frame_idx'], r['bboxes']) for r in score_rows(path)]


def pruned_key(video: str, param: str) -> tuple | None:
    """(video, sample rate, tolerance, T_r, tracker) of a pruned branch's param string; None when unpruned.

    Pruned strings read <classifier>_<tile>_<s>_<tol>_r<T_r>[_<pad>_s<scale>]_<tracker>; unpruned ones have no
    tolerance field (the fourth field is already r<T_r>)."""
    parts = param.split('_')
    if parts[3].startswith('r'):
        return None
    return video, parts[2], parts[3], parts[4], parts[-1]


def compare_score_files(ours: str, theirs: str) -> dict:
    (oi, og), (ti, tg) = grids(score_rows(ours)), grids(score_rows(theirs))
    if oi != ti or og.shape != tg.shape:
        return {'equal': False, 'frames_match': oi == ti}
    diff = np.abs(og.astype(int) - tg.astype(int))
    return {'equal': not diff.any(), 'max_abs_diff': int(diff.max()), 'differing_values': int(np.count_nonzero(diff))}


def hota_from_cache(cache_execution: str, video: str, tracks_param: str, reference_file: str, scratch: str) -> np.ndarray:
    """p110's run_trackeval on the scripts' cached tracks for one video."""
    from evaluation.p110_accuracy_compute import build_metric_objects, run_trackeval
    track_dir = os.path.join(scratch, 'cached', tracks_param)
    os.makedirs(os.path.join(track_dir, video), exist_ok=True)
    src = os.path.join(cache_execution, video, '060_uncompressed_tracks', tracks_param, 'tracking.jsonl')
    dst = os.path.join(track_dir, video, 'tracking.jsonl')
    if not os.path.exists(dst):
        os.symlink(src, dst)
    results = run_trackeval({'output_fol': os.path.join(scratch, 'te'), 'output_sub_fol': 'x', 'input_gt': reference_file,
                             'input_track': 'tracking.jsonl', 'skip': 1, 'tracker': 'x', 'seq_list': [video],
                             'input_dir': track_dir, 'input_gt_dir': cache_execution, 'input_track_dir': track_dir},
                            build_metric_objects(['HOTA']), os.path.join(scratch, 'te'), 1)
    return np.asarray(results[video]['vehicle']['HOTA']['HOTA'], dtype=float)


def main_cli() -> None:
    args = parse_args()
    raw = yaml.safe_load(open(args.config))
    raw = copy.deepcopy(raw)
    raw['output_root'] = args.output_root
    # Every tolerance runs (no merging), so each script output has its own counterpart.
    raw['params'].update(SMALL_GRID, prune_time_limit_s=args.time_limit_s, reduce_search_space=False)
    cfg = parse_sweep_config(raw)
    ds = cfg.dataset
    videos = [str(store.dataset(ds, cfg.videoset, v)) for v in args.videos]
    summary = pmain.execute(cfg, args.config, videos=videos)

    ours_ex = os.path.join(args.output_root, ds, 'execution')
    cache_ex = os.path.join(args.cache, ds, 'execution')
    report: dict = {'grid': SMALL_GRID, 'time_limit_s': args.time_limit_s, 'cache_time_limit_s': args.cache_time_limit_s,
                    'summary': summary, 'stages': {}}
    cached_optimal_below_ms = LIMIT_SHARE * args.cache_time_limit_s * 1000
    ours_optimal_below_ms = LIMIT_SHARE * args.time_limit_s * 1000

    def rel(path: str) -> str:
        return os.path.relpath(path, ours_ex)

    # 020 and 022: every saved score file.
    for stage in ('020_relevancy', '022_pruned_polyominoes'):
        rows = []
        for ours in sorted(glob.glob(os.path.join(ours_ex, '*', stage, '*', 'score', 'score.jsonl'))):
            theirs = os.path.join(cache_ex, rel(ours))
            entry = {'path': rel(ours)}
            if stage == '022_pruned_polyominoes':
                ilp = json.load(open(os.path.join(os.path.dirname(os.path.dirname(ours)), 'ilp.json')))
                cached_ms = cached_solve_ms(os.path.dirname(os.path.dirname(theirs)))
                entry.update(ours_status=ilp['status'], ours_kept=ilp['kept_tiles'], cached_solve_ms=cached_ms)
                both_optimal = (ilp['status'] == 2 and ilp['solve_ms'] < ours_optimal_below_ms
                                and cached_ms is not None and cached_ms < cached_optimal_below_ms)
                entry['compared'] = both_optimal
                if not both_optimal:
                    rows.append(entry)
                    continue
                cached_kept = int(np.count_nonzero(grids(score_rows(theirs))[1]))
                entry['cached_kept'] = cached_kept
            entry.update(compare_score_files(ours, theirs))
            rows.append(entry)
        report['stages'][stage] = rows
    # Branches below a prune pair that was not compared (either solve hit its limit) are not compared either.
    skipped = {pruned_key(*Path(r['path']).parts[0:3:2]) for r in report['stages']['022_pruned_polyominoes']
               if not r['compared']}

    def compared(path: str) -> bool:
        video, _, param = Path(path).parts[:3]
        return pruned_key(video, param) not in skipped

    # 033: index maps and offset lookups.
    rows = []
    for d in sorted(glob.glob(os.path.join(ours_ex, '*', '033_compressed_frames', '*'))):
        theirs = os.path.join(cache_ex, rel(d))
        names = sorted(Path(d, 'index_maps').glob('*.npy'))
        cached = sorted(Path(theirs, 'index_maps').glob('*.npy'))
        equal = [n.name for n in names] == [c.name for c in cached] and all(
            np.array_equal(np.load(n), np.load(c)) for n, c in zip(names, cached)) and all(
            Path(d, 'offset_lookups', n.stem + '.jsonl').read_text() ==
            Path(theirs, 'offset_lookups', n.stem + '.jsonl').read_text() for n in names)
        rows.append({'path': rel(d), 'compared': compared(rel(d)), 'collages': [len(names), len(cached)],
                     'equal': bool(equal)})
    report['stages']['033_compressed_frames'] = rows

    # 050: detections per frame.
    rows = []
    for ours in sorted(glob.glob(os.path.join(ours_ex, '*', '050_uncompressed_detections', '*', 'detections.jsonl'))):
        theirs = os.path.join(cache_ex, rel(ours))
        mine = {r['frame_idx']: r['bboxes'] for r in score_rows(ours)}
        # Exact: same frames, and per frame the same boxes in the same order (p050's merge order).
        rows.append({'path': rel(ours), 'compared': compared(rel(ours)),
                     'equal_in_order': detection_rows(ours) == detection_rows(theirs),
                     **compare_detections(mine, score_rows(theirs))})
    report['stages']['050_uncompressed_detections'] = rows

    # 060 tracks and per-video HOTA.
    records = [json.loads(l) for l in open(os.path.join(args.output_root, ds, 'profiling', 'records.jsonl'))]
    ours_hota = {}
    from profiling.messages import Branch
    for r in records:
        b = Branch(tuple(r['branch'].items()))
        param = b.param_str('sample_rate', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding', 'canvas_scale',
                            'tracker')
        ours_hota[(r['video'], param)] = np.asarray(r['metrics']['HOTA']['HOTA'], dtype=float)
    rows = []
    scratch = os.path.join(args.output_root, 'hota-scratch')
    for ours in sorted(glob.glob(os.path.join(ours_ex, '*', '060_uncompressed_tracks', '*', 'tracking.jsonl'))):
        video, _, param = Path(rel(ours)).parts[:3]
        theirs = os.path.join(cache_ex, rel(ours))
        entry = {'path': rel(ours), 'compared': compared(rel(ours)),
                 **compare_track_lines(open(ours).read().splitlines(), open(theirs).read().splitlines())}
        cached = hota_from_cache(cache_ex, video, param, cfg.evaluation.reference_file, scratch)
        mine = ours_hota[(video, param)]
        entry.update(hota_equal=bool(np.allclose(mine, cached, rtol=0, atol=1e-12)),
                     hota_ours=float(mine.mean()), hota_cached=float(cached.mean()))
        rows.append(entry)
    report['stages']['060_uncompressed_tracks'] = rows

    # Totals per stage.
    totals = {}
    for stage, rows in report['stages'].items():
        key = {'060_uncompressed_tracks': 'same_up_to_id_renaming', '050_uncompressed_detections': None}.get(stage, 'equal')
        compared = [r for r in rows if r.get('compared', True)]
        if key:
            totals[stage] = {'compared': len(compared), 'equal': sum(bool(r.get(key)) for r in compared),
                             'rows': len(rows)}
            if stage == '060_uncompressed_tracks':
                # Byte-identical tracking.jsonl, track ids included (stricter than equal up to renaming).
                totals[stage]['text_identical'] = sum(bool(r.get('equal')) for r in compared)
        else:
            totals[stage] = {'compared': len(compared), 'rows': len(rows),
                             'equal_in_order': sum(r['equal_in_order'] for r in compared),
                             'frames_identical': sum(r['frames_identical'] for r in compared),
                             'frames': sum(r['frames'] for r in compared),
                             'count_mismatch_frames': sum(r['frames_with_different_box_counts'] for r in compared),
                             'max_abs_box_diff': max((r['max_abs_box_diff_same_count'] for r in compared), default=0.0)}
    t = [r for r in report['stages']['060_uncompressed_tracks'] if r['compared']]
    totals['hota_per_video'] = {'compared': len(t), 'equal': sum(r['hota_equal'] for r in t),
                                'max_abs_diff': max((abs(r['hota_ours'] - r['hota_cached']) for r in t), default=0.0)}
    report['totals'] = totals
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(totals, indent=1))


if __name__ == '__main__':
    main_cli()
