"""
Check a profiling sweep against the scripts' validation outputs (experiments/fanout_profiling/PLAN.md 14A, step 7).

Runs experiments.fanout_profiling.main on a small grid over a few validation videos into a scratch output_root, with prune's ILP
replaced by the scripts' saved 022 grids (a solve stopped by its 0.1 s limit is not unique, so solving would
make every pruned branch below it incomparable; the ILP is checked by execution's compare_scripts).  Then it
compares every saved result with the scripts' cache (read only):

  020 scores           exact (classify runs once on every frame; p020 ran once per sample rate)
  022 pruned grids     exact (the scripts' own, saved under each branch's name: checks the fan-out's labels)
  033 index maps,      exact
      offset lookups
  050 detections       exact: same frames, per frame the same boxes in the same order (p040 reads canvases in
                       name order, as profiling detects them); also box-set statistics
  060 tracks           byte-identical, and equal up to track-id renaming
  HOTA per video       ours vs tetris.trackeval.evaluate.run_trackeval on the cached 060 tracks, same reference

    python -m experiments.tools.profiling.compare_scripts --config experiments/fanout_profiling/configs/caldot1-y05.yaml \
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

from experiments.tools.execution.compare_scripts import compare_detections, compare_track_lines
from experiments.fanout_profiling import main as pmain
from experiments.fanout_profiling.config import parse_sweep_config
from tetris.io import store

# The small grid: sample rates x T_r x (no pruning + 2 tolerances x 2 trackers) x 2 paddings.
SMALL_GRID = {'sample_rates': [1, 4], 'relevance_thresholds': [0.25, 0.5], 'mistrack_tolerances': [None, 0.4, 0.8],
              'tile_paddings': ['none', 'tr'], 'canvas_scales': [1.0], 'trackers': ['sortcython', 'bytetrackcython']}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--cache', default='/polyis-cache')
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def score_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path).read().splitlines() if l.strip()]


def grids(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    return [r['idx'] for r in rows], np.stack([np.frombuffer(bytes.fromhex(r['classification_hex']), np.uint8)
                                               .reshape(r['classification_size']) for r in rows])


def compare_score_files(ours: str, theirs: str) -> dict:
    (oi, og), (ti, tg) = grids(score_rows(ours)), grids(score_rows(theirs))
    if oi != ti or og.shape != tg.shape:
        return {'equal': False, 'frames_match': oi == ti}
    diff = np.abs(og.astype(int) - tg.astype(int))
    return {'equal': not diff.any(), 'max_abs_diff': int(diff.max()), 'differing_values': int(np.count_nonzero(diff))}


def detection_rows(path: str) -> list[tuple[int, list]]:
    """(frame_idx, bboxes) per line of a detections.jsonl, in file order."""
    return [(r['frame_idx'], r['bboxes']) for r in score_rows(path)]


def script_solve(dataset: str, state: dict, sc, tracker: str, tolerance: float, relevance_threshold: float) -> tuple:
    """Test-only stand-in for experiments.fanout_profiling.stages.solve_ilp: the scripts' saved 022 grids for this branch."""
    from tetris.utilities import load_pruned_classification_results
    b = sc.branch
    idx, grid = grids(load_pruned_classification_results(dataset, os.path.basename(sc.video), b['tile_size'],
                                                         b['classifier'], tracker, tolerance, relevance_threshold,
                                                         b['sample_rate']))
    # The saved grids must cover exactly this branch's sampled frames, in order.
    if idx != sc.decoded.sampled.tolist():
        raise ValueError(f'{sc.video}: script 022 frames {idx[:5]}... differ from the sampled frames')
    return grid, {'status': 'script', 'solve_ms': 0.0}


def script_prune_setup(cfg, worker) -> dict:
    """profiling's prune setup, with the ILP solve swapped for script_solve."""
    from functools import partial
    from experiments.fanout_profiling import stages as st
    state = st.prune_setup(cfg, worker)
    state['solve'] = partial(script_solve, cfg.dataset)
    return state


def build_with_script_prune(cfg, prefix, completion) -> list:
    """profiling's stages with prune's solve reading the scripts' grids (fan-out and labels unchanged)."""
    from functools import partial
    from experiments.fanout_profiling import stages as st
    return [s._replace(setup=partial(script_prune_setup, cfg)) if s.name == 'prune' else s
            for s in st.build_stages(cfg, prefix, completion)]


def hota_from_cache(cache_execution: str, video: str, tracks_param: str, reference_file: str, scratch: str) -> np.ndarray:
    """tetris.trackeval.evaluate.run_trackeval on the scripts' cached tracks for one video."""
    from tetris.trackeval.evaluate import build_metric_objects, run_trackeval
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
    raw['params'].update(SMALL_GRID, reduce_search_space=False)
    cfg = parse_sweep_config(raw)
    ds = cfg.dataset
    videos = [str(store.dataset(ds, cfg.videoset, v)) for v in args.videos]
    summary = pmain.execute(cfg, args.config, build=build_with_script_prune, videos=videos)
    report = {'grid': SMALL_GRID, 'prune': 'scripts 022 grids', 'summary': summary,
              **compare_with_scripts(args.output_root, ds, args.cache, cfg.evaluation.reference_file)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['totals'], indent=1))


def compare_with_scripts(output_root: str, ds: str, cache_root: str, reference_file: str) -> dict:
    """Every result a sweep saved under output_root vs the scripts' outputs under cache_root: per-stage rows and
    totals ({'stages': ..., 'totals': ...}).  Shared by this CLI and experiments/tests/integration."""
    ours_ex = os.path.join(output_root, ds, 'execution')
    cache_ex = os.path.join(cache_root, ds, 'execution')
    report: dict = {'stages': {}}

    def rel(path: str) -> str:
        return os.path.relpath(path, ours_ex)

    # 020 and 022: every saved score file.
    for stage in ('020_relevancy', '022_pruned_polyominoes'):
        rows = []
        for ours in sorted(glob.glob(os.path.join(ours_ex, '*', stage, '*', 'score', 'score.jsonl'))):
            theirs = os.path.join(cache_ex, rel(ours))
            entry = {'path': rel(ours)}
            entry.update(compare_score_files(ours, theirs))
            rows.append(entry)
        report['stages'][stage] = rows

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
        rows.append({'path': rel(d), 'collages': [len(names), len(cached)], 'equal': bool(equal)})
    report['stages']['033_compressed_frames'] = rows

    # 050: detections per frame.
    rows = []
    for ours in sorted(glob.glob(os.path.join(ours_ex, '*', '050_uncompressed_detections', '*', 'detections.jsonl'))):
        theirs = os.path.join(cache_ex, rel(ours))
        mine = {r['frame_idx']: r['bboxes'] for r in score_rows(ours)}
        # Exact: same frames, and per frame the same boxes in the same order (p050's merge order).
        rows.append({'path': rel(ours), 'equal_in_order': detection_rows(ours) == detection_rows(theirs),
                     **compare_detections(mine, score_rows(theirs))})
    report['stages']['050_uncompressed_detections'] = rows

    # 060 tracks and per-video HOTA.
    records = [json.loads(l) for l in open(os.path.join(output_root, ds, 'profiling', 'records.jsonl'))]
    ours_hota = {}
    from experiments.fanout_profiling.messages import Branch
    for r in records:
        b = Branch(tuple(r['branch'].items()))
        param = b.param_str('sample_rate', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding', 'canvas_scale',
                            'tracker')
        ours_hota[(r['video'], param)] = np.asarray(r['metrics']['HOTA']['HOTA'], dtype=float)
    rows = []
    scratch = os.path.join(output_root, 'hota-scratch')
    for ours in sorted(glob.glob(os.path.join(ours_ex, '*', '060_uncompressed_tracks', '*', 'tracking.jsonl'))):
        video, _, param = Path(rel(ours)).parts[:3]
        theirs = os.path.join(cache_ex, rel(ours))
        entry = {'path': rel(ours), **compare_track_lines(open(ours).read().splitlines(), open(theirs).read().splitlines())}
        cached = hota_from_cache(cache_ex, video, param, reference_file, scratch)
        mine = ours_hota[(video, param)]
        entry.update(hota_equal=bool(np.allclose(mine, cached, rtol=0, atol=1e-12)),
                     hota_ours=float(mine.mean()), hota_cached=float(cached.mean()))
        rows.append(entry)
    report['stages']['060_uncompressed_tracks'] = rows

    # Totals per stage.
    totals = {}
    for stage, rows in report['stages'].items():
        key = {'060_uncompressed_tracks': 'same_up_to_id_renaming', '050_uncompressed_detections': None}.get(stage, 'equal')
        # Every row is compared (callers make prune deterministic: the CLI reads the scripts' grids).
        compared = rows
        if key:
            totals[stage] = {'compared': len(compared), 'equal': sum(bool(r.get(key)) for r in compared),
                             'rows': len(rows)}
            if stage == '060_uncompressed_tracks':
                # Byte-identical tracking.jsonl, track ids included (stricter than equal up to renaming).
                totals[stage]['text_identical'] = sum(bool(r.get('equal')) for r in compared)
        else:
            totals[stage] = {'rows': len(rows), 'equal_in_order': sum(r['equal_in_order'] for r in rows),
                             'frames_identical': sum(r['frames_identical'] for r in rows),
                             'frames': sum(r['frames'] for r in rows),
                             'count_mismatch_frames': sum(r['frames_with_different_box_counts'] for r in rows),
                             'max_abs_box_diff': max((r['max_abs_box_diff_same_count'] for r in rows), default=0.0)}
    t = report['stages']['060_uncompressed_tracks']
    totals['hota_per_video'] = {'compared': len(t), 'equal': sum(r['hota_equal'] for r in t),
                                'max_abs_diff': max((abs(r['hota_ours'] - r['hota_cached']) for r in t), default=0.0)}
    report['totals'] = totals
    return report


if __name__ == '__main__':
    main_cli()
