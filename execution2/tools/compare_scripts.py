"""
Compare execution2 against the cached p020 -> p060 outputs, stage by stage (PLAN.md step 6, 13A).

Reads the scripts' cache read-only; never writes to it.  Each stage is compared in isolation,
fed the script's own output from the stage before, so a difference cannot cascade:

  classify  our scores           vs 020 score.jsonl           (input: the video)
  prune     our pruned grids      vs 022 score.jsonl           (input: 020 scores)
  pack      index maps, offsets,  vs 033 files                 (input: 022 or 020 scores)
            canvases
  detect    per-frame boxes       vs 050 detections.jsonl      (input: our pack output)
  track     tracking.jsonl text   vs 060 tracking.jsonl        (input: 050 detections)

Then the whole chain: main.execute on the same videos (real pipeline, real processes), each
output file vs 060's.  Run inside the polyis container with one visible GPU:

  CUDA_VISIBLE_DEVICES=0 python -m execution2.tools.compare_scripts \
      --config execution2/configs/caldot1-y05-s4.yaml --dataset caldot1-y05 \
      --videos te00.mp4 te01.mp4 --out measurements-results/execution2-compare/s4.json
"""

import argparse
import dataclasses
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from execution2 import frames, main, stages
from execution2.config import Config, load_config
from execution2.stages import Classified, Detected
from polyis.io import cache, store
from polyis.utilities import (build_param_str, load_classification_results, load_pruned_classification_results,
                              save_tracking_results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--videoset', default='test')
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def grids(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    """Frame indices and uint8 grids from score.jsonl rows."""
    idx = [r['idx'] for r in rows]
    arr = np.stack([np.frombuffer(bytes.fromhex(r['classification_hex']), np.uint8).reshape(r['classification_size'])
                    for r in rows])
    return idx, arr


def compare_grids(ours: np.ndarray, theirs: np.ndarray, threshold: float | None) -> dict:
    """Exactness and size of differences between two score/bitmap stacks."""
    if ours.shape != theirs.shape:
        return {'equal': False, 'shape': [list(ours.shape), list(theirs.shape)]}
    diff = np.abs(ours.astype(int) - theirs.astype(int))
    out = {'equal': bool(not diff.any()), 'max_abs_diff': int(diff.max()), 'differing_values': int(np.count_nonzero(diff)),
           'values': int(diff.size)}
    if threshold is not None:
        # Tiles whose relevance decision (p030's > T_r*255) differs.
        out['decision_flips'] = int(np.count_nonzero((ours > threshold * 255) != (theirs > threshold * 255)))
    return out


def script_paths(cfg: Config, dataset: str, video: str) -> dict:
    """Cache locations of every script output for this config."""
    p = cfg.params
    tracker = p.tracker if cfg.use_prune else None
    packed = build_param_str(classifier=cfg.artifacts.classifier.name, tilesize=p.tile_size, sample_rate=p.sample_rate,
                             tilepadding=p.tile_padding, canvas_scale=p.canvas_scale, tracker=tracker,
                             tracking_accuracy_threshold=p.mistrack_tolerance, relevance_threshold=p.relevance_threshold)
    tracked = build_param_str(classifier=cfg.artifacts.classifier.name, tilesize=p.tile_size, sample_rate=p.sample_rate,
                              tilepadding=p.tile_padding, canvas_scale=p.canvas_scale, tracker=p.tracker,
                              tracking_accuracy_threshold=p.mistrack_tolerance, relevance_threshold=p.relevance_threshold)
    return {
        'comp_frames': cache.exec(dataset, 'comp-frames', video, packed),
        'ucomp_dets': cache.exec(dataset, 'ucomp-dets', video, packed, 'detections.jsonl'),
        'tracks': cache.exec(dataset, 'ucomp-tracks', video, tracked, 'tracking.jsonl'),
    }


def load_p033(directory: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[list]]:
    """p030's canvases (as p040 reads them), index maps, and offset lookups, in collage order."""
    names = sorted((f.stem for f in (directory / 'index_maps').glob('*.npy')), key=lambda n: int(n.split('_')[0]))
    canvases = [cv2.imread(str(directory / 'images' / f'{n}.jpg')) for n in names]
    maps = [np.load(directory / 'index_maps' / f'{n}.npy') for n in names]
    lookups = [[json.loads(line) for line in (directory / 'offset_lookups' / f'{n}.jsonl').read_text().splitlines()]
               for n in names]
    return canvases, maps, lookups


def compare_detections(ours: dict[int, list], theirs_rows: list[dict]) -> dict:
    """Per-frame box sets (order-free) between our detect output and p050's."""
    theirs = {r['frame_idx']: r['bboxes'] for r in theirs_rows}
    frames_equal = count_diff = 0
    max_diff = 0.0
    for f in sorted(set(ours) | set(theirs)):
        a = sorted(map(tuple, ours.get(f, [])))
        b = sorted(map(tuple, theirs.get(f, [])))
        if len(a) != len(b):
            count_diff += 1
            continue
        d = float(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float)).max()) if a else 0.0
        max_diff = max(max_diff, d)
        frames_equal += d == 0.0
    return {'frames': len(set(ours) | set(theirs)), 'frames_identical': frames_equal,
            'frames_with_different_box_counts': count_diff, 'max_abs_box_diff_same_count': max_diff,
            'boxes_ours': sum(map(len, ours.values())), 'boxes_theirs': sum(map(len, theirs.values()))}


def compare_track_lines(a: list[str], b: list[str]) -> dict:
    """tracking.jsonl text equality, and numerically: same track ids per frame, largest box difference."""
    out = {'equal': a == b, 'lines': [len(a), len(b)],
           'differing_lines': sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))}
    ra = {r['frame_idx']: {t[0]: t[1:] for t in r['tracks']} for r in map(json.loads, a)}
    rb = {r['frame_idx']: {t[0]: t[1:] for t in r['tracks']} for r in map(json.loads, b)}
    frames = set(ra) | set(rb)
    same_ids = [f for f in frames if ra.get(f, {}).keys() == rb.get(f, {}).keys()]
    diffs = [abs(x - y) for f in same_ids for tid in ra.get(f, {}) for x, y in zip(ra[f][tid], rb[f][tid])]
    out.update({'frames_with_different_track_ids': len(frames) - len(same_ids),
                'max_abs_box_diff_same_ids': max(diffs, default=0.0)})
    # Track ids are arbitrary labels: check whether one fixed id mapping over the whole video makes
    # every frame match (boxes within 1e-3 px), i.e. the tracks are the same up to renaming.
    mapping: dict = {}
    consistent, unmatched = True, 0
    for f in frames:
        for tid, box in ra.get(f, {}).items():
            match = [u for u, other in rb.get(f, {}).items()
                     if len(other) == len(box) and max((abs(x - y) for x, y in zip(box, other)), default=0) <= 1e-3]
            if len(match) != 1:
                unmatched += 1
                continue
            if mapping.setdefault(tid, match[0]) != match[0]:
                consistent = False
    out.update({'same_up_to_id_renaming': consistent and unmatched == 0
                and len(set(mapping.values())) == len(mapping), 'unmatched_track_boxes': unmatched})
    return out


def compare_track_files(ours: dict[int, list], theirs_path: Path, tmp: Path) -> dict:
    """Our frame_tracks written as p060 writes them, compared with p060's file."""
    path = tmp / 'ours.jsonl'
    save_tracking_results(ours, path)
    return compare_track_lines(path.read_text().splitlines(), theirs_path.read_text().splitlines())


def isolated(cfg: Config, states: dict, dataset: str, videoset: str, video: str, prefix: str, tmp: Path) -> dict:
    """Every stage fed the script's previous output; returns per-stage comparisons."""
    p, paths = cfg.params, script_paths(cfg, dataset, video)
    path = str(store.dataset(dataset, videoset, video))
    info = main.probe(path)
    report: dict = {}
    decoded = stages.decode(stages.decode_setup(cfg, prefix, None), path)
    try:
        # classify
        ours = stages.classify(states['classify'], decoded)
        idx, theirs = grids(load_classification_results(dataset, video, p.tile_size, cfg.artifacts.classifier.name,
                                                       p.sample_rate))
        report['classify'] = {'frames_match': idx == decoded.sampled,
                              **compare_grids(ours.scores, theirs, p.relevance_threshold)}
        pack_input = theirs
        # prune
        if cfg.use_prune:
            pruned = stages.prune(states['prune'], Classified(decoded, theirs))
            pidx, ptheirs = grids(load_pruned_classification_results(
                dataset, video, p.tile_size, cfg.artifacts.classifier.name, p.tracker, p.mistrack_tolerance,
                p.relevance_threshold, p.sample_rate))
            report['prune'] = {'frames_match': pidx == decoded.sampled, **compare_grids(pruned.scores, ptheirs, None)}
            pack_input = ptheirs
        # pack (frees the frames block)
        packed = stages.pack_video(stages.pack_setup(cfg, prefix, None), Classified(decoded, pack_input))
        decoded = None
    finally:
        if decoded is not None:
            frames.unlink(decoded.frames)
    canvases, maps, lookups = load_p033(paths['comp_frames'])
    with frames.attach(packed.canvases) as m:
        ours_canvases = m.array.copy()
    same_count = len(maps) == len(packed.offset_lookups)
    report['pack'] = {
        'collages': [len(packed.offset_lookups), len(maps)],
        'index_maps_equal': same_count and bool(np.array_equal(packed.index_maps, np.stack(maps))) if maps else same_count,
        'offset_lookups_equal': json.loads(json.dumps(packed.offset_lookups)) == lookups,
        'canvases_equal': same_count and all(np.array_equal(a, b) for a, b in zip(ours_canvases, canvases)),
    }
    # detect (frees the canvases block)
    detected = stages.detect(states['detect'], packed)
    rows = [json.loads(l) for l in paths['ucomp_dets'].read_text().splitlines() if l.strip()]
    report['detect'] = compare_detections(detected.detections, rows)
    # track, fed p050's detections
    theirs_dets = Detected(video, info.width, info.height, len(rows), {r['frame_idx']: r['bboxes'] for r in rows})
    tracked = stages.track(stages.track_setup(cfg, None), theirs_dets)
    report['track'] = compare_track_files(tracked.frame_tracks, paths['tracks'], tmp)
    # Track fed our own detections: shows how detector noise alone carries into the tracks.
    ours_tracked = stages.track(stages.track_setup(cfg, None), detected)
    report['track_on_our_detections'] = compare_track_files(ours_tracked.frame_tracks, paths['tracks'], tmp)
    return report


def chained(cfg: Config, dataset: str, videoset: str, videos: list[str], tmp: Path) -> dict:
    """The real multi-process pipeline end to end; each output vs 060's tracking.jsonl."""
    vdir, out = tmp / 'videos', tmp / 'out'
    vdir.mkdir()
    for v in videos:
        os.symlink(store.dataset(dataset, videoset, v), vdir / v)
    summary = main.execute(cfg, str(vdir), str(out))
    per_video = {}
    for v in videos:
        per_video[v] = compare_track_lines((out / f'{Path(v).stem}.jsonl').read_text().splitlines(),
                                           script_paths(cfg, dataset, v)['tracks'].read_text().splitlines())
    return {'summary': summary, 'videos': per_video}


def main_cli() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # Models are loaded once and reused across videos, as the pipeline's workers do.
    states = {'classify': stages.classify_setup(cfg, None), 'detect': stages.detect_setup(cfg, None),
              'prune': stages.prune_setup(cfg, None) if cfg.use_prune else None}
    prefix = frames.new_prefix()
    report: dict = {'config': args.config, 'dataset': args.dataset, 'params': dataclasses.asdict(cfg.params),
                    'isolated': {}}
    try:
        with tempfile.TemporaryDirectory(prefix='execution2-compare-') as tmp:
            for v in args.videos:
                vtmp = Path(tmp) / v
                vtmp.mkdir()
                report['isolated'][v] = isolated(cfg, states, args.dataset, args.videoset, v, prefix, vtmp)
                print(v, json.dumps(report['isolated'][v]), flush=True)
            report['chained'] = chained(cfg, args.dataset, args.videoset, args.videos, Path(tmp))
            print('chained', json.dumps(report['chained']), flush=True)
    finally:
        frames.unlink_all(prefix)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main_cli()
