"""
Stage tests that compare each execution stage with its original p0xx script: the script's file loaders and
cache paths are redirected to temporary files, so both see the same inputs.  Needs the polyis container
(Cython builds, Gurobi, one GPU).  The stage-only tests are in tests/execution/test_stages.py.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from execution import frames, operators, stages
from execution.stages import Classified, Decoded, Detected
from tetris.utilities import save_tracking_results
from tests.execution.test_stages import (H, W, FakeCache, N_FRAMES, TS, make_cfg, prefix, score_rows, synthetic_scores, video)  # noqa: F401  (fixtures included)


def test_constants_match_the_scripts(monkeypatch):
    import experiments.scripts.p020_exec_classify as p020
    import experiments.scripts.p022_exec_prune_polyominoes as p022
    import experiments.scripts.p030_exec_compress as p030
    import experiments.scripts.p040_exec_detect as p040
    assert stages.CLASSIFY_BATCH == p020.BATCH_SIZE
    assert stages.PRUNE_TILEPADDING_MODE == p022.TILEPADDING_MODE
    assert stages.PACK_MODE_BEST_FIT == int(p030.PackMode.Best_Fit)
    monkeypatch.setattr('sys.argv', ['p040'])
    assert stages.DETECT_BATCH == p040.parse_args().batch_size
    # p040 reads canvases in name (collage) order on both of its paths, not in os.listdir's arbitrary order.
    p040_source = Path(p040.__file__).read_text()
    assert p040_source.count("sorted(f for f in os.listdir(compressed_frames_dir) if f.endswith('.jpg'))") == 2
    assert "[f for f in os.listdir(compressed_frames_dir)" not in p040_source
    # p022 passes `time_limit or 0.1` to solve_ilp.
    assert 'time_limit_seconds=time_limit or 0.1' in Path(p022.__file__).read_text()
    assert stages.PRUNE_TIME_LIMIT_S == 0.1


PRUNE_TEST_TIME_LIMIT_S = 5.0


def run_p022(monkeypatch, tmp_path, cfg, scores, sampled) -> np.ndarray:
    """p022.process_video on these scores, with PRUNE_TEST_TIME_LIMIT_S; returns its pruned 0/255 grids."""
    import experiments.scripts.p022_exec_prune_polyominoes as p022
    table = np.load(cfg.artifacts.max_rate_table)
    idx = p022._ACCURACY_THRESHOLD_TO_IDX[cfg.params.mistrack_tolerance]
    monkeypatch.setattr(p022, 'load_classification_results', lambda *a, **k: score_rows(scores, sampled))
    monkeypatch.setattr(p022, 'load_max_sampling_rate', lambda *a, **k: table[:, :, idx])
    cache = FakeCache(tmp_path / 'p022')
    monkeypatch.setattr(p022, 'cache', cache)
    # p022 first checks that its input score file exists on disk.
    cache.exec('ds', 'relevancy', 'v.avi', f'Fake_{TS}_{cfg.params.sample_rate}', 'score', 'score.jsonl').touch()
    p022.process_video('ds', 'test', 'v.avi', 'Fake', TS, cfg.params.sample_rate, cfg.params.tracker,
                       cfg.params.mistrack_tolerance, cfg.params.relevance_threshold, PRUNE_TEST_TIME_LIMIT_S)
    # With a time limit p022 writes to its separate time-limit stage directory.
    (score_file,) = (tmp_path / 'p022' / 'ds' / 'pruned-polyominoes-tl').rglob('score.jsonl')
    rows = [json.loads(line) for line in score_file.read_text().splitlines()]
    assert [r['idx'] for r in rows] == list(sampled)
    return np.stack([np.frombuffer(bytes.fromhex(r['classification_hex']), np.uint8).reshape(r['classification_size'])
                     for r in rows])


@pytest.mark.parametrize('sample_rate, tolerance, threshold', [(2, 0.4, 0.5), (1, 0.9, 0.25), (4, 0.6, 0.75)])
def test_prune_matches_p022(monkeypatch, tmp_path, sample_rate, tolerance, threshold):
    cfg = make_cfg(tmp_path, sample_rate=sample_rate, mistrack_tolerance=tolerance, relevance_threshold=threshold)
    sampled, partner, _ = stages.frame_selection(N_FRAMES, sample_rate)
    scores = synthetic_scores(len(sampled))
    decoded = Decoded('v.avi', W, H, N_FRAMES, sampled, partner, np.array([]), frames.Block('unused', (0,), '|u1'))
    monkeypatch.setattr(operators, 'PRUNE_TIME_LIMIT_S', PRUNE_TEST_TIME_LIMIT_S)
    ours = stages.prune(stages.prune_setup(cfg, None), Classified(decoded, scores)).scores
    expected = run_p022(monkeypatch, tmp_path, cfg, scores, sampled)
    assert ours.dtype == np.uint8 and np.array_equal(ours, expected)
    # Pruning only removes tiles.
    assert not np.any((ours > 0) & (scores < np.floor(threshold * 255)))


def run_p030(monkeypatch, tmp_path, cfg, video, scores, sampled) -> tuple[list, list, list]:
    """p030.compress on these scores; returns its (JPEG-decoded canvases, index maps, offset lookups) in collage order."""
    import experiments.scripts.p030_exec_compress as p030
    monkeypatch.setattr(p030, 'load_classification_results', lambda *a, **k: score_rows(scores, sampled))
    monkeypatch.setattr(p030, 'cache', FakeCache(tmp_path / 'p030'))
    monkeypatch.setattr(p030.store, 'dataset', lambda *a: video)
    p = cfg.params
    p030.compress('ds', 'test', 'v.avi', 'Fake', TS, p.sample_rate, p.tile_padding, p.canvas_scale,
                  None, None, p.relevance_threshold, p030.PackMode.Best_Fit)
    (out,) = {f.parent.parent for f in (tmp_path / 'p030').rglob('*.npy')}
    names = sorted(f.stem for f in (out / 'index_maps').glob('*.npy'))
    canvases = [cv2.imread(str(out / 'images' / f'{n}.jpg')) for n in names]
    maps = [np.load(out / 'index_maps' / f'{n}.npy') for n in names]
    lookups = [[json.loads(line) for line in (out / 'offset_lookups' / f'{n}.jsonl').read_text().splitlines()]
               for n in names]
    return canvases, maps, lookups


@pytest.mark.parametrize('canvas_format', ['jpeg', 'raw'])
@pytest.mark.parametrize('tile_padding, canvas_scale, sample_rate', [('none', 1.0, 2), ('tr', 1.0, 1), ('plus', 1.25, 4)])
def test_pack_matches_p030(monkeypatch, tmp_path, video, prefix, tile_padding, canvas_scale, sample_rate, canvas_format):
    cfg = make_cfg(tmp_path, tile_padding=tile_padding, canvas_scale=canvas_scale, sample_rate=sample_rate,
                   canvas_format=canvas_format)
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
    scores = synthetic_scores(len(d.sampled))
    p = stages.pack_video(stages.pack_setup(cfg, prefix, None), Classified(d, scores))
    assert frames.leftover(prefix) == [p.canvases.name], 'pack must free the frames block'
    canvases, maps, lookups = run_p030(monkeypatch, tmp_path, cfg, video, scores, d.sampled)

    assert len(maps) == len(p.offset_lookups) > 0
    assert np.array_equal(p.index_maps, np.stack(maps))
    # JSON turns the scripts' tuples into lists.
    assert json.loads(json.dumps(p.offset_lookups)) == lookups
    with frames.attach(p.canvases) as m:
        ours = m.array.copy()
    frames.unlink(p.canvases)
    assert ours.shape == (len(canvases), *canvases[0].shape)
    if canvas_format == 'jpeg':
        # Exactly the pixels p040 reads back from p030's JPEG files.
        assert np.array_equal(ours, np.stack(canvases))
    else:
        # Raw canvases differ from p030's JPEGs only by compression error; exact pixels are checked by
        # test_packed_tiles_are_exact_copies_of_their_source_tiles (tests/execution/test_stages.py).
        assert not np.array_equal(ours, np.stack(canvases))
        assert np.abs(np.stack(canvases).astype(int) - ours.astype(int)).mean() < 10.0


def test_track_matches_p060(monkeypatch, tmp_path):
    import experiments.scripts.p060_exec_track as p060
    rng = np.random.default_rng(5)
    detections = {f: [] for f in range(30)}
    for f in range(0, 30, 2):                       # sampled frames only, like real input
        for obj in range(3):
            x, y = 10 + 5 * f + 40 * obj, 20 + 3 * obj
            detections[f].append([x, y, x + 30 + rng.uniform(-1, 1), y + 20, 0.8 + 0.01 * obj, 2.0])
    ours = stages.track(stages.track_setup(make_cfg(tmp_path), None), Detected('v.avi', 640, 480, 30, detections))

    cache = FakeCache(tmp_path / 'p060')
    monkeypatch.setattr(p060, 'cache', cache)
    monkeypatch.setattr(p060, 'get_video_resolution', lambda *a: (640, 480))
    monkeypatch.setattr(p060, 'load_detection_results',
                        lambda *a, **k: [{'frame_idx': f, 'bboxes': detections[f]} for f in range(30)])
    for path in [cache.exec('ds', 'ucomp-dets', 'v.avi', 'x', 'detections.jsonl')]:
        path.touch()
    monkeypatch.setattr(p060, 'build_param_str', lambda **k: 'x')
    p060.track('ds', 'v.avi', 'Fake', TS, 2, 'none', 1.0, 'sortcython', None, 0.5, False)
    ours_file = tmp_path / 'ours.jsonl'
    save_tracking_results(ours.frame_tracks, ours_file)
    assert ours.frame_tracks, 'the tracker must produce tracks'
    assert ours_file.read_text() == (tmp_path / 'p060' / 'ds' / 'ucomp-tracks' / 'v.avi' / 'x' / 'tracking.jsonl').read_text()
