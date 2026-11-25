"""Profiling stages with the heavy operators faked: fan-out, branches, charging, freeing, saving (PLAN step 4)."""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from execution import frames, operators as ops
from execution.config import Detector
from experiments.fanout_profiling import save, stages as st
from experiments.fanout_profiling.completion import VideoCompletion
from experiments.fanout_profiling.config import Artifacts, Evaluation, Params, StageSystem, SweepConfig, System
from experiments.fanout_profiling.messages import Branch, Decoded, Detected, Packed, Scored, StageTiming, Tracked

N, H, W, TS = 20, 60, 80, 10


def make_cfg(tmp_path: Path, tolerances=(None, 0.4, 0.8), trackers=('sortcython', 'bytetrackcython'),
             sample_rates=(1, 4), thresholds=(0.25, 0.5), paddings=('none', 'tr'), scales=(1.0,),
             reduce=False, table_values=None) -> SweepConfig:
    ref = tmp_path / 'ref'
    ref.mkdir(exist_ok=True)
    table = tmp_path / 'rates.npy'
    # Default: every tolerance's gap is 8 frames; table_values gives one value per tolerance-axis entry.
    values = np.full(9, 8) if table_values is None else np.asarray(table_values)
    np.save(table, np.broadcast_to(values, (H // TS, W // TS, 9)).astype(np.int64))
    never = tmp_path / 'never.npy'
    np.save(never, np.ones((H // TS, W // TS), dtype=np.uint8))
    tables = {t: str(table) for t in trackers} if any(x is not None for x in tolerances) else {}
    sys_ = {n: StageSystem(1, (0,) if n in ('classify', 'detect') else (), 1)
            for n in ('decode', 'classify', 'prune', 'pack', 'detect', 'track', 'evaluate')}
    return SweepConfig('ds', 'valid', str(tmp_path / 'out'),
                       Params('Fake', TS, 'raw', sample_rates, thresholds, tolerances, paddings, scales, trackers,
                              reduce_search_space=reduce),
                       Artifacts('w', 'b', str(never), tables, Detector('ultralytics', {'model_path': 'm'})),
                       Evaluation(str(ref), '002_naive/tracking.jsonl'), System(2, 8, 2, sys_))


@pytest.fixture
def prefix():
    p = frames.new_prefix()
    yield p
    frames.unlink_all(p)


@pytest.fixture
def manager():
    m = mp.get_context('spawn').Manager()
    yield m
    m.shutdown()


def all_frames(prefix) -> ops.Decoded:
    """A fake all-frames video: frame i is filled with i."""
    with frames.create(prefix, (N, H, W, 3), 'uint8') as m:
        for i in range(N):
            m.array[i] = i
    sampled, partner, needed = ops.frame_selection(N, 1)
    return ops.Decoded('/v/a.mp4', W, H, N, sampled, partner, needed, m.block)


def base_branch():
    return Branch().with_(classifier='Fake', tile_size=TS)


def test_expected_counts_for_the_papers_grid(tmp_path):
    cfg = make_cfg(tmp_path, tolerances=(None, 0.4, 0.6, 0.8), sample_rates=(1, 2, 4, 8, 16),
                   thresholds=(0.25, 0.5, 0.75), paddings=('none', 'tr', 'bl', 'square'))
    assert st.expected_pack_inputs(cfg) == 5 * 3 * (3 * 2 + 1) == 105
    assert st.expected_results(cfg) == 5 * 3 * (3 * 2 * 4 + 2 * 4) == 480
    unpruned = make_cfg(tmp_path, tolerances=(None,))
    assert st.expected_pack_inputs(unpruned) == 2 * 2 and st.expected_results(unpruned) == 2 * 2 * 2 * 2


def test_classify_slices_once_per_sample_rate_and_charges_only_its_frames(tmp_path, prefix, monkeypatch):
    v = all_frames(prefix)
    scores = np.arange(N, dtype=np.uint8)[:, None, None] * np.ones((1, H // TS, W // TS), np.uint8)
    calls = []
    monkeypatch.setattr(ops, 'classify_frames', lambda state, video, time_batches: (calls.append(1) or scores,
                                                                                    np.full(N, 0.01)))
    d = Decoded('/v/a.mp4', base_branch(), (StageTiming('decode', 0, 1, 0.0),), v)
    out = st.classify({'model': None, 'sample_rates': (1, 4)}, d)
    assert len(calls) == 1 and [o.branch['sample_rate'] for o in out] == [1, 4]
    s4 = out[1]
    expected = ops.frame_selection(N, 4)[0]
    assert np.array_equal(s4.decoded.sampled, expected) and np.array_equal(s4.scores[:, 0, 0], expected)
    assert s4.timings[-1].stage == 'classify' and s4.timings[-1].charged == pytest.approx(0.01 * len(expected))
    assert out[0].timings[-1].charged == pytest.approx(0.01 * N) and s4.timings[0].charged == 0.0
    frames.unlink(v.frames)


class FakeILP:
    status, solve_ms = 2, 1.5


def test_prune_fans_out_and_passes_unpruned_scores_through(tmp_path, prefix, monkeypatch):
    cfg = make_cfg(tmp_path)
    v = all_frames(prefix)
    scores = np.full((N, H // TS, W // TS), 200, np.uint8)
    monkeypatch.setattr(ops, 'prune_scores', lambda s, distance, t_r, limit=None: (np.where(s > 0, 255, 0).astype(np.uint8),
                                                                        FakeILP()))
    sc = Scored('/v/a.mp4', base_branch().with_(sample_rate=4), (), v, scores)
    out = st.prune(st.prune_setup(cfg, None), sc)
    assert len(out) == 2 * (1 + 2 * 2)
    unpruned = [o for o in out if o.branch['mistrack_tolerance'] is None]
    assert len(unpruned) == 2 and all(o.scores is scores and o.timings[-1].charged == 0.0 and o.ilp is None
                                      for o in unpruned)
    pruned = [o for o in out if o.branch['mistrack_tolerance'] is not None]
    assert {(o.branch['relevance_threshold'], o.branch['mistrack_tolerance'], o.branch['tracker']) for o in pruned} == \
        {(t, m, k) for t in (0.25, 0.5) for m in (0.4, 0.8) for k in ('sortcython', 'bytetrackcython')}
    assert all(o.ilp == {'status': 2, 'solve_ms': 1.5, 'kept_tiles': scores.size, 'group': [o.branch['mistrack_tolerance']]}
               for o in pruned)
    frames.unlink(v.frames)


def test_prune_solves_through_state_solve(tmp_path, prefix):
    # The equivalence test swaps state['solve'] for a reader of the scripts' grids; prune must use it for every
    # pruned branch and keep its fan-out and labels.
    cfg = make_cfg(tmp_path)
    v = all_frames(prefix)
    scores = np.full((N, H // TS, W // TS), 200, np.uint8)
    state = st.prune_setup(cfg, None)
    calls = []
    state['solve'] = lambda state, sc, tracker, tol, t_r: (calls.append((tracker, tol, t_r)) or scores,
                                                         {'status': 'script', 'solve_ms': 0.0})
    out = st.prune(state, Scored('/v/a.mp4', base_branch().with_(sample_rate=4), (), v, scores))
    assert sorted(calls) == sorted((k, m, t) for t in (0.25, 0.5) for m in (0.4, 0.8)
                                   for k in ('sortcython', 'bytetrackcython'))
    assert all(o.ilp['status'] == 'script' for o in out if o.branch['mistrack_tolerance'] is not None)
    frames.unlink(v.frames)


def test_pack_frees_frames_only_after_the_videos_last_input(tmp_path, prefix, manager, monkeypatch):
    cfg = make_cfg(tmp_path, tolerances=(None,), sample_rates=(4,), thresholds=(0.5,), paddings=('none', 'tr'))
    v = all_frames(prefix)

    def fake_pack(scores, video, pfx, ts, padding, scale, t_r, jpeg):
        with frames.create(pfx, (1,), 'uint8') as m:
            pass
        return ops.Packed(video.video, W, H, N, m.block, np.zeros((0, 1, 1), np.uint16), [])

    monkeypatch.setattr(ops, 'pack_collages', fake_pack)
    state = st.pack_setup(cfg, prefix, VideoCompletion(manager), None)
    state['expected'] = 2   # pretend this video has two pack inputs
    sc = Scored('/v/a.mp4', base_branch().with_(sample_rate=4, relevance_threshold=0.5, mistrack_tolerance=None),
                (), v, np.zeros((5, 6, 8), np.uint8))
    first = st.pack(state, sc)
    assert len(first) == 2 and [o.branch['tile_padding'] for o in first] == ['none', 'tr']
    assert v.frames.name in frames.leftover(prefix)
    st.pack(state, sc)
    assert v.frames.name not in frames.leftover(prefix)


def test_track_fans_out_over_trackers_only_when_unpruned(monkeypatch):
    monkeypatch.setattr(ops, 'track_detections', lambda tracker, w, h, dets: {0: [[1, 0, 0, 1, 1]]})
    unpruned = Detected('/v/a.mp4', base_branch().with_(sample_rate=1, relevance_threshold=0.5, mistrack_tolerance=None,
                                                         tile_padding='none', canvas_scale=1.0), (), W, H, N, {})
    out = st.track({'trackers': ('sortcython', 'bytetrackcython')}, unpruned)
    assert [o.branch['tracker'] for o in out] == ['sortcython', 'bytetrackcython']
    pruned = Detected('/v/a.mp4', base_branch().with_(sample_rate=1, relevance_threshold=0.5, mistrack_tolerance=0.4,
                                                       tracker='sortcython', tile_padding='none', canvas_scale=1.0),
                      (), W, H, N, {})
    assert [o.branch['tracker'] for o in st.track({'trackers': ('sortcython', 'bytetrackcython')}, pruned)] == ['sortcython']


def test_detect_frees_the_canvases(prefix, monkeypatch):
    monkeypatch.setattr(ops, 'detect_canvases', lambda det, packed, ts: {i: [] for i in range(packed.frame_count)})
    with frames.create(prefix, (1,), 'uint8') as m:
        pass
    p = Packed('/v/a.mp4', base_branch(), (), ops.Packed('/v/a.mp4', W, H, N, m.block, np.zeros((0, 1, 1)), []))
    out = st.detect({'detector': None, 'tile_size': TS}, p)
    assert sorted(out.detections) == list(range(N)) and frames.leftover(prefix) == []


def full_branch(tolerance=0.8, tracker='sortcython'):
    return base_branch().with_(sample_rate=4, relevance_threshold=0.5, mistrack_tolerance=tolerance,
                               **({'tracker': tracker} if tolerance is not None else {}),
                               tile_padding='bl', canvas_scale=1.0)


def test_savers_write_the_scripts_paths_and_formats(tmp_path, prefix):
    root, v = str(tmp_path / 'out'), all_frames(prefix)
    sampled = ops.frame_selection(N, 4)[0]
    view = v._replace(sampled=sampled, partner=sampled)
    scores = np.full((len(sampled), 6, 8), 7, np.uint8)
    pruned_branch = base_branch().with_(sample_rate=4, relevance_threshold=0.5, mistrack_tolerance=0.8, tracker='sortcython')
    save.save_scores(root, 'ds', Scored('/v/a.mp4', base_branch().with_(sample_rate=4), (), view, scores))
    save.save_pruned(root, 'ds', Scored('/v/a.mp4', pruned_branch, (), view, scores))
    ex = Path(root) / 'ds' / 'execution' / 'a.mp4'
    rows = [json.loads(l) for l in (ex / '020_relevancy' / 'Fake_10_4' / 'score' / 'score.jsonl').read_text().splitlines()]
    assert [r['idx'] for r in rows] == sampled.tolist() and rows[0]['classification_size'] == [6, 8]
    assert (ex / '022_pruned_polyominoes' / 'Fake_10_4_080_r050_sortcython' / 'score' / 'score.jsonl').exists()
    lookup = [((0, 0), (1, 2), 8), ((0, 1), (0, 0), 12)]
    save.save_packed(root, 'ds', Packed('/v/a.mp4', full_branch(), (),
                                        ops.Packed('/v/a.mp4', W, H, N, v.frames, np.ones((1, 6, 8), np.uint16),
                                                   [lookup])))
    packed_dir = ex / '033_compressed_frames' / 'Fake_10_4_080_r050_bl_s100_sortcython'
    assert np.array_equal(np.load(packed_dir / 'index_maps' / '0000_0008_0012.npy'), np.ones((6, 8)))
    assert [json.loads(l) for l in (packed_dir / 'offset_lookups' / '0000_0008_0012.jsonl').read_text().splitlines()] == \
        [[[0, 0], [1, 2], 8], [[0, 1], [0, 0], 12]]
    save.save_detections(root, 'ds', Detected('/v/a.mp4', full_branch(None), (), W, H, 3, {2: [[1.0]], 0: [], 1: []}))
    dets = (ex / '050_uncompressed_detections' / 'Fake_10_4_r050_bl_s100' / 'detections.jsonl').read_text().splitlines()
    assert [json.loads(l)['frame_idx'] for l in dets] == [0, 1, 2]
    unpruned_tracked = full_branch(None).with_(tracker='bytetrackcython')
    save.save_tracks(root, 'ds', Tracked('/v/a.mp4', unpruned_tracked, (), {0: [[1, 0, 0, 1, 1]]}))
    assert (ex / '060_uncompressed_tracks' / 'Fake_10_4_r050_bl_s100_bytetrackcython' / 'tracking.jsonl').exists()
    frames.unlink(v.frames)


def test_evaluate_scores_identical_tracks_as_perfect(tmp_path):
    from tetris.utilities import save_tracking_results
    cfg = make_cfg(tmp_path)
    tracks = {f: [[1, 10 + f, 10, 40 + f, 30], [2, 100, 50 - f, 130, 80 - f]] for f in range(15)}
    ref = Path(cfg.evaluation.reference_dir) / 'a.mp4' / '002_naive'
    ref.mkdir(parents=True)
    save_tracking_results(tracks, ref / 'tracking.jsonl')
    state = st.evaluate_setup(cfg, type('W', (), {'index': 0})())
    out = st.evaluate(state, Tracked('/v/a.mp4', full_branch(), (), tracks))
    assert set(out.metrics) == {'HOTA', 'Count'} and float(np.mean(out.metrics['HOTA']['HOTA'])) == pytest.approx(1.0)
    assert out.timings[-1].stage == 'evaluate' and out.timings[-1].charged == 0.0


# ---------------------------------------------------------------- search-space reduction (DECISIONS.md A1-A3)

# Tolerance axis: 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 1.00; gaps for 0.4 / 0.6 / 0.8 = 12 / 8 / 8 frames.
AXIS_VALUES = [16, 12, 10, 8, 8, 8, 4, 2, 1]


def test_reduction_groups_identical_gaps_and_counts_follow(tmp_path):
    from experiments.fanout_profiling.reduction import sweep_groups
    cfg = make_cfg(tmp_path, tolerances=(None, 0.4, 0.6, 0.8), sample_rates=(1, 8, 16), reduce=True,
                   table_values=AXIS_VALUES)
    g = sweep_groups(cfg)
    # s=1: 12 vs 8 vs 8 -> {0.4}, {0.6, 0.8}; s=8: 1 vs 1 vs 1 -> one all-ones group; s=16 likewise.
    assert [grp.tolerances for grp in g[('sortcython', 1)]] == [(0.4,), (0.6, 0.8)]
    assert [(grp.tolerances, grp.all_ones) for grp in g[('sortcython', 8)]] == [((0.4, 0.6, 0.8), True)]
    assert st.expected_pack_inputs(cfg, g) == 2 * ((1 + 2 + 2) + (1 + 1 + 1) + (1 + 1 + 1))
    assert st.expected_results(cfg, g) == 2 * 2 * ((2 + 2 + 2) + (2 + 1 + 1) + (2 + 1 + 1))
    off = make_cfg(tmp_path, tolerances=(None, 0.4, 0.6, 0.8), sample_rates=(1, 8, 16), reduce=False,
                   table_values=AXIS_VALUES)
    assert st.expected_pack_inputs(off) == 3 * 2 * (1 + 3 * 2)


def test_reduction_ignores_tiles_the_mask_excludes(tmp_path):
    from experiments.fanout_profiling.reduction import group_tolerances
    table = np.broadcast_to(np.array(AXIS_VALUES), (2, 2, 9)).copy()
    table[0, 0, 1] = 99          # 0.4 differs from 0.6/0.8 only on tile (0, 0)
    table[0, 0, 3] = 1
    mask = np.ones((2, 2), bool)
    assert len(group_tolerances(table, mask, [0.4, 0.6, 0.8], 16)) == 2
    mask[0, 0] = False
    assert [g.tolerances for g in group_tolerances(table, mask, [0.4, 0.6, 0.8], 16)] == [(0.4, 0.6, 0.8)]


def test_prune_runs_one_representative_per_group(tmp_path, prefix, monkeypatch):
    cfg = make_cfg(tmp_path, tolerances=(None, 0.4, 0.6, 0.8), sample_rates=(1, 16), thresholds=(0.5,),
                   trackers=('sortcython',), reduce=True, table_values=AXIS_VALUES)
    v = all_frames(prefix)
    solves = []
    monkeypatch.setattr(ops, 'prune_scores', lambda s, distance, t_r, limit=None: (solves.append(1) or s, FakeILP()))
    state = st.prune_setup(cfg, None)
    scores = np.full((N, H // TS, W // TS), 200, np.uint8)
    out1 = st.prune(state, Scored('/v/a.mp4', base_branch().with_(sample_rate=1), (), v, scores))
    assert [o.branch['mistrack_tolerance'] for o in out1] == [None, 0.4, 0.6] and len(solves) == 2
    assert out1[2].ilp['group'] == [0.6, 0.8]
    out16 = st.prune(state, Scored('/v/a.mp4', base_branch().with_(sample_rate=16), (), v, scores))
    # An all-1s group still solves once (all-1s gaps do not mean everything is kept).
    assert [o.branch['mistrack_tolerance'] for o in out16] == [None, 0.4] and len(solves) == 3
    assert out16[1].ilp['group'] == [0.4, 0.6, 0.8]
    frames.unlink(v.frames)


def test_all_ones_gaps_drop_polyominoes_whose_tiles_appear_only_once():
    # The ILP constrains only tiles covered in >= 2 sampled frames; with every gap 1 it keeps exactly the
    # polyominoes that contain such a tile and drops the rest (DECISIONS.md A4).  So all-1s groups still solve.
    from tetris.pack.adapters import group_tiles_all
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 256, (12, 6, 8)).astype(np.uint8)
    for t_r in (0.25, 0.5, 0.75):
        pruned, result = ops.prune_scores(scores, np.ones((6, 8), dtype=np.int64), t_r, 5.0)
        bitmaps = (scores >= np.floor(t_r * 255)).astype(np.uint8)
        ids, _ = group_tiles_all(bitmaps, ops.PRUNE_TILEPADDING_MODE)
        ids = np.asarray(ids)
        appearances = (ids >= 0).sum(axis=0)              # frames covering each tile
        expected = np.zeros_like(scores)
        for b in range(len(scores)):
            keep = {int(k) for k in np.unique(ids[b][(ids[b] >= 0) & (appearances >= 2)])}
            expected[b] = (np.isin(ids[b], list(keep)) & (ids[b] >= 0)).astype(np.uint8) * 255
        assert result.status == 2 and np.array_equal(pruned, expected), t_r
        # Some single-appearance polyomino exists, so this is not "keep everything".
        assert not np.array_equal(pruned, bitmaps * 255) or not ((appearances == 1) & (bitmaps.sum(0) > 0)).any()
