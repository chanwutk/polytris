"""
Stage tests (PLAN.md step 4).  Where a p0xx script function exists, the stage is compared with
the script itself: the script's file loaders and cache paths are redirected to temporary files,
so both see the same inputs.  Needs the polyis container (Cython builds, Gurobi, one GPU).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from execution import frames, operators, stages
from execution.config import (Artifacts, Classifier, Config, Detector, Params, StageSystem, System)
from execution.stages import Classified, Decoded, Detected, Packed
from tetris.classify import classify_batch
from tetris.utilities import save_tracking_results
from tests.test_shared_helpers_golden import FakeClassifier

TS = 10                     # tile size: small synthetic videos, many tiles
H, W = 60, 80               # 6x8 tile grid
N_FRAMES = 40


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def prefix():
    p = frames.new_prefix()
    yield p
    assert frames.leftover(p) == [], 'a stage leaked a shared block'
    frames.unlink_all(p)


@pytest.fixture(scope='module')
def video(tmp_path_factory) -> str:
    """A 40-frame MJPG video with two moving rectangles on a noisy background."""
    path = str(tmp_path_factory.mktemp('video') / 'synthetic.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))
    rng = np.random.default_rng(0)
    for t in range(N_FRAMES):
        frame = rng.integers(0, 40, (H, W, 3), dtype=np.uint8)
        frame[10:25, 2 + t:20 + t] = (0, 0, 255)
        frame[35:55, 60 - t:75 - t] = (255, 200, 0)
        writer.write(frame)
    writer.release()
    return path


def read_all(path: str) -> list[np.ndarray]:
    """Every BGR frame, as the scripts read them."""
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def make_cfg(tmp_path: Path, *, sample_rate=2, relevance_threshold=0.5, mistrack_tolerance=None,
             tile_padding='none', tracker='sortcython', canvas_scale=1.0, canvas_format='raw') -> Config:
    """A Config built directly (artifact files are synthetic)."""
    bench = tmp_path / 'bench.jsonl'
    bench.write_text('')
    never = tmp_path / 'never.npy'
    np.save(never, np.ones((H // TS, W // TS), dtype=np.uint8))
    table = tmp_path / 'rates.npy'
    rates = np.full((H // TS, W // TS, 9), 4, dtype=np.int64)
    rates[:, :4] = 8
    np.save(table, rates)
    return Config(
        params=Params(sample_rate, relevance_threshold, mistrack_tolerance, tile_padding, tracker, TS, canvas_scale,
                      canvas_format),
        artifacts=Artifacts(Classifier('Fake', str(tmp_path / 'w.pth'), str(bench)), str(never),
                            str(table) if mistrack_tolerance is not None else None,
                            Detector('ultralytics', {'model_path': 'unused'})),
        system=System(1, {}),
    )


def synthetic_scores(n: int, seed: int = 3) -> np.ndarray:
    """Classifier-like scores with blobs, so frames have several polyominoes."""
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 120, (n, H // TS, W // TS), dtype=np.uint8)
    for b in range(n):
        for _ in range(3):
            y, x = rng.integers(0, H // TS - 1), rng.integers(0, W // TS - 1)
            scores[b, y:y + 2, x:x + rng.integers(1, 3)] = rng.integers(140, 256)
    # Exactly-at-threshold values exercise the >= vs > difference between p022 and p030.
    scores[0, 0, 0] = 127
    scores[0, 0, 1] = 128
    return scores


def score_rows(scores: np.ndarray, sampled) -> list[dict]:
    """scores in the score.jsonl row format the scripts load."""
    return [{'classification_size': list(s.shape), 'classification_hex': s.flatten().tobytes().hex(), 'idx': int(idx)}
            for s, idx in zip(scores, sampled)]


class FakeCache:
    """Stands in for tetris.io.cache: every exec() path lands under a temporary root."""

    def __init__(self, root: Path):
        self.root = root

    def exec(self, dataset, stage, *parts):
        path = self.root.joinpath(dataset, stage, *map(str, parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------- constants and frame selection


@pytest.mark.parametrize('n, s, sampled, needed', [
    (10, 4, [0, 4, 8, 9], [0, 1, 3, 4, 7, 8, 9]),
    (10, 1, list(range(10)), list(range(10))),
    (10, 16, [0, 9], [0, 1, 8, 9]),
    (2, 1, [0, 1], [0, 1]),
    (9, 4, [0, 4, 8], [0, 1, 3, 4, 7, 8]),
])
def test_frame_selection_matches_p020(n, s, sampled, needed):
    got_sampled, partner, got_needed = stages.frame_selection(n, s)
    assert (got_sampled.tolist(), got_needed.tolist()) == (sampled, needed)
    # Partner is parallel to sampled: the next frame for frame 0, the previous frame otherwise.
    assert partner.tolist() == [1] + [i - 1 for i in sampled[1:]]


@pytest.mark.parametrize('s', [1, 2, 3, 4, 8, 16, 100])
def test_needed_is_the_sorted_unique_union_without_sorting(s):
    # Against a set-based reference over many lengths, including 2-frame videos, s=1, and appended last frames.
    for n in range(2, 60):
        sampled, partner, needed = stages.frame_selection(n, s)
        assert needed.tolist() == sorted(set(sampled.tolist()) | set(partner.tolist())), (n, s)
        assert len(partner) == len(sampled) and partner.max() < n


def test_one_frame_video_is_rejected():
    with pytest.raises(ValueError, match='at least 2'):
        stages.frame_selection(1, 1)


# ---------------------------------------------------------------- decode

def test_decode_stores_exactly_the_needed_frames(tmp_path, video, prefix):
    cfg = make_cfg(tmp_path, sample_rate=4)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    reference = read_all(video)
    assert (d.width, d.height, d.frame_count) == (W, H, N_FRAMES)
    assert np.array_equal(d.sampled, stages.frame_selection(N_FRAMES, 4)[0]) and d.sampled[-1] == N_FRAMES - 1
    with frames.attach(d.frames) as m:
        assert m.array.shape == (len(d.needed), H, W, 3)
        for k, idx in enumerate(d.needed):
            assert np.array_equal(m.array[k], reference[idx]), idx
    frames.unlink(d.frames)


class FakeCapture:
    """Frames t = 0..n-1 filled with value t; read(dst) writes into dst only when in_place is set."""

    def __init__(self, n: int, in_place: bool = True):
        self.n, self.t, self.in_place = n, 0, in_place
        self.grabs = 0

    def grab(self) -> bool:
        self.grabs += 1
        self.t += 1
        return self.t <= self.n

    def read(self, dst):
        if self.t >= self.n:
            return False, None
        self.t += 1
        if self.in_place:
            dst[...] = self.t - 1
            return True, dst
        return True, np.full(dst.shape, self.t - 1, dtype=np.uint8)


def test_read_needed_decodes_needed_frames_in_place_and_skips_the_rest():
    needed = np.array([0, 1, 3, 4, 7, 8, 9])
    block = np.zeros((len(needed), 2, 2, 3), dtype=np.uint8)
    cap = FakeCapture(10)
    assert operators._read_needed(cap, block, needed, 'v') == 10
    assert [int(block[k].max()) for k in range(len(needed))] == needed.tolist()
    # Unneeded frames are only grabbed (frames 2, 5, 6), then one final grab fails at the end.
    assert cap.grabs == 3 + 1


def test_read_needed_reports_short_reads():
    needed = np.array([0, 1, 3, 4])
    block = np.zeros((len(needed), 2, 2, 3), dtype=np.uint8)
    assert operators._read_needed(FakeCapture(3), block, needed, 'v') == 3


def test_read_needed_rejects_frames_opencv_did_not_write_in_place():
    # OpenCV returns a new array when the frame does not fit the row: a frame-size mismatch.
    block = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='frame 0 has shape'):
        operators._read_needed(FakeCapture(2, in_place=False), block, np.array([0, 1]), 'v')


def test_decode_rejects_unreadable_files(tmp_path, prefix):
    bad = tmp_path / 'notes.mp4'
    bad.write_text('not a video')
    with pytest.raises(OSError, match='could not open'):
        stages.decode(stages.decode_setup(make_cfg(tmp_path), frames.Stores(prefix, prefix), None), str(bad))
    with pytest.raises(OSError, match='could not open'):
        stages.decode(stages.decode_setup(make_cfg(tmp_path), frames.Stores(prefix, prefix), None), str(tmp_path / 'missing.mp4'))


# ---------------------------------------------------------------- classify

@pytest.fixture
def fake_classifier(monkeypatch):
    """Replace the trained model and its optimization pass with the deterministic fake."""
    monkeypatch.setattr(operators, 'load_classifier', lambda path: FakeClassifier().eval())
    monkeypatch.setattr(operators, 'select_model_optimization', lambda model, *a: (model, 'none'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_matches_p020_batching(tmp_path, video, prefix, fake_classifier):
    cfg = make_cfg(tmp_path, sample_rate=2)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    c = stages.classify(stages.classify_setup(cfg, None), d)
    frames.unlink(d.frames)

    # p020's own loop: RGB frames, batches of 16 over the sampled frames, diff vs previous (or next).
    rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in read_all(video)]
    gh, gw = H // TS, W // TS
    y = torch.arange(gh, device='cuda:0', dtype=torch.uint8).repeat_interleave(gw)
    x = torch.arange(gw, device='cuda:0', dtype=torch.uint8).repeat(gh)
    positions = torch.stack([y, x], dim=1).float()
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, device='cuda:0', dtype=torch.float16).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, device='cuda:0', dtype=torch.float16).view(1, 6, 1, 1)
    mask = torch.ones(gh * gw, device='cuda:0', dtype=torch.uint8)
    expected = []
    with torch.no_grad():
        for start in range(0, len(d.sampled), 16):
            idxs = d.sampled[start:start + 16]
            probs, _ = classify_batch(gw, gh, positions, [rgb[i] for i in idxs],
                                      [rgb[i - 1] if i > 0 else rgb[i + 1] for i in idxs],
                                      FakeClassifier(), TS, 'cuda:0', mean, std, mask)
            expected.append(probs.cpu().numpy())
    assert len(d.sampled) > 16, 'the video must span more than one batch'
    assert c.scores.dtype == np.uint8 and np.array_equal(c.scores, np.concatenate(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='pinned memory needs CUDA')
def test_classify_staging_buffer_is_pinned_reused_and_remade_for_another_frame_size():
    state = {}
    first = operators._staging(state, (H, W, 3))
    assert first.is_pinned() and tuple(first.shape) == (2 * operators.CLASSIFY_BATCH, H, W, 3)
    assert operators._staging(state, (H, W, 3)) is first
    other = operators._staging(state, (H * 2, W, 3))
    assert other is not first and tuple(other.shape[1:]) == (H * 2, W, 3) and state['staging'] is other


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_reads_pooled_frames_like_fresh_ones(tmp_path, video, prefix, fake_classifier):
    cfg = make_cfg(tmp_path, sample_rate=1)
    state = stages.classify_setup(cfg, None)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    fresh = stages.classify(state, d).scores
    frames.unlink(d.frames)
    pool_prefix = frames.new_prefix()
    try:
        pool = frames.create_pool(pool_prefix, 'frames', 1, N_FRAMES * H * W * 3)
        d = stages.decode(stages.decode_setup(cfg, frames.Stores(pool, prefix), None), video)
        assert np.array_equal(stages.classify(state, d).scores, fresh)
        frames.release(pool, d.frames)
    finally:
        frames.unlink_all(pool_prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_setup_optimizes_once_for_the_bitmaps_tile_count(tmp_path, video, prefix, monkeypatch):
    calls = []
    monkeypatch.setattr(operators, 'load_classifier', lambda path: FakeClassifier().eval())

    def fake_optimize(model, benchmark_results, device, tile_size, batch_size):
        calls.append((tile_size, batch_size))
        return model, 'none'

    monkeypatch.setattr(operators, 'select_model_optimization', fake_optimize)
    cfg = make_cfg(tmp_path)
    state = stages.classify_setup(cfg, None)
    # One call, in setup, with the grid's tile count (6x8 here), as p020 passes its first video's grid.
    assert calls == [(TS, (H // TS) * (W // TS))]
    for _ in range(2):
        d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
        stages.classify(state, d)
        frames.unlink(d.frames)
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
@pytest.mark.parametrize('bitmap_shape', [(3, 3), (W // TS, H // TS)])
def test_classify_rejects_a_mismatched_never_relevant_bitmap(tmp_path, video, prefix, fake_classifier, bitmap_shape):
    # A different grid, and the transposed grid (same tile count, different shape), are both rejected.
    cfg = make_cfg(tmp_path)
    np.save(cfg.artifacts.never_relevant_bitmap, np.ones(bitmap_shape, dtype=np.uint8))
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    try:
        with pytest.raises(ValueError, match=r'video grid \(6, 8\) does not match the never-relevant bitmap grid'):
            stages.classify(stages.classify_setup(cfg, None), d)
    finally:
        frames.unlink(d.frames)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_setup_rejects_a_model_in_training_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, 'load_classifier', lambda path: FakeClassifier())   # nn.Module: training by default
    monkeypatch.setattr(operators, 'select_model_optimization', lambda model, *a: (model, 'none'))
    with pytest.raises(RuntimeError, match='training mode'):
        stages.classify_setup(make_cfg(tmp_path), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_setup_rejects_a_flat_never_relevant_bitmap(tmp_path, fake_classifier):
    cfg = make_cfg(tmp_path)
    np.save(cfg.artifacts.never_relevant_bitmap, np.ones((H // TS) * (W // TS), dtype=np.uint8))
    with pytest.raises(ValueError, match='must be 2-D'):
        stages.classify_setup(cfg, None)


# ---------------------------------------------------------------- prune

# Solve time limit for the prune-vs-p022 comparison: two independent 0.1 s solves can differ under machine
# load (one hits the limit before finding any solution and keeps nothing), so both get time to reach optimal.


def test_prune_rejects_a_mismatched_rate_table(tmp_path):
    cfg = make_cfg(tmp_path, mistrack_tolerance=0.4)
    np.save(cfg.artifacts.max_rate_table, np.ones((2, 2, 9), dtype=np.int64))
    decoded = Decoded('v.avi', W, H, N_FRAMES, np.array([0, 1]), np.array([1, 0]), np.array([]),
                      frames.Block('unused', (0,), '|u1'))
    with pytest.raises(ValueError, match='max_rate_table grid'):
        stages.prune(stages.prune_setup(cfg, None), Classified(decoded, synthetic_scores(2)))


# ---------------------------------------------------------------- pack


def test_packed_tiles_are_exact_copies_of_their_source_tiles(tmp_path, video, prefix):
    # Independent of p030: every tile an offset lookup places must equal its source tile pixel for pixel.
    cfg = make_cfg(tmp_path, sample_rate=1)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    p = stages.pack_video(stages.pack_setup(cfg, frames.Stores(prefix, prefix), None), Classified(d, synthetic_scores(len(d.sampled))))
    reference = read_all(video)
    with frames.attach(p.canvases) as m:
        for k, lookup in enumerate(p.offset_lookups):
            for gid, ((py, px), (oy, ox), frame_idx) in enumerate(lookup, start=1):
                for i, j in zip(*np.nonzero(p.index_maps[k] == gid)):
                    src = reference[frame_idx][(oy + i - py) * TS:(oy + i - py + 1) * TS,
                                               (ox + j - px) * TS:(ox + j - px + 1) * TS]
                    assert np.array_equal(m.array[k, i * TS:(i + 1) * TS, j * TS:(j + 1) * TS], src)
    frames.unlink(p.canvases)


@pytest.mark.parametrize('canvas_format', ['raw', 'jpeg'])
def test_pooled_slots_give_the_same_frames_and_canvases_as_fresh_blocks(tmp_path, video, prefix, canvas_format):
    # One slot per pool, so the second video reuses both slots; the dense video leaves canvas pixels behind
    # that the sparse one's fewer collages must not show.
    cfg = make_cfg(tmp_path, sample_rate=1, canvas_format=canvas_format)
    fresh = frames.Stores(prefix, prefix)
    pool_prefix = frames.new_prefix()
    try:
        n = N_FRAMES
        pools = frames.Stores(frames.create_pool(pool_prefix, 'frames', 1, n * H * W * 3),
                              frames.create_pool(pool_prefix, 'canvases', 1, n * H * W * 3))
        sparse = synthetic_scores(n)
        sparse[:, :, 2:] = 0
        collages = []
        for scores in (synthetic_scores(n), sparse):
            got = {}
            for name, stores in (('fresh', fresh), ('pooled', pools)):
                d = stages.decode(stages.decode_setup(cfg, stores, None), video)
                with frames.attach(d.frames) as m:
                    decoded = m.array.copy()
                p = stages.pack_video(stages.pack_setup(cfg, stores, None), Classified(d, scores))
                with frames.attach(p.canvases) as m:
                    got[name] = (decoded, m.array.copy(), p.index_maps, p.offset_lookups)
                frames.free(stores.canvases, p.canvases)
            (fd, fc, fm, fl), (pd, pc, pm, pl) = got['fresh'], got['pooled']
            assert np.array_equal(fd, pd) and np.array_equal(fc, pc) and np.array_equal(fm, pm) and fl == pl
            collages.append(len(pl))
        # The reuse case is real: the sparse video draws fewer collages into the slot the dense one filled.
        assert collages[0] > collages[1] > 0
        # Pack gave the frames slot back and detect's stand-in the canvases slot.
        frames.check_returned(pools.frames)
        frames.check_returned(pools.canvases)
    finally:
        frames.unlink_all(pool_prefix)


def test_pack_with_nothing_relevant_makes_zero_collages(tmp_path, video, prefix):
    cfg = make_cfg(tmp_path)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    p = stages.pack_video(stages.pack_setup(cfg, frames.Stores(prefix, prefix), None),
                          Classified(d, np.zeros((len(d.sampled), H // TS, W // TS), np.uint8)))
    assert p.index_maps.shape[0] == 0 and p.offset_lookups == []
    frames.unlink(p.canvases)


# ---------------------------------------------------------------- detect

def test_detect_unpacks_into_every_frame_and_frees_canvases(tmp_path, video, prefix, monkeypatch):
    cfg = make_cfg(tmp_path, sample_rate=2)
    d = stages.decode(stages.decode_setup(cfg, frames.Stores(prefix, prefix), None), video)
    p = stages.pack_video(stages.pack_setup(cfg, frames.Stores(prefix, prefix), None), Classified(d, synthetic_scores(len(d.sampled))))
    batches: list[int] = []

    def fake_detect_batch(images, detector):
        # One box inside the first placed tile of each canvas, in canvas order.
        first = sum(batches)
        batches.append(len(images))
        out = []
        for k in range(first, first + len(images)):
            i, j = map(int, np.argwhere(p.index_maps[k] > 0)[0])
            out.append(np.array([[j * TS + 2, i * TS + 2, j * TS + 8, i * TS + 8, 0.9]]))
        return out

    monkeypatch.setattr(operators, 'detect_batch', fake_detect_batch)
    det = stages.detect({'detector': None, 'tile_size': TS, 'canvases': prefix}, p)
    assert frames.leftover(prefix) == []
    assert sorted(det.detections) == list(range(N_FRAMES))
    assert all(b <= stages.DETECT_BATCH for b in batches) and sum(batches) == len(p.offset_lookups)
    # Each box lands in the frame of the polyomino it was drawn on, shifted back to that frame's tile.
    for k, lookup in enumerate(p.offset_lookups):
        i, j = map(int, np.argwhere(p.index_maps[k] > 0)[0])
        (py, px), (oy, ox), frame_idx = lookup[int(p.index_maps[k][i, j]) - 1]
        x0, y0 = (j - px + ox) * TS, (i - py + oy) * TS
        assert [x0 + 2, y0 + 2, x0 + 8, y0 + 8, 0.9] in det.detections[frame_idx]
    assert sum(len(v) for v in det.detections.values()) == len(p.offset_lookups)


# ---------------------------------------------------------------- track


# ---------------------------------------------------------------- wiring

def test_build_stages_wires_procs_gpus_and_optional_prune(tmp_path):
    base = make_cfg(tmp_path)
    system = System(2, {'decode': StageSystem(3, None), 'classify': StageSystem(1, 0), 'prune': StageSystem(1, None),
                        'pack': StageSystem(2, None), 'detect': StageSystem(1, 5), 'track': StageSystem(4, None)})
    no_prune = Config(base.params, base.artifacts, system)
    got = stages.build_stages(no_prune, frames.Stores('p-', 'p-'))
    assert [(s.name, s.procs, s.gpu) for s in got] == [
        ('decode', 3, None), ('classify', 1, 0), ('pack', 2, None), ('detect', 1, 5), ('track', 4, None)]
    pruning = make_cfg(tmp_path, mistrack_tolerance=0.4)
    got = stages.build_stages(Config(pruning.params, pruning.artifacts, system), frames.Stores('p-', 'p-'))
    assert [s.name for s in got] == ['decode', 'classify', 'prune', 'pack', 'detect', 'track']
