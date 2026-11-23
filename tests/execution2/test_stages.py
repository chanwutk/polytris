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

from execution2 import frames, operators, stages
from execution2.config import (Artifacts, Classifier, Config, Detector, Params, StageSystem, System)
from execution2.stages import Classified, Decoded, Detected, Packed
from polyis.classify import classify_batch
from polyis.utilities import save_tracking_results
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
    """Stands in for polyis.io.cache: every exec() path lands under a temporary root."""

    def __init__(self, root: Path):
        self.root = root

    def exec(self, dataset, stage, *parts):
        path = self.root.joinpath(dataset, stage, *map(str, parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------- constants and frame selection

def test_constants_match_the_scripts(monkeypatch):
    import scripts.p020_exec_classify as p020
    import scripts.p022_exec_prune_polyominoes as p022
    import scripts.p030_exec_compress as p030
    import scripts.p040_exec_detect as p040
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
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
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
        stages.decode(stages.decode_setup(make_cfg(tmp_path), prefix, None), str(bad))
    with pytest.raises(OSError, match='could not open'):
        stages.decode(stages.decode_setup(make_cfg(tmp_path), prefix, None), str(tmp_path / 'missing.mp4'))


# ---------------------------------------------------------------- classify

@pytest.fixture
def fake_classifier(monkeypatch):
    """Replace the trained model and its optimization pass with the deterministic fake."""
    monkeypatch.setattr(operators, 'load_classifier', lambda path: FakeClassifier().eval())
    monkeypatch.setattr(operators, 'select_model_optimization', lambda model, *a: (model, 'none'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
def test_classify_matches_p020_batching(tmp_path, video, prefix, fake_classifier):
    cfg = make_cfg(tmp_path, sample_rate=2)
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
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
        d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
        stages.classify(state, d)
        frames.unlink(d.frames)
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='classify runs on the GPU')
@pytest.mark.parametrize('bitmap_shape', [(3, 3), (W // TS, H // TS)])
def test_classify_rejects_a_mismatched_never_relevant_bitmap(tmp_path, video, prefix, fake_classifier, bitmap_shape):
    # A different grid, and the transposed grid (same tile count, different shape), are both rejected.
    cfg = make_cfg(tmp_path)
    np.save(cfg.artifacts.never_relevant_bitmap, np.ones(bitmap_shape, dtype=np.uint8))
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
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
PRUNE_TEST_TIME_LIMIT_S = 5.0


def run_p022(monkeypatch, tmp_path, cfg, scores, sampled) -> np.ndarray:
    """p022.process_video on these scores, with PRUNE_TEST_TIME_LIMIT_S; returns its pruned 0/255 grids."""
    import scripts.p022_exec_prune_polyominoes as p022
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


def test_prune_rejects_a_mismatched_rate_table(tmp_path):
    cfg = make_cfg(tmp_path, mistrack_tolerance=0.4)
    np.save(cfg.artifacts.max_rate_table, np.ones((2, 2, 9), dtype=np.int64))
    decoded = Decoded('v.avi', W, H, N_FRAMES, np.array([0, 1]), np.array([1, 0]), np.array([]),
                      frames.Block('unused', (0,), '|u1'))
    with pytest.raises(ValueError, match='max_rate_table grid'):
        stages.prune(stages.prune_setup(cfg, None), Classified(decoded, synthetic_scores(2)))


# ---------------------------------------------------------------- pack

def run_p030(monkeypatch, tmp_path, cfg, video, scores, sampled) -> tuple[list, list, list]:
    """p030.compress on these scores; returns its (JPEG-decoded canvases, index maps, offset lookups) in collage order."""
    import scripts.p030_exec_compress as p030
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
        # test_packed_tiles_are_exact_copies_of_their_source_tiles.
        assert not np.array_equal(ours, np.stack(canvases))
        assert np.abs(np.stack(canvases).astype(int) - ours.astype(int)).mean() < 10.0


def test_packed_tiles_are_exact_copies_of_their_source_tiles(tmp_path, video, prefix):
    # Independent of p030: every tile an offset lookup places must equal its source tile pixel for pixel.
    cfg = make_cfg(tmp_path, sample_rate=1)
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
    p = stages.pack_video(stages.pack_setup(cfg, prefix, None), Classified(d, synthetic_scores(len(d.sampled))))
    reference = read_all(video)
    with frames.attach(p.canvases) as m:
        for k, lookup in enumerate(p.offset_lookups):
            for gid, ((py, px), (oy, ox), frame_idx) in enumerate(lookup, start=1):
                for i, j in zip(*np.nonzero(p.index_maps[k] == gid)):
                    src = reference[frame_idx][(oy + i - py) * TS:(oy + i - py + 1) * TS,
                                               (ox + j - px) * TS:(ox + j - px + 1) * TS]
                    assert np.array_equal(m.array[k, i * TS:(i + 1) * TS, j * TS:(j + 1) * TS], src)
    frames.unlink(p.canvases)


def test_pack_with_nothing_relevant_makes_zero_collages(tmp_path, video, prefix):
    cfg = make_cfg(tmp_path)
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
    p = stages.pack_video(stages.pack_setup(cfg, prefix, None),
                          Classified(d, np.zeros((len(d.sampled), H // TS, W // TS), np.uint8)))
    assert p.index_maps.shape[0] == 0 and p.offset_lookups == []
    frames.unlink(p.canvases)


# ---------------------------------------------------------------- detect

def test_detect_unpacks_into_every_frame_and_frees_canvases(tmp_path, video, prefix, monkeypatch):
    cfg = make_cfg(tmp_path, sample_rate=2)
    d = stages.decode(stages.decode_setup(cfg, prefix, None), video)
    p = stages.pack_video(stages.pack_setup(cfg, prefix, None), Classified(d, synthetic_scores(len(d.sampled))))
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
    det = stages.detect({'detector': None, 'tile_size': TS}, p)
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

def test_track_matches_p060(monkeypatch, tmp_path):
    import scripts.p060_exec_track as p060
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


# ---------------------------------------------------------------- wiring

def test_build_stages_wires_procs_gpus_and_optional_prune(tmp_path):
    base = make_cfg(tmp_path)
    system = System(2, {'decode': StageSystem(3, None), 'classify': StageSystem(1, 0), 'prune': StageSystem(1, None),
                        'pack': StageSystem(2, None), 'detect': StageSystem(1, 5), 'track': StageSystem(4, None)})
    no_prune = Config(base.params, base.artifacts, system)
    got = stages.build_stages(no_prune, 'p-')
    assert [(s.name, s.procs, s.gpu) for s in got] == [
        ('decode', 3, None), ('classify', 1, 0), ('pack', 2, None), ('detect', 1, 5), ('track', 4, None)]
    pruning = make_cfg(tmp_path, mistrack_tolerance=0.4)
    got = stages.build_stages(Config(pruning.params, pruning.artifacts, system), 'p-')
    assert [s.name for s in got] == ['decode', 'classify', 'prune', 'pack', 'detect', 'track']
