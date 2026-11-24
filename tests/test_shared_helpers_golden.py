"""
Golden tests for the helpers execution shares with the p0xx scripts (PLAN.md 5A/11A/19B).

The goldens in tests/golden/shared_helpers.pkl were recorded from the helpers while they
still lived in experiments/scripts/p020, p030 and p050.  After the move to tetris/, these tests
assert byte-equal outputs, so the move cannot have changed behaviour.

Record (only before the move, or after a deliberate behaviour change):
    python tests/test_shared_helpers_golden.py --record
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris.classify import classify_batch, classify_batch_tensors
from tetris.pack.collage import (
    _compute_polyomino_tile_boundaries,
    _copy_polyomino_tiles_to_canvas,
    _update_polyomino_metadata,
    unpack_detections,
)

GOLDEN = Path(__file__).parent / 'golden' / 'shared_helpers.pkl'


class FakeClassifier(torch.nn.Module):
    """Deterministic stand-in for the trained classifier: one logit per tile from its pixels and position."""

    def forward(self, tiles: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Mean over channels and pixels, plus a position-dependent offset, as an (N, 1) logit.
        return (tiles.float().mean(dim=(1, 2, 3)) + positions.float().sum(dim=1) * 0.1).unsqueeze(1).half()


def run_classify(frame_hw: tuple[int, int]) -> np.ndarray:
    """classify_batch on 3 seeded frames on CPU; returns the uint8 relevance grids."""
    return classify_both_ways(frame_hw, use_tensors=False)[0]


def classify_both_ways(frame_hw: tuple[int, int], use_tensors: bool) -> tuple[np.ndarray, dict]:
    """The golden classify inputs through classify_batch (numpy lists) or classify_batch_tensors (stacked tensors)."""
    tile = 10
    rng = np.random.default_rng(0)
    h, w = frame_hw
    # Three frames and their diff partners; one frame is all black to hit the non-black mask.
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(3)]
    frames[1][:] = 0
    prevs = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(3)]
    grid_h, grid_w = h // tile, w // tile
    # Tile positions exactly as p020 builds them.
    y = torch.arange(grid_h, dtype=torch.uint8).repeat_interleave(grid_w)
    x = torch.arange(grid_w, dtype=torch.uint8).repeat(grid_h)
    positions = torch.stack([y, x], dim=1).float()
    # p020's normalization constants and a never-relevant mask with some tiles disabled.
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, dtype=torch.float16).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, dtype=torch.float16).view(1, 6, 1, 1)
    mask = torch.from_numpy((np.arange(grid_h * grid_w) % 3 != 0).astype(np.uint8))
    if use_tensors:
        probs, runtime = classify_batch_tensors(grid_w, grid_h, positions, torch.from_numpy(np.stack(frames)),
                                                torch.from_numpy(np.stack(prevs)), FakeClassifier(), tile, 'cpu',
                                                mean, std, mask)
    else:
        probs, runtime = classify_batch(grid_w, grid_h, positions, frames, prevs, FakeClassifier(),
                                        tile, 'cpu', mean, std, mask)
    return probs.numpy(), runtime


def grid_bounds(n: int, tile: int, short_last: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tile start/end pixel arrays; short_last trims the last tile to exercise edge repeat."""
    starts = np.arange(n, dtype=np.int32) * tile
    ends = starts + tile
    ends[-1] -= short_last
    return starts, ends


def run_render(short_last: int) -> dict:
    """Render one collage the way p030's loop does, from seeded polyominoes."""
    tile = 10
    rng = np.random.default_rng(1)
    # Source grid 4x5 tiles, destination canvas 3x4 tiles.
    src_y, src_x = grid_bounds(4, tile, short_last), grid_bounds(5, tile, short_last)
    dst_y, dst_x = grid_bounds(3, tile), grid_bounds(4, tile)
    frames = {7: rng.integers(0, 256, (40, 50, 3), dtype=np.uint8),
              9: rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)}
    # (oy, ox, py, px, frame_idx, shape) placements; shapes are (i, j) tile offsets.
    placements = [
        (0, 0, 0, 0, 7, np.array([[0, 0], [0, 1], [1, 0]])),
        (2, 3, 0, 2, 9, np.array([[0, 0], [1, 0], [1, 1]])),   # reaches the short last row/col
        (3, 4, 2, 3, 7, np.array([[0, 0]])),                   # corner tile, short on both axes
    ]
    canvas = np.zeros((3 * tile, 4 * tile, 3), dtype=np.uint8)
    index_map = np.zeros((3, 4), dtype=np.uint16)
    offset_lookup: list = []
    for gid, (oy, ox, py, px, frame_idx, shape) in enumerate(placements, start=1):
        i, j = shape[:, 0], shape[:, 1]
        bounds = _compute_polyomino_tile_boundaries(oy, ox, py, px, i, j, *src_y, *src_x, *dst_y, *dst_x)
        _copy_polyomino_tiles_to_canvas(canvas, frames[frame_idx], *bounds)
        _update_polyomino_metadata(index_map, offset_lookup, gid, py, px, oy, ox, frame_idx, i, j)
    return {'canvas': canvas, 'index_map': index_map, 'offset_lookup': offset_lookup}


def run_unpack(drop_center_outside: bool) -> tuple:
    """unpack_detections on a fixed collage layout and seeded boxes, including off-canvas ones."""
    tile = 10
    index_map = np.array([[1, 1, 0], [2, 0, 3]], dtype=np.uint16)
    offset_lookup = [((0, 0), (2, 3), 4), ((1, 0), (0, 0), 5), ((1, 2), (5, 1), 4)]
    rng = np.random.default_rng(2)
    boxes = []
    for _ in range(40):
        x1, y1 = rng.uniform(-5, 30), rng.uniform(-5, 20)
        boxes.append([x1, y1, x1 + rng.uniform(1, 15), y1 + rng.uniform(1, 15), rng.uniform(), 0.0])
    return unpack_detections(boxes, index_map, offset_lookup, tile, drop_center_outside)


def compute_all() -> dict:
    """Every golden case, keyed by name."""
    return {
        'classify_aligned': run_classify((30, 40)),
        'classify_resized': run_classify((34, 47)),
        'render_exact': run_render(0),
        'render_edge_repeat': run_render(3),
        'unpack_keep': run_unpack(False),
        'unpack_drop': run_unpack(True),
    }


def assert_same(actual, expected, path: str):
    """Recursive exact equality over dicts, tuples, lists, and arrays."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            assert_same(actual[k], expected[k], f'{path}.{k}')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f'{path}[{i}]')
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected), path
    else:
        assert actual == expected, path


def load_golden() -> dict:
    with open(GOLDEN, 'rb') as f:
        return pickle.load(f)


def test_classify_batch_matches_golden():
    golden = load_golden()
    actual = compute_all()
    for case in ('classify_aligned', 'classify_resized'):
        assert_same(actual[case], golden[case], case)


def test_collage_render_matches_golden():
    golden = load_golden()
    actual = compute_all()
    for case in ('render_exact', 'render_edge_repeat'):
        assert_same(actual[case], golden[case], case)


def test_unpack_detections_matches_golden():
    golden = load_golden()
    actual = compute_all()
    for case in ('unpack_keep', 'unpack_drop'):
        assert_same(actual[case], golden[case], case)


@pytest.mark.parametrize('frame_hw', [(30, 40), (34, 47)])
def test_classify_batch_tensors_equals_classify_batch(frame_hw):
    # The tensor core (execution's entry point) and the list wrapper (p020's) give identical grids,
    # and report the same runtime fields, so p020's runtime.jsonl keeps its shape.
    by_list, list_runtime = classify_both_ways(frame_hw, use_tensors=False)
    by_tensor, tensor_runtime = classify_both_ways(frame_hw, use_tensors=True)
    assert np.array_equal(by_list, by_tensor)
    assert [r['op'] for r in list_runtime] == [r['op'] for r in tensor_runtime]


def test_golden_cases_are_not_trivial():
    # Guard against goldens that would pass for broken code (all-zero canvases, no detections).
    golden = load_golden()
    assert golden['render_exact']['canvas'].any()
    assert not np.array_equal(golden['render_exact']['canvas'], golden['render_edge_repeat']['canvas'])
    assert golden['classify_aligned'].any() and len(set(golden['classify_aligned'].ravel())) > 2
    assert golden['unpack_keep'][0] and golden['unpack_keep'][1] and golden['unpack_keep'][2]


if __name__ == '__main__' and '--record' in sys.argv:
    GOLDEN.parent.mkdir(exist_ok=True)
    with open(GOLDEN, 'wb') as f:
        pickle.dump(compute_all(), f)
    print(f'recorded {GOLDEN}')
