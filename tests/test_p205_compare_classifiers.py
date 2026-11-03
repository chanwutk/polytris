import json

import numpy as np
import pytest

from evaluation import p205_compare_classifiers as p205
from polyis.io import cache

TILE = 60
# A 2x3 tile grid (120x180 pixels) where one object moves one tile right per frame.
BOXES = {0: [[5, 5, 50, 50]], 1: [[65, 5, 110, 50]], 2: [[125, 5, 170, 50]]}


def grid_for(boxes):
    # Classifier output that marks exactly the tiles the boxes touch (score 255) as relevant.
    grid = np.zeros((2, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        grid[int(y1) // TILE:int(y2 - 1e-9) // TILE + 1, int(x1) // TILE:int(x2 - 1e-9) // TILE + 1] = 255
    return grid


def entry(idx, grid):
    return {'idx': idx, 'classification_hex': grid.tobytes().hex(), 'classification_size': list(grid.shape)}


def write_tracks(dataset, video, frames):
    path = cache.exec(dataset, 'groundtruth', video, 'tracking.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': f, 'tracks': [[i, *b] for i, b in enumerate(boxes)]}) + '\n'
                            for f, boxes in frames.items()))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path))
    # Classifier output that follows the moving object frame by frame.
    results = [entry(f, grid_for(BOXES[f])) for f in sorted(BOXES)]
    monkeypatch.setattr(p205, 'load_classification_results', lambda *args: results)
    return tmp_path


def run(dataset):
    return p205.evaluate_video_pair(dataset, 'te00.mp4', TILE, 1, 'exp', 'base', 0.5, np.ones((2, 3), dtype=np.uint8))


def test_frame_k_is_scored_against_ground_truth_frame_k(env):
    write_tracks('ams', 'te00.mp4', BOXES)
    exp, base = run('ams')
    # A perfect classifier is perfect only when frames are paired one to one.
    assert exp == base == {'tp': 3, 'fp': 0, 'fn': 0, 'tn': 15}


def test_detector_variants_read_the_root_ground_truth(env):
    write_tracks('ams', 'te00.mp4', BOXES)
    # The variant's own copy is stale and must be ignored.
    write_tracks('ams-y05', 'te00.mp4', {0: [], 1: [], 2: []})
    assert run('ams-y05')[0]['fn'] == 0 and run('ams-y05')[0]['tp'] == 3


def test_frames_without_ground_truth_rows_count_as_empty(env):
    # Frame 1 has no row: its predicted tile becomes a false positive instead of being skipped.
    write_tracks('ams', 'te00.mp4', {0: BOXES[0], 2: BOXES[2]})
    exp, _ = run('ams')
    assert exp == {'tp': 2, 'fp': 1, 'fn': 0, 'tn': 15}
    assert sum(exp.values()) == 3 * 6


def test_ground_truth_at_a_different_frame_rate_is_rejected(env):
    # A 30 FPS file reaches twice as far as the 15 FPS classifier output.
    write_tracks('ams', 'te00.mp4', {f: [] for f in range(6)})
    with pytest.raises(ValueError, match='frame rates differ'):
        run('ams')


def test_missing_ground_truth_is_an_error(env):
    with pytest.raises(FileNotFoundError, match='tracking.jsonl'):
        run('ams')


def test_tile_confusion_counts_relevant_tiles_as_positive():
    from polyis.utilities import tile_confusion
    # Asymmetric case: 1 TP, 2 FP, 1 FN, 2 TN, so any swap of the cells is visible.
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    boxes = [[5, 5, 50, 50], [5, 65, 50, 110]]  # tiles (0, 0) and (1, 0)
    assert tile_confusion(scores, boxes, TILE, 0.5) == {'tp': 1, 'fp': 2, 'fn': 1, 'tn': 2}


def test_tile_confusion_skips_masked_tiles_and_checks_the_mask_shape():
    from polyis.utilities import tile_confusion
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    mask = np.array([[1, 0, 0], [1, 1, 1]])
    assert tile_confusion(scores, [[5, 65, 50, 110]], TILE, 0.5, mask) == {'tp': 0, 'fp': 1, 'fn': 1, 'tn': 2}
    with pytest.raises(ValueError, match='eval_mask shape'):
        tile_confusion(scores, [], TILE, 0.5, np.ones((3, 3)))


def test_p024_uses_the_same_counts_on_track_rows():
    from evaluation import p024_exec_classify_tradeoff as p024
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    result = p024.evaluate_classification_accuracy(scores, [[9, 5, 5, 50, 50], [3, 5, 65, 50, 110]], TILE, 0.5)
    assert {k: result[k] for k in ('tp', 'fp', 'fn', 'tn')} == {'tp': 1, 'fp': 2, 'fn': 1, 'tn': 2}
    assert result['precision'] == pytest.approx(1 / 3) and result['recall'] == pytest.approx(1 / 2)
