import numpy as np
import pytest

from scripts import p022_exec_prune_polyominoes as prune


@pytest.mark.parametrize('threshold,first_kept', [
    (0.0, 0), (0.2, 51), (0.25, 63), (0.5, 127), (0.75, 191), (1.0, 255),
])
def test_process_video_preserves_cached_relevance_boundary(tmp_path, monkeypatch, threshold, first_kept):
    # Exercise the pipeline's decoding and thresholding across every possible quantized score.
    scores = np.arange(256, dtype=np.uint8).reshape(16, 16)
    score_path = tmp_path / 'score.jsonl'
    score_path.touch()
    monkeypatch.setattr(prune.cache, 'exec', lambda *args: str(score_path))
    monkeypatch.setattr(prune, 'load_classification_results', lambda *args, **kwargs: [{
        'idx': 4, 'classification_size': [16, 16], 'classification_hex': scores.tobytes().hex(),
    }])
    monkeypatch.setattr(prune, 'load_max_sampling_rate', lambda *args: np.ones((16, 16)))

    class GroupingReached(Exception):
        pass

    def check_bitmaps(bitmaps, padding):
        # Check the actual grouping input, then stop before solving or writing any cached output.
        expected = np.zeros(256, dtype=np.uint8)
        expected[first_kept:] = 1
        np.testing.assert_array_equal(bitmaps, expected.reshape(1, 16, 16))
        assert bitmaps.dtype == np.uint8
        assert padding == 0
        raise GroupingReached

    monkeypatch.setattr(prune, 'group_tiles_all', check_bitmaps)
    with pytest.raises(GroupingReached):
        prune.process_video('demo', 'test', 'te00.mp4', 'C', 60, 1, 'sortcython', 0.4, threshold, None)
