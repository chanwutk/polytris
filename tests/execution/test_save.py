"""execution/save.py: directory names, the scripts' formats, the timing dummy skipped, and saver wiring."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from execution import save
from execution.operators import Decoded, Packed
from execution.pipeline import Stage
from execution.stages import Classified, Detected, Tracked
from tests.execution.test_stages import make_cfg

PARAMS = {'classifier': 'C', 'tile_size': 60, 'sample_rate': 4, 'relevance_threshold': 0.5,
          'mistrack_tolerance': 0.4, 'tile_padding': 'tr', 'canvas_scale': 1.0, 'tracker': 'sortcython'}
UNPRUNED = {**PARAMS, 'mistrack_tolerance': None}


def decoded(video='/videos/va00.mp4') -> Decoded:
    # Only the fields the savers read are meaningful; frames is never touched (pixels are not saved).
    return Decoded(video, 120, 60, 9, np.array([0, 4, 8]), np.array([1, 3, 7]), np.array([0, 1, 3, 4, 7, 8]), None)


def scores() -> np.ndarray:
    return np.arange(3 * 1 * 2, dtype=np.uint8).reshape(3, 1, 2)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def stage_path(root: Path, stage_dir: str, name: str, video='va00.mp4') -> Path:
    return root / 'execution' / video / stage_dir / name


def test_param_strings_match_the_scripts_names():
    assert save.param_str(PARAMS, *save.SCORE_PARAMS) == 'C_60_4'
    assert save.param_str(PARAMS, *save.PRUNE_PARAMS) == 'C_60_4_040_r050_sortcython'
    # The tracker joins pack's name only when pruned; the unpruned tolerance is left out entirely.
    assert save.pack_param_str(PARAMS) == 'C_60_4_040_r050_tr_s100_sortcython'
    assert save.pack_param_str(UNPRUNED) == 'C_60_4_r050_tr_s100'
    assert save.param_str(UNPRUNED, *save.TRACK_PARAMS) == 'C_60_4_r050_tr_s100_sortcython'


def test_param_str_names_a_missing_parameter():
    with pytest.raises(KeyError, match='has no'):
        save.param_str({'classifier': 'C', 'tile_size': 60}, 'sample_rate')


def test_classify_and_prune_write_score_rows_and_the_ilp_record(tmp_path):
    ilp = {'status': 2, 'solve_ms': 12.5, 'kept_tiles': 3}
    save.save_classified(str(tmp_path), PARAMS, None, Classified(decoded(), scores()))
    save.save_pruned(str(tmp_path), PARAMS, None, Classified(decoded(), scores() * 2, ilp))
    classified = rows(stage_path(tmp_path, '020_relevancy', 'C_60_4') / 'score' / 'score.jsonl')
    pruned_dir = stage_path(tmp_path, '022_pruned_polyominoes', 'C_60_4_040_r050_sortcython')
    pruned = rows(pruned_dir / 'score' / 'score.jsonl')
    # One row per sampled frame, with p020's fields; the hex decodes back to the grid.
    assert [r['idx'] for r in classified] == [0, 4, 8]
    assert all(r['classification_size'] == [1, 2] for r in classified)
    decoded_grids = np.stack([np.frombuffer(bytes.fromhex(r['classification_hex']), np.uint8).reshape(1, 2)
                              for r in pruned])
    assert np.array_equal(decoded_grids, scores() * 2)
    assert json.loads((pruned_dir / 'ilp.json').read_text()) == ilp


def test_pack_writes_index_maps_and_offset_lookups_named_as_p030(tmp_path):
    maps = np.arange(2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 3)
    lookups = [[[[0, 0], [1, 1], 4], [[0, 1], [2, 2], 8]], [[[1, 0], [0, 0], 0]]]
    save.save_packed(str(tmp_path), PARAMS, None, Packed('/videos/va00.mp4', 120, 60, 9, None, maps, lookups))
    d = stage_path(tmp_path, '033_compressed_frames', 'C_60_4_040_r050_tr_s100_sortcython')
    # <collage>_<first frame>_<last frame>, as p030 names each collage.
    assert sorted(p.name for p in (d / 'index_maps').iterdir()) == ['0000_0004_0008.npy', '0001_0000_0000.npy']
    assert np.array_equal(np.load(d / 'index_maps' / '0000_0004_0008.npy'), maps[0])
    assert rows(d / 'offset_lookups' / '0001_0000_0000.jsonl') == [[[1, 0], [0, 0], 0]]


def test_detect_and_track_write_p050_and_p060_files(tmp_path):
    detections = {1: [[0.0, 1.0, 2.0, 3.0, 0.9]], 0: []}
    tracks = {0: [[1.0, 0.0, 0.0, 10.0, 10.0]], 1: [[1.0, 1.0, 1.0, 11.0, 11.0]]}
    save.save_detected(str(tmp_path), UNPRUNED, None, Detected('/videos/va00.mp4', 120, 60, 2, detections))
    save.save_tracked(str(tmp_path), UNPRUNED, None, Tracked('/videos/va00.mp4', tracks))
    dets = rows(stage_path(tmp_path, '050_uncompressed_detections', 'C_60_4_r050_tr_s100') / 'detections.jsonl')
    # Every frame, in frame order.
    assert dets == [{'frame_idx': 0, 'bboxes': []}, {'frame_idx': 1, 'bboxes': [[0.0, 1.0, 2.0, 3.0, 0.9]]}]
    track_rows = rows(stage_path(tmp_path, '060_uncompressed_tracks', 'C_60_4_r050_tr_s100_sortcython') / 'tracking.jsonl')
    assert [r['frame_idx'] for r in track_rows] == [0, 1]


def test_every_saver_skips_the_timing_dummy(tmp_path):
    dummy = '/tmp/execution-x/timing-dummy.mp4'
    save.save_classified(str(tmp_path), PARAMS, dummy, Classified(decoded(dummy), scores()))
    save.save_pruned(str(tmp_path), PARAMS, dummy, Classified(decoded(dummy), scores(), {'status': 2}))
    save.save_packed(str(tmp_path), PARAMS, dummy, Packed(dummy, 1, 1, 1, None, np.zeros((0, 1, 1)), []))
    save.save_detected(str(tmp_path), PARAMS, dummy, Detected(dummy, 1, 1, 1, {}))
    save.save_tracked(str(tmp_path), PARAMS, dummy, Tracked(dummy, {}))
    # Nothing at all is written for the dummy.
    assert list(tmp_path.iterdir()) == []


def test_with_savers_attaches_one_picklable_saver_per_stage_with_results(tmp_path):
    cfg = make_cfg(tmp_path, mistrack_tolerance=0.4)
    stages = [Stage(n, None, None) for n in ('decode', 'classify', 'prune', 'pack', 'detect', 'track')]
    out = save.with_savers(stages, cfg, str(tmp_path / 'root'), skip='dummy.mp4')
    by_name = {s.name: s for s in out}
    # Decode has only pixels, so it gets no saver; every other stage gets exactly one saver process.
    assert by_name['decode'].save is None
    for name in ('classify', 'prune', 'pack', 'detect', 'track'):
        assert by_name[name].save.func is save.SAVERS[name] and by_name[name].savers == 1
        assert by_name[name].save.args[2] == 'dummy.mp4'
        # The saver crosses into a spawned process, so it must pickle.
        pickle.dumps(by_name[name].save)
    # The parameters come from the config.
    assert by_name['track'].save.args[1] == save.config_params(cfg)
    assert save.config_params(cfg)['classifier'] == 'Fake'
