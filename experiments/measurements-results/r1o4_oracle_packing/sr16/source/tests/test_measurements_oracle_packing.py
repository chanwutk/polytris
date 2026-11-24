import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pytest

from measurements import r1o4_packing_detection_accuracy as pa


spec = importlib.util.spec_from_file_location('oracle_packing', Path(__file__).parents[1] /
                                             'measurements-gpu/r1o4_oracle_packing.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_oracle_uses_boxes_without_track_id_or_coordinate_scaling():
    bitmap = oracle.oracle_bitmap([[987, 1201, 1, 1259, 59]], 1280, 720)
    assert bitmap.shape == (12, 21)
    assert np.argwhere(bitmap).tolist() == [[0, 20]]
    assert bitmap[0, 20] == 255
    assert not oracle.oracle_bitmap([], 720, 480).any()


def test_all_sampled_frames_counts_background_fp_and_does_not_fill_missing_boxes():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE frames AS SELECT 'v' AS video, unnest([0,32,64,96]) AS frame")
        con.execute("""CREATE TABLE gt AS SELECT 'v' AS video, unnest([0,32,64]) AS frame,
                       1 AS gid, 0.0 AS x1, 0.0 AS y1, 10.0 AS x2, 10.0 AS y2""")
        con.execute('CREATE TABLE dets(source VARCHAR, video VARCHAR, frame BIGINT, did BIGINT, '
                    'x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE, score DOUBLE)')
        con.executemany('INSERT INTO dets VALUES (?,?,?,?,?,?,?,?,?)', [
            ('reference', 'v', 0, 1, 0, 0, 10, 10, 0.9),
            ('reference', 'v', 32, 1, 0, 0, 10, 10, 0.8),
            ('reference', 'v', 64, 1, 0, 0, 10, 10, 0.7),
            ('reference', 'v', 96, 1, 0, 0, 10, 10, 0.1),
            ('oracle', 'v', 0, 1, 0, 0, 10, 10, 0.9),
            ('oracle', 'v', 64, 1, 0, 0, 10, 10, 0.8),
        ])
        pa.greedy_match(con)
        scores = oracle.all_frame_scores(con, 'oracle')
        assert scores['reference']['n_gt'] == scores['packed']['n_gt'] == 3
        assert scores['reference']['precision'] == 0.75
        assert scores['reference']['recall'] == 1
        assert scores['packed']['recall'] == pytest.approx(2 / 3, abs=5e-5)
        assert scores['packed']['ap50'] == pytest.approx(2 / 3, abs=5e-5)


def test_each_padding_has_its_own_configuration_and_result_file():
    # P=0 and P=1 must never share stage outputs or a result file; the stride is part of both.
    none, square = oracle.config_name(16, 'none'), oracle.config_name(16, 'square')
    assert none != square and 'none' in none and 'square' in square
    assert oracle.config_name(32, 'none') != none
    assert oracle.output_path(16, 'square', 'jnc0') == oracle.RESULTS / 'sr16' / 'square' / 'jnc0.json'
    assert oracle.output_path(16, 'none', 'jnc0') != oracle.output_path(16, 'square', 'jnc0')
    assert oracle.PADDINGS == ('none', 'square')
