import duckdb
import gzip
import json
import numpy as np
import pytest

from measurements.r1o2_leap_sampling_diagnostics import (
    assign_boxes, box_iou, continuity_by_stationarity, derive_tracks, oracle_interpolation,
)


def test_matching_is_one_to_one_and_excludes_invalid_boxes():
    # Duplicate predictions cannot both claim one annotated object.
    predictions = [[0, 0, 2, 2], [0, 0, 2, 2], [9, 9, 9, 9]]
    truth = [[0, 0, 2, 2]]
    rows, columns, iou = assign_boxes(predictions, truth)
    assert len(rows) == len(columns) == len(iou) == 1
    assert rows[0] in (0, 1)
    assert columns.tolist() == [0]
    assert iou.tolist() == [1]
    assert np.isfinite(box_iou(predictions, [[9, 9, 9, 9]])).all()
    assert len(assign_boxes([], truth)[0]) == 0


def test_matching_keeps_exact_iou_boundary_and_maximizes_cardinality():
    # At IoU 0.5, either object still has a valid match; rejecting the boundary loses a match.
    rows, columns, iou = assign_boxes([[0, 0, 2, 1], [1, 0, 3, 1]],
                                      [[0, 0, 1, 1], [1, 0, 3, 1]])
    assert set(zip(rows.tolist(), columns.tolist())) == {(0, 0), (1, 1)}
    assert iou.tolist() == [0.5, 1.0]


def test_sampling_exposure_is_distinct_from_identity_fragmentation():
    # One long, fragmented trajectory coexists with tracks missed entirely between samples.
    con = duckdb.connect()
    con.execute("CREATE TABLE clips(video VARCHAR,frames BIGINT)")
    con.execute("INSERT INTO clips VALUES ('v',11)")
    con.execute("""CREATE TABLE gt AS
      SELECT 'v' video,range AS frame,1::BIGINT id,range*3.0 x1,0.0 y1,range*3+2.0 x2,2.0 y2 FROM range(11)
      UNION ALL SELECT 'v',range,2,20+range,0,22+range,2 FROM range(2,5)
      UNION ALL SELECT 'v',range,3,30,0,32,2 FROM range(6,9)
      UNION ALL SELECT 'v',5,4,40,0,42,2""")
    con.execute("CREATE TABLE samples AS SELECT 'v' video,unnest([0,5,10])::BIGINT frame")
    con.execute("""CREATE TABLE matches AS SELECT * FROM (VALUES
      ('v',0,101,1,1.0),('v',5,102,1,1.0),('v',10,102,1,1.0))
      t(video,frame,id,gt_id,iou)""")
    derive_tracks(con)
    assert con.execute('SELECT id,sampled,observations,predicted_ids,censored FROM track_diagnostics ORDER BY id').fetchall() == [
        (1, 3, 3, 2, True), (2, 0, 0, 0, False), (3, 0, 0, 0, False), (4, 1, 0, 0, False)]
    oracle = oracle_interpolation(con)
    assert oracle['evaluated_gt_boxes'] == 10
    assert oracle['mean_iou'] == pytest.approx(1)
    assert oracle['iou_ge_half_share'] == pytest.approx(1)
    assert oracle['motion']['mean_perpendicular_box_diagonals'] == pytest.approx(0)
    assert oracle['motion']['mean_along_line_progress_error'] == pytest.approx(0)
    assert all(v['mean_iou'] == pytest.approx(1) for v in oracle['center_and_size'])
    con.execute('''CREATE TABLE observations AS SELECT m.video,m.id,m.frame,g.x1,g.y1,g.x2,g.y2
      FROM matches m JOIN gt g ON m.video=g.video AND m.frame=g.frame AND m.gt_id=g.id''')
    continuity = continuity_by_stationarity(con, {'stop_iou_thresh': 0.7})
    assert len(continuity) == 1
    assert not continuity[0]['passes_stationary_rule']
    assert continuity[0]['pairs'] == 2
    assert continuity[0]['same_id_share'] == pytest.approx(0.5)
    con.close()


def test_pattern_anchors_count_missing_frames_as_failures(tmp_path, monkeypatch):
    import measurements.r1o2_leap_sampling_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, 'ROOT', tmp_path)
    monkeypatch.setattr(diagnostics, 'canonical_leap_result', lambda dataset: None)
    directory = tmp_path / 'measurements-results/r1o2_leap_matched/raw/caldot1-y05'
    directory.mkdir(parents=True)
    box = lambda identity, left: dict(track_id=identity, left=left, top=0, right=left+2, bottom=2)
    # An empty exported frame is a missing reconstruction, not an omitted observation in the denominator.
    with gzip.open(directory / 'leap_rerun_export.jsonl.gz', 'wt') as stream:
        json.dump(dict(video_name='0', frames=[[box(1, 0)], None, [box(1, 10), box(2, 4)]]), stream)
        stream.write('\n')
    con = duckdb.connect()
    con.execute("CREATE TABLE clips AS SELECT 'v' video,'0' leap_name,3 frames")
    con.execute("CREATE TABLE objects AS SELECT * FROM (VALUES ('v',1,0,3),('v',2,2,3)) t(video,id,s,e)")
    con.execute("""CREATE TABLE observations AS SELECT * FROM (VALUES
        ('v',1,0,0,0,2,2),('v',1,1,0,0,2,2),('v',1,2,0,0,2,2),('v',2,2,4,0,6,2))
        t(video,id,frame,x1,y1,x2,y2)""")
    result = diagnostics.pattern_anchors(con, 'caldot1-y05')
    assert result['observations'] == 4
    assert result['reconstructed_anchor_boxes'] == 3
    assert result['mean_iou_when_present'] == pytest.approx(2/3)
    assert result['iou_ge_half_share_all_observations'] == pytest.approx(0.5)
    con.close()


def test_diagnostics_use_selected_raw_run_for_metadata_and_pattern_export(tmp_path, monkeypatch):
    import measurements.r1o2_leap_sampling_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, 'ROOT', tmp_path)
    chosen = tmp_path/'reviewed/raw/caldot1-y05'
    monkeypatch.setattr(diagnostics, 'canonical_leap_result', lambda dataset:
                        {'native_intervals': chosen/'leap_rerun.jsonl.gz'} if dataset=='caldot1-y05' else None)
    assert diagnostics.raw_directory('caldot1-y05') == chosen
    assert diagnostics.raw_directory('jnc6') == tmp_path/'measurements-results/r1o2_leap_otif_rerun/raw/jnc6'
