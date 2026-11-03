import csv
import hashlib
import json

import duckdb
import pytest

from measurements import b3d_human_gt_report as report

HEADER = ['dataset', 'variant', 'variant_id', 'throughput_fps', 'time', 'HOTA_HOTA', 'HOTA_AssA', 'HOTA_DetA',
          'Count_DetsMAPE', 'Count_TracksMAPE']
PRIOR_HEADER = ['param_id', 'runtime', 'dataset', 'HOTA_HOTA', 'HOTA_AssA', 'HOTA_DetA', 'Count_DetsMAPE', 'Count_TracksMAPE']


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def tetris_rows(hotas):
    # (variant_id, fps, hota) per row; the reference is the 'naive' row.
    return [['jnc0', 'naive' if v == 'ref' else 'polytris', v, fps, 1.0, h, h, h, 1, 1] for v, fps, h in hotas]


OLD = [('ref', 15, 0.80), ('a', 100, 0.79), ('b', 200, 0.70), ('c', 150, 0.77)]


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / 'cache'
    write(root / 'jnc0/evaluation/090_tradeoff.auto-gt/tradeoff.csv', HEADER, tetris_rows(OLD))
    for system in ('otif', 'leap'):
        write(root / f'SOTA/{system}/jnc0/tradeoff.csv.auto-gt', PRIOR_HEADER, [[0, 5.0, 'jnc0', 0.5, 0.5, 0.5, 1, 1]])
        write(root / f'SOTA/{system}/jnc0/tradeoff.csv', PRIOR_HEADER, [[0, 5.0, 'jnc0', 0.4, 0.4, 0.4, 1, 1]])
    untouched = tmp_path / 'caldot.csv'
    untouched.write_text('x')
    (tmp_path / 'sums.json').write_text(json.dumps({str(untouched): hashlib.sha256(b'x').hexdigest()}))
    return tmp_path


def run(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['r', '--cache', str(tmp_path / 'cache'), '--datasets', 'jnc0',
                                     '--untouched-checksums', str(tmp_path / 'sums.json'), '--output', str(tmp_path / 'out.json')])
    report.main()
    return json.loads((tmp_path / 'out.json').read_text())


def test_report_moves_operating_points_with_the_new_reference(cache, monkeypatch):
    # New labels raise the reference and lower 'c', so the 5% point moves from c to a.
    write(cache / 'cache/jnc0/evaluation/090_tradeoff/tradeoff.csv', HEADER,
          tetris_rows([('ref', 15, 0.90), ('a', 100, 0.88), ('b', 200, 0.80), ('c', 150, 0.84)]))
    out = run(cache, monkeypatch)
    entry = out['datasets']['jnc0']
    assert out['gate_failures'] == [] and out['untouched_checked'] == 1
    assert entry['tetris']['reference_hota'] == {'old': 0.80, 'new': 0.90}
    five = entry['tetris']['operating_points']['0.05']
    assert five['old']['variant_id'] == 'c' and five['new']['variant_id'] == 'a' and five['moved']
    ten = entry['tetris']['operating_points']['0.1']
    # At 10% the bound is 0.81 on new labels: 'b' (0.80) still misses, so 'c' stays the point.
    assert ten['old']['variant_id'] == 'c' and ten['new']['variant_id'] == 'c' and not ten['moved']
    assert entry['otif'] == [{'param_id': 0, 'old_hota': 0.5, 'new_hota': 0.4}]


@pytest.mark.parametrize('rows, message', [
    ([('ref', 15, 0.9), ('a', 100, 0.8), ('b', 200, 0.8)], 'dropped'),                       # config lost
    ([('ref', 15, 0.9), ('a', 101, 0.8), ('b', 200, 0.8), ('c', 150, 0.8)], 'outside the accuracy'),  # fps moved
    ([('ref', 15, 0.9), ('a', 100, 0.8), ('a', 100, 0.8), ('b', 200, 0.8), ('c', 150, 0.8)], 'duplicate'),
])
def test_gates_fail_on_anything_but_accuracy_changes(cache, monkeypatch, rows, message):
    write(cache / 'cache/jnc0/evaluation/090_tradeoff/tradeoff.csv', HEADER, tetris_rows(rows))
    with pytest.raises(SystemExit, match='gate failure'):
        run(cache, monkeypatch)
    assert any(message in f for f in json.loads((cache / 'out.json').read_text())['gate_failures'])


def test_gate_fails_when_a_non_b3d_table_changed(cache, monkeypatch):
    write(cache / 'cache/jnc0/evaluation/090_tradeoff/tradeoff.csv', HEADER, tetris_rows(OLD))
    (cache / 'caldot.csv').write_text('y')
    with pytest.raises(SystemExit):
        run(cache, monkeypatch)
    assert any('non-B3D' in f for f in json.loads((cache / 'out.json').read_text())['gate_failures'])


def test_operating_point_ties_break_by_name(tmp_path):
    old = tmp_path / 'o.csv'
    write(old, HEADER, tetris_rows([('ref', 15, 0.8), ('z', 100, 0.79), ('y', 100, 0.79)]))
    with duckdb.connect() as con:
        out = report.tetris_report(con, old, old, [0.05])
    assert out['operating_points']['0.05']['new']['variant_id'] == 'y'
    assert not out['operating_points']['0.05']['moved']
