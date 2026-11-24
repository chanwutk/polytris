"""execution2.tools.run_variants: configs from selection rows, and runtimes recorded where p130 reads them."""

import csv
import json
from pathlib import Path

import pytest

from execution2.tools import run_variants as rv

ROW = {'variant_id': 'ShuffleNet05_60_4_040_r025_tr_s100_sortcython', 'classifier': 'ShuffleNet05', 'tilesize': '60',
       'sample_rate': '4', 'tracking_accuracy_threshold': '0.4', 'relevance_threshold': '0.25', 'tilepadding': 'tr',
       'canvas_scale': '1.0', 'tracker': 'sortcython'}


def test_selection_rows_parse_and_name_the_same_configuration():
    p = rv.parse_row(ROW)
    assert p['tracking_accuracy_threshold'] == 0.4 and p['sample_rate'] == 4 and rv.variant_id(p) == ROW['variant_id']
    # An empty tolerance (pandas writes NaN as '') is the unpruned branch.
    unpruned = rv.parse_row({**ROW, 'tracking_accuracy_threshold': ''})
    assert unpruned['tracking_accuracy_threshold'] is None and '_040_' not in rv.variant_id(unpruned)


def test_config_uses_one_gpu_and_prunes_only_with_a_tolerance():
    cfg = rv.build_config('jnc0', rv.parse_row(ROW))
    stages = cfg['system']['stages']
    assert stages['classify']['gpu'] == stages['detect']['gpu'] == 0 and stages['prune'] == {'procs': 1}
    assert cfg['artifacts']['max_rate_table'].endswith('/jnc0/indexing/track_rate/sortcython_60/max_rate_table.npy')
    assert cfg['artifacts']['detector'] == {'type': 'retina', 'config_path': '/polyis/configs/retinanet.json'}
    assert cfg['params']['mistrack_tolerance'] == 0.4 and cfg['params']['canvas_format'] == 'jpeg'
    unpruned = rv.build_config('jnc0', rv.parse_row({**ROW, 'tracking_accuracy_threshold': ''}))
    assert 'prune' not in unpruned['system']['stages'] and 'max_rate_table' not in unpruned['artifacts']


def test_runtime_is_split_by_frame_share_and_sums_to_the_runtime():
    record = {'dataset': 'jnc0', 'videoset': 'test', 'variant_id': ROW['variant_id'], 'params': rv.parse_row(ROW),
              'runtime_s': 10.0, 'frames': 400, 'video_frames': {'te01.mp4': 100, 'te00.mp4': 300}}
    rows = rv.runtime_rows(record)
    assert [r['video'] for r in rows] == ['te00.mp4', 'te01.mp4']
    assert [r['time'] for r in rows] == pytest.approx([7.5, 2.5]) and sum(r['time'] for r in rows) == pytest.approx(10.0)
    assert all(r['variant'] == 'polytris' and r['stage'] == 'execution2' for r in rows)


@pytest.fixture
def fake_eval(tmp_path, monkeypatch):
    # cache.eval(dataset, stage, *rest) -> tmp/<dataset>/<stage>/<rest...>
    monkeypatch.setattr(rv.cache, 'eval', lambda ds, stage, *rest: str(tmp_path.joinpath(ds, stage, *rest)))
    return tmp_path


def write_table(root: Path, rows: list[dict]) -> Path:
    path = root / 'jnc0' / 'tp' / 'measurements' / 'query_execution_overall.csv'
    path.parent.mkdir(parents=True)
    columns = ['stage', 'dataset', 'videoset', 'video', 'variant', 'variant_id', 'classifier', 'tilesize',
               'tilepadding', 'sample_rate', 'tracking_accuracy_threshold', 'canvas_scale', 'tracker', 'time']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows({c: r.get(c, '') for c in columns} for r in rows)
    return path


def write_record(root: Path, runtime_s: float) -> None:
    path = root / 'jnc0' / 'tp' / 'execution2' / 'test' / f"{ROW['variant_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'dataset': 'jnc0', 'videoset': 'test', 'variant_id': ROW['variant_id'],
                                'params': rv.parse_row(ROW), 'runtime_s': runtime_s, 'frames': 200,
                                'video_frames': {'te00.mp4': 100, 'te01.mp4': 100}}))


def test_record_replaces_stale_rows_of_the_recorded_configurations_only(fake_eval):
    naive = {'stage': '002', 'dataset': 'jnc0', 'videoset': 'test', 'video': 'te00.mp4', 'variant': 'naive',
             'variant_id': 'naive', 'time': '50'}
    stale_script = {'stage': '040', 'dataset': 'jnc0', 'videoset': 'test', 'video': 'te00.mp4', 'variant': 'polytris',
                    'variant_id': ROW['variant_id'], 'time': '99'}
    valid_row = {**stale_script, 'videoset': 'valid'}
    table = write_table(fake_eval, [naive, stale_script, valid_row])
    write_record(fake_eval, 8.0)
    assert rv.record_runtimes('jnc0', 'test') == 2
    # Recording again (a rerun) replaces its own rows instead of adding them twice.
    write_record(fake_eval, 6.0)
    assert rv.record_runtimes('jnc0', 'test') == 2
    rows = list(csv.DictReader(open(table, newline='')))
    test_rows = [r for r in rows if r['videoset'] == 'test' and r['variant_id'] == ROW['variant_id']]
    assert [r['stage'] for r in test_rows] == ['execution2', 'execution2']
    assert sum(float(r['time']) for r in test_rows) == pytest.approx(6.0)
    # Other variants and splits are untouched.
    assert any(r['variant_id'] == 'naive' for r in rows) and any(r['videoset'] == 'valid' for r in rows)


def test_record_needs_runtimes_and_p121s_table(fake_eval):
    with pytest.raises(SystemExit, match='run `run` first'):
        rv.record_runtimes('jnc0', 'test')
    write_record(fake_eval, 8.0)
    with pytest.raises(SystemExit, match='p121_throughput_compute'):
        rv.record_runtimes('jnc0', 'test')
