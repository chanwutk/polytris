import json

import pytest

from measurements import refresh_fixed_reference_timing as timing


def test_refresh_uses_counted_reference_ops_and_keeps_measurements(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    inputs = {'d': {'label': 'Dataset', 'videos': ['te00.mp4']}}
    result = dict(dataset='d', clips=1, frames=100, seconds=2.0, fps=50.0, hota=.8,
                  speedup=5.0, reference=dict(reference_fps=10.0))
    for name, value in [('inputs.json', inputs), ('results/d.json', result),
                        ('summary.json', [dict(dataset='Dataset', frames=100, seconds=4.0, speedup=2.5)])]:
        (tmp_path / name).write_text(json.dumps(value))
    csv = tmp_path / 'ops.csv'
    csv.write_text('dataset,video,videoset,variant,stage,op,time\n'
                   'd,te00.mp4,test,naive,001_preprocess_groundtruth_detection,detect,6\n'
                   'd,te00.mp4,test,naive,002_preprocess_groundtruth_tracking,track,2\n'
                   'd,te00.mp4,test,tetris,040_exec_detect,read,99\n'
                   'd,te01.mp4,test,naive,001_preprocess_groundtruth_detection,detect,99\n')
    monkeypatch.setattr(timing.cache, 'eval', lambda *args: csv)
    timing.refresh(tmp_path)
    fresh = json.loads((tmp_path / 'results/d.json').read_text())
    assert fresh == dict(result, speedup=4.0, reference=dict(reference_fps=12.5))
    assert json.loads((tmp_path / 'inputs.json').read_text()) == inputs
    assert json.loads((tmp_path / 'naive_operator_times.json').read_text()) == [
        dict(dataset='d', naive_seconds=8.0, detector_seconds=6.0, frames=100)]
    assert json.loads((tmp_path / 'summary.json').read_text())[0] == dict(
        dataset='Dataset', frames=100, seconds=4.0, speedup=2.0)
    # Refuse an old accounting table before touching derived outputs.
    csv.write_text(csv.read_text() + 'd,te00.mp4,test,naive,001_preprocess_groundtruth_detection,read,2\n')
    with pytest.raises(AssertionError):
        timing.refresh(tmp_path)
    assert json.loads((tmp_path / 'results/d.json').read_text()) == fresh
