import altair as alt
import pandas as pd
import pytest

from evaluation import p038_packing_efficacy as stats
from evaluation import p206_classifier_and_packing as figure


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(stats, 'DATASETS', ['d'])
    monkeypatch.setattr(stats, 'list_split_videos', lambda dataset, split: ['te00.mp4', 'te01.mp4'])
    return [dict(dataset='d', video=video, classifier='ShuffleNet05', tilesize=60,
                 _poly_n_canvases=2, occupied_tiles=100, empty_tiles=20)
            for video in ('te00.mp4', 'te01.mp4')]


def test_control_records_every_expected_clip(control):
    rows = stats._test_control_coverage(pd.DataFrame(control))
    assert [r['video'] for r in rows] == ['te00.mp4', 'te01.mp4']
    assert all(r['canvases'] == 2 and r['occupied_tiles'] == 100 for r in rows)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'extra', 'classifier', 'tilesize', 'empty', 'histogram'])
def test_control_rejects_incomplete_or_mixed_inputs(control, problem):
    if problem == 'missing':
        control.pop()
    elif problem == 'duplicate':
        control.append(dict(control[0]))
    elif problem == 'extra':
        control.append(dict(control[0], video='te02.mp4'))
    elif problem == 'classifier':
        control[0]['classifier'] = 'Perfect'
    elif problem == 'tilesize':
        control[0]['tilesize'] = 120
    elif problem == 'empty':
        control[0]['occupied_tiles'] = 0
    else:
        control[0]['_poly_n_canvases'] = None
    with pytest.raises(AssertionError):
        stats._test_control_coverage(pd.DataFrame(control))


def test_efficacy_bars_use_only_unpruned_unpadded_results():
    frame = pd.DataFrame([
        dict(dataset='caldot1-y05', tilepadding='none', tracking_accuracy_threshold=None, packing_efficacy_pct=80.),
        dict(dataset='caldot2-y05', tilepadding='none', tracking_accuracy_threshold=None, packing_efficacy_pct=90.),
        dict(dataset='caldot1-y05', tilepadding='none', tracking_accuracy_threshold=.8, packing_efficacy_pct=50.),
        dict(dataset='caldot2-y05', tilepadding='square', tracking_accuracy_threshold=None, packing_efficacy_pct=60.),
    ])
    domain = figure._shared_dataset_domain([frame])
    scale = alt.Scale(domain=domain, range=['blue', 'orange'])
    chart = figure._build_efficacy_chart(frame, scale, domain).to_dict()
    rows = [row for data in chart['datasets'].values() for row in data]
    assert sorted(row['packing_efficacy_pct'] for row in rows) == [80., 90.]
    x = chart['layer'][0]['encoding']['x']
    assert x['field'] == 'packing_efficacy_pct' and 'aggregate' not in x
    with pytest.raises(ValueError):
        figure._build_efficacy_chart(frame[frame['tracking_accuracy_threshold'].notna()], scale, domain)
    with pytest.raises(AssertionError):
        figure._build_efficacy_chart(pd.concat([frame, frame.iloc[:1]]), scale, domain)
