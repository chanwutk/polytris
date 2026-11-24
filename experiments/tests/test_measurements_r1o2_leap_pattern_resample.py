import json

from experiments.measurements import r1o2_leap_pattern_resample as resample


def test_interval_counts_classify_patterns_against_intervals(tmp_path, monkeypatch):
    # Clip 1: pattern 0 has 3 boxes, pattern 1 has 5; objects cover shorter, equal, longer, empty, inverted, missing.
    parsed = tmp_path / 'inputs' / 'ds' / 'parsed_results'
    parsed.mkdir(parents=True)
    (parsed / '1_results.json').write_text(json.dumps({
        'a': [[0, 10], 0, 2], 'b': [[0, 3], 0, 2], 'c': [[0, 2], 1, 2], 'd': [[4, 4], 1, 2], 'e': [[6, 2], 0, 2],
        'f': [[0, 3], 7, 2]}))
    (parsed / '1_tracks.json').write_text(json.dumps([[[0, 0, 1, 1]] * 3, [[0, 0, 1, 1]] * 5]))
    monkeypatch.setattr(resample, 'RESULTS', tmp_path)
    assert resample.interval_counts('ds') == dict(objects=6, empty_or_inverted=2, shorter=1, equal=1, longer=1,
                                                  missing_pattern=1)


def test_recorded_hota_follows_the_key_path(tmp_path, monkeypatch):
    (tmp_path / 'rec.json').write_text(json.dumps({'datasets': {'X': {'hota': {'pattern': {'HOTA': 0.5}}}}}))
    monkeypatch.setattr(resample, 'MEASURED', tmp_path)
    monkeypatch.setitem(resample.DATASETS, 'x', dict(recorded=('rec.json', ('datasets', 'X', 'hota'))))
    assert resample.recorded_hota('x') == {'pattern': {'HOTA': 0.5}}


def test_every_dataset_has_a_recorded_section_and_saved_export():
    # The configured recording files and exports exist in the repository's results.
    for dataset, config in resample.DATASETS.items():
        assert (resample.MEASURED / config['export']).exists(), dataset
        section = resample.recorded_hota(dataset)
        assert {'raw', 'linear', 'pattern'} <= set(section), dataset
