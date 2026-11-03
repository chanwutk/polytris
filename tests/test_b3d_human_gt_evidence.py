import hashlib
import json

import pytest

from measurements import b3d_human_gt_evidence as ev


def leap_entry(label, linear=0.2, pattern=0.05, raw=0.01, clips=5):
    hota = {v: {'HOTA': h} for v, h in (('raw', raw), ('linear', linear), ('pattern', pattern))}
    return {'agreement': 'a', 'classes': 'c', 'clips': clips, 'dataset': label, 'frame_base': {'lag': 0},
            'hota': hota, 'timing': 't', 'video_identity': 'v'}


def leap_report(**overrides):
    return {'datasets': {label: leap_entry(label, **overrides) for label in ev.B3D.values()} | {'CalDoT1': {'x': 1}},
            'environment': 'e'}


def configs(leap_hota=0.04):
    return {d: [{'system': 'reference', 'config': 'naive', 'fps': 15.0, 'hota': 0.8},
                {'system': 'tetris', 'config': 'k', 'fps': 60.0, 'hota': 0.78},
                {'system': 'leap', 'config': 'param 0', 'fps': 200.0, 'hota': leap_hota}] for d in ev.B3D}


def test_leap_rerun_takes_new_ground_truth_sections_only():
    new = ev.splice_leap_rerun(leap_report(), leap_report(linear=0.3))
    assert new['datasets']['B3D2']['hota']['linear']['HOTA'] == 0.3
    # Non-B3D entries and top-level fields are untouched.
    assert new['datasets']['CalDoT1'] == {'x': 1} and new['environment'] == 'e'


def test_leap_rerun_refuses_changes_outside_ground_truth_sections():
    with pytest.raises(ValueError, match='ground-truth-independent'):
        ev.splice_leap_rerun(leap_report(), leap_report(clips=6))


def test_leap_rerun_refuses_different_sections():
    rerun = leap_report()
    del rerun['datasets']['B3D1']['frame_base']
    with pytest.raises(ValueError, match='sections'):
        ev.splice_leap_rerun(leap_report(), rerun)


@pytest.mark.parametrize('linear, pattern, raw, best', [(0.2, 0.05, 0.01, 0.2), (0.1, 0.3, 0.01, 0.3), (0.0, 0.0, 0.02, 0.02)])
def test_leap_row_uses_the_best_reconstruction(linear, pattern, raw, best):
    leap = leap_report(linear=linear, pattern=pattern, raw=raw)
    old = configs()
    out = ev.rebuild_extraction_configs(old, configs(leap_hota=0.01), leap)
    assert [r['hota'] for r in out['jnc6'] if r['system'] == 'leap'] == [best]
    # The cached system_configs rows are not mutated.
    assert old['jnc6'][2]['hota'] == 0.04


def test_extraction_configs_refuse_changed_configurations():
    rebuilt = configs()
    rebuilt['jnc0'] = rebuilt['jnc0'][:2] + [{'system': 'tetris', 'config': 'new', 'fps': 1.0, 'hota': 0.5},
                                            rebuilt['jnc0'][2]]
    with pytest.raises(ValueError, match='configurations changed'):
        ev.rebuild_extraction_configs(configs(), rebuilt, leap_report())


def test_sampling_replaces_b3d_only():
    old = {'datasets': {'CalDoT1': 1, **{l: 0 for l in ev.B3D.values()}}, 'method': 'm'}
    new = ev.splice_sampling(old, {'datasets': {l: 9 for l in ev.B3D.values()}})
    assert new['datasets'] == {'CalDoT1': 1, **{l: 9 for l in ev.B3D.values()}} and old['datasets']['B3D1'] == 0


def test_refresh_sources_recopies_outputs_and_records_hashes(tmp_path):
    evidence, results = tmp_path / 'evidence', tmp_path / 'results'
    (results / 'r7o1').mkdir(parents=True)
    evidence.mkdir()
    (results / 'r7o1' / 'm.json').write_text('new')
    (evidence / 'matched.json').write_text('old')
    (evidence / 'leap_rerun.json').write_text('spliced')
    (evidence / 'other.json').write_text('same')
    entries = [{'file': 'matched.json', 'source': '../tetris/measurements-results/r7o1/m.json', 'sha256': hashlib.sha256(b'old').hexdigest()},
               {'file': 'leap_rerun.json', 'source': 'scripts/refresh_leap_matched.py', 'sha256': 'stale'},
               {'file': 'other.json', 'source': 'scripts/x.py', 'sha256': hashlib.sha256(b'same').hexdigest()}]
    (evidence / 'sources.json').write_text(json.dumps(entries))
    assert ev.refresh_sources(evidence, results) == ['matched.json', 'leap_rerun.json']
    assert (evidence / 'matched.json').read_text() == 'new'
    recorded = {e['file']: e['sha256'] for e in json.loads((evidence / 'sources.json').read_text())}
    assert recorded['matched.json'] == hashlib.sha256(b'new').hexdigest()
    assert recorded['leap_rerun.json'] == hashlib.sha256(b'spliced').hexdigest()


def test_refresh_sources_fails_on_a_missing_output(tmp_path):
    (tmp_path / 'evidence').mkdir()
    (tmp_path / 'evidence' / 'sources.json').write_text(json.dumps(
        [{'file': 'a.json', 'source': '../tetris/measurements-results/none/a.json', 'sha256': 'x'}]))
    with pytest.raises(FileNotFoundError):
        ev.refresh_sources(tmp_path / 'evidence', tmp_path / 'results')
