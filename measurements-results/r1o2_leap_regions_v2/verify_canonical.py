import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, '/polyis')
import duckdb
from measurements import common
from polyis.leap_results import canonical_leap_result
from evaluation.p200_compare_compute import load_sota_tradeoff_data

root = Path('/polyis')
version = root/'measurements-results/r1o2_leap_regions_v2'
before = version/'canonical-archive/paper-before-promotion/revision_evidence'
after = version/'paper-workspace/scripts/data/revision_evidence'
registry = json.loads((root/'measurements-results/leap_canonical.json').read_text())
con = duckdb.connect()
con.execute('CREATE TABLE old AS SELECT unnest(records) r FROM read_json(?)', [str(before/'native_query_configs.json')])
con.execute('CREATE TABLE new AS SELECT unnest(records) r FROM read_json(?)', [str(after/'native_query_configs.json')])
changed = con.execute('''SELECT o.r.dataset, o.r.system FROM old o JOIN new n
 ON (o.r.dataset,o.r.system,o.r.config)=(n.r.dataset,n.r.system,n.r.config)
 WHERE o.r<>n.r ORDER BY 1''').fetchall()
assert changed == [('ams-y05','leap'),('caldot1-y05','leap'),('caldot2-y05','leap')], changed
assert con.execute('SELECT count(*) FROM new').fetchone()[0] == 536
assert json.loads((before/'baseline_preparation.json').read_text()) == json.loads((after/'baseline_preparation.json').read_text())
for dataset, entry in registry['datasets'].items():
    selected = canonical_leap_result(dataset)
    assert hashlib.sha256(selected['source_comparison'].read_bytes()).hexdigest() == entry['source_sha256']
    csv = Path(common.cache.sota('leap', dataset, 'evaluation', 'tradeoff.csv'))
    # The loader's actual cache location is captured rather than assumed.
    old_cache = common.cache.sota
    seen = []
    def capture(*args, **kwargs):
        path = old_cache(*args, **kwargs)
        if str(path).endswith('.csv'):
            seen.append((Path(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()))
        return path
    common.cache.sota = capture
    try:
        rows = load_sota_tradeoff_data([dataset], 'leap')
    finally:
        common.cache.sota = old_cache
    assert seen and all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in seen)
    point = next(r for r in common.system_configs(dataset) if r['system']=='leap')
    assert abs(point['hota']-entry['hota']['HOTA']) < 1e-12
    assert abs(point['fps']-entry['fps']) < 1e-8, (dataset,point['fps'],entry['fps'])
    tracks = common.output_path(dataset, 'leap', 'param 0', 'te00.mp4')
    assert tracks == selected['tracking_root']/'te00.mp4/tracking.jsonl' and tracks.is_file()
    print(dataset, 'canonical:', point, tracks)
for dataset in ('jnc0','jnc2','jnc6','jnc7'):
    assert canonical_leap_result(dataset) is None
    assert '/r1o2_leap_regions_v2/' not in str(common.output_path(dataset,'leap','param 0','te00.mp4'))
print('PASS: exactly three LEAP records changed; 533 query records, all preparation, original caches and B3D selection preserved.')
