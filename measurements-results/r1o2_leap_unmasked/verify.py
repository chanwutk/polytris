"""Verify canonical routing and that only the two requested benchmark records changed."""
import hashlib
import json
from pathlib import Path
import sys

import duckdb

VERSION=Path(__file__).resolve().parent
ROOT=VERSION.parents[1]
sys.path.insert(0,str(ROOT))
from measurements import common
from polyis.leap_results import canonical_leap_result

before=VERSION/'paper-before-unmasking/revision_evidence'
after=VERSION/'paper-workspace/scripts/data/revision_evidence'
con=duckdb.connect()
con.execute('CREATE TABLE old AS SELECT unnest(records) r FROM read_json(?)',[str(before/'native_query_configs.json')])
con.execute('CREATE TABLE new AS SELECT unnest(records) r FROM read_json(?)',[str(after/'native_query_configs.json')])
changed=con.execute('''SELECT o.r.dataset,o.r.system FROM old o JOIN new n
    ON (o.r.dataset,o.r.system,o.r.config)=(n.r.dataset,n.r.system,n.r.config)
    WHERE o.r<>n.r ORDER BY 1''').fetchall()
assert changed==[('caldot1-y05','leap'),('caldot2-y05','leap')],changed
assert con.execute('SELECT count(*) FROM new').fetchone()[0]==536
assert json.loads((before/'baseline_preparation.json').read_text())==json.loads((after/'baseline_preparation.json').read_text())
old_report=json.loads((before/'leap_rerun.json').read_text())['datasets']
new_report=json.loads((after/'leap_rerun.json').read_text())['datasets']
for label in ('Amsterdam','B3D1','B3D2','B3D3','B3D4'):
    assert old_report[label]==new_report[label],label
registry=json.loads((VERSION.parent/'leap_canonical.json').read_text())
previous=json.loads((VERSION/'canonical-before.json').read_text())
assert registry['datasets']['ams-y05']==previous['datasets']['ams-y05']
for dataset,entry in registry['datasets'].items():
    csv=Path(common.cache.sota('leap',dataset,'tradeoff.csv'))
    digest=hashlib.sha256(csv.read_bytes()).hexdigest()
    point=next(row for row in common.system_configs(dataset) if row['system']=='leap')
    assert abs(point['fps']-entry['fps'])<1e-8 and abs(point['hota']-entry['hota']['HOTA'])<1e-12
    assert hashlib.sha256(csv.read_bytes()).hexdigest()==digest
    selected=canonical_leap_result(dataset)
    assert hashlib.sha256(selected['source_comparison'].read_bytes()).hexdigest()==entry['source_sha256']
    assert common.output_path(dataset,'leap','param 0','te00.mp4')==selected['tracking_root']/'te00.mp4/tracking.jsonl'
    if dataset.startswith('caldot'):
        assert entry['mask_enabled'] is False and entry['best_reconstruction']=='linear'
    print(dataset,point,flush=True)
for dataset in ('jnc0','jnc2','jnc6','jnc7'):
    assert canonical_leap_result(dataset) is None
(VERSION/'verification.json').write_text(json.dumps(dict(passed=True,changed=changed,configurations=536,
    unchanged_query_configurations=534,preparation_unchanged=True,other_datasets_unchanged=True),indent=2)+'\n')
print('PASS: only two LEAP records changed; preparation, other datasets and original caches are preserved.')
