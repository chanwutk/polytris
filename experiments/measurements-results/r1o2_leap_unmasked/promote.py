"""Select validated unmasked CalDoT results while preserving every previous run."""
import gzip
import hashlib
import json
from pathlib import Path

VERSION=Path(__file__).resolve().parent
ROOT=VERSION.parents[1]
registry_path=VERSION.parent/'leap_canonical.json'
registry=json.loads(registry_path.read_text())
assert registry==json.loads((VERSION/'canonical-before.json').read_text()),'Canonical selection changed during the rerun.'
comparison_path=VERSION/'comparison/results.json'
comparison=json.loads(comparison_path.read_text())
inputs=json.loads((VERSION/'inputs.json').read_text())
sha=hashlib.sha256(comparison_path.read_bytes()).hexdigest()
for dataset,label in [('caldot1-y05','CalDoT1'),('caldot2-y05','CalDoT2')]:
    section=comparison['datasets'][label]
    point=section['versions']['unmasked']
    assert section['clips']==60 and section['frames']==54000
    assert point['hota']['raw']['HOTA']<=point['best_hota']
    assert not point['timing']['contaminated'] and point['timing']['compute_s']>0
    with gzip.open(VERSION/f'raw/{dataset}/leap_rerun_run.jsonl.gz','rt') as stream:
        metadata=json.loads(next(stream))
    entry=inputs['datasets'][dataset]
    assert metadata['mask_enabled'] is False and metadata['config']==entry['new_config']
    assert metadata['patterns_sha256']==entry['patterns_sha256'] and metadata['video_files']==entry['video_files']
    old=registry['datasets'][dataset]
    old_path=ROOT/old['source_comparison']
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==old['source_sha256']
    variant=point['best_reconstruction']
    registry['datasets'][dataset]=dict(param_id=0,version='reviewed-regions-unmasked',version_id=VERSION.name,
        result_key='unmasked',mask_enabled=False,frames=section['frames'],compute_s=point['timing']['compute_s'],
        fps=point['timing']['fps'],best_reconstruction=variant,hota=point['hota'][variant],
        tracking_root=f'measurements-results/{VERSION.name}/tracks/{dataset}/{variant}',
        native_intervals=f'measurements-results/{VERSION.name}/raw/{dataset}/leap_rerun.jsonl.gz',
        source_comparison=str(comparison_path.relative_to(ROOT)),source_sha256=sha)
    assert (ROOT/registry['datasets'][dataset]['tracking_root']/'te00.mp4/tracking.jsonl').exists()
    marker=VERSION/f'raw/{dataset}/complete.json'
    complete=json.loads(marker.read_text())
    complete['scored']=True
    marker.write_text(json.dumps(complete,indent=2)+'\n')
registry.pop('version_id',None)
registry['selected_version']='reviewed regions; unmasked CalDoT'
temporary=registry_path.with_suffix('.tmp')
temporary.write_text(json.dumps(registry,indent=2)+'\n')
temporary.replace(registry_path)
print('Canonical LEAP updated for CalDoT1/2 only; Amsterdam, B3D and archived results preserved.')
