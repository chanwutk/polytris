"""Freeze the reviewed regions and reused preparation before the unmasked extraction."""
import hashlib
import json
from pathlib import Path
import shutil

VERSION = Path(__file__).resolve().parent
PREVIOUS = VERSION.parent/'r1o2_leap_regions_v2'
DATASETS = ('caldot1-y05','caldot2-y05')
assert not (VERSION/'inputs.json').exists(), 'Inputs already frozen; reuse them instead of overwriting.'
old = json.loads((PREVIOUS/'inputs.json').read_text())
manifest = dict(display_name='LEAP',version=VERSION.name,changed='image masking disabled; reviewed stop_area unchanged',
                preparation='reuse the unmasked detector-matched preparation read-only',datasets={})
(VERSION/'configs').mkdir(exist_ok=True)
(VERSION/'sources').mkdir(exist_ok=True)
for dataset in DATASETS:
    entry = old['datasets'][dataset]
    config = 'caldot1.yaml' if dataset.startswith('caldot1') else 'caldot2.yaml'
    shutil.copyfile(PREVIOUS/'configs'/config,VERSION/'configs'/config)
    assert hashlib.sha256((VERSION/'configs'/config).read_bytes()).hexdigest() == entry['config_sha256']
    # Current launcher hashes change; archived data and learned artifacts must retain their original hashes.
    for path in entry['input_sha256']:
        if '/measurements-gpu/' in path:
            entry['input_sha256'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for filename in ('leap_rerun.jsonl.gz','leap_rerun_export.jsonl.gz','leap_rerun_run.jsonl.gz','leap_rerun_launch.json'):
        path=PREVIOUS/'raw'/dataset/filename
        entry['input_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry['mask_enabled'] = False
    entry['comparison_base'] = str(PREVIOUS)
    manifest['datasets'][dataset] = entry
    for path,expected in entry['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
for filename in ('r1o2_leap_otif_rerun_leap.py','r1o2_leap_otif_rerun.py','r1o2_leap_regions_v2.py','r1o2_leap_matched_detector.py'):
    shutil.copyfile(VERSION.parents[1]/'measurements-gpu'/filename,VERSION/'sources'/filename)
shutil.copyfile(PREVIOUS/'sources/evaluate.py',VERSION/'sources/evaluate.py')
shutil.copyfile(VERSION.parent/'leap_canonical.json',VERSION/'canonical-before.json')
(VERSION/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Inputs frozen: CalDoT1/2, reviewed regions unchanged, masks disabled, original data preserved.')
