"""Validate and summarize the saved, detection-only oracle-packing experiment."""
import hashlib
import json
import shutil
from pathlib import Path

import duckdb

from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements.common import DATASET_LABELS
from polyis.io import cache


root = Path(__file__).resolve().parent
out = root / 'sr32'
inputs = json.loads((root / 'inputs.json').read_text())
files = []
for dataset in DATASET_LABELS:
    path = out / f'{dataset}.json'
    result = json.loads(path.read_text())
    assert result['dataset'] == dataset and result['method']['split'] == 'test'
    assert result['method']['stride'] == 32 and result['method']['tile_size'] == 60
    assert result['method']['tracking'] is False and result['method']['track_interpolation'] is False
    assert sorted(result['videos']) == inputs[dataset]['videos']
    assert result['all_sampled_frames']['reference']['n_gt'] == result['all_sampled_frames']['packed']['n_gt']
    gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
    # Verify input files individually; all numerical aggregation below uses DuckDB.
    for video, meta in result['videos'].items():
        assert meta['sampled_frames'] == (meta['frames'] + 31) // 32
        with cache.exec(gt_dataset, 'groundtruth', video, 'tracking.jsonl').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['inputs']['ground_truth_sha256']
    files.append(str(path))

con = duckdb.connect(config={'threads': 4})
# Read clip counts from the manifest; unioned nested video structs also contain absent keys.
con.execute("""CREATE TABLE clip_counts AS
    SELECT key AS dataset, json_array_length(value, '$.videos') AS clips
    FROM json_each(?)""", [json.dumps(inputs)])
con.execute("""CREATE TABLE summary AS SELECT dataset,label,provisional,sampled_frames,clips,
 all_sampled_frames.reference.n_gt AS gt_boxes,
 retained_content.center.reference.n_gt AS retained_gt_boxes,
 round(100*all_sampled_frames.reference.precision,2) AS original_precision_pct,
 round(100*all_sampled_frames.packed.precision,2) AS packed_precision_pct,
 round(100*all_sampled_frames.reference.recall,2) AS original_recall_pct,
 round(100*all_sampled_frames.packed.recall,2) AS packed_recall_pct,
 round(100*all_sampled_frames.reference.ap50,2) AS original_map50_pct,
 round(100*all_sampled_frames.packed.ap50,2) AS packed_map50_pct,
 round(100*(all_sampled_frames.packed.ap50-all_sampled_frames.reference.ap50),2) AS map50_change_pp,
 round(100*retained_content.center.reference.precision,2) AS retained_original_precision_pct,
 round(100*retained_content.center.packed.precision,2) AS retained_packed_precision_pct,
 round(100*retained_content.center.reference.recall,2) AS retained_original_recall_pct,
 round(100*retained_content.center.packed.recall,2) AS retained_packed_recall_pct,
 round(100*retained_content.center.reference.ap50,2) AS retained_original_map50_pct,
 round(100*retained_content.center.packed.ap50,2) AS retained_packed_map50_pct
 FROM read_json(?,union_by_name=true) JOIN clip_counts USING(dataset) ORDER BY label""", [files])
cursor = con.execute('SELECT * FROM summary')
columns = [item[0] for item in cursor.description]
rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
assert con.execute('SELECT count(*),sum(clips),sum(sampled_frames) FROM summary').fetchone() == (7,200,5870)
report = dict(
    method='Test-annotation tiles; frames 0,32,64,...; 60px tiles; no extra tile padding; no ILP; detection only.',
    scope='all_sampled_frames counts all annotations and outputs on selected frames; retained fields use Table 3 center filtering.',
    rows=rows, source_files=files)
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

# Freeze the relevant sources alongside the input and checkpoint hashes in each result.
provenance = {}
for relative in ['measurements-gpu/r1o4_oracle_packing.py', 'measurements/r1o4_packing_detection_accuracy.py',
                 'tests/test_measurements_oracle_packing.py', 'scripts/p030_exec_compress.py',
                 'scripts/p040_exec_detect.py', 'scripts/p050_exec_uncompress.py',
                 'measurements-results/r1o4_oracle_packing/summarize.py']:
    source = Path('/polyis') / relative
    target = root / 'source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    provenance[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
(root / 'source_sha256.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(rows, indent=2))
print('Verified 7 datasets, 200 clips, 5870 frames; copied annotation hashes match canonical inputs.')
