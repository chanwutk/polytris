"""Validate and summarize the saved, detection-only oracle-packing experiment."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import duckdb

from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements.common import DATASET_LABELS
from polyis.io import cache


# The sampling stride to summarize (sr<stride>/), given as the only argument.
STRIDE = int(sys.argv[1])
PADDINGS = ('none', 'square')
# Corner paddings some deployed operating points use; summarized when a dataset has them.
EXTRA_PADDINGS = ('tr', 'bl')
root = Path(__file__).resolve().parent
out = root / f'sr{STRIDE}'
inputs = json.loads((root / 'inputs.json').read_text())
files = []
for dataset in DATASET_LABELS:
    gt_dataset = resolve_groundtruth_tracking_dataset(dataset)
    paddings = list(PADDINGS) + [p for p in EXTRA_PADDINGS if (out / p / f'{dataset}.json').exists()]
    results = {padding: json.loads((out / padding / f'{dataset}.json').read_text()) for padding in paddings}
    for padding, result in results.items():
        assert result['dataset'] == dataset and result['method']['split'] == 'test'
        assert result['method']['stride'] == STRIDE and result['method']['tile_size'] == 60
        assert result['method']['tile_padding'] == padding and result['method']['ilp_pruning'] is False
        assert result['method']['tracking'] is False and result['method']['track_interpolation'] is False
        assert sorted(result['videos']) == inputs[dataset]['videos']
        # Verify input files individually; all numerical aggregation below uses DuckDB.
        for video, meta in result['videos'].items():
            assert meta['sampled_frames'] == (meta['frames'] + STRIDE - 1) // STRIDE
            with cache.exec(gt_dataset, 'groundtruth', video, 'tracking.jsonl').open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['inputs']['ground_truth_sha256']
        files.append(str(out / padding / f'{dataset}.json'))
    # Every padding scores the same original-frame detections on the same frames.
    first, *rest = results.values()
    for other in rest:
        assert other['sampled_frames'] == first['sampled_frames']
        assert other['all_sampled_frames']['reference'] == first['all_sampled_frames']['reference']
        assert all(other['videos'][v]['inputs']['reference_sha256'] == first['videos'][v]['inputs']['reference_sha256']
                   for v in first['videos'])

con = duckdb.connect(config={'threads': 4})
# Read clip counts from the manifest; unioned nested video structs also contain absent keys.
con.execute("""CREATE TABLE clip_counts AS
    SELECT key AS dataset, json_array_length(value, '$.videos') AS clips
    FROM json_each(?)""", [json.dumps(inputs)])
# Packed canvases per result file, summed over its videos (canvases are frame-sized, canvas_scale=1.0).
con.execute("""CREATE TABLE canvas_counts AS
    SELECT f.json->>'dataset' AS dataset, f.json->'method'->>'tile_padding' AS padding,
           sum(CAST(v.value->>'canvases' AS BIGINT)) AS canvases
    FROM read_json_objects(?) f, json_each(f.json->'videos') v GROUP BY ALL""", [files])
# Object and polyomino statistics (measurements/r1o4_oracle_packing_stats.py), one file per dataset.
stats_files = [str(out / 'stats' / f'{dataset}.json') for dataset in DATASET_LABELS]
con.execute("""CREATE TABLE stats AS
    SELECT f.json->>'dataset' AS dataset, p.key AS padding, f.json->'objects' AS objects, p.value AS packing
    FROM read_json_objects(?) f, json_each(f.json->'padding') p""", [stats_files])
con.execute("""CREATE TABLE summary AS SELECT r.dataset,label,method.tile_padding AS padding,sampled_frames,clips,
 c.canvases, round(c.canvases/sampled_frames,3) AS canvases_per_sampled_frame,
 CAST(s.packing->>'packing_efficacy' AS DOUBLE) AS packing_efficacy,
 CAST(s.packing->>'polyominoes_per_frame' AS DOUBLE) AS polyominoes_per_frame,
 CAST(s.packing->>'tiles_per_polyomino' AS DOUBLE) AS tiles_per_polyomino,
 CAST(s.packing->>'objects_per_polyomino' AS DOUBLE) AS objects_per_polyomino,
 CAST(s.packing->>'occupied_fraction_of_grid' AS DOUBLE) AS occupied_fraction_of_grid,
 CAST(s.objects->>'boxes_per_frame' AS DOUBLE) AS objects_per_frame,
 CAST(s.objects->>'median_longest_side_tiles' AS DOUBLE) AS median_object_longest_side_tiles,
 CAST(s.objects->>'fraction_within_one_tile' AS DOUBLE) AS objects_within_one_tile,
 CAST(s.objects->>'mean_tiles_overlapped' AS DOUBLE) AS mean_tiles_per_object,
 all_sampled_frames.reference.n_gt AS gt_boxes,
 retained_content.center.reference.n_gt AS retained_gt_boxes,
 round(100*all_sampled_frames.reference.precision,2) AS original_precision_pct,
 round(100*all_sampled_frames.packed.precision,2) AS packed_precision_pct,
 round(100*all_sampled_frames.reference.recall,2) AS original_recall_pct,
 round(100*all_sampled_frames.packed.recall,2) AS packed_recall_pct,
 round(100*all_sampled_frames.reference.ap50,2) AS original_map50_pct,
 round(100*all_sampled_frames.packed.ap50,2) AS packed_map50_pct,
 round(100*retained_content.center.reference.precision,2) AS retained_original_precision_pct,
 round(100*retained_content.center.packed.precision,2) AS retained_packed_precision_pct,
 round(100*retained_content.center.reference.recall,2) AS retained_original_recall_pct,
 round(100*retained_content.center.packed.recall,2) AS retained_packed_recall_pct,
 round(100*retained_content.center.reference.ap50,2) AS retained_original_map50_pct,
 round(100*retained_content.center.packed.ap50,2) AS retained_packed_map50_pct
 FROM read_json(?,union_by_name=true) r JOIN clip_counts USING(dataset)
 JOIN canvas_counts c ON c.dataset = r.dataset AND c.padding = r.method.tile_padding
 JOIN stats s ON s.dataset = r.dataset AND s.padding = r.method.tile_padding ORDER BY label, padding""", [files])
cursor = con.execute('SELECT * FROM summary')
columns = [item[0] for item in cursor.description]
rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
# Every dataset appears once for each required padding, over the manifest's 200 test clips.
assert con.execute(f"SELECT count(*), count(DISTINCT dataset), sum(clips) FROM summary WHERE padding IN {PADDINGS}"
                   ).fetchone() == (7 * len(PADDINGS), 7, 200 * len(PADDINGS))
report = dict(
    method=f'Test-annotation tiles; frames 0,{STRIDE},{2 * STRIDE},...; 60px tiles; padding none (P=0) and square (P=1); '
           'no ILP; detection only.',
    scope='all_sampled_frames counts all annotations and outputs on selected frames; retained fields use Table 3 center filtering.',
    rows=rows, source_files=files)
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

# Freeze the relevant sources alongside the input and checkpoint hashes in each result.
provenance = {}
for relative in ['measurements-gpu/r1o4_oracle_packing.py', 'measurements/r1o4_packing_detection_accuracy.py',
                 'measurements/r1o4_oracle_packing_stats.py',
                 'tests/test_measurements_oracle_packing.py', 'scripts/p030_exec_compress.py',
                 'scripts/p040_exec_detect.py', 'scripts/p050_exec_uncompress.py',
                 'measurements-results/r1o4_oracle_packing/summarize.py']:
    source = Path('/polyis') / relative
    target = out / 'source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    provenance[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
(out / 'source_sha256.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(rows, indent=2))
print(f'Verified 7 datasets x {len(PADDINGS)} paddings, 200 clips; annotation and reference hashes match.')
