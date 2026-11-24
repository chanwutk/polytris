#!/usr/local/bin/python
"""R1.O3: classifier and reference-detector time per frame as the frame size grows.

Detector: measurements-gpu/r1o3_resolution_scaling.py's timings, each detector run at the frame's own resolution.
Classifier: not rerun. It classifies each tile independently, so its time per frame is the per-tile time from
measurements-results/r9o1_matched_timing/ (mean ms per frame over the tiles it infers, same GPU model) times the
tiles of the larger frame (p020's grid drops partial tiles). That per-tile time carries the per-frame fixed cost
spread over today's tile count, so the projection overstates the classifier at larger frames.
measured_max_batch: the classifier measured by the same GPU job (--classifier --max-batch) at 1x, 2x, and 4x the
pixels at its largest batch, paired with the detector at its largest batch at the same size, with growth over 1x.

Writes measurements-results/r1o3_resolution_scaling/summary.json.
"""
import duckdb

from measurements.common import DATASET_LABELS, RESULTS_ROOT, assert_files_exist, query_dicts, results_dir, write_json


# Inputs: the GPU job's timings and R9.O1's matched classifier timing.
DETECTOR = RESULTS_ROOT / 'r1o3_resolution_scaling' / 'detector_timing.jsonl'
MATCHED = RESULTS_ROOT / 'r9o1_matched_timing' / 'matched_timing.json'
# The batched and max-batch sweeps' timings (optional).
BATCHED = [RESULTS_ROOT / 'r1o3_resolution_scaling' / f'detector_timing_{s}.jsonl' for s in ('batched', 'maxbatch')]
# The classifier measured at max batch (optional).
CLASSIFIER_MAX = RESULTS_ROOT / 'r1o3_resolution_scaling' / 'detector_timing_classifier_maxbatch.jsonl'


def main():
    assert_files_exist([DETECTOR, MATCHED], 'resolution-scaling inputs')
    con = duckdb.connect()
    # Dataset ids to their paper labels, to join the two inputs.
    con.execute('CREATE TABLE labels (dataset VARCHAR, label VARCHAR)')
    con.executemany('INSERT INTO labels VALUES (?, ?)', list(DATASET_LABELS.items()))
    # Per-tile classifier time (ms) and tile size, per dataset label, from the matched timing.
    con.execute(f"""
        CREATE TABLE classifier AS
        SELECT key AS label,
               (value->'runs'->'classifier'->>'mean_ms_per_frame')::DOUBLE
                 / (value->'classifier'->>'tiles_inferred_at_most')::DOUBLE AS ms_per_tile,
               (value->'classifier'->>'tilesize')::INT AS tilesize
        FROM (SELECT unnest(map_entries(json_transform(datasets, '"MAP(VARCHAR, JSON)"'))) AS e
              FROM read_json('{MATCHED}', columns={{datasets: 'JSON'}})),
             LATERAL (SELECT e.key AS key, e.value AS value)""")
    # Detector rows at the frame's own resolution, with the projected classifier time at the same size.
    rows = query_dicts(con, f"""
        SELECT l.label, d.width, d.height, d.input_shape, d.mean_ms AS detector_ms, d.median_ms AS detector_median_ms,
               (d.width // c.tilesize) * (d.height // c.tilesize) AS tiles,
               c.ms_per_tile * (d.width // c.tilesize) * (d.height // c.tilesize) AS classifier_ms,
               d.mean_ms / (c.ms_per_tile * (d.width // c.tilesize) * (d.height // c.tilesize)) AS detector_over_classifier
        FROM read_json('{DETECTOR}') d JOIN labels l USING (dataset) JOIN classifier c USING (label)
        WHERE d.mode = 'native'
        ORDER BY l.label, d.width * d.height""")
    # The loader-default rows, as a cross-check against r9o1_matched_timing's detector means.
    defaults = query_dicts(con, f"""
        SELECT l.label, d.width, d.height, d.input_shape, d.mean_ms
        FROM read_json('{DETECTOR}') d JOIN labels l USING (dataset) WHERE d.mode = 'default' ORDER BY l.label""")
    for r in rows:
        print(f"{r['label']:10} {r['width']:>4}x{r['height']:<4} tiles {r['tiles']:>5}  classifier {r['classifier_ms']:7.2f} ms"
              f"  detector {r['detector_ms']:8.2f} ms  ratio {r['detector_over_classifier']:6.1f}")
    summary = dict(rows=rows, loader_defaults=defaults)
    # Batched sweep (own size x1/x2/x4 pixels; equal batch 16, or 16 divided by the scale), when it has run.
    batched_files = [str(p) for p in BATCHED if p.exists()]
    if batched_files:
        batched = query_dicts(con, f"""
            SELECT l.label, d.scale, d.schedule, d.batch, d.width, d.height, d.input_shape, d.oom,
                   d.mean_ms AS detector_ms, d.peak_allocated_mib, d.peak_reserved_mib,
                   (d.width // c.tilesize) * (d.height // c.tilesize) AS tiles,
                   c.ms_per_tile * (d.width // c.tilesize) * (d.height // c.tilesize) AS classifier_ms,
                   d.mean_ms / (c.ms_per_tile * (d.width // c.tilesize) * (d.height // c.tilesize)) AS detector_over_classifier
            FROM read_json({batched_files}, union_by_name=true) d JOIN labels l USING (dataset) JOIN classifier c USING (label)
            ORDER BY l.label, d.schedule, d.scale""")
        for r in batched:
            print(f"{r['label']:10} x{r['scale']} {r['schedule']:6} batch {r['batch']:>2} {r['width']:>4}x{r['height']:<4} "
                  + ('out of memory' if r['oom'] else
                     f"classifier {r['classifier_ms']:6.2f}  detector {r['detector_ms']:7.2f} ms/frame  "
                     f"ratio {r['detector_over_classifier']:5.1f}  peak {r['peak_allocated_mib']:6.0f} MiB"))
        summary['batched'] = batched
    # Measured classifier at max batch, paired with the detector at max batch at the same dataset and scale.
    if CLASSIFIER_MAX.exists() and BATCHED[1].exists():
        measured = query_dicts(con, f"""
            WITH c AS (SELECT * FROM read_json('{CLASSIFIER_MAX}') WHERE NOT oom),
                 d AS (SELECT * FROM read_json('{BATCHED[1]}') WHERE NOT oom),
                 j AS (SELECT l.label, c.scale, c.width, c.height, c.tiles, c.batch AS classifier_batch,
                              d.batch AS detector_batch, c.mean_ms AS classifier_ms, d.mean_ms AS detector_ms,
                              c.peak_allocated_mib AS classifier_peak_mib, d.peak_allocated_mib AS detector_peak_mib
                       FROM c JOIN d USING (dataset, scale) JOIN labels l USING (dataset))
            SELECT j.*, detector_ms / classifier_ms AS detector_over_classifier,
                   classifier_ms / first_value(classifier_ms) OVER w AS classifier_growth,
                   detector_ms / first_value(detector_ms) OVER w AS detector_growth,
                   tiles / first_value(tiles) OVER w AS tiles_growth
            FROM j WINDOW w AS (PARTITION BY label ORDER BY scale)
            ORDER BY label, scale""")
        # Both sides must cover the same datasets and scales.
        assert len(measured) == con.execute(f"SELECT count(*) FROM read_json('{CLASSIFIER_MAX}')").fetchone()[0]
        for r in measured:
            print(f"{r['label']:10} x{r['scale']} {r['width']:>4}x{r['height']:<4} classifier batch {r['classifier_batch']:>4} "
                  f"{r['classifier_ms']:6.3f} ms (x{r['classifier_growth']:.2f})  detector batch {r['detector_batch']:>3} "
                  f"{r['detector_ms']:7.2f} ms (x{r['detector_growth']:.2f})  ratio {r['detector_over_classifier']:5.1f}")
        summary['measured_max_batch'] = measured
    write_json(results_dir(__file__), 'summary.json', summary)


if __name__ == '__main__':
    main()
