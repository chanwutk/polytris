#!/usr/local/bin/python
"""Add isolated preparation timings to the saved R9.O2 report, with two detector-reuse assumptions.

Uses measurements-gpu/r9o2_preparation_overheads.py outputs. This is an accounting sensitivity,
not a measured end-to-end serial run. Cached inference costs and published throughput are unchanged.
"""
import argparse
import hashlib
from pathlib import Path

import duckdb

from measurements.common import query_dicts, results_dir, write_json


BASE_REPORT = Path('measurements-results/r9o2_preparation_cost/preparation_cost.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings', required=True)
    parser.add_argument('--base-report', default=str(BASE_REPORT))
    return parser.parse_args()


def account(con, base_report, timings):
    # Preserve the saved report as JSON while doing all joins, counts, and arithmetic in DuckDB.
    con.execute('CREATE OR REPLACE TABLE base AS SELECT key AS dataset, value AS report '
                "FROM read_text(?), LATERAL json_each(content::JSON, '$.datasets')", [str(base_report)])
    con.execute('CREATE OR REPLACE TABLE samples AS SELECT json FROM read_json_objects(?, format=\'newline_delimited\')',
                [str(timings)])
    con.execute('''CREATE OR REPLACE TABLE measured AS
        SELECT json->>'dataset' AS dataset, json->>'step' AS step,
               (json->>'seconds')::DOUBLE AS seconds,
               (json->>'cuda_init_s')::DOUBLE AS cuda_s,
               (json->>'load_s')::DOUBLE AS load_s,
               (json->>'warmup_s')::DOUBLE AS warmup_s,
               (json->>'tradeoff_s')::DOUBLE AS tradeoff_s,
               (json->>'pareto_s')::DOUBLE AS pareto_s
        FROM samples''')
    # Refuse incomplete or repeated measurements rather than silently using zero or selecting an arbitrary run.
    expected_steps = ('train', 'reference', 'sweep', 'persistent', 'classifier', 'tables')
    problems = con.execute('''
        SELECT b.dataset, s.step, count(m.seconds) AS records
        FROM base b CROSS JOIN unnest(?) s(step)
        LEFT JOIN measured m ON m.dataset = b.dataset AND m.step = s.step
        GROUP BY b.dataset, s.step HAVING count(m.seconds) != 1
    ''', [list(expected_steps)]).fetchall()
    assert not problems, f'Missing or duplicate overhead timings: {problems}'
    assert con.execute('SELECT count(*) FROM measured WHERE seconds < 0 OR NOT isfinite(seconds)').fetchone()[0] == 0
    invalid_samples = con.execute("""
        SELECT dataset, step FROM measured
        WHERE step IN ('train', 'reference', 'sweep', 'persistent') AND
              (cuda_s IS NULL OR load_s IS NULL OR warmup_s IS NULL OR
               cuda_s < 0 OR load_s < 0 OR warmup_s < 0 OR NOT isfinite(cuda_s + load_s + warmup_s))
           OR step = 'tables' AND (tradeoff_s IS NULL OR pareto_s IS NULL OR tradeoff_s < 0 OR pareto_s < 0)
    """).fetchall()
    assert not invalid_samples, f'Invalid overhead timing components: {invalid_samples}'
    # Count actual p040 tasks, including shared configurations, from cached stage work units.
    con.execute('''CREATE OR REPLACE TABLE costs AS
        SELECT b.dataset, b.report,
               (report->>'$.video.train.videos')::INTEGER AS train_videos,
               (report->>'$.video.valid.videos')::INTEGER AS valid_videos,
               (report->>'$.components.validation_sweep.stages.040_exec_detect.work_units')::INTEGER AS detect_units,
               (report->>'$.components.validation_sweep.coverage.complete_videos')::INTEGER AS complete_videos,
               (report->>'$.components.classifier_optimization.instrumented_s')::DOUBLE AS old_benchmark_s,
               c.seconds AS classifier_benchmark_s, t.tradeoff_s AS measured_tradeoff_s,
               t.tradeoff_s * valid_videos / complete_videos AS tradeoff_s, t.pareto_s,
               p.seconds AS persistent_startup_s,
               a.cuda_s + r.cuda_s + w.cuda_s AS repeated_cuda_s,
               a.load_s AS train_load_s, r.load_s + r.warmup_s AS reference_startup_s,
               r.warmup_s AS reference_warmup_s, w.load_s + w.warmup_s AS sweep_startup_s
        FROM base b
        JOIN measured a ON a.dataset = b.dataset AND a.step = 'train'
        JOIN measured r ON r.dataset = b.dataset AND r.step = 'reference'
        JOIN measured w ON w.dataset = b.dataset AND w.step = 'sweep'
        JOIN measured p ON p.dataset = b.dataset AND p.step = 'persistent'
        JOIN measured c ON c.dataset = b.dataset AND c.step = 'classifier'
        JOIN measured t ON t.dataset = b.dataset AND t.step = 'tables'
    ''')
    invalid = con.execute('SELECT dataset FROM costs WHERE complete_videos IS NULL OR detect_units IS NULL '
                          'OR complete_videos <= 0 OR detect_units % complete_videos != 0 '
                          'OR (report->\'$.retiming.totals_with_retiming\') IS NULL').fetchall()
    assert not invalid, f'Invalid work-unit counts or missing base re-timing: {invalid}'
    # Replace the old p015 estimate once; add p130/p135 and the selected detector-startup assumption.
    rows = query_dicts(con, '''
        WITH adjusted AS (
            SELECT *, detect_units // complete_videos AS sweep_detector_loads,
                   repeated_cuda_s + train_videos * train_load_s + valid_videos * reference_startup_s
                       + (detect_units // complete_videos) * sweep_startup_s AS repeated_startup_s,
                   classifier_benchmark_s - old_benchmark_s + tradeoff_s + pareto_s AS common_adjustment_s
            FROM costs
        )
        SELECT dataset, train_videos, valid_videos, sweep_detector_loads,
               classifier_benchmark_s, old_benchmark_s, measured_tradeoff_s, tradeoff_s, pareto_s,
               boundary, scenario, startup_s,
               (report->>('$.retiming.totals_with_retiming.' || boundary || '_s'))::DOUBLE AS base_s,
               CASE WHEN boundary = 'process' THEN valid_videos * reference_warmup_s ELSE 0 END
                   AS already_counted_reference_warmup_s,
               base_s + common_adjustment_s + startup_s - already_counted_reference_warmup_s AS total_s,
               total_s / 3600 AS total_hours,
               total_s / (report->>('$.break_even.' || boundary || '.saving_s_per_video_minute'))::DOUBLE / 60
                   AS break_even_video_hours
        FROM adjusted CROSS JOIN (VALUES ('instrumented'), ('process')) b(boundary)
        CROSS JOIN LATERAL (VALUES ('once_per_dataset', persistent_startup_s),
                                   ('per_video_and_configuration', repeated_startup_s)) s(scenario, startup_s)
        ORDER BY dataset, boundary, scenario
    ''')
    return dict(method=dict(
        base_report=str(base_report), base_report_sha256=hashlib.sha256(Path(base_report).read_bytes()).hexdigest(),
        timings=str(timings), unit='summed process seconds',
        persistent='one detector load/CUDA initialization per dataset; reference and representative packed-input '
                   'shapes each warmed once; assumes a persistent model shared across preparation stages',
        repeated='detector loaded per training video, validation reference video, and distinct p040 configuration; '
                 'CUDA initialized once per stage; p011 has no extra warm-up since first inference is already timed',
        warmup_accounting='reference process spans already include the 32-frame warm-up; subtract its measured '
                          'per-video proxy before adding either scenario, avoiding double counting; '
                          'instrumented times exclude that warm-up so no subtraction is needed',
        limitation='accounting estimates using representative packed frames, not an end-to-end serial replay; '
                   'cached inference and throughput unchanged; unmeasured first-output startup still excluded',
        classifier='measured p015 replaces its existing estimate; no retraining',
        tables='measured p130/p135 on reconstructed validation rows only; scratch frontiers are cost probes and '
               'never replace the saved selections; raw timings record configuration-set differences; '
               'p130 time scales by full/complete validation videos when runtime logs are missing, '
               'while p135 uses the full configuration count without scaling'),
        rows=rows)


def main():
    args = parse_args()
    report = account(duckdb.connect(), args.base_report, args.timings)
    write_json(results_dir(__file__), 'preparation_overheads.json', report)
    for row in report['rows']:
        if row['boundary'] == 'instrumented':
            print(f"{row['dataset']} {row['scenario']}: {row['total_hours']:.3f} h, "
                  f"startup {row['startup_s']:.2f} s, break-even {row['break_even_video_hours']:.3f} video h")


if __name__ == '__main__':
    main()
