#!/usr/local/bin/python
"""R9.O2: per-stage process time of CalDoT2's re-swept validation clips va00-va09, by the cost report's rules.

CalDoT2's sweep outputs on va00-va09 were cleared on ace on 2026-05-29, so measurements/r9o2_preparation_cost.py times
its sweep on the other 50 clips and scales by 60/50. measurements-gpu/r9o2_caldot2_resweep_bombe.sh re-runs the sweep
on those 10 clips on bombe; this script, run in the same bombe container (its /polyis-cache is that scratch cache),
times them with the cost report's own functions: p120's validation manifest, load_runtime_ops, load_units (each unit's
span from file timestamps, chained to the previous video of the same task), and stage_summary (shared outputs counted
once). Videos are ranked among all 60 validation clips, as on ace. measurements/r9o2_preparation_total.py adds these
stages to the 50 clips' instead of scaling.

Writes measurements-results/r9o2_caldot2_resweep_units/units.json.
"""
import argparse
import os

import duckdb
import pandas as pd

from evaluation.manifests import build_polytris_variant_manifest, build_split_video_manifest
from measurements.common import host_info, query_dicts, results_dir, write_json
from measurements.r9o2_preparation_cost import (STAGE_BUILDERS, dropped_ops, load_manifest, load_runtime_ops,
                                                load_units, stage_summary)


def parse_args():
    parser = argparse.ArgumentParser(description="R9.O2: CalDoT2 re-swept clips' per-stage process time")
    parser.add_argument('--dataset', default='caldot2-y05')
    parser.add_argument('--videos', nargs='+', default=[f'va{i:02d}.mp4' for i in range(10)])
    return parser.parse_args()


def main():
    args = parse_args()
    con = duckdb.connect()
    # Every validation configuration on every clip, as p120 lists them, with which runtime files exist here.
    # split_manifest also lists the reference pipeline's files, which this scratch cache does not hold.
    configs = build_split_video_manifest(datasets=[args.dataset], videosets=['valid']).merge(
        build_polytris_variant_manifest(), how='cross')
    manifest = pd.concat([build(configs) for build in STAGE_BUILDERS], ignore_index=True)
    manifest['exists'] = manifest['runtime_file'].map(os.path.exists)
    load_manifest(con, 'valid_manifest', manifest)
    # The re-swept clips' sweep files; every one must exist.
    con.execute('CREATE OR REPLACE TABLE sweep_manifest AS SELECT * FROM valid_manifest '
                "WHERE variant = 'polytris' AND video IN (SELECT unnest($videos))", dict(videos=args.videos))
    missing = query_dicts(con, 'SELECT stage, runtime_file FROM sweep_manifest WHERE NOT "exists" LIMIT 5')
    assert not missing, f'{args.dataset}: re-swept files missing: {missing}'
    # The work units, and the split's clips in the pipeline's sorted order (all 60, so neighbours are the true ones).
    con.execute('CREATE OR REPLACE TABLE work_units AS SELECT DISTINCT stage, runtime_file, videoset, video FROM sweep_manifest')
    con.execute('CREATE OR REPLACE TABLE split_videos AS SELECT videoset, video, '
                'row_number() OVER (PARTITION BY videoset ORDER BY video) AS rank '
                'FROM (SELECT DISTINCT videoset, video FROM valid_manifest)')
    # Op times and per-unit process seconds, by the cost report's rules.
    files = [(r['stage'], r['runtime_file']) for r in query_dicts(con, 'SELECT stage, runtime_file FROM work_units')]
    load_runtime_ops(con, files)
    load_units(con, 'work_units', 'split_videos')
    stages = {r.pop('stage'): r for r in stage_summary(con, 'sweep_manifest')}
    write_json(results_dir(__file__), 'units.json',
               dict(dataset=args.dataset, videos=args.videos, host=host_info(), stages=stages,
                    dropped_ops_s=dropped_ops(con, 'sweep_manifest')))
    for stage, r in stages.items():
        print(f"{stage}: {r['work_units']} units, {r['chained_units']} chained, process {r['process_s'] / 3600:.2f} h")


if __name__ == '__main__':
    main()
