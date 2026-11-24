#!/usr/local/bin/python
"""R9.O2: each system's preparation time per dataset, as a sum of measured components.

Every component is one measurement added as is: a sum of runtimes over work units, or one measured run. All times
include I/O. The places that leave work out are marked EXCLUDED with their reason. Nothing is subtracted.

Tetris, from measurements/r9o2_preparation_cost.py (cache runtimes and the serial re-runs on bombe),
measurements/r9o2_preparation_overheads.py, measurements/r9o2_packing_serial.py, and, for CalDoT2's 10 clips whose sweep
outputs were cleared on ace, measurements/r9o2_caldot2_resweep_units.py (re-swept and timed on bombe):
  training    p011 reference detection, p012 training data, p013 pruning, p014 classifier training,
              p015 classifier benchmark, p016 mistrack profiling, detector loading
  validation  reference outputs (002), sweep p020-p060, p110 scoring, p130/p135 tables, detector loading
OTIF and LEAP, from measurements-gpu/r1o2_leap_otif_rerun.py prep (LEAP on Amsterdam and CalDoT from its
matched-detector reruns, measurements-results/r1o2_leap_matched): the GNU time wall clock of each preparation step.

Writes measurements-results/r9o2_preparation_total/preparation_total.json.
"""
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from measurements.common import DATASET_LABELS, RESULTS_ROOT, results_dir, write_json
from measurements.r1o2_leap_otif_rerun import RAW_DIR, preparation


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: preparation time per system and dataset')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS), choices=list(DATASET_LABELS))
    return parser.parse_args()


# Inputs, each written by the measurement script named above.
COST = RESULTS_ROOT / 'r9o2_preparation_cost' / 'preparation_cost.json'
OVERHEADS = RESULTS_ROOT / 'r9o2_preparation_overheads' / 'preparation_overheads.json'
PACKING = RESULTS_ROOT / 'r9o2_packing_serial'
RESWEEP = RESULTS_ROOT / 'r9o2_caldot2_resweep_units' / 'units.json'
LEAP_MATCHED = RESULTS_ROOT / 'r1o2_leap_matched' / 'summary'
# LEAP's reported results on these datasets come from its matched-detector reruns, so its preparation does too.
LEAP_MATCHED_DATASETS = ('ams-y05', 'caldot1-y05', 'caldot2-y05')
# Sweep stages timed per work unit from the cache; p030 comes from its serial re-run instead.
SWEEP_STAGES = ('020_exec_classify', '022_exec_prune_polyominoes', '040_exec_detect', '050_exec_uncompress',
                '060_exec_track')
# Baseline steps on the validation split; every other step prepares on the training split.
BASELINE_VALIDATION_STEPS = ('detect_valid', 'tune_valid')


def tetris(dataset: str, cost: dict, overhead: dict, resweeps: dict, con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Tetris's preparation components for one dataset: (phase, component, seconds, source)."""
    d = cost['datasets'][dataset]
    cached, retimed = d['components'], d['retiming']['components']
    sweep = cached['validation_sweep']
    # CalDoT2's sweep outputs on va00-va09 were cleared on ace on 2026-05-29, so the cost report times its sweep on the
    # other 50 clips; those 10 were re-swept on bombe and timed there by the same rules (RESWEEP), and each stage adds
    # both parts. Every other dataset's sweep is complete on ace.
    resweep = resweeps.get(dataset, {})
    assert sweep['coverage']['complete_videos'] + len(resweep.get('videos', ())) == sweep['coverage']['videos'], dataset
    def sweep_part(stage, key):
        return sweep['stages'][stage][key] + resweep.get('stages', {}).get(stage, {}).get(key, 0)
    # A stage's output shared by several sweep configurations is computed once, so its time is summed once.
    # p015 has no recorded time in the cache, so it comes from its re-run.
    rows = [
        ('training', 'p011 reference detection', retimed['reference_detection_train']['process_s'], 'serial re-run'),
        ('training', 'p012 training data', cached['training_data_creation']['process_s'], 'cache, per video'),
        ('training', 'p013 training-data pruning', retimed['training_data_pruning']['process_s'], 'serial re-run'),
        ('training', 'p014 classifier training', cached['classifier_training']['process_s'], 'cache, wall clock'),
        ('training', 'p015 classifier benchmark', overhead['classifier_benchmark_s'], 'overheads re-run'),
        ('training', 'p016 mistrack profiling', retimed['mistrack_profiling']['process_s'], 'serial re-run'),
        ('validation', '002 reference outputs', cached['reference_outputs_valid']['process_s'], 'cache, per video'),
        *(('validation', f'sweep {stage[:4]}', sweep_part(stage, 'process_s'),
           'cache, per unit' + (' (+ bombe re-sweep)' if resweep else '')) for stage in SWEEP_STAGES),
        ('validation', 'p110 scoring', retimed['validation_scoring']['process_s'], 'serial re-run'),
        ('validation', 'p130/p135 tables', overhead['tradeoff_s'] + overhead['pareto_s'], 'overheads re-run'),
    ]
    # p030 ran with 20 concurrent workers, whose per-unit spans include slowdown from sharing CPUs and disk; its
    # serial re-run times every unit from the start to the end of compress().
    ((units, packing_s),) = con.execute(
        'SELECT count(DISTINCT runtime_file), sum(total_ms) / 1000 FROM read_json(?, format=\'newline_delimited\')',
        [str(PACKING / f'{dataset}.jsonl')]).fetchall()
    assert units == sweep_part('030_exec_compress', 'work_units'), (dataset, units)
    rows.append(('validation', 'sweep p030', packing_s, 'serial re-run, per unit'))
    # The detector is loaded as the scripts load it: per training video (p011), per validation reference video (002),
    # and per sweep configuration (p040). The loading measurement includes a 32-frame warm-up per reference video,
    # which the reference outputs' spans also contain; it is kept in both, so it may be counted twice.
    loading = overhead['startup_s']
    training_share = overhead['train_videos'] / (overhead['train_videos'] + overhead['valid_videos']
                                                 + overhead['sweep_detector_loads'])
    rows += [('training', 'detector loading', training_share * loading, 'overheads re-run'),
             ('validation', 'detector loading', (1 - training_share) * loading, 'overheads re-run')]
    # EXCLUDED: classifiers and trackers the reported configuration does not use (not_needed_by_reported_configuration
    # in the cost report: the baseline ShuffleNet classifier and OC-SORT profiles), since a deployment trains only
    # what it runs. Detector fine-tuning is excluded for every system, since all of them use the same detector.
    return rows


def baselines(dataset: str, rerun: dict) -> list[tuple]:
    """OTIF's and LEAP's preparation components for one dataset: (system, phase, component, seconds, source)."""
    rows = []
    for system in ('otif', 'leap'):
        if system == 'leap' and dataset in LEAP_MATCHED_DATASETS:
            prep = json.loads((LEAP_MATCHED / f'results-{dataset}.json').read_text())['preparation']
            source = 'matched-detector rerun'
        else:
            prep, source = rerun, 'rerun'
        total = next(t for t in prep['totals'] if (t['system'], t['dataset']) == (system, dataset))
        assert total['complete'] and total['clean'], (system, dataset)
        for step in prep['steps']:
            if (step['system'], step['dataset']) != (system, dataset):
                continue
            # EXCLUDED: evaluation-only steps (LEAP's pseudo labels on the test split), which only score the outputs.
            if step['evaluation_only']:
                continue
            phase = 'validation' if step['step'] in BASELINE_VALIDATION_STEPS else 'training'
            rows.append((system, phase, step['step'], step['wall_s'], source))
    return rows


def main():
    args = parse_args()
    cost = json.loads(COST.read_text())
    # The loading scenario that reloads the detector per video and configuration, at the process boundary.
    overheads = {r['dataset']: r for r in json.loads(OVERHEADS.read_text())['rows']
                 if r['scenario'] == 'per_video_and_configuration' and r['boundary'] == 'process'}
    rerun = preparation(RAW_DIR / 'prep')
    # Clips re-swept outside the cache, by dataset.
    resweep = json.loads(RESWEEP.read_text())
    resweeps = {resweep['dataset']: resweep}
    con = duckdb.connect()
    rows = []
    for dataset in args.datasets:
        rows += [dict(dataset=dataset, system='tetris', phase=p, component=c, seconds=s, source=src)
                 for p, c, s, src in tetris(dataset, cost, overheads[dataset], resweeps, con)]
        rows += [dict(dataset=dataset, system=sy, phase=p, component=c, seconds=s, source=src)
                 for sy, p, c, s, src in baselines(dataset, rerun)]
    # Totals are plain sums of the components above.
    con.register('components', pd.DataFrame(rows))
    totals = [dict(zip(('dataset', 'system', 'phase', 'seconds'), r)) for r in con.execute(
        'SELECT dataset, system, phase, sum(seconds) FROM components GROUP BY ALL ORDER BY ALL').fetchall()]
    write_json(results_dir(__file__), 'preparation_total.json',
               dict(inputs=dict(cost=str(COST), overheads=str(OVERHEADS), packing=str(PACKING), resweep=str(RESWEEP),
                                leap_matched=str(LEAP_MATCHED), rerun=str(RAW_DIR / 'prep')),
                    components=rows, totals=totals))


if __name__ == '__main__':
    main()
