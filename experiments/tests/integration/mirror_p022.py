"""
Test-only stand-in for experiments/scripts/p022_exec_prune_polyominoes.py in the integration tests.

For every (video, pruned combination) of the configured grid it runs p022's own process_video, one at a time, so at
most one Gurobi session is open (p022 solves its tasks in parallel; the web license allows two).  It records each
solve's Gurobi status by wrapping the solve_ilp that p022 calls.  With --pipeline-execution (the validation test),
a pair where either the pipeline's ILP (its saved ilp.json) or this solve stopped before the optimum gets the
pipeline's pruned grid instead, so the scripts' later stages see the same input and stay comparable.  Without it
(the test-split test) nothing is replaced and the report shows whether every solve finished.

Run with CONFIG set to the test's config (experiments/tests/integration/harness.py):

    python -m experiments.tests.integration.mirror_p022 --split valid --videos va00.mp4 va01.mp4 \
        --pipeline-execution <output_root>/<dataset>/execution --report mirror.json
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path

GUROBI_OPTIMAL = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--pipeline-execution', default=None)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here: p022 and tetris.io.cache read CONFIG when imported.
    import experiments.scripts.p022_exec_prune_polyominoes as p022
    from tetris.io import cache
    from tetris.utilities import build_param_str, get_config

    ex = get_config()['EXEC']
    (dataset,) = ex['DATASETS']
    real_solve = p022.solve_ilp
    last: dict = {}

    def recording_solve(*a, **k):
        # p022's own solve, with its status and time kept for the report.
        result = real_solve(*a, **k)
        last.update(status=int(result.status), solve_ms=float(result.solve_ms))
        return result

    p022.solve_ilp = recording_solve
    rows = []
    combos = itertools.product(args.videos, ex['CLASSIFIERS'], ex['TILE_SIZES'], ex['SAMPLE_RATES'], ex['TRACKERS'],
                               [t for t in ex['TRACKING_ACCURACY_THRESHOLDS'] if t is not None],
                               ex['RELEVANCE_THRESHOLDS'])
    for video, classifier, tile, s, tracker, tol, t_r in combos:
        last.clear()
        p022.process_video(dataset, args.split, video, classifier, tile, s, tracker, tol, t_r, None)
        param = build_param_str(classifier=classifier, tilesize=tile, sample_rate=s, tracker=tracker,
                                tracking_accuracy_threshold=tol, relevance_threshold=t_r)
        row = {'video': video, 'param': param, 'status': last['status'], 'solve_ms': last['solve_ms'],
               'copied': False}
        if args.pipeline_execution is not None:
            pipeline = Path(args.pipeline_execution) / video / '022_pruned_polyominoes' / param
            ilp = json.loads((pipeline / 'ilp.json').read_text())
            row['pipeline_status'] = ilp['status']
            # A solve stopped by its limit is not unique: both sides then use the pipeline's grid.
            if ilp['status'] != GUROBI_OPTIMAL or last['status'] != GUROBI_OPTIMAL:
                shutil.copyfile(pipeline / 'score' / 'score.jsonl',
                                cache.exec(dataset, 'pruned-polyominoes', video, param, 'score', 'score.jsonl'))
                row['copied'] = True
        rows.append(row)
    Path(args.report).write_text(json.dumps(rows, indent=1) + '\n')
    print(f'{len(rows)} prune results, {sum(r["copied"] for r in rows)} took the pipeline\'s grid', flush=True)


if __name__ == '__main__':
    main()
