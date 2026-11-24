"""
Test split: execution against the scripts (p020 -> p060, the real p022), on committed clips, byte-identical tracks.

The configurations are few and the clips short, so every ILP finishes well inside p022's 0.1 s limit; the test checks
that first (on the pipeline's own classifier scores, and from the scripts' solve statuses) and fails if a solve came
close, rather than comparing outputs a stopped solve could make differ.  execution runs first, then the scripts.
"""

import json
import os

import numpy as np
import pytest
import yaml

from tests.integration import harness

# A few operating points: with and without pruning, sample rates 1 and 4, both trackers and paddings.
CONFIGS = [
    {'sample_rate': 1, 'relevance_threshold': 0.5, 'mistrack_tolerance': None, 'tile_padding': 'none',
     'tracker': 'sortcython'},
    # (T_r 0.25 here made jnc0 te01's ILP run past p022's 0.1 s limit; at 0.5 every solve takes about 10 ms.)
    {'sample_rate': 1, 'relevance_threshold': 0.5, 'mistrack_tolerance': 0.4, 'tile_padding': 'tr',
     'tracker': 'bytetrackcython'},
    {'sample_rate': 4, 'relevance_threshold': 0.5, 'mistrack_tolerance': 0.8, 'tile_padding': 'none',
     'tracker': 'sortcython'},
    {'sample_rate': 4, 'relevance_threshold': 0.25, 'mistrack_tolerance': None, 'tile_padding': 'tr',
     'tracker': 'bytetrackcython'},
]
# A solve counts as safely finished only under half of p022's 0.1 s limit.
SAFE_SOLVE_MS = 50.0


def full(c: dict) -> dict:
    """make_paper_configs.build_params's keys for one configuration."""
    return {'classifier': 'ShuffleNet05', 'tile_size': 60, 'canvas_scale': 1.0, **c}


def param_str(c: dict) -> str:
    """The configuration's stage-060 directory name."""
    from polyis.utilities import build_param_str
    p = full(c)
    return build_param_str(classifier=p['classifier'], tilesize=p['tile_size'], sample_rate=p['sample_rate'],
                           tilepadding=p['tile_padding'], canvas_scale=p['canvas_scale'], tracker=p['tracker'],
                           tracking_accuracy_threshold=p['mistrack_tolerance'],
                           relevance_threshold=p['relevance_threshold'])


def prune_param_str(c: dict) -> str:
    """The configuration's p022 directory name (no padding or canvas scale: pruning comes before packing)."""
    from polyis.utilities import build_param_str
    p = full(c)
    return build_param_str(classifier=p['classifier'], tilesize=p['tile_size'], sample_rate=p['sample_rate'],
                           tracker=p['tracker'], tracking_accuracy_threshold=p['mistrack_tolerance'],
                           relevance_threshold=p['relevance_threshold'])


def grid() -> dict:
    """The scripts' grid: every value any test configuration uses (the selection CSV picks the configurations)."""
    return {'sample_rates': sorted({c['sample_rate'] for c in CONFIGS}),
            'relevance_thresholds': sorted({c['relevance_threshold'] for c in CONFIGS}),
            'mistrack_tolerances': [None] + sorted({c['mistrack_tolerance'] for c in CONFIGS} - {None}),
            'tile_paddings': sorted({c['tile_padding'] for c in CONFIGS}), 'canvas_scales': [1.0],
            'trackers': sorted({c['tracker'] for c in CONFIGS})}


def execution_config(tree: harness.Tree, c: dict) -> dict:
    """execution's config for one configuration, on the scratch cache's artifacts."""
    from execution.tools.make_paper_configs import build_params
    detectors = yaml.safe_load((harness.REPO / 'configs' / 'detectors.yaml').read_text())
    text = yaml.safe_dump(build_params(tree.dataset, full(c), detectors), sort_keys=False)
    return yaml.safe_load(text.replace('/polyis-cache/', f'{tree.cache}/'))


def solve_times(tree: harness.Tree, configs: list[dict]) -> list[dict]:
    """Each pruned configuration's ILP per clip, solved as execution does on execution's own classifier scores."""
    from execution import frames
    from execution import operators as ops
    from execution.config import MISTRACK_TOLERANCES
    out = []
    prefix = frames.new_prefix()
    try:
        for c in configs:
            cfg = execution_config(tree, c)
            a = cfg['artifacts']
            state = ops.load_classifier_state(a['classifier']['weights'], a['classifier']['compilation_benchmark'],
                                              a['never_relevant_bitmap'], 60)
            distance = ops.max_sampling_distance(np.load(a['max_rate_table']),
                                                 MISTRACK_TOLERANCES.index(c['mistrack_tolerance']), c['sample_rate'])
            for video in tree.videos:
                decoded = ops.decode_video(str(tree.clips / video), prefix, c['sample_rate'])
                try:
                    scores, _ = ops.classify_frames(state, decoded)
                finally:
                    frames.unlink(decoded.frames)
                _, result = ops.prune_scores(scores, distance, c['relevance_threshold'])
                out.append({'prune': prune_param_str(c), 'video': video, 'status': int(result.status),
                            'solve_ms': float(result.solve_ms)})
    finally:
        frames.unlink_all(prefix)
    return out


def write_selection(tree: harness.Tree) -> None:
    """The test split's configuration selection, where the scripts read it (polyis.pareto.load_pareto_params:
    <cache>/<dataset>/evaluation/095_pareto_params/pareto_params.csv), in its columns."""
    rows = ['variant_id,classifier,tilesize,sample_rate,tracking_accuracy_threshold,relevance_threshold,'
            'tilepadding,canvas_scale,tracker']
    for c in CONFIGS:
        tol = '' if c['mistrack_tolerance'] is None else c['mistrack_tolerance']
        rows.append(f"{param_str(c)},ShuffleNet05,60,{c['sample_rate']},{tol},{c['relevance_threshold']},"
                    f"{c['tile_padding']},1.0,{c['tracker']}")
    path = tree.cache / tree.dataset / 'evaluation' / '095_pareto_params' / 'pareto_params.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n')


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['caldot1-y05', 'jnc0'])
def test_execution_matches_the_scripts(dataset, tmp_path):
    harness.require_detector(dataset)
    tree = harness.make_tree(tmp_path, dataset, 'test', grid())
    out = tmp_path / 'execution-out'
    # The pipeline first: every configuration through execution over the clips.
    for c in CONFIGS:
        path = tmp_path / f'{param_str(c)}.yaml'
        path.write_text(yaml.safe_dump(execution_config(tree, c), sort_keys=False))
        tree.run('-m', 'execution.main', '--config', str(path), '--videos', str(tree.clips),
                 '--out', str(out / param_str(c)))
    # Its ILPs must finish well inside the limit, or a stopped solve could make the outputs differ.
    ours = solve_times(tree, [c for c in CONFIGS if c['mistrack_tolerance'] is not None])
    assert all(r['status'] == 2 and r['solve_ms'] < SAFE_SOLVE_MS for r in ours), ours

    # Then the scripts, on the same clips, restricted to these configurations; p022 one combination at a time.
    write_selection(tree)
    harness.scripts(tree, 'p020_exec_classify')
    harness.mirror_p022(tree, None, tmp_path / 'mirror.json')
    harness.scripts(tree, 'p030_exec_compress', 'p040_exec_detect', 'p050_exec_uncompress', 'p060_exec_track')
    theirs = {(r['video'], r['param']): r for r in json.loads((tmp_path / 'mirror.json').read_text())}
    # The scripts' solves for these configurations finished well inside the limit too.
    for r in ours:
        script = theirs[(r['video'], r['prune'])]
        assert script['status'] == 2 and script['solve_ms'] < SAFE_SOLVE_MS, script

    # Byte-identical tracks, track ids included, for every configuration and clip.
    mismatches = []
    for c in CONFIGS:
        for video in tree.videos:
            mine = (out / param_str(c) / (os.path.splitext(video)[0] + '.jsonl')).read_text()
            script = (tree.execution / video / '060_uncompressed_tracks' / param_str(c) / 'tracking.jsonl').read_text()
            if mine != script:
                mismatches.append(f'{param_str(c)} {video}')
    assert not mismatches, mismatches
