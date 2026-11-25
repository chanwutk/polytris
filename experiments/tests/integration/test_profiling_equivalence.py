"""
Validation: the profiling sweep against the scripts (p020 -> p060), on committed clips, every stage exactly.

The sweep runs first and saves every result in the scripts' layout, with each ILP's status (ilp.json).  The scripts
then run on the same clips, p022 replaced by experiments/tests/integration/mirror_p022.py: where either side's ILP stopped at its
time limit, both use the sweep's pruned grid, so every later stage is comparable.  Compared (profiling's own
compare_with_scripts): scores, pruned grids, index maps and offsets, detections in order, byte-identical tracks, HOTA.
"""

import json
from pathlib import Path

import pytest
import yaml

from experiments.tests.integration import harness

# The grid of experiments/tools/profiling/compare_scripts.py: 2 sample rates x 2 T_r x (no pruning + 2 tolerances x 2 trackers)
# x 2 paddings.
GRID = {'sample_rates': [1, 4], 'relevance_thresholds': [0.25, 0.5], 'mistrack_tolerances': [None, 0.4, 0.8],
        'tile_paddings': ['none', 'tr'], 'canvas_scales': [1.0], 'trackers': ['sortcython', 'bytetrackcython']}


def sweep_config(tree: harness.Tree, path: Path, output_root: Path) -> Path:
    """The dataset's sweep config with the test's grid, the scratch cache, and its own output root."""
    raw = (harness.REPO / 'experiments' / 'fanout_profiling' / 'configs' / f'{tree.dataset}.yaml').read_text()
    # Artifacts and the reference tracks live in the scratch cache.
    config = yaml.safe_load(raw.replace('/polyis-cache/', f'{tree.cache}/'))
    config['output_root'] = str(output_root)
    # Every tolerance runs (no merging), so each script output has its own counterpart.
    config['params'].update(GRID, reduce_search_space=False)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['caldot1-y05', 'jnc0'])
def test_profiling_matches_the_scripts(dataset, tmp_path):
    harness.require_detector(dataset)
    tree = harness.make_tree(tmp_path, dataset, 'valid', GRID)
    output_root = tmp_path / 'profiling-out'
    # The reference tracks the sweep's evaluate stage and HOTA score against.
    harness.reference(tree)
    # The pipeline first: it saves every stage's results and each ILP's status.
    tree.run('-m', 'experiments.fanout_profiling.main', '--config', str(sweep_config(tree, tmp_path / 'sweep.yaml', output_root)))
    # Then the scripts, p022 through the mirror.
    harness.scripts(tree, 'p020_exec_classify')
    harness.mirror_p022(tree, output_root / dataset / 'execution', tmp_path / 'mirror.json')
    harness.scripts(tree, 'p030_exec_compress', 'p040_exec_detect', 'p050_exec_uncompress', 'p060_exec_track')

    from experiments.tools.profiling.compare_scripts import compare_with_scripts
    report = compare_with_scripts(str(output_root), dataset, str(tree.cache), harness.REFERENCE_FILE)
    mirror = json.loads((tmp_path / 'mirror.json').read_text())
    (tmp_path / 'report.json').write_text(json.dumps({'mirror': mirror, **report}, indent=1) + '\n')
    stages, totals = report['stages'], report['totals']
    print(json.dumps({'totals': totals, 'mirror_copied': sum(r['copied'] for r in mirror)}, indent=1))

    videos, per_pack = len(tree.videos), 2 * 2 * 2 * (1 + 2 * 2)
    # Every branch was produced and compared: 2 videos x 40 pack branches, and 6 tracked branches per pack setting.
    assert len(stages['033_compressed_frames']) == videos * per_pack
    assert len(stages['060_uncompressed_tracks']) == videos * 2 * 2 * 2 * (2 + 2 * 2)
    # Scores: classify runs once on every frame, p020 once per sample rate; at s = 1 the two are the same batches.
    for row in stages['020_relevancy']:
        if row['path'].split('/')[2].endswith('_1'):
            assert row['equal'], row
        else:
            # Other rates batch different frames together on the GPU: values may differ by 1 of 255.
            assert row.get('max_abs_diff', 0) <= 1, row
    # Everything after classify is exact.
    for stage in ('022_pruned_polyominoes', '033_compressed_frames'):
        assert all(r['equal'] for r in stages[stage]), [r for r in stages[stage] if not r['equal']][:3]
    assert all(r['equal_in_order'] for r in stages['050_uncompressed_detections'])
    assert totals['060_uncompressed_tracks']['text_identical'] == totals['060_uncompressed_tracks']['rows']
    assert totals['hota_per_video']['equal'] == totals['hota_per_video']['compared'] > 0
