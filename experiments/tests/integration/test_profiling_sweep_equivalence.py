"""
Validation: the profiling sweep (one execution run per configuration, with --save) against the scripts (p020 -> p060),
on committed clips, every stage exactly.

The sweep runs first: every configuration as `python -m execution.main --save`, which writes each stage's results in the
scripts' layout with each ILP's status (ilp.json); then profiling.evaluate scores every configuration.  The scripts
then run on the same clips, p022 replaced by mirror_p022.py: where either side's ILP stopped at its time limit, both
use the sweep's pruned grid, so every later stage is comparable.  Compared (compare_with_scripts): scores, pruned
grids, index maps and offsets, detections in order, byte-identical tracks, and per-video HOTA.
"""

import json
from pathlib import Path

import pytest
import yaml

from experiments.tests.integration import harness

# The fan-out equivalence test's grid: 2 sample rates x 2 T_r x (no pruning + 2 tolerances x 2 trackers) x 2 paddings.
GRID = {'sample_rates': [1, 4], 'relevance_thresholds': [0.25, 0.5], 'mistrack_tolerances': [None, 0.4, 0.8],
        'tile_paddings': ['none', 'tr'], 'canvas_scales': [1.0], 'trackers': ['sortcython', 'bytetrackcython']}


def sweep_config(tree: harness.Tree, path: Path, output_root: Path) -> Path:
    """The dataset's generated sweep config with the test's grid, the scratch cache, and its own output root."""
    from experiments.tools.profiling.make_configs import sweep_config as generated
    grid = yaml.safe_load((harness.REPO / 'configs' / 'global.yaml').read_text())['EXEC']
    detectors = yaml.safe_load((harness.REPO / 'configs' / 'detectors.yaml').read_text())
    raw = yaml.safe_dump(generated(tree.dataset, grid, detectors), sort_keys=False)
    # Artifacts and the reference tracks live in the scratch cache.
    config = yaml.safe_load(raw.replace('/polyis-cache/', f'{tree.cache}/'))
    config['output_root'] = str(output_root)
    # Every tolerance runs (no dedup), so each script output has its own counterpart.
    config['params'].update(GRID, dedup_pruning=False)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['caldot1-y05', 'jnc0'])
def test_profiling_sweep_matches_the_scripts(dataset, tmp_path):
    harness.require_detector(dataset)
    tree = harness.make_tree(tmp_path, dataset, 'valid', GRID)
    output_root = tmp_path / 'profiling-out'
    # The reference tracks the evaluation step scores against.
    harness.reference(tree)
    config = str(sweep_config(tree, tmp_path / 'sweep.yaml', output_root))
    # The sweep first, saving every stage; then its evaluation step.
    tree.run('-m', 'profiling.main', '--config', config, '--save')
    tree.run('-m', 'profiling.evaluate', '--config', config, '--procs', '4')
    # Then the scripts, p022 through the mirror.
    harness.scripts(tree, 'p020_exec_classify')
    harness.mirror_p022(tree, output_root / dataset / 'execution', tmp_path / 'mirror.json')
    harness.scripts(tree, 'p030_exec_compress', 'p040_exec_detect', 'p050_exec_uncompress', 'p060_exec_track')

    from experiments.tools.profiling.compare_scripts import compare_with_scripts, profiling_hota
    report = compare_with_scripts(str(output_root), dataset, str(tree.cache), harness.REFERENCE_FILE,
                                  ours_hota=profiling_hota(str(output_root), dataset))
    mirror = json.loads((tmp_path / 'mirror.json').read_text())
    (tmp_path / 'report.json').write_text(json.dumps({'mirror': mirror, **report}, indent=1) + '\n')
    stages, totals = report['stages'], report['totals']
    print(json.dumps({'totals': totals, 'mirror_copied': sum(r['copied'] for r in mirror)}, indent=1))

    videos = len(tree.videos)
    # Every configuration ran and saved: 2 videos x 40 pack settings, and 48 configurations (tracks) per video.
    assert len(stages['033_compressed_frames']) == videos * 2 * 2 * 2 * (1 + 2 * 2)
    assert len(stages['060_uncompressed_tracks']) == videos * 2 * 2 * 2 * (2 + 2 * 2)
    # Each run classifies at its own sample rate with p020's batches, so scores are exact at every rate.
    assert all(r['equal'] for r in stages['020_relevancy']), [r for r in stages['020_relevancy'] if not r['equal']][:3]
    for stage in ('022_pruned_polyominoes', '033_compressed_frames'):
        assert all(r['equal'] for r in stages[stage]), [r for r in stages[stage] if not r['equal']][:3]
    assert all(r['equal_in_order'] for r in stages['050_uncompressed_detections'])
    assert totals['060_uncompressed_tracks']['text_identical'] == totals['060_uncompressed_tracks']['rows']
    assert totals['hota_per_video']['equal'] == totals['hota_per_video']['compared'] == videos * 48
