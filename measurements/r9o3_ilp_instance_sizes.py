#!/usr/local/bin/python
"""R9.O3: size of the polyomino-pruning ILP that p022 solves, counted on the ILP's input.

scripts/p022_exec_prune_polyominoes.py solves one ILP per clip and configuration (an instance). Its
input is built from the classifier scores in 020_relevancy/{classifier}_{tilesize}_{sample_rate}/score/
score.jsonl: each frame's tile scores are binarized at the configuration's relevance threshold r and
grouped into 4-connected polyominoes by polyis.pack.adapters.group_tiles_all with p022's
TILEPADDING_MODE (0, no padding). The ILP (polyis/sample/ilp/c/gurobi.pyx) has one binary keep/prune
variable per polyomino per frame. For every tile covered in at least two frames it adds two equality
constraints forcing the tile's first and last covering polyominoes, and for each consecutive pair of
covering frames either two equality constraints (when the gap exceeds the tile's maximum sampling
distance, profiled in track_rate/{tracker}_{tilesize}/max_rate_table.npy and divided by the sample
rate) or one window constraint (the covering polyominoes within that distance sum to at least the
current one). This script rebuilds that input with p022's own functions and counts variables,
constraint rows, and nonzeros exactly as the C code adds them (before presolve, duplicates included),
plus the distinct rows, without calling Gurobi or writing the cache.

The feedback repo's ilp_instance_sizes.py counted 4-connected components in 022_pruned_polyominoes
score.jsonl, which holds the ILP's kept output. Those components are the kept polyominoes, so its
"variables" were the solution size, not the instance size; checks.json recounts its sample both ways.

The unit is an instance, a (clip, ILP configuration) pair. Pooled statistics weight datasets by
test videos x configurations and mix sample rates, so the per dataset x sample rate rows and the
operating points are the ones to cite. On test the instances are the validated (Pareto)
configurations that use the ILP (tracking accuracy threshold set); every test video must have all of
them, and test outputs outside that set are excluded. On validation they are every configuration p022
left in the cache, which is not a complete video x configuration grid everywhere (checks.json
coverage). Configurations that differ only in packing settings (tile padding, canvas scale) share
one ILP instance and are counted once. Nothing is sampled.

The binarization differs between code versions. p022 as committed since 2026-05-22 keeps
score > r * 255, but the April 2026 outputs that make up most of the cache keep score >= floor(r * 255)
(tiles scoring exactly 63, 127, or 191 at r = 0.25, 0.5, 0.75), and that p022 version was never
committed. Every instance is therefore rebuilt under both rules (RULES), and the one its saved
selection is consistent with is reported; rule_effect in checks.json says how much the rules differ.

Validation, per instance: the saved pruned grid must have the same frames, contain no tile outside the
reconstructed polyominoes, split none of them, and satisfy every reconstructed constraint, unless the
solver returned no solution (empty selection although some variable is forced). Its 4-connected
components, counted independently, must equal the kept polyominoes. The 020 scores and max-rate
tables must predate the 022 output. p022's worker log, for the runs whose logged timings still match
the cached runtime.jsonl, gives p022's own polyomino and selection counts to compare.

Timings are p022's runtime.jsonl, read by r9o3_ilp_timings.load_runs (strict: one build_ilp and one
solve_ilp per file, no negative time) with p022's default budget read from its source. build_ilp starts
before the Gurobi environment is created and started, and solve_ilp ends after the solution is read and
the model and environment are freed (gurobi.pyx), so both include fixed Gurobi overheads. A solve at or
above its budget is counted as reaching it; that is an estimate, since Gurobi's status is not logged.

Writes measurements-results/r9o3_ilp_instance_sizes/:
  summary.json  per split, per dataset (and pooled, and by sample rate): tile grid, frames per clip,
                polyominoes per frame, variables, constraints, and nonzeros per instance, ILP timings,
                their correlation (pooled and within dataset x sample rate) and binned medians against
                the timings, budget sensitivity, and the instances at each dataset's operating points
  configs.json  per split, dataset, and ILP configuration: median and max sizes and timings over its clips
  checks.json   validation against p022's outputs and log, input freshness, the rule decided per
                instance (and per day written), the size change under the current rule, coverage of
                the video x configuration grid, and the feedback repo's sample recounted
"""
import argparse
import itertools
import multiprocessing as mp
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from measurements.common import (DATASET_LABELS, fastest_within, load_test_tradeoff, query_dicts, results_dir,
                                 write_json)
from measurements.r9o3_ilp_timings import VARIANT_PATTERN, VIDEO_EXTENSIONS, default_time_limit, load_points, load_runs
from polyis.io import cache, store
from polyis.pack.adapters import group_tiles_all
from polyis.utilities import (build_param_str, load_classification_results, load_pruned_classification_results,
                              parse_execution_param_str)
from scripts.p022_exec_prune_polyominoes import _ACCURACY_THRESHOLD_TO_IDX, TILEPADDING_MODE, load_max_sampling_rate


# Splits p022 runs on, as store videoset names.
SPLITS = ('test', 'valid')
# Relevance binarization rules. 'current' is p022 as committed since 2026-05-22 (score > r * 255).
# 'floor' (score >= floor(r * 255)) is what the April 2026 outputs in the cache match: they keep tiles
# scoring exactly 63, 127, or 191 at r = 0.25, 0.5, 0.75, which 'current' drops. The rule of each saved
# output is decided by which rebuilt instance its selection is consistent with.
RULES = {'current': lambda scores, r: scores > r * 255,
         'floor': lambda scores, r: scores >= np.floor(r * 255)}
# Where p022's ProgressBar workers write their stdout.
P022_LOG_DIR = Path(cache.CACHE_DIR) / 'LOGS' / 'p022_exec_prune_polyominoes'
# p022 prints build_ilp and solve_ilp with one decimal, so a current run is within half a unit (plus float slack).
LOG_TOLERANCE_MS = 0.05 + 1e-9
# A solve reaching its budget (p022's `time_limit or default`, in seconds): an estimate, Gurobi's status is not logged.
AT_BUDGET = 'solve_ms >= 1000 * budget_s'
# Multiples of the budget for the at-budget sensitivity: solve_ilp also times solution readout and env release.
BUDGET_MULTIPLES = (0.95, 1.0, 1.05, 1.1)
# Smallest dataset x sample rate stratum whose within-stratum correlation enters the summary.
MIN_STRATUM = 30
# A saved selection is consistent with a rebuilt instance when it keeps only whole input polyominoes and
# either satisfies every constraint or is empty although a variable is forced (no solution found).
CONSISTENT = ('(tiles_outside_input = 0 AND split_polyominoes = 0 AND '
              '((kept = 0 AND forced_variables > 0) OR (forced_dropped = 0 AND window_violations = 0)))')
# Majority of the decided rules ('floor' or 'current') in a group; NULL on a tie or when none is decided.
MAJORITY = ("CASE WHEN count(*) FILTER (rule_match = 'floor') > count(*) FILTER (rule_match = 'current') THEN 'floor' "
            "WHEN count(*) FILTER (rule_match = 'floor') < count(*) FILTER (rule_match = 'current') THEN 'current' END")
# Lines of a p022 worker log: the task header, then two lines per video around the ILP.
LOG_HEADER = re.compile(r"Function: process_all \( dataset='(?P<dataset>[^']+)', videoset='(?P<split>[^']+)', "
                        r"videos=\[.*\], classifier='(?P<classifier>[^']+)', tile_size=(?P<tilesize>\d+), "
                        r"sample_rate=(?P<sample_rate>\d+), tracker='(?P<tracker>[^']+)', "
                        r"tracking_accuracy_threshold=(?P<tat>[\d.]+), relevance_threshold=(?P<r>[\d.]+), "
                        r"time_limit=(?P<time_limit>\S+) \)")
LOG_SOLVING = re.compile(r'\[(?P<video>[^\]]+)\] Step 3: Solving ILP \((?P<polyominoes>\d+) polyominoes '
                         r'across (?P<frames>\d+) frames\)')
LOG_DONE = re.compile(r'\[(?P<video>[^\]]+)\] Step 3 done: build_ilp=(?P<build>[\d.]+)ms '
                      r'solve_ilp=(?P<solve>[\d.]+)ms \((?P<selected>\d+) polyominoes selected\)')


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O3: ILP instance sizes on the ILP input, per clip and configuration')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS))
    parser.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--loss', type=float, default=0.05)
    return parser.parse_args()


def decode_grids(entries: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Frames of a 020 or 022 score.jsonl as uint8 [frames, rows, cols] and their frame indices."""
    # p022 cannot build an instance from a file without frames.
    assert entries, 'score file has no frames'
    # Every frame must share one tile grid.
    sizes = {tuple(e['classification_size']) for e in entries}
    assert len(sizes) == 1, f'inconsistent grid sizes {sorted(sizes)}'
    (rows, cols), = sizes
    # Decode all frames' hex at once; each frame is rows * cols bytes in row-major order.
    grids = np.frombuffer(bytes.fromhex(''.join(e['classification_hex'] for e in entries)), dtype=np.uint8)
    # A truncated or padded hex string does not fill the declared grids exactly.
    assert grids.size == len(entries) * rows * cols, 'hex length does not match the grid size'
    return grids.reshape(len(entries), rows, cols), [e['idx'] for e in entries]


def binarize(scores: np.ndarray, relevance_threshold: float, rule: str) -> np.ndarray:
    """p022's bitmaps under a binarization rule (see RULES): 1 for a relevant tile, else 0."""
    # Apply the rule's comparison, as 0/1 instead of p022's 0/255 // 255.
    return RULES[rule](scores, relevance_threshold).astype(np.uint8)


def max_distance(dataset: str, tracker: str, tilesize: int, tracking_accuracy_threshold: float,
                 sample_rate: int) -> np.ndarray:
    """p022's per-tile maximum gap in sampled frames: profiled max rate // sample rate, at least 1."""
    # Load the profiled table at the configuration's accuracy threshold with p022's own loader.
    table = load_max_sampling_rate(dataset, tracker, tilesize, _ACCURACY_THRESHOLD_TO_IDX[tracking_accuracy_threshold])
    # Convert to sampled-frame units and clamp at 1, as p022 does; the C solver reads it as float64.
    return np.maximum(table // sample_rate, 1).astype(np.float64)


def group(bitmaps: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """p022's grouping: polyomino id per frame and tile (-1 where none) and each polyomino's tile count.

    group_tiles clears the bitmap it is given (group_tiles.c), so it gets a copy, as p022's astype makes.
    """
    # Same call and padding mode as p022, on a C-contiguous copy.
    tile_ids, lengths = group_tiles_all(np.array(bitmaps, dtype=np.uint8, order='C'), TILEPADDING_MODE)
    # The Cython function returns a memoryview.
    return np.asarray(tile_ids), lengths


def ilp_structure(tile_ids: np.ndarray, lengths: list[list[int]], max_dist: np.ndarray) -> dict[str, np.ndarray]:
    """Variables and constraints of the ILP, laid out as solve_ilp in polyis/sample/ilp/c/gurobi.pyx adds them.

    Entries are the (tile, frame) pairs a polyomino covers, ordered by tile then frame, the order the
    C code scans. Variable ids follow the C code (frame offset + polyomino id). Returns:
      per_frame    polyominoes (variables) per frame
      offsets      first variable id of each frame
      lengths      tile count of each variable (objective coefficient)
      var          variable id of each entry
      forced       variable of each equality constraint (x = 1), one per constraint, repeats kept
      window       entry index i of each window constraint (sum of window >= x[i])
      window_end   exclusive entry index ending that constraint's window
    """
    # Instance dimensions.
    frames, rows, cols = tile_ids.shape
    # The grouping returns one polyomino list per frame, and p022 requires one distance per tile.
    assert len(lengths) == frames, f'{len(lengths)} polyomino lists for {frames} frames'
    assert max_dist.shape == (rows, cols), f'max distance {max_dist.shape} vs grid {(rows, cols)}'
    # Polyominoes per frame.
    per_frame = np.fromiter(map(len, lengths), dtype=np.int64, count=frames)
    # Each frame's first variable id (frame_offsets in the C code).
    offsets = np.cumsum(per_frame) - per_frame
    # Tile count of every variable, in variable order.
    sizes = np.fromiter(itertools.chain.from_iterable(lengths), dtype=np.int64, count=int(per_frame.sum()))
    # Covered (tile, frame) pairs, tile-major with frames ascending, and each pair's variable.
    flat = tile_ids.reshape(frames, rows * cols)
    tile, frame = np.nonzero(flat.T >= 0)
    var = offsets[frame] + flat[frame, tile]
    # Maximum sampling distance of each entry's tile.
    dist = max_dist.reshape(-1)[tile]
    # First and last entry of each tile's run.
    starts = np.flatnonzero(np.diff(tile, prepend=-1) != 0)
    ends = np.flatnonzero(np.diff(tile, append=-1) != 0)
    # Tiles covered in two or more frames get both endpoints forced.
    multi = ends > starts
    # Consecutive covering frames of one tile.
    pair = np.flatnonzero(tile[1:] == tile[:-1])
    # A gap beyond the distance forces both endpoints (a bridge); otherwise it is a window constraint.
    bridge = frame[pair + 1] > frame[pair] + dist[pair]
    window = pair[~bridge]
    # A window covers later entries of the same tile with frame <= frame + distance (integer frames,
    # so <= floor(frame + distance)); the stride keeps tiles apart in one sorted key.
    stride = frames + int(max_dist.max(initial=1)) + 2
    key = tile * stride + frame
    limit = tile[window] * stride + np.floor(frame[window] + dist[window]).astype(np.int64)
    # Exclusive end of each window in the sorted entries.
    window_end = np.searchsorted(key, limit, side='right')
    # One equality constraint per forced endpoint, in the C code's order of addition.
    forced = np.concatenate([var[starts[multi]], var[ends[multi]], var[pair[bridge]], var[pair[bridge] + 1]])
    return dict(per_frame=per_frame, offsets=offsets, lengths=sizes, var=var, forced=forced,
                window=window, window_end=window_end)


def distinct_windows(s: dict[str, np.ndarray]) -> np.ndarray:
    """The distinct window constraints of ilp_structure as rows [current variable, window variables, -1 padding].

    Tiles of one polyomino that persist across frames add the same window constraint, so this is the
    deduplicated set. Window variables are in ascending order (later frames have larger ids), so equal
    constraints give equal rows.
    """
    # Window variables per constraint: entries window + 1 .. window_end - 1, at least one.
    sizes = s['window_end'] - s['window'] - 1
    # Rows padded to the widest window; -1 is never a variable id.
    rows = np.full((len(sizes), int(sizes.max(initial=0)) + 1), -1, dtype=np.int64)
    # Without window constraints there is nothing to deduplicate.
    if not len(sizes):
        return rows
    # First column: the current variable.
    rows[:, 0] = s['var'][s['window']]
    # Constraint and position of every window variable.
    which = np.repeat(np.arange(len(sizes)), sizes)
    position = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    # Fill each constraint's window variables after the current one.
    rows[which, position + 1] = s['var'][s['window'][which] + 1 + position]
    # Identical rows are one constraint.
    return np.unique(rows, axis=0)


def instance_size(s: dict[str, np.ndarray]) -> dict[str, int]:
    """Counts of the ILP rebuilt by ilp_structure: rows as added to Gurobi, and distinct rows."""
    # A window constraint has its window's variables plus the current one: (end - (i + 1)) + 1.
    window_nonzeros = s['window_end'] - s['window']
    # Distinct variables fixed to 1, i.e. the distinct equality rows.
    forced_variables = len(np.unique(s['forced']))
    # Distinct window rows.
    unique_windows = distinct_windows(s)
    return dict(
        variables=int(s['per_frame'].sum()),
        relevant_tiles=int(s['lengths'].sum()),
        max_polyominoes_per_frame=int(s['per_frame'].max(initial=0)),
        equality_constraints=len(s['forced']),
        window_constraints=len(s['window']),
        constraints=len(s['forced']) + len(s['window']),
        nonzeros=len(s['forced']) + int(window_nonzeros.sum()),
        forced_variables=forced_variables,
        distinct_window_constraints=len(unique_windows),
        distinct_constraints=forced_variables + len(unique_windows),
        distinct_nonzeros=forced_variables + int((unique_windows >= 0).sum()),
    )


def check_solution(s: dict[str, np.ndarray], tile_ids: np.ndarray, pruned: np.ndarray) -> dict[str, int]:
    """How p022's saved pruned grid (nonzero = kept tile) relates to the rebuilt instance."""
    # The saved grid must have the instance's frames and tile grid.
    assert pruned.shape == tile_ids.shape, f'pruned grid {pruned.shape} vs instance {tile_ids.shape}'
    # Tiles inside some input polyomino.
    covered = tile_ids >= 0
    # Saved tiles that belong to an input polyomino.
    kept_tile = (pruned > 0) & covered
    # Variable id of every covered tile.
    var_grid = s['offsets'][:, None, None] + tile_ids
    # Kept tiles of every variable, against its size: a kept polyomino has all of its tiles kept.
    kept_tiles = np.bincount(var_grid[kept_tile], minlength=len(s['lengths']))
    kept = kept_tiles == s['lengths']
    # Kept flag of every entry, and its running count over entries.
    entry_kept = kept[s['var']]
    cumulative = np.concatenate([[0], np.cumsum(entry_kept)])
    # Kept entries inside each window constraint's window.
    window_hits = cumulative[s['window_end']] - cumulative[s['window'] + 1]
    # Distinct forced variables, so that each is checked once.
    forced = np.unique(s['forced'])
    return dict(
        kept=int(kept.sum()),
        tiles_outside_input=int(((pruned > 0) & ~covered).sum()),
        split_polyominoes=int(((kept_tiles > 0) & ~kept).sum()),
        forced_dropped=int((~kept[forced]).sum()),
        window_violations=int((entry_kept[s['window']] & (window_hits == 0)).sum()),
    )


def mtime(path) -> float:
    """Modification time of a file, in seconds since the epoch."""
    # A thin wrapper so every timestamp comes from one place.
    return os.path.getmtime(path)


def measure_video(task: tuple[str, str, str, list[str]]) -> tuple[list[dict], list[tuple[str, np.ndarray]]]:
    """Every ILP instance of one video under each rule: sizes, checks of p022's output, and polyominoes per frame.

    The task is (dataset, split, video, 022 parameter strings). Scores are loaded once per
    (classifier, tile size, sample rate) and grouped once per relevance threshold and rule, since
    the tracker and accuracy threshold change only the constraints. Each row also counts the
    boundary tiles (score exactly floor(r * 255)), the only tiles on which the two rules differ,
    the components of the saved grid, and the modification times of the inputs and the output.
    """
    # Unpack the task.
    dataset, split, video, params = task
    # Decoded 020 scores per source, and polyominoes per grouping, reused across configurations.
    scores: dict[tuple, tuple[np.ndarray, list[int]]] = {}
    groupings: dict[str, tuple[np.ndarray, list[list[int]]]] = {}
    # One row per configuration and rule.
    rows = []
    for param in params:
        # The configuration's parameters from its 022 directory name.
        p = parse_execution_param_str(param)
        # Classifier scores of this video at the configuration's sample rate (p022's input file).
        source = (p['classifier'], p['tilesize'], p['sample_rate'])
        if source not in scores:
            scores[source] = decode_grids(load_classification_results(dataset, video, p['tilesize'], p['classifier'],
                                                                      p['sample_rate']))
        grids, frame_idx = scores[source]
        # p022's saved selection for this instance must cover the same frames.
        pruned, pruned_idx = decode_grids(load_pruned_classification_results(
            dataset, video, p['tilesize'], p['classifier'], p['tracker'], p['tracking_accuracy_threshold'],
            p['relevance_threshold'], p['sample_rate']))
        assert pruned_idx == frame_idx, f'{dataset} {video} {param}: 022 frames differ from 020 frames'
        # The configuration's sampling distances, the same under both rules.
        max_dist = max_distance(dataset, p['tracker'], p['tilesize'], p['tracking_accuracy_threshold'], p['sample_rate'])
        # Tiles scoring exactly floor(r * 255), the only ones the two rules treat differently.
        boundary_tiles = int((grids == np.floor(p['relevance_threshold'] * 255)).sum())
        # Polyominoes of the saved grid, counted independently of the rebuilt instance (the feedback's count).
        saved_components = sum(map(len, group(pruned > 0)[1]))
        # p022's output files and the input files it read.
        output_dir = cache.exec(dataset, 'pruned-polyominoes', video, param, 'score')
        scores_path = cache.exec(dataset, 'relevancy', video, f"{p['classifier']}_{p['tilesize']}_{p['sample_rate']}",
                                 'score', 'score.jsonl')
        table_path = cache.index(dataset, 'track_rates', f"{p['tracker']}_{p['tilesize']}", 'max_rate_table.npy')
        # The UTC day p022 wrote this output, which groups outputs of one code version.
        runtime_path = output_dir / 'runtime.jsonl'
        written = datetime.fromtimestamp(mtime(runtime_path), timezone.utc).date().isoformat()
        # Modification times for the input freshness check.
        times = dict(scores_mtime=mtime(scores_path), table_mtime=mtime(table_path),
                     output_mtime=mtime(output_dir / 'score.jsonl'))
        for rule in RULES:
            # Polyominoes at the relevance threshold, shared by every tracker and accuracy threshold.
            gid = (f"{dataset}/{split}/{video}/{p['classifier']}_{p['tilesize']}_{p['sample_rate']}"
                   f"/r{p['relevance_threshold']}/{rule}")
            if gid not in groupings:
                groupings[gid] = group(binarize(grids, p['relevance_threshold'], rule))
            tile_ids, lengths = groupings[gid]
            # Rebuild the instance.
            s = ilp_structure(tile_ids, lengths, max_dist)
            # Its sizes and the checks of the saved selection against it.
            rows.append(dict(
                dataset=dataset, split=split, video=video, config=param, rule=rule, gid=gid,
                classifier=p['classifier'], tilesize=p['tilesize'], sample_rate=p['sample_rate'], tracker=p['tracker'],
                tracking_accuracy_threshold=p['tracking_accuracy_threshold'],
                relevance_threshold=p['relevance_threshold'], boundary_tiles=boundary_tiles, written=written,
                grid_rows=tile_ids.shape[1], grid_cols=tile_ids.shape[2], frames=tile_ids.shape[0],
                saved_components=saved_components, **times,
                **instance_size(s), **check_solution(s, tile_ids, pruned),
                runtime_path=str(runtime_path)))
    # Polyominoes per frame of every grouping, for frame-level statistics.
    return rows, [(gid, np.fromiter(map(len, lengths), dtype=np.int64, count=len(lengths)))
                  for gid, (_, lengths) in groupings.items()]


def validated_ilp_configs(tetris: pd.DataFrame) -> set[str]:
    """022 parameter strings of the validated Tetris configurations (load_test_tradeoff rows) that run the ILP."""
    # Only configurations with a tracking accuracy threshold run p022.
    ilp = tetris[tetris['tracking_accuracy_threshold'].notna()]
    # Build each one's 022 directory name; packing settings do not enter it, so their variants collapse.
    return {build_param_str(classifier=c, tilesize=int(t), sample_rate=int(s), tracker=k,
                            tracking_accuracy_threshold=float(a), relevance_threshold=float(r))
            for c, t, s, k, a, r in zip(ilp['classifier'], ilp['tilesize'], ilp['sample_rate'], ilp['tracker'],
                                        ilp['tracking_accuracy_threshold'], ilp['relevance_threshold'])}


def video_tasks(dataset: str, split: str, validated: set[str]) -> tuple[list[tuple], dict]:
    """One task per video of the split with the expected 022 outputs, and the coverage of the video x configuration grid.

    An output is complete when it has both score.jsonl and runtime.jsonl. The expected configurations
    are the validated ones on test (every test video must have all of them; other test outputs are
    excluded and reported) and every configuration found on any video on validation (missing pairs
    are reported, not filled in).
    """
    # Videos as p022 lists them.
    videoset_dir = store.dataset(dataset, split)
    assert os.path.isdir(videoset_dir), f'missing videoset {videoset_dir}'
    videos = sorted(f for f in os.listdir(videoset_dir) if f.endswith(VIDEO_EXTENSIONS))
    # Complete outputs per video, and the incomplete ones.
    complete: dict[str, list[str]] = {}
    incomplete: list[str] = []
    for video in videos:
        # Every configuration directory p022 left for this video.
        stage = cache.exec(dataset, 'pruned-polyominoes', video)
        params = sorted(os.listdir(stage)) if stage.is_dir() else []
        # Directories holding both of p022's files.
        complete[video] = [p for p in params
                           if all((stage / p / 'score' / f).is_file() for f in ('score.jsonl', 'runtime.jsonl'))]
        # Directories missing one of them.
        incomplete += [f'{video}/{p}' for p in params if p not in complete[video]]
    # The configurations every video should have.
    expected = validated if split == 'test' else set().union(*map(set, complete.values()))
    # Expected configurations absent per video, and outputs outside the expected set (test only).
    missing = {v: sorted(expected - set(ps)) for v, ps in complete.items() if expected - set(ps)}
    excluded = [f'{v}/{p}' for v, ps in complete.items() for p in ps if p not in expected]
    # A validated configuration without an instance on some test video would bias the test summary.
    if split == 'test':
        assert not missing, f'{dataset}: validated ILP configurations missing on test: {sorted(missing.items())[:3]}'
    # One task per video with at least one expected output.
    tasks = [(dataset, split, v, [p for p in ps if p in expected]) for v, ps in complete.items()
             if any(p in expected for p in ps)]
    coverage = dict(
        videos=len(videos), videos_with_outputs=len(tasks), configs=len(expected),
        expected_instances=len(videos) * len(expected), instances=sum(len(t[3]) for t in tasks),
        missing_instances=sum(map(len, missing.values())),
        configs_with_missing_videos=len(set().union(*map(set, missing.values()))),
        missing_configs_by_video={v: len(m) for v, m in missing.items()},
        incomplete_outputs=incomplete, excluded_outside_validated=excluded)
    return tasks, coverage


def p022_log_runs(log_dir: Path) -> list[dict]:
    """ILP runs recorded in p022's worker logs: configuration, video, polyominoes, frames, selection, timings.

    The log is a few dozen lines of state (a header applies to the video lines after it), so it is
    parsed line by line; this is the necessary loop, not data analysis.
    """
    # Runs parsed from every worker log.
    runs = []
    for path in sorted(log_dir.glob('worker_*.stdout')):
        # The current task header, and the 'Solving' line of each video still waiting for its 'done' line.
        header, solving = None, {}
        for line in open(path):
            if m := LOG_HEADER.search(line):
                # A new task: its configuration applies to the video lines that follow.
                header = m.groupdict()
                solving = {}
            elif header and (m := LOG_SOLVING.search(line)):
                # The ILP of this video started.
                solving[m['video']] = m
            elif header and (m := LOG_DONE.search(line)) and m['video'] in solving:
                # The ILP of this video finished: record both lines' numbers.
                before = solving.pop(m['video'])
                runs.append(dict(
                    dataset=header['dataset'], split=header['split'], video=m['video'],
                    config=build_param_str(classifier=header['classifier'], tilesize=int(header['tilesize']),
                                           sample_rate=int(header['sample_rate']), tracker=header['tracker'],
                                           tracking_accuracy_threshold=float(header['tat']),
                                           relevance_threshold=float(header['r'])),
                    time_limit=header['time_limit'], log=path.name,
                    logged_polyominoes=int(before['polyominoes']), logged_frames=int(before['frames']),
                    logged_selected=int(m['selected']), logged_build_ms=float(m['build']),
                    logged_solve_ms=float(m['solve'])))
    return runs


def load_tables(con: duckdb.DuckDBPyConnection, rows: list[dict], groupings: list[tuple[str, np.ndarray]],
                validated: dict[str, set[str]], default_limit: float):
    """Tables instances_all (every rule), instances (the rule each saved output was solved with),
    frames(gid, polyominoes), runs (r9o3_ilp_timings.load_runs over the instances' runtime.jsonl),
    and the joined view ilp.

    rule_match says how the rule was decided: 'identical' (no boundary tile, so both rules give the
    same instance, and it is consistent), 'floor' or 'current' (only that rule's instance is consistent
    with the saved selection), 'ambiguous' (both consistent although the instances differ), 'neither'.
    The last two take the majority decided rule among outputs of the same dataset and split written
    the same UTC day (one p022 code version), else of the dataset and split, else 'current'; a tied
    majority counts as none.
    """
    # An empty sweep would make every aggregate meaningless.
    assert rows, 'no ILP instances found'
    # Instance rows (pandas only carries them into DuckDB).
    con.register('instance_rows', pd.DataFrame(rows))
    # Dataset labels.
    con.execute('CREATE OR REPLACE TABLE labels (dataset VARCHAR, label VARCHAR)')
    con.executemany('INSERT INTO labels VALUES (?, ?)', list(DATASET_LABELS.items()))
    # The validated configurations.
    con.execute('CREATE OR REPLACE TABLE validated (dataset VARCHAR, config VARCHAR)')
    pairs = [(d, c) for d, cs in validated.items() for c in sorted(cs)]
    # DuckDB rejects an empty executemany; a dataset may have no validated ILP configuration.
    if pairs:
        con.executemany('INSERT INTO validated VALUES (?, ?)', pairs)
    # Every (instance, rule) with its label, validated flag, and whether the saved selection is consistent with it.
    con.execute(f"""
        CREATE OR REPLACE TABLE instances_all AS
        SELECT r.*, l.label, v.config IS NOT NULL AS validated,
               r.kept = 0 AND r.forced_variables > 0 AS no_solution, {CONSISTENT} AS consistent
        FROM instance_rows r JOIN labels l ON l.dataset = r.dataset
        LEFT JOIN validated v ON v.dataset = r.dataset AND v.config = r.config
    """)
    # Decide each instance's rule from the consistency of its saved selection under each rule.
    con.execute(f"""
        CREATE OR REPLACE TABLE instances AS
        WITH per_instance AS (
            SELECT dataset, split, video, config, any_value(boundary_tiles) AS boundary_tiles,
                   any_value(written) AS written,
                   bool_or(consistent) FILTER (WHERE rule = 'current') AS current_ok,
                   bool_or(consistent) FILTER (WHERE rule = 'floor') AS floor_ok
            FROM instances_all GROUP BY dataset, split, video, config
        ),
        matched AS (
            SELECT *, CASE WHEN boundary_tiles = 0 AND current_ok THEN 'identical'
                           WHEN floor_ok AND NOT current_ok THEN 'floor'
                           WHEN current_ok AND NOT floor_ok THEN 'current'
                           WHEN current_ok AND floor_ok THEN 'ambiguous'
                           ELSE 'neither' END AS rule_match
            FROM per_instance
        ),
        day_majority AS (SELECT dataset, split, written, {MAJORITY} AS day_rule FROM matched GROUP BY ALL),
        split_majority AS (SELECT dataset, split, {MAJORITY} AS split_rule FROM matched GROUP BY ALL),
        chosen AS (
            SELECT m.*, CASE WHEN rule_match IN ('floor', 'current') THEN rule_match
                             WHEN rule_match = 'identical' THEN 'current'
                             ELSE coalesce(d.day_rule, s.split_rule, 'current') END AS chosen_rule
            FROM matched m LEFT JOIN day_majority d USING (dataset, split, written)
            LEFT JOIN split_majority s USING (dataset, split)
        )
        SELECT a.*, c.rule_match
        FROM instances_all a JOIN chosen c USING (dataset, split, video, config)
        WHERE a.rule = c.chosen_rule
    """)
    # One row per frame of every grouping.
    con.register('frame_rows', pd.DataFrame(dict(
        gid=np.repeat([g for g, _ in groupings], [len(c) for _, c in groupings]),
        polyominoes=np.concatenate([c for _, c in groupings]))))
    con.execute('CREATE OR REPLACE TABLE frames AS SELECT * FROM frame_rows')
    # p022's recorded times per output, parsed and validated as in r9o3_ilp_timings (one parser for both).
    # Its video listing is only used for clip positions, which this script does not report.
    paths = sorted({r['runtime_path'] for r in rows})
    listing = sorted({(r['dataset'], r['split'], r['video']) for r in rows})
    load_runs(con, paths, default_limit, listing)
    # Instances with their output's timings.
    con.execute("""
        CREATE OR REPLACE VIEW ilp AS
        SELECT i.*, r.time_limit, r.budget_s, r.build_ms, r.solve_ms, r.stage_ms
        FROM instances i JOIN runs r ON r.dataset = i.dataset AND r.split = i.split AND r.video = i.video
                                    AND r.param = i.config
    """)
    # Every instance must have exactly one run.
    counts = query_dicts(con, 'SELECT (SELECT count(*) FROM instances) AS instances, (SELECT count(*) FROM ilp) AS joined')
    assert counts[0]['instances'] == counts[0]['joined'], f'instances without exactly one p022 run: {counts}'


# Instance-level aggregates shared by the summary, per-configuration, and operating-point reports.
SIZE_AGGREGATES = f"""
    count(*) AS instances, count(DISTINCT (dataset, video)) AS videos, count(DISTINCT (dataset, config)) AS configs,
    count(*) FILTER (variables = 0) AS empty_instances,
    min(frames) AS frames_min, median(frames) AS frames_median, max(frames) AS frames_max,
    min(variables) AS variables_min, median(variables) AS variables_median,
    quantile_disc(variables, 0.95) AS variables_p95, max(variables) AS variables_max,
    median(constraints) AS constraints_median, quantile_disc(constraints, 0.95) AS constraints_p95,
    max(constraints) AS constraints_max,
    median(distinct_constraints) AS distinct_constraints_median,
    quantile_disc(distinct_constraints, 0.95) AS distinct_constraints_p95, max(distinct_constraints) AS distinct_constraints_max,
    median(distinct_constraints / nullif(constraints, 0)) AS distinct_constraint_share_median,
    median(window_constraints) AS window_constraints_median, median(equality_constraints) AS equality_constraints_median,
    median(nonzeros) AS nonzeros_median, max(nonzeros) AS nonzeros_max, max(distinct_nonzeros) AS distinct_nonzeros_max,
    median(forced_variables / nullif(variables, 0)) AS forced_share_median,
    median(kept / nullif(variables, 0)) AS kept_share_median,
    median(relevant_tiles / nullif(variables, 0)) AS tiles_per_polyomino_median,
    median(build_ms) AS build_ms_median, quantile_cont(build_ms, 0.95) AS build_ms_p95, max(build_ms) AS build_ms_max,
    median(solve_ms) AS solve_ms_median, quantile_cont(solve_ms, 0.95) AS solve_ms_p95, max(solve_ms) AS solve_ms_max,
    median(build_ms + solve_ms) AS ilp_ms_median, max(build_ms + solve_ms) AS ilp_ms_max,
    median((build_ms + solve_ms) / nullif(stage_ms, 0)) AS ilp_share_of_stage_median,
    count(*) FILTER ({AT_BUDGET}) AS solve_at_budget, avg(CAST({AT_BUDGET} AS DOUBLE)) AS solve_at_budget_share
"""


def size_summary(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Instance sizes and timings per split at four levels: dataset, dataset x sample rate, pooled, pooled x sample rate.

    Rows have level ('dataset', 'dataset_sample_rate', 'all', 'all_sample_rate'); polyominoes per frame
    are pooled over every instance's frames under its chosen rule (instances sharing a grouping each
    count their frames). polyominoes_per_frame_bound is ceil(rows * cols / 2), the most 4-connected
    polyominoes a frame can hold (a checkerboard).
    """
    # The four grouping levels, always within a split.
    sets = 'GROUPING SETS ((split, dataset), (split, dataset, sample_rate), (split), (split, sample_rate))'
    return query_dicts(con, f"""
        WITH sizes AS (
            SELECT split, dataset, sample_rate, grouping(dataset, sample_rate) AS g,
                   CASE WHEN grouping(dataset) = 0 THEN any_value(label) END AS label,
                   count(DISTINCT concat(grid_rows, 'x', grid_cols)) AS grids,
                   CASE WHEN count(DISTINCT concat(grid_rows, 'x', grid_cols)) = 1
                        THEN any_value(concat(grid_rows, 'x', grid_cols)) END AS grid,
                   CASE WHEN count(DISTINCT grid_rows * grid_cols) = 1
                        THEN any_value(grid_rows * grid_cols) END AS tiles_per_frame,
                   CASE WHEN count(DISTINCT grid_rows * grid_cols) = 1
                        THEN any_value((grid_rows * grid_cols + 1) // 2) END AS polyominoes_per_frame_bound,
                   {SIZE_AGGREGATES}
            FROM ilp GROUP BY {sets}
        ),
        per_frame AS (
            SELECT split, dataset, sample_rate, grouping(dataset, sample_rate) AS g,
                   avg(f.polyominoes) AS polyominoes_per_frame_mean,
                   quantile_disc(f.polyominoes, 0.95) AS polyominoes_per_frame_p95,
                   max(f.polyominoes) AS polyominoes_per_frame_max,
                   avg(CAST(f.polyominoes = 0 AS DOUBLE)) AS empty_frame_share
            FROM instances JOIN frames f USING (gid) GROUP BY {sets}
        )
        SELECT CASE s.g WHEN 0 THEN 'dataset_sample_rate' WHEN 1 THEN 'dataset' WHEN 2 THEN 'all_sample_rate'
                        ELSE 'all' END AS level,
               s.*, p.polyominoes_per_frame_mean, p.polyominoes_per_frame_p95, p.polyominoes_per_frame_max,
               p.empty_frame_share
        FROM sizes s JOIN per_frame p
          ON s.split = p.split AND s.g = p.g AND s.dataset IS NOT DISTINCT FROM p.dataset
         AND s.sample_rate IS NOT DISTINCT FROM p.sample_rate
        ORDER BY s.split, s.g, s.dataset, s.sample_rate
    """)


def avg_rank(column: str, partition: str) -> str:
    """SQL for the average rank (ties share the mean of their ranks) of a column within a partition."""
    # rank() gives a tie its first rank; adding half the tie size minus one gives the mean rank.
    return (f'rank() OVER (PARTITION BY {partition} ORDER BY {column}) '
            f'+ (count(*) OVER (PARTITION BY {partition}, {column}) - 1) / 2.0')


def finite(expr: str) -> str:
    """SQL for expr with NaN (DuckDB's corr of a constant column) replaced by NULL."""
    # JSON has no NaN, and a NaN would poison weighted means.
    return f'CASE WHEN isnan({expr}) THEN NULL ELSE {expr} END'


# Size and timing columns related by correlations().
CORRELATION_PAIRS = tuple(itertools.product(('variables', 'constraints', 'distinct_constraints'), ('build_ms', 'solve_ms')))


def correlation_sql(table: str, partition: str) -> str:
    """SQL for Pearson and Spearman correlations of instance sizes with build and solve times, per partition."""
    # Average ranks of each size and timing column within the partition.
    columns = sorted({c for pair in CORRELATION_PAIRS for c in pair})
    ranks = ', '.join(f'{avg_rank(c, partition)} AS rank_{c}' for c in columns)
    # Both correlations of every pair, NaN as NULL.
    pairs = ', '.join(f'{finite(f"corr({x}, {y})")} AS pearson_{x}_{y}, '
                      f'{finite(f"corr(rank_{x}, rank_{y})")} AS spearman_{x}_{y}' for x, y in CORRELATION_PAIRS)
    return f"""
        SELECT {partition}, count(*) AS instances, {pairs}
        FROM (SELECT *, {ranks} FROM {table})
        GROUP BY {partition}
    """


def correlations(con: duckdb.DuckDBPyConnection, table: str, partition: str) -> list[dict]:
    """Rows of correlation_sql, ordered by the partition."""
    # Run the correlation query in a stable order.
    return query_dicts(con, f'{correlation_sql(table, partition)} ORDER BY {partition}')


def within_strata(con: duckdb.DuckDBPyConnection, table: str, strata: str, min_instances: int) -> list[dict]:
    """Per split: instance-weighted mean, min, and max of the within-stratum Spearman correlations.

    Only strata with at least min_instances instances enter; a stratum whose correlation is undefined
    (a constant column) is left out of that correlation's mean. strata must start with split.
    """
    # Weighted mean, min, and max of each Spearman column over the strata.
    columns = [f'spearman_{x}_{y}' for x, y in CORRELATION_PAIRS]
    stats = ', '.join(f'sum(instances * {c}) / sum(instances) FILTER (WHERE {c} IS NOT NULL) AS {c}_weighted_mean, '
                      f'min({c}) AS {c}_min, max({c}) AS {c}_max' for c in columns)
    return query_dicts(con, f"""
        SELECT split, count(*) AS strata, sum(instances) AS instances, {stats}
        FROM ({correlation_sql(table, strata)}) WHERE instances >= $min_instances
        GROUP BY split ORDER BY split
    """, {'min_instances': min_instances})


def binned_timings(con: duckdb.DuckDBPyConnection, table: str, bins: int) -> list[dict]:
    """Per split, timing medians in `bins` equal-count bins of variables per instance, with each bin's mix.

    Ties on (variables, constraints) are broken by dataset, video, and configuration, so the bins are
    deterministic. The bins pool datasets and sample rates, which confound size and time; the
    dataset and sample-rate counts per bin show the mix.
    """
    return query_dicts(con, f"""
        SELECT split, bin, count(*) AS instances, min(variables) AS variables_min, max(variables) AS variables_max,
               median(constraints) AS constraints_median,
               median(build_ms) AS build_ms_median, quantile_cont(build_ms, 0.95) AS build_ms_p95,
               median(solve_ms) AS solve_ms_median, quantile_cont(solve_ms, 0.95) AS solve_ms_p95,
               avg(CAST({AT_BUDGET} AS DOUBLE)) AS solve_at_budget_share,
               histogram(sample_rate) AS instances_by_sample_rate, histogram(label) AS instances_by_dataset
        FROM (SELECT *, ntile({bins}) OVER (PARTITION BY split ORDER BY variables, constraints, dataset, video, config)
                        AS bin
              FROM {table})
        GROUP BY split, bin ORDER BY split, bin
    """)


def budget_sensitivity(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per split and pooled: instances whose solve time reaches each multiple of its budget (BUDGET_MULTIPLES)."""
    # One count and share per multiple.
    cells = ', '.join(f"count(*) FILTER (solve_ms >= {m} * 1000 * budget_s) AS at_{round(100 * m)}pct, "
                      f"avg(CAST(solve_ms >= {m} * 1000 * budget_s AS DOUBLE)) AS share_at_{round(100 * m)}pct"
                      for m in BUDGET_MULTIPLES)
    return query_dicts(con, f"""
        SELECT split, count(*) AS instances, {cells}
        FROM ilp GROUP BY GROUPING SETS ((split), ()) ORDER BY split NULLS LAST
    """)


def config_summary(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Instance sizes and timings per split, dataset, and ILP configuration (a subset of the summary fields)."""
    return query_dicts(con, f"""
        SELECT split, dataset, config, any_value(validated) AS validated, count(*) AS instances,
               median(frames) AS frames_median, median(variables) AS variables_median,
               max(variables) AS variables_max, median(constraints) AS constraints_median,
               max(constraints) AS constraints_max, median(distinct_constraints) AS distinct_constraints_median,
               median(forced_variables / nullif(variables, 0)) AS forced_share_median,
               median(kept / nullif(variables, 0)) AS kept_share_median,
               median(build_ms) AS build_ms_median, median(solve_ms) AS solve_ms_median,
               avg(CAST({AT_BUDGET} AS DOUBLE)) AS solve_at_budget_share
        FROM ilp GROUP BY split, dataset, config ORDER BY split, dataset, config
    """)


def operating_point_sizes(con: duckdb.DuckDBPyConnection, tradeoffs: dict, loss: float) -> dict:
    """Per dataset label, the test instances of the operating points r9o3_ilp_timings reports.

    operating_point is the fastest validated configuration within `loss` of the reference test HOTA,
    and fastest_pruning_point the fastest such configuration that runs the ILP. A point that does not
    prune has no instance; a pruning point must have one on every test video.
    """
    # Every validated configuration with its throughput, HOTA, and whether it prunes.
    load_points(con, tradeoffs)
    # Fastest within the bound, among all configurations and among pruning ones.
    chosen = fastest_within(con, 'points', loss, ['tetris', 'tetris_pruned'])
    # The selections that meet the bound, named by their role.
    con.execute('CREATE OR REPLACE TABLE chosen_points (dataset VARCHAR, point VARCHAR, variant VARCHAR)')
    selected = [(r['dataset'], 'operating_point' if r['system'] == 'tetris' else 'fastest_pruning_point', r['config'])
                for r in chosen if r['meets_bound']]
    # Insert them only when there are any (executemany needs at least one row).
    if selected:
        con.executemany('INSERT INTO chosen_points VALUES (?, ?, ?)', selected)
    # Each selection's p022 parameter string (packing settings dropped) and its test instances, if it prunes.
    rows = query_dicts(con, f"""
        WITH c AS (
            SELECT c.dataset AS point_dataset, c.point, c.variant, p.prunes, p.fps, p.accuracy AS hota,
                   regexp_replace(c.variant, '{VARIANT_PATTERN}', '\\1_\\2') AS ilp_config
            FROM chosen_points c JOIN points p ON p.dataset = c.dataset AND p.system = 'tetris' AND p.config = c.variant
        ),
        sizes AS (
            SELECT point_dataset, point, {SIZE_AGGREGATES}
            FROM c JOIN ilp i ON i.split = 'test' AND i.dataset = c.point_dataset AND i.config = c.ilp_config
            GROUP BY point_dataset, point
        )
        SELECT c.point_dataset AS dataset, c.point, c.variant, c.ilp_config, c.prunes, c.fps, c.hota,
               s.* EXCLUDE (point_dataset, point)
        FROM c LEFT JOIN sizes s ON s.point_dataset = c.point_dataset AND s.point = c.point
        ORDER BY dataset, c.point
    """)
    # Test videos per dataset, which every pruning point must cover.
    videos = dict(con.execute("SELECT dataset, count(DISTINCT video) FROM instances WHERE split = 'test' GROUP BY ALL")
                  .fetchall())
    bad = [r for r in rows if r['prunes'] and r['videos'] != videos.get(r['dataset'])]
    assert not bad, f'pruning operating points without an instance on every test video: {bad}'
    # Bounds and unmet selections per dataset, keyed by the manuscript label.
    out = {}
    for r in chosen:
        entry = out.setdefault(DATASET_LABELS[r['dataset']], dict(dataset=r['dataset'], hota_bound=r['bound']))
        # Placeholders, replaced below by the met selections.
        entry.setdefault('operating_point' if r['system'] == 'tetris' else 'fastest_pruning_point',
                         dict(meets_bound=False, best_hota=r['best_accuracy']))
    # The met selections with their sizes.
    for r in rows:
        out[DATASET_LABELS[r['dataset']]][r['point']] = dict(meets_bound=True, **rounded([r])[0])
    return out


def solution_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per split and dataset, how each instance's rule was decided and whether its saved selection is consistent.

    saved_components (4-connected components of the saved grid, counted without the rebuilt instance)
    must equal kept, which also shows kept <= variables independently of the rebuild.
    """
    return query_dicts(con, """
        SELECT split, dataset, count(*) AS instances,
               count(*) FILTER (WHERE rule_match = 'identical') AS rule_identical,
               count(*) FILTER (WHERE rule_match = 'floor') AS rule_floor,
               count(*) FILTER (WHERE rule_match = 'current') AS rule_current,
               count(*) FILTER (WHERE rule_match = 'ambiguous') AS rule_ambiguous,
               count(*) FILTER (WHERE rule_match = 'neither') AS rule_neither,
               count(*) FILTER (WHERE saved_components <> kept) AS components_differ_from_kept,
               count(*) FILTER (WHERE saved_components > variables) AS components_above_variables,
               count(*) FILTER (WHERE tiles_outside_input > 0) AS with_tiles_outside_input,
               count(*) FILTER (WHERE split_polyominoes > 0) AS with_split_polyominoes,
               count(*) FILTER (WHERE no_solution) AS no_solution,
               count(*) FILTER (WHERE NOT no_solution AND forced_dropped > 0) AS with_forced_dropped,
               count(*) FILTER (WHERE NOT no_solution AND window_violations > 0) AS with_window_violations,
               count(*) FILTER (WHERE consistent) AS consistent,
               max(kept / nullif(variables, 0)) AS kept_share_max
        FROM instances GROUP BY split, dataset ORDER BY split, dataset
    """)


def input_freshness(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per split and dataset, instances whose 020 scores or max-rate table changed after p022 wrote the output.

    Zero means the rebuilt inputs are the files p022 read (or identical rewrites of them).
    """
    return query_dicts(con, """
        SELECT split, dataset, count(*) AS instances,
               count(*) FILTER (scores_mtime > output_mtime) AS scores_newer_than_output,
               count(*) FILTER (table_mtime > output_mtime) AS table_newer_than_output,
               CAST(to_timestamp(min(table_mtime)) AS VARCHAR) AS table_written_first,
               CAST(to_timestamp(max(table_mtime)) AS VARCHAR) AS table_written_last,
               CAST(to_timestamp(min(output_mtime)) AS VARCHAR) AS output_written_first,
               CAST(to_timestamp(max(output_mtime)) AS VARCHAR) AS output_written_last
        FROM instances GROUP BY split, dataset ORDER BY split, dataset
    """)


def rule_by_day(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """How each instance's rule was decided, per dataset, split, and the UTC day p022 wrote the output."""
    return query_dicts(con, """
        SELECT dataset, split, written, count(*) AS instances,
               count(*) FILTER (WHERE rule_match = 'floor') AS floor,
               count(*) FILTER (WHERE rule_match = 'current') AS current,
               count(*) FILTER (WHERE rule_match = 'identical') AS identical,
               count(*) FILTER (WHERE rule_match = 'ambiguous') AS ambiguous,
               count(*) FILTER (WHERE rule_match = 'neither') AS neither,
               list(DISTINCT rule ORDER BY rule) AS reported_rules,
               list(DISTINCT config ORDER BY config)[:5] AS configs_first5, count(DISTINCT config) AS configs
        FROM instances GROUP BY dataset, split, written ORDER BY dataset, split, written
    """)


def rule_effect(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per split and dataset, how much the current p022 rule changes the instance relative to the floor rule."""
    return query_dicts(con, """
        WITH paired AS (
            SELECT c.split, c.dataset, c.variables AS current_variables, f.variables AS floor_variables,
                   c.constraints AS current_constraints, f.constraints AS floor_constraints
            FROM instances_all c JOIN instances_all f USING (dataset, split, video, config)
            WHERE c.rule = 'current' AND f.rule = 'floor'
        )
        SELECT split, dataset, count(*) AS instances,
               median(current_variables / nullif(floor_variables, 0) - 1) AS variables_change_median,
               min(current_variables / nullif(floor_variables, 0) - 1) AS variables_change_min,
               median(current_constraints / nullif(floor_constraints, 0) - 1) AS constraints_change_median,
               min(current_constraints / nullif(floor_constraints, 0) - 1) AS constraints_change_min
        FROM paired GROUP BY split, dataset ORDER BY split, dataset
    """)


def log_comparison(con: duckdb.DuckDBPyConnection, runs: list[dict]) -> dict:
    """p022's logged polyomino, frame, and selection counts against the rebuilt instances.

    A logged run is current when both logged timings (printed with one decimal) are within
    LOG_TOLERANCE_MS of the cached runtime.jsonl, so the cached output came from that run; older runs
    are counted but not compared. scope lists which instances the current runs cover.
    """
    # Without a log there is nothing to compare.
    if not runs:
        return dict(log_dir=str(P022_LOG_DIR), runs=0, current=0, scope=[], rows=[])
    # Join each logged run to its rebuilt instance and decide whether the cache holds that run's output.
    con.register('log_run_rows', pd.DataFrame(runs))
    con.execute("""
        CREATE OR REPLACE TABLE log_runs AS
        SELECT l.*, i.variables, i.frames, i.kept, i.rule, i.written, i.build_ms, i.solve_ms,
               coalesce(abs(i.build_ms - l.logged_build_ms) <= $tol
                        AND abs(i.solve_ms - l.logged_solve_ms) <= $tol, false) AS current
        FROM log_run_rows l LEFT JOIN ilp i USING (dataset, split, video, config)
    """, {'tol': LOG_TOLERANCE_MS})
    # Match counts over the current runs.
    counts = query_dicts(con, """
        SELECT count(*) AS runs, count(*) FILTER (WHERE current) AS current,
               count(*) FILTER (WHERE current AND logged_polyominoes = variables) AS polyominoes_match,
               count(*) FILTER (WHERE current AND logged_frames = frames) AS frames_match,
               count(*) FILTER (WHERE current AND logged_selected = kept) AS selected_match
        FROM log_runs
    """)[0]
    # Which instances the current runs cover: the log validates only these.
    scope = query_dicts(con, """
        SELECT dataset, split, config, rule, written, count(*) AS runs,
               min(variables) AS variables_min, max(variables) AS variables_max
        FROM log_runs WHERE current GROUP BY ALL ORDER BY ALL
    """)
    # Every logged run with its comparison.
    rows = query_dicts(con, 'SELECT * FROM log_runs ORDER BY dataset, split, config, video')
    return dict(log_dir=str(P022_LOG_DIR), **counts, scope=scope, rows=rows)


def feedback_sample(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """The feedback script's sample (first 6 sorted test outputs per dataset at sample rate 1 and r = 0.5), recounted.

    Its count was the 4-connected components of the pruned grid (saved_components here, equal to the
    kept polyominoes); variables is the instance size.
    """
    return query_dicts(con, """
        SELECT dataset, video, config, frames, saved_components AS feedback_style_components, kept, variables,
               constraints, distinct_constraints
        FROM instances
        WHERE split = 'test' AND config GLOB '*_60_1_*_r050_*'
        QUALIFY row_number() OVER (PARTITION BY dataset ORDER BY video, config) <= 6
        ORDER BY dataset, video, config
    """)


def rounded(rows: list[dict], digits: int = 4) -> list[dict]:
    """Rows with floats rounded for a readable report."""
    # Round every float value; other values pass through.
    return [{k: round(v, digits) if isinstance(v, float) else v for k, v in r.items()} for r in rows]


def main():
    args = parse_args()
    # Unknown datasets are an error rather than an empty report.
    unknown = set(args.datasets) - set(DATASET_LABELS)
    assert not unknown, f'unknown datasets {sorted(unknown)}'
    # This script's result directory.
    out_dir = results_dir(__file__)

    # Validated configurations per dataset, for the test split and the operating points.
    tradeoffs = {d: load_test_tradeoff(d) for d in args.datasets}
    validated = {d: validated_ilp_configs(t) for d, (t, _) in tradeoffs.items()}
    # The per-video tasks and grid coverage of every requested split.
    tasks, coverage = [], {}
    for dataset, split in itertools.product(args.datasets, args.splits):
        found, coverage[f'{split}/{dataset}'] = video_tasks(dataset, split, validated[dataset])
        tasks += found
    print(f'{len(tasks)} videos, {sum(len(t[3]) for t in tasks)} ILP instances')

    # Rebuild every instance, one video per task (the grouping call is a necessary per-clip loop).
    with mp.Pool(args.workers) as pool:
        results = pool.map(measure_video, tasks, chunksize=1)
    # Flatten the per-video rows and groupings.
    rows = [r for rs, _ in results for r in rs]
    groupings = [g for _, gs in results for g in gs]

    # Load instances, frames, and timings into DuckDB, with p022's default budget read from its source.
    con = duckdb.connect()
    default_limit = default_time_limit()
    load_tables(con, rows, groupings, validated, default_limit)
    # On test, every instance must be a validated configuration's (video_tasks excludes the others).
    unvalidated = query_dicts(con, "SELECT count(*) AS n FROM instances WHERE split = 'test' AND NOT validated")[0]['n']
    assert unvalidated == 0, f'{unvalidated} test instances outside the validated configurations'

    # Sizes and timings per dataset, and their relation over the instances.
    summary = dict(
        method=dict(
            unit='an instance is one (clip, ILP configuration) pair; pooled rows weight datasets by videos x '
                 'configurations and mix sample rates, so cite level dataset_sample_rate or operating_points',
            instances='every complete 022_pruned_polyominoes output in the cache that is in the split\'s expected set; '
                      'no sampling',
            test='validated (Pareto) Tetris configurations that run the ILP, on every test video; other test outputs '
                 'excluded (checks.json coverage)',
            valid='every configuration p022 left on the validation videos; not a complete video x configuration '
                  'grid on every dataset (checks.json coverage missing_instances)',
            input='020_relevancy scores binarized at relevance_threshold, grouped by group_tiles_all '
                  f'with TILEPADDING_MODE {TILEPADDING_MODE} (4-connected polyominoes)',
            rules='current p022: score > r * 255; floor (what the April 2026 cache matches): '
                  'score >= floor(r * 255); each instance uses the rule its saved selection is consistent with '
                  '(checks.json solutions, rule_* counts)',
            variables='one binary per polyomino per sampled frame, summed over the clip',
            constraints='constraint rows as added by polyis/sample/ilp/c/gurobi.pyx, before presolve and including '
                        'duplicates: per tile covered in >= 2 frames, 2 equalities (first, last), and per consecutive '
                        'covering pair 2 equalities if the gap exceeds the tile max distance (max_rate_table // '
                        'sample_rate, >= 1) else 1 window constraint',
            distinct_constraints='distinct rows: distinct forced variables plus distinct (current variable, window '
                                 'variables) window constraints',
            nonzeros='1 per equality, window size + 1 per window constraint (distinct_nonzeros over distinct rows)',
            forced_variables='distinct variables fixed to 1 by equality constraints',
            polyominoes_per_frame='pooled over the sampled frames of every instance; bound ceil(rows * cols / 2)',
            timings='p022 runtime.jsonl via r9o3_ilp_timings.load_runs: build_ilp (includes Gurobi env start) and '
                    'solve_ilp (includes solution read and env release)',
            solve_at_budget=f'estimate: {AT_BUDGET}, budget = time_limit or {default_limit} s; Gurobi termination '
                            'status is not logged, so a slow optimal solve also counts (budget_sensitivity)',
            correlations='Spearman with average ranks. Pooled correlations mix datasets and sample rates; '
                         'correlation_within_strata gives the within dataset x sample rate values '
                         f'(strata with >= {MIN_STRATUM} instances). s = 1 solve times are censored by the budget '
                         'and s >= 4 ones sit at the fixed overhead floor',
            deciles='equal-count bins of variables per split, pooled over datasets and sample rates (confounded; see '
                    'each bin\'s instances_by_sample_rate and instances_by_dataset)',
            operating_points=f'fastest validated configuration within {args.loss:.0%} of the reference test HOTA, '
                             'and the fastest that prunes, as in r9o3_ilp_timings extraction_share.json',
            quantiles='median interpolated; p95 of counts is an observed value (quantile_disc), p95 of times interpolated'),
        time_limits=query_dicts(con, 'SELECT split, time_limit, budget_s, count(*) AS instances FROM ilp '
                                     'GROUP BY ALL ORDER BY ALL'),
        budget_sensitivity=rounded(budget_sensitivity(con)),
        sizes=rounded(size_summary(con)),
        correlation_by_split=rounded(correlations(con, 'ilp', 'split')),
        correlation_by_dataset=rounded(correlations(con, 'ilp', 'split, dataset')),
        correlation_by_dataset_sample_rate=rounded(correlations(con, 'ilp', 'split, dataset, sample_rate')),
        correlation_within_strata=rounded(within_strata(con, 'ilp', 'split, dataset, sample_rate', MIN_STRATUM)),
        timing_by_variable_decile=rounded(binned_timings(con, 'ilp', 10)),
        operating_points=operating_point_sizes(con, tradeoffs, args.loss) if 'test' in args.splits else {},
    )
    write_json(out_dir, 'summary.json', summary)
    write_json(out_dir, 'configs.json', rounded(config_summary(con)))

    # Validation against p022's saved outputs and log, and the feedback sample recounted.
    log = log_comparison(con, p022_log_runs(P022_LOG_DIR))
    checks = dict(
        coverage=coverage, test_instances_outside_validated=unvalidated,
        solutions=rounded(solution_checks(con)),
        input_freshness=input_freshness(con),
        rule_by_day=rule_by_day(con),
        rule_effect=rounded(rule_effect(con)),
        p022_log=dict(log, rows=rounded(log['rows']),
                      note='covers only the instances in scope; the other instances, including every test one, are '
                           'validated by their saved selections (solutions) and input freshness alone'),
        feedback_sample=feedback_sample(con),
    )
    write_json(out_dir, 'checks.json', checks)

    # One line per split and dataset for the run log.
    for r in summary['sizes']:
        if r['level'] in ('dataset', 'all'):
            print(f"{r['split']:5s} {r['label'] or 'all':9s} grid {r['grid'] or 'mixed':5s} instances {r['instances']:5d} "
                  f"videos {r['videos']:3d} frames {r['frames_min']}-{r['frames_max']} poly/frame mean "
                  f"{r['polyominoes_per_frame_mean']:.1f} p95 {r['polyominoes_per_frame_p95']} max "
                  f"{r['polyominoes_per_frame_max']} | variables median {r['variables_median']:.0f} p95 "
                  f"{r['variables_p95']} max {r['variables_max']} | constraints median {r['constraints_median']:.0f} "
                  f"max {r['constraints_max']} (distinct max {r['distinct_constraints_max']}) | build "
                  f"{r['build_ms_median']:.1f} solve {r['solve_ms_median']:.1f} ms, at budget "
                  f"{r['solve_at_budget']} ({r['solve_at_budget_share']:.1%})")
    # One line per split and dataset of the checks.
    for r in checks['solutions']:
        print(f"check {r['split']:5s} {r['dataset']:12s} consistent {r['consistent']}/{r['instances']} "
              f"no_solution {r['no_solution']} components!=kept {r['components_differ_from_kept']} | rule identical "
              f"{r['rule_identical']} floor {r['rule_floor']} current {r['rule_current']} ambiguous "
              f"{r['rule_ambiguous']} neither {r['rule_neither']}")
    # The log comparison.
    print(f"p022 log: {log['current']}/{log['runs']} current runs; polyominoes match {log.get('polyominoes_match')}, "
          f"selected match {log.get('selected_match')}")


if __name__ == '__main__':
    main()
