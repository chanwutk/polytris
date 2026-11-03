import itertools
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from measurements import r9o3_ilp_instance_sizes as ilp
from polyis.io import store


# p022 parameter strings used as configurations in the synthetic tables (they must parse like real ones).
A = 'C_60_1_040_r050_sortcython'
B = 'C_60_4_080_r025_bytetrackcython'
Z = 'C_60_16_040_r050_sortcython'


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def entries(grids, idx=None):
    # score.jsonl lines for a [frames, rows, cols] uint8 array, as 020 and 022 write them.
    grids = np.asarray(grids, dtype=np.uint8)
    idx = list(range(len(grids))) if idx is None else idx
    return [{'classification_size': list(g.shape), 'classification_hex': g.tobytes().hex(), 'idx': i}
            for g, i in zip(grids, idx)]


def reference_constraints(tile_ids, lengths, max_dist):
    # Line-by-line transcription of the constraint loop in polyis/sample/ilp/c/gurobi.pyx.
    frames, rows, cols = tile_ids.shape
    offsets = [0]
    for frame_lengths in lengths:
        offsets.append(offsets[-1] + len(frame_lengths))
    constraints = []
    for n in range(rows):
        for m in range(cols):
            pos = [b for b in range(frames) if tile_ids[b, n, m] >= 0]
            if len(pos) <= 1:
                continue
            var = lambda b: offsets[b] + int(tile_ids[b, n, m])
            constraints.append(('eq', var(pos[0])))
            if pos[-1] != pos[0]:
                constraints.append(('eq', var(pos[-1])))
            for i in range(len(pos) - 1):
                limit = pos[i] + max_dist[n, m]
                if pos[i + 1] > limit:
                    constraints.append(('eq', var(pos[i])))
                    constraints.append(('eq', var(pos[i + 1])))
                else:
                    j = i + 1
                    while j < len(pos) and pos[j] <= limit:
                        j += 1
                    window = [var(b) for b in pos[i + 1:j]]
                    if window:
                        constraints.append(('ge', window, var(pos[i])))
    return constraints, offsets[-1]


def reference_counts(tile_ids, lengths, max_dist):
    # Instance counts from the transcription, in instance_size's keys; distinct rows deduplicated as sets.
    constraints, variables = reference_constraints(tile_ids, lengths, max_dist)
    eq = [c for c in constraints if c[0] == 'eq']
    ge = [c for c in constraints if c[0] == 'ge']
    distinct_eq = {c[1] for c in eq}
    distinct_ge = {(c[2], frozenset(c[1])) for c in ge}
    return dict(variables=variables, equality_constraints=len(eq), window_constraints=len(ge),
                constraints=len(constraints), nonzeros=len(eq) + sum(len(c[1]) + 1 for c in ge),
                forced_variables=len(distinct_eq), distinct_window_constraints=len(distinct_ge),
                distinct_constraints=len(distinct_eq) + len(distinct_ge),
                distinct_nonzeros=len(distinct_eq) + sum(len(w) + 1 for _, w in distinct_ge))


def feasible(constraints, selected):
    # Whether a set of kept variables satisfies every transcribed constraint.
    for c in constraints:
        if c[0] == 'eq' and c[1] not in selected:
            return False
        if c[0] == 'ge' and c[2] in selected and not any(v in selected for v in c[1]):
            return False
    return True


def pruned_grid(tile_ids, lengths, selected):
    # The 022 grid p022 would save for a selection: 255 on tiles of kept polyominoes.
    offsets = np.cumsum([0] + [len(l) for l in lengths])[:-1]
    var_grid = offsets[:, None, None] + tile_ids
    return np.where((tile_ids >= 0) & np.isin(var_grid, list(selected)), 255, 0).astype(np.uint8)


def random_instance(rng, frames, rows, cols, density, max_rate):
    # Random bitmaps grouped by the real grouping, with random per-tile distances in [1, max_rate].
    bitmaps = (rng.random((frames, rows, cols)) < density).astype(np.uint8)
    tile_ids, lengths = ilp.group(bitmaps)
    max_dist = rng.integers(1, max_rate + 1, size=(rows, cols)).astype(np.float64)
    return tile_ids, lengths, max_dist


def test_binarize_historical_current_key_retains_strict_cutoff():
    scores = np.array([0, 1, 51, 52, 127, 128, 255], dtype=np.uint8)
    # r = 0.5: the cutoff 127.5 keeps 128 and above.
    assert ilp.binarize(scores, 0.5, 'current').tolist() == [0, 0, 0, 0, 0, 1, 1]
    # r = 0.2: the cutoff is exactly 51, and a score equal to it is not relevant.
    assert ilp.binarize(scores, 0.2, 'current').tolist() == [0, 0, 0, 1, 1, 1, 1]
    # Extremes: r = 0 keeps any positive score, r = 1 keeps nothing.
    assert ilp.binarize(scores, 0.0, 'current').tolist() == [0, 1, 1, 1, 1, 1, 1]
    assert ilp.binarize(scores, 1.0, 'current').sum() == 0
    # Output is 0/1 uint8, the bitmap p022 passes to the grouping.
    assert ilp.binarize(scores, 0.5, 'current').dtype == np.uint8


def test_binarize_floor_rule_keeps_the_boundary_score():
    scores = np.array([62, 63, 64, 126, 127, 128, 190, 191, 255], dtype=np.uint8)
    # The floor rule keeps floor(r * 255) itself: 63, 127, 191 at r = 0.25, 0.5, 0.75.
    assert ilp.binarize(scores, 0.25, 'floor').tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert ilp.binarize(scores, 0.5, 'floor').tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert ilp.binarize(scores, 0.75, 'floor').tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]
    # The two rules differ exactly on the boundary score.
    for r in (0.25, 0.5, 0.75):
        differ = ilp.binarize(scores, r, 'floor') != ilp.binarize(scores, r, 'current')
        assert scores[differ].tolist() == [int(np.floor(r * 255))]
    # An unknown rule is an error.
    with pytest.raises(KeyError):
        ilp.binarize(scores, 0.5, 'other')


def test_max_distance_floor_divides_and_clamps(monkeypatch):
    calls = []

    def fake(dataset, tracker, tilesize, accuracy_idx):
        # Record the lookup and return an int32 table like max_rate_table.npy's slice.
        calls.append((dataset, tracker, tilesize, accuracy_idx))
        return np.array([[16, 3, 6], [1, 0, 7], [10, 15, 9]], dtype=np.int32)

    monkeypatch.setattr(ilp, 'load_max_sampling_rate', fake)
    dist = ilp.max_distance('d', 'sortcython', 60, 0.4, 4)
    # Floor division by the sample rate (6 and 7 give 1, 10 gives 2, 15 gives 3), then at least one frame.
    assert dist.tolist() == [[4.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 2.0]] and dist.dtype == np.float64
    # The accuracy threshold 0.4 is index 1 of p022's table.
    assert calls == [('d', 'sortcython', 60, 1)]
    # Sample rate 1 keeps the profiled values, with zero clamped to one.
    assert ilp.max_distance('d', 'sortcython', 60, 0.4, 1).tolist() == [[16.0, 3.0, 6.0], [1.0, 1.0, 7.0],
                                                                          [10.0, 15.0, 9.0]]
    # An accuracy threshold p022 does not profile is an error.
    with pytest.raises(KeyError):
        ilp.max_distance('d', 'sortcython', 60, 0.33, 1)


def test_group_is_four_connected():
    # Diagonal neighbors are separate polyominoes; an L shape is one.
    bitmaps = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[1, 1, 0], [0, 1, 0], [0, 1, 1]]], dtype=np.uint8)
    tile_ids, lengths = ilp.group(bitmaps)
    assert [sorted(l) for l in lengths] == [[1, 1], [5]]
    # Uncovered tiles are -1, covered ones point at their polyomino.
    assert (tile_ids[0][bitmaps[0] == 0] == -1).all() and len(set(tile_ids[0][bitmaps[0] == 1])) == 2
    # The caller's bitmaps are left intact (group_tiles clears the array it is given).
    assert bitmaps.sum() == 7
    # A boolean grid (the saved selection > 0) groups the same way.
    assert [sorted(l) for l in ilp.group(bitmaps > 0)[1]] == [[1, 1], [5]]


def test_ilp_structure_hand_example():
    # One row of three tiles over four frames:
    # frame 0 covers tiles 0-1 (one polyomino), frame 1 tile 0, frame 2 nothing, frame 3 tiles 0 and 2 (two).
    bitmaps = np.array([[[1, 1, 0]], [[1, 0, 0]], [[0, 0, 0]], [[1, 0, 1]]], dtype=np.uint8)
    tile_ids, lengths = ilp.group(bitmaps)
    s = ilp.ilp_structure(tile_ids, lengths, np.array([[1.0, 5.0, 5.0]]))
    size = ilp.instance_size(s)
    # Tile 0 is covered at frames 0, 1, 3: first and last forced, (0, 1) a window of one, (1, 3) a bridge.
    # Tiles 1 and 2 are covered once, so they add nothing. Frame 3's tile-0 polyomino is forced twice.
    assert size == dict(variables=4, relevant_tiles=5, max_polyominoes_per_frame=2, equality_constraints=4,
                        window_constraints=1, constraints=5, nonzeros=6, forced_variables=3,
                        distinct_window_constraints=1, distinct_constraints=4, distinct_nonzeros=5)
    assert s['per_frame'].tolist() == [1, 1, 0, 2] and s['offsets'].tolist() == [0, 1, 2, 2]


def test_instance_size_counts_duplicate_rows_once_as_distinct():
    # A two-tile polyomino persisting over three frames, distance 2: both tiles add the same constraints.
    tile_ids, lengths = ilp.group(np.ones((3, 1, 2), dtype=np.uint8))
    s = ilp.ilp_structure(tile_ids, lengths, np.full((1, 2), 2.0))
    size = ilp.instance_size(s)
    # Per tile: x0 = 1, x2 = 1, x1 + x2 >= x0, x2 >= x1; the second tile repeats all four.
    assert (size['constraints'], size['distinct_constraints']) == (8, 4)
    assert (size['window_constraints'], size['distinct_window_constraints']) == (4, 2)
    assert (size['nonzeros'], size['distinct_nonzeros']) == (14, 7)
    # The distinct rows are [current, window..., padding].
    assert ilp.distinct_windows(s).tolist() == [[0, 1, 2], [1, 2, -1]]


def test_ilp_structure_empty_and_single_frame():
    # No relevant tile: no variables, no constraints.
    tile_ids, lengths = ilp.group(np.zeros((5, 3, 4), dtype=np.uint8))
    assert ilp.instance_size(ilp.ilp_structure(tile_ids, lengths, np.ones((3, 4)))) == dict(
        variables=0, relevant_tiles=0, max_polyominoes_per_frame=0, equality_constraints=0,
        window_constraints=0, constraints=0, nonzeros=0, forced_variables=0, distinct_window_constraints=0,
        distinct_constraints=0, distinct_nonzeros=0)
    # One frame: variables but no temporal constraint.
    tile_ids, lengths = ilp.group(np.array([[[1, 0, 1]]], dtype=np.uint8))
    size = ilp.instance_size(ilp.ilp_structure(tile_ids, lengths, np.ones((1, 3))))
    assert size['variables'] == 2 and size['constraints'] == 0 and size['nonzeros'] == 0
    assert size['distinct_constraints'] == 0


def test_ilp_structure_rejects_mismatched_lengths():
    # One polyomino list per frame is required.
    tile_ids, lengths = ilp.group(np.ones((2, 1, 1), dtype=np.uint8))
    with pytest.raises(AssertionError):
        ilp.ilp_structure(tile_ids, lengths[:1], np.ones((1, 1)))
    # The distance table must match the tile grid, as p022 asserts.
    with pytest.raises(AssertionError):
        ilp.ilp_structure(tile_ids, lengths, np.ones((1, 2)))


@pytest.mark.parametrize('frames, rows, cols, density, max_rate', [
    (1, 1, 1, 1.0, 1), (2, 1, 1, 1.0, 1), (40, 1, 1, 0.5, 3), (30, 3, 4, 0.3, 4),
    (60, 8, 12, 0.1, 16), (25, 12, 21, 0.6, 2), (50, 4, 4, 0.9, 1), (50, 4, 4, 0.2, 100),
    (40, 3, 3, 0.95, 6),
])
def test_ilp_structure_matches_transcription(frames, rows, cols, density, max_rate):
    # Vectorized counts equal the C loop's on random instances, including the forced and distinct sets.
    rng = np.random.default_rng(frames * 1000 + rows * 10 + cols)
    for _ in range(5):
        tile_ids, lengths, max_dist = random_instance(rng, frames, rows, cols, density, max_rate)
        s = ilp.ilp_structure(tile_ids, lengths, max_dist)
        size = ilp.instance_size(s)
        expected = reference_counts(tile_ids, lengths, max_dist)
        assert {k: size[k] for k in expected} == expected
        constraints, _ = reference_constraints(tile_ids, lengths, max_dist)
        assert set(s['forced'].tolist()) == {c[1] for c in constraints if c[0] == 'eq'}
        # Distinct window rows are exactly the transcription's distinct (current, window) pairs.
        rows_ = {(r[0], tuple(v for v in r[1:] if v >= 0)) for r in ilp.distinct_windows(s).tolist()}
        assert rows_ == {(c[2], tuple(c[1])) for c in constraints if c[0] == 'ge'}
        # Objective coefficients are the polyomino sizes, which sum to the relevant tiles.
        assert size['relevant_tiles'] == int((tile_ids >= 0).sum())


def test_check_solution_agrees_with_brute_force_feasibility():
    # For small random instances, every selection is checked against the transcription's feasibility.
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 30:
        tile_ids, lengths, max_dist = random_instance(rng, 6, 2, 3, 0.35, 3)
        variables = sum(map(len, lengths))
        if not 1 <= variables <= 10:
            continue
        checked += 1
        s = ilp.ilp_structure(tile_ids, lengths, max_dist)
        constraints, _ = reference_constraints(tile_ids, lengths, max_dist)
        for bits in itertools.product([0, 1], repeat=variables):
            selected = {v for v, b in enumerate(bits) if b}
            grid = pruned_grid(tile_ids, lengths, selected)
            check = ilp.check_solution(s, tile_ids, grid)
            # Whole polyominoes from the input are kept, so nothing is outside or split.
            assert check['kept'] == len(selected) and check['tiles_outside_input'] == 0
            assert check['split_polyominoes'] == 0
            # The grid's own components are the kept polyominoes.
            assert sum(map(len, ilp.group(grid > 0)[1])) == len(selected)
            ok = check['forced_dropped'] == 0 and check['window_violations'] == 0
            assert ok == feasible(constraints, selected)


def test_check_solution_flags_inconsistent_grids():
    # Frame 0 has a two-tile polyomino and frame 1 repeats it, so both copies are forced.
    bitmaps = np.array([[[1, 1, 0]], [[1, 1, 0]]], dtype=np.uint8)
    tile_ids, lengths = ilp.group(bitmaps)
    s = ilp.ilp_structure(tile_ids, lengths, np.ones((1, 3)))
    # Keeping both is consistent.
    full = ilp.check_solution(s, tile_ids, bitmaps * 255)
    assert full == dict(kept=2, tiles_outside_input=0, split_polyominoes=0, forced_dropped=0, window_violations=0)
    # A kept tile where the input had none.
    outside = bitmaps * 255
    outside[0, 0, 2] = 255
    assert ilp.check_solution(s, tile_ids, outside)['tiles_outside_input'] == 1
    # Half of frame 1's polyomino: it is split, not kept, and its forced copy counts as dropped.
    half = bitmaps * 255
    half[1, 0, 1] = 0
    check = ilp.check_solution(s, tile_ids, half)
    assert check['split_polyominoes'] == 1 and check['kept'] == 1 and check['forced_dropped'] == 1
    # An empty selection drops both forced variables.
    assert ilp.check_solution(s, tile_ids, np.zeros_like(bitmaps))['forced_dropped'] == 2
    # A grid with a different shape is an error.
    with pytest.raises(AssertionError):
        ilp.check_solution(s, tile_ids, np.zeros((1, 1, 3), dtype=np.uint8))


def test_check_solution_window_violation():
    # One tile covered at frames 0, 1, 2 with distance 1: frame 1 is optional only if frame 2 follows it.
    bitmaps = np.ones((3, 1, 1), dtype=np.uint8)
    tile_ids, lengths = ilp.group(bitmaps)
    s = ilp.ilp_structure(tile_ids, lengths, np.ones((1, 1)))
    # Dropping frame 1 violates frame 0's window (its only window variable is frame 1).
    grid = bitmaps * 255
    grid[1] = 0
    assert ilp.check_solution(s, tile_ids, grid)['window_violations'] == 1
    # With distance 2, frame 2 is also in frame 0's window, so dropping frame 1 is feasible.
    s2 = ilp.ilp_structure(tile_ids, lengths, np.full((1, 1), 2.0))
    assert ilp.check_solution(s2, tile_ids, grid)['window_violations'] == 0


def fake_inputs(monkeypatch, scores, saved, table=4, mtimes=None):
    # Patch p022's loaders and the file times: saved maps a relevance threshold to its 022 grid.
    monkeypatch.setattr(ilp, 'load_classification_results', lambda *a, **k: entries(scores))
    monkeypatch.setattr(ilp, 'load_pruned_classification_results', lambda d, v, t, c, k, a, r, s, **kw: entries(saved[r]))
    monkeypatch.setattr(ilp, 'load_max_sampling_rate',
                        lambda *a: np.full(scores.shape[1:], table, dtype=np.int32))
    # Every file is written at 1.7e9 s (2023-11-14 UTC) unless mtimes maps a path fragment to another time.
    times = mtimes or {}
    monkeypatch.setattr(ilp.os.path, 'getmtime',
                        lambda path: next((t for k, t in times.items() if k in str(path)), 1.7e9))


def test_measure_video_rebuilds_and_checks(monkeypatch):
    # Two frames of a 1x3 grid with scores; r = 0.5 keeps tiles 0-1 in both frames.
    scores = np.array([[[200, 130, 0]], [[255, 140, 100]]], dtype=np.uint8)
    # The 022 file of each configuration: keep everything at r = 0.5, nothing at r = 0.75.
    saved = {0.5: (scores > 127.5).astype(np.uint8) * 255, 0.75: np.zeros_like(scores)}
    fake_inputs(monkeypatch, scores, saved, mtimes={'020_relevancy': 1.6e9, 'track_rate': 1.5e9})
    params = [A, 'C_60_1_080_r050_bytetrackcython', 'C_60_1_040_r075_sortcython']
    rows, groupings = ilp.measure_video(('d', 'test', 'te00.mp4', params))
    # Written day from the runtime file's modification time (1.7e9 s is 2023-11-14 UTC).
    assert {r['written'] for r in rows} == {'2023-11-14'}
    # Input and output times come from the 020 file, the max-rate table, and the 022 score file.
    assert {(r['scores_mtime'], r['table_mtime'], r['output_mtime']) for r in rows} == {(1.6e9, 1.5e9, 1.7e9)}
    # One row per configuration and rule; no score sits on a boundary (127 or 191), so the rules agree.
    assert len(rows) == 6 and all(r['boundary_tiles'] == 0 for r in rows)
    by_config = {r['config']: r for r in rows if r['rule'] == 'current'}
    # r = 0.5: one two-tile polyomino per frame, both forced (first and last of tiles 0 and 1).
    first = by_config[params[0]]
    assert (first['variables'], first['kept'], first['forced_variables'], first['frames']) == (2, 2, 2, 2)
    assert first['tiles_outside_input'] == 0 and first['split_polyominoes'] == 0
    # The saved grid has two components, both kept; the two tiles' duplicate rows collapse.
    assert first['saved_components'] == 2 and first['constraints'] == 6 and first['distinct_constraints'] == 3
    # r = 0.75: only tile 0 in both frames; the empty 022 grid drops both forced copies.
    third = by_config[params[2]]
    assert (third['variables'], third['kept'], third['forced_dropped'], third['saved_components']) == (2, 0, 2, 0)
    # Two relevance thresholds and two rules give four groupings, shared by the two r = 0.5 configurations.
    assert len(groupings) == 4 and first['gid'] == by_config[params[1]]['gid']
    assert sorted(g.tolist() for _, g in groupings) == [[1, 1]] * 4
    # The runtime path points at the configuration's 022 directory.
    assert first['runtime_path'].endswith(f'te00.mp4/022_pruned_polyominoes/{A}/score/runtime.jsonl')


def test_measure_video_rules_differ_on_boundary_tiles(monkeypatch):
    # Tile 1 scores exactly 127, relevant only under the floor rule at r = 0.5; the saved grid keeps it.
    scores = np.array([[[200, 127, 0]], [[200, 127, 0]]], dtype=np.uint8)
    saved = {0.5: np.array([[[255, 255, 0]], [[255, 255, 0]]], dtype=np.uint8)}
    fake_inputs(monkeypatch, scores, saved, table=1)
    rows, _ = ilp.measure_video(('d', 'test', 'te00.mp4', [A]))
    by_rule = {r['rule']: r for r in rows}
    # Two boundary tiles; the floor instance has the two-tile polyominoes, the current one single tiles.
    assert by_rule['floor']['boundary_tiles'] == 2 and by_rule['floor']['relevant_tiles'] == 4
    assert by_rule['current']['relevant_tiles'] == 2
    # Only the floor instance explains the saved grid.
    assert by_rule['floor']['tiles_outside_input'] == 0 and by_rule['current']['tiles_outside_input'] == 2


def test_measure_video_rejects_frame_mismatch(monkeypatch):
    # A 022 file whose frames differ from the 020 file it was built from is an error.
    monkeypatch.setattr(ilp, 'load_classification_results', lambda *a, **k: entries(np.zeros((2, 1, 1)), [0, 1]))
    monkeypatch.setattr(ilp, 'load_pruned_classification_results', lambda *a, **k: entries(np.zeros((2, 1, 1)), [0, 2]))
    monkeypatch.setattr(ilp, 'load_max_sampling_rate', lambda *a: np.ones((1, 1), dtype=np.int32))
    with pytest.raises(AssertionError):
        ilp.measure_video(('d', 'test', 'te00.mp4', [A]))


def test_validated_ilp_configs_collapses_packing_variants():
    # Two packing variants of one ILP configuration, one without a threshold, and thresholds 0.95 and 0.25.
    tetris = pd.DataFrame(dict(
        classifier=['C', 'C', 'C', 'D'], tilesize=[60.0, 60.0, 60.0, 30.0], sample_rate=[1.0, 1.0, 2.0, 8.0],
        tracker=['sortcython', 'sortcython', 'sortcython', 'bytetrackcython'],
        tracking_accuracy_threshold=[0.95, 0.95, np.nan, 0.4], relevance_threshold=[0.25, 0.25, 0.5, 0.75],
        tilepadding=['none', 'tr', 'none', 'none']))
    # The variants collapse, the unpruned row drops, and floats format like p022's directory names.
    assert ilp.validated_ilp_configs(tetris) == {'C_60_1_095_r025_sortcython', 'D_30_8_040_r075_bytetrackcython'}
    # No pruning configuration at all gives an empty set.
    assert ilp.validated_ilp_configs(tetris.iloc[[2]]) == set()


def make_cache(tmp_path, monkeypatch, layout, split='test'):
    # A fake store and cache: layout maps video -> {config: complete?}.
    store_dir, cache_dir = tmp_path / 'store', tmp_path / 'cache'
    (store_dir / split).mkdir(parents=True)
    for video, configs in layout.items():
        (store_dir / split / video).write_text('')
        for config, complete in configs.items():
            score = cache_dir / video / '022_pruned_polyominoes' / config / 'score'
            score.mkdir(parents=True)
            (score / 'score.jsonl').write_text('')
            if complete:
                (score / 'runtime.jsonl').write_text('')
    # A non-video file in the videoset directory is ignored, as p022 ignores it.
    (store_dir / split / 'notes.txt').write_text('')
    monkeypatch.setattr(store, 'dataset', lambda dataset, *args: store_dir.joinpath(*args))
    monkeypatch.setattr(ilp.cache, 'exec', lambda dataset, stage, video, *args:
                        cache_dir.joinpath(video, '022_pruned_polyominoes', *args))


def test_video_tasks_test_coverage_excludes_unvalidated(tmp_path, monkeypatch):
    # te00 has both validated configs plus an extra; te01 has both plus an incomplete one.
    make_cache(tmp_path, monkeypatch, {'te00.mp4': {'A': True, 'B': True, 'C': True},
                                       'te01.mp4': {'A': True, 'B': True, 'D': False}})
    tasks, coverage = ilp.video_tasks('d', 'test', {'A', 'B'})
    # The extra output is reported and left out of the tasks.
    assert tasks == [('d', 'test', 'te00.mp4', ['A', 'B']), ('d', 'test', 'te01.mp4', ['A', 'B'])]
    assert coverage['excluded_outside_validated'] == ['te00.mp4/C'] and coverage['incomplete_outputs'] == ['te01.mp4/D']
    assert (coverage['videos'], coverage['configs'], coverage['expected_instances'], coverage['instances'],
            coverage['missing_instances']) == (2, 2, 4, 4, 0)


def test_video_tasks_requires_every_validated_config(tmp_path, monkeypatch):
    # A validated configuration missing (or incomplete) on one test video fails the run.
    make_cache(tmp_path, monkeypatch, {'te00.mp4': {'A': True, 'B': False}})
    with pytest.raises(AssertionError):
        ilp.video_tasks('d', 'test', {'A', 'B'})


def test_video_tasks_valid_split_reports_missing_pairs(tmp_path, monkeypatch):
    # Validation expects every configuration seen on any video; va00 lacks B and C, va02 has nothing.
    make_cache(tmp_path, monkeypatch, {'va00.mp4': {'A': True}, 'va01.mp4': {'A': True, 'B': True, 'C': True},
                                       'va02.mp4': {}}, split='valid')
    tasks, coverage = ilp.video_tasks('d', 'valid', {'Z'})
    # Missing pairs are reported, not required; a video without outputs gives no task.
    assert tasks == [('d', 'valid', 'va00.mp4', ['A']), ('d', 'valid', 'va01.mp4', ['A', 'B', 'C'])]
    assert (coverage['videos'], coverage['videos_with_outputs'], coverage['configs'], coverage['expected_instances'],
            coverage['instances'], coverage['missing_instances']) == (3, 2, 3, 9, 4, 5)
    assert coverage['missing_configs_by_video'] == {'va00.mp4': 2, 'va02.mp4': 3}
    assert coverage['configs_with_missing_videos'] == 3 and coverage['excluded_outside_validated'] == []


def test_video_tasks_empty_and_missing_videoset(tmp_path, monkeypatch):
    # A videoset without any output has no task and no expected configuration.
    make_cache(tmp_path, monkeypatch, {'va00.mp4': {}}, split='valid')
    tasks, coverage = ilp.video_tasks('d', 'valid', set())
    assert tasks == [] and coverage['configs'] == 0 and coverage['missing_instances'] == 0
    # A missing videoset directory is an error.
    monkeypatch.setattr(store, 'dataset', lambda dataset, *args: tmp_path / 'nowhere')
    with pytest.raises(AssertionError):
        ilp.video_tasks('d', 'valid', set())


LOG = """================
Function: process_all ( dataset='caldot2-y05', videoset='valid', videos=['va00.mp4', 'va01.mp4'], classifier='ShuffleNet05', tile_size=60, sample_rate=16, tracker='sortcython', tracking_accuracy_threshold=0.4, relevance_threshold=0.5, time_limit=None )
================
[va00.mp4] Step 3: Solving ILP (79 polyominoes across 58 frames)...
[va00.mp4] Step 3 done: build_ilp=6.0ms solve_ilp=10.1ms (79 polyominoes selected)
[va01.mp4] Step 3: Solving ILP (78 polyominoes across 58 frames)...
Function: process_all ( dataset='jnc0', videoset='test', videos=['te00.mp4'], classifier='ShuffleNet05', tile_size=60, sample_rate=1, tracker='bytetrackcython', tracking_accuracy_threshold=0.8, relevance_threshold=0.25, time_limit=None )
[te00.mp4] Step 3: Solving ILP (3000 polyominoes across 1075 frames)...
[te00.mp4] Step 3 done: build_ilp=20.5ms solve_ilp=101.2ms (900 polyominoes selected)
"""


def test_p022_log_runs_parses_headers_and_skips_unfinished(tmp_path):
    (tmp_path / 'worker_0.stdout').write_text(LOG)
    runs = ilp.p022_log_runs(tmp_path)
    # va01 never finished its ILP, so only two runs are recorded, each under its own header.
    assert [(r['dataset'], r['video'], r['config']) for r in runs] == [
        ('caldot2-y05', 'va00.mp4', 'ShuffleNet05_60_16_040_r050_sortcython'),
        ('jnc0', 'te00.mp4', 'ShuffleNet05_60_1_080_r025_bytetrackcython')]
    assert runs[0]['split'] == 'valid' and runs[0]['logged_polyominoes'] == 79 and runs[0]['logged_frames'] == 58
    assert runs[1]['logged_selected'] == 900 and runs[1]['logged_solve_ms'] == 101.2
    # No log directory or files: no runs.
    assert ilp.p022_log_runs(tmp_path / 'missing') == []


def instance_row(dataset, split, video, config, variables, build_ms, solve_ms, tmp_path, rule='current',
                 written='2026-04-19', time_limit=None, **extra):
    # One instance row with its runtime.jsonl written where p022 writes it, as measure_video and p022 would produce.
    score = tmp_path / dataset / 'execution' / video / '022_pruned_polyominoes' / config / 'score'
    score.mkdir(parents=True, exist_ok=True)
    path = score / 'runtime.jsonl'
    path.write_text(json.dumps({'runtime': [{'op': 'group_tiles', 'time': 1.0}, {'op': 'build_ilp', 'time': build_ms},
                                            {'op': 'solve_ilp', 'time': solve_ms}], 'time_limit': time_limit}) + '\n')
    row = dict(dataset=dataset, split=split, video=video, config=config, rule=rule, boundary_tiles=0, written=written,
               gid=f'{dataset}/{split}/{video}',
               classifier='C', tilesize=60, sample_rate=1, tracker='sortcython', tracking_accuracy_threshold=0.4,
               relevance_threshold=0.5, grid_rows=8, grid_cols=12, frames=10, variables=variables,
               relevant_tiles=2 * variables, max_polyominoes_per_frame=3, equality_constraints=variables,
               window_constraints=variables, constraints=2 * variables, nonzeros=3 * variables,
               forced_variables=variables // 2, distinct_window_constraints=variables // 2,
               distinct_constraints=variables, distinct_nonzeros=2 * variables, kept=variables // 2,
               saved_components=variables // 2, tiles_outside_input=0, split_polyominoes=0,
               forced_dropped=0, window_violations=0, scores_mtime=1.0, table_mtime=1.0, output_mtime=2.0,
               runtime_path=str(path))
    row.update(extra)
    return row


def load(con, rows, groupings=None, validated=None):
    # load_tables with a 0.1 s default budget; by default every configuration of the rows is validated.
    if validated is None:
        validated = {}
        for r in rows:
            validated.setdefault(r['dataset'], set()).add(r['config'])
    ilp.load_tables(con, rows, groupings or [('g', np.array([1]))], validated, 0.1)


def test_load_tables_joins_timings_and_flags(con, tmp_path):
    rows = [instance_row('caldot1-y05', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path),
            instance_row('caldot1-y05', 'valid', 'va00.mp4', B, 20, 6.0, 100.0, tmp_path, time_limit=0.2)]
    load(con, rows, [('caldot1-y05/test/te00.mp4', np.array([1, 2])),
                     ('caldot1-y05/valid/va00.mp4', np.array([0, 3]))], {'caldot1-y05': {A}})
    got = con.execute('SELECT config, label, validated, build_ms, solve_ms, stage_ms, time_limit, budget_s '
                      'FROM ilp ORDER BY config').fetchall()
    # Timings come from runtime.jsonl, stage time sums every op, only A is validated, and the budget is
    # the time limit or the default.
    assert got == [(A, 'CalDoT1', True, 5.0, 10.0, 16.0, None, 0.1), (B, 'CalDoT1', False, 6.0, 100.0, 107.0, 0.2, 0.2)]
    assert con.execute('SELECT count(*), sum(polyominoes) FROM frames').fetchone() == (4, 6)


def test_load_tables_rejects_bad_timings_and_empty(con, tmp_path):
    # A runtime.jsonl without the ILP ops cannot be related to timings.
    row = instance_row('jnc0', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path)
    open(row['runtime_path'], 'w').write(json.dumps({'runtime': [{'op': 'group_tiles', 'time': 1.0}], 'time_limit': None}))
    with pytest.raises(AssertionError):
        load(con, [row])
    # A duplicated build_ilp would double-count the build time.
    ops = [{'op': 'build_ilp', 'time': 5.0}, {'op': 'build_ilp', 'time': 5.0}, {'op': 'solve_ilp', 'time': 9.0}]
    open(row['runtime_path'], 'w').write(json.dumps({'runtime': ops, 'time_limit': None}))
    with pytest.raises(AssertionError):
        load(con, [row])
    # A negative time.
    ops = [{'op': 'build_ilp', 'time': -1.0}, {'op': 'solve_ilp', 'time': 9.0}]
    open(row['runtime_path'], 'w').write(json.dumps({'runtime': ops, 'time_limit': None}))
    with pytest.raises(AssertionError):
        load(con, [row])
    # No instances at all.
    with pytest.raises(AssertionError):
        ilp.load_tables(con, [], [], {}, 0.1)


def load_synthetic(con, tmp_path):
    # Two datasets on test: CalDoT1 with two clips at sample rate 1, B3D1 with one at sample rate 4 (12x18 grid).
    rows = [instance_row('caldot1-y05', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path),
            instance_row('caldot1-y05', 'test', 'te01.mp4', A, 30, 7.0, 100.0, tmp_path),
            instance_row('jnc0', 'test', 'te00.mp4', B, 50, 9.0, 99.99, tmp_path, sample_rate=4, grid_rows=12,
                         grid_cols=18, kept=0, forced_variables=3, forced_dropped=3, saved_components=0)]
    frames = [('caldot1-y05/test/te00.mp4', np.array([0, 1, 2, 3])), ('caldot1-y05/test/te01.mp4', np.array([4, 4])),
              ('jnc0/test/te00.mp4', np.array([10]))]
    load(con, rows, frames)


def test_size_summary_levels_and_values(con, tmp_path):
    load_synthetic(con, tmp_path)
    rows = {(r['level'], r['dataset'], r['sample_rate']): r for r in ilp.size_summary(con)}
    # Every grouping level is present.
    assert {k[0] for k in rows} == {'dataset', 'dataset_sample_rate', 'all', 'all_sample_rate'}
    caldot = rows[('dataset', 'caldot1-y05', None)]
    assert caldot['grid'] == '8x12' and caldot['tiles_per_frame'] == 96 and caldot['instances'] == 2
    # The checkerboard bound of an 8x12 grid.
    assert caldot['polyominoes_per_frame_bound'] == 48
    assert caldot['variables_median'] == 20 and caldot['variables_max'] == 30
    # A solve at exactly the 100 ms budget counts as reaching it.
    assert caldot['solve_at_budget_share'] == 0.5 and caldot['solve_at_budget'] == 1
    # Polyominoes per frame are pooled over the instance frames: (0+1+2+3+4+4) / 6.
    assert caldot['polyominoes_per_frame_mean'] == pytest.approx(14 / 6) and caldot['polyominoes_per_frame_max'] == 4
    assert caldot['empty_frame_share'] == pytest.approx(1 / 6)
    # Pooled over datasets, the grid is mixed and therefore not reported.
    pooled = rows[('all', None, None)]
    assert pooled['grid'] is None and pooled['tiles_per_frame'] is None and pooled['polyominoes_per_frame_bound'] is None
    # te00.mp4 exists in both datasets: pooled videos count (dataset, video) pairs, configurations likewise.
    assert pooled['instances'] == 3 and pooled['videos'] == 3 and pooled['configs'] == 2
    # 99.99 ms is below the budget.
    assert rows[('all_sample_rate', None, 4)]['solve_at_budget_share'] == 0.0


def test_size_summary_uses_only_the_chosen_rule_frames(con, tmp_path):
    # One instance whose saved grid only the floor rule explains; the two rules' groupings differ.
    rows = [instance_row('jnc6', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path, rule='current', boundary_tiles=2,
                         tiles_outside_input=1, gid='current-grouping'),
            instance_row('jnc6', 'test', 'te00.mp4', A, 12, 5.0, 10.0, tmp_path, rule='floor', boundary_tiles=2,
                         gid='floor-grouping')]
    load(con, rows, [('current-grouping', np.array([9, 9, 9])), ('floor-grouping', np.array([1, 2, 3]))])
    r = next(r for r in ilp.size_summary(con) if r['level'] == 'dataset')
    # Only the floor grouping's frames enter the per-frame statistics, and only the floor instance's size.
    assert (r['polyominoes_per_frame_mean'], r['polyominoes_per_frame_max'], r['variables_max']) == (2.0, 3, 12)


def correlation_table(con, rows):
    # A table with the columns correlation_sql reads: (split, dataset, sample_rate, variables, solve_ms).
    con.execute('CREATE TABLE t (split VARCHAR, dataset VARCHAR, sample_rate INTEGER, variables BIGINT, '
                'constraints BIGINT, distinct_constraints BIGINT, build_ms DOUBLE, solve_ms DOUBLE)')
    con.executemany('INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    [(s, d, k, v, 2 * v, v, b, b) for s, d, k, v, b in rows])


def test_correlations_match_numpy_with_ties(con):
    # Variables with a tie; Spearman uses average ranks.
    variables = [1, 2, 2, 3, 5]
    build = [1.0, 3.0, 2.0, 4.0, 3.5]
    correlation_table(con, [('test', 'd', 1, v, b) for v, b in zip(variables, build)])
    r = ilp.correlations(con, 't', 'split')[0]

    # Average ranks computed independently.
    def ranks(x):
        order = np.argsort(x, kind='stable')
        ranked = np.empty(len(x))
        ranked[order] = np.arange(1, len(x) + 1)
        return np.array([ranked[np.array(x) == v].mean() for v in x])
    assert r['spearman_variables_build_ms'] == pytest.approx(np.corrcoef(ranks(variables), ranks(build))[0, 1])
    assert r['pearson_variables_build_ms'] == pytest.approx(np.corrcoef(variables, build)[0, 1])
    # Constraints are 2 x variables here, so their rank correlation is the same.
    assert r['spearman_constraints_solve_ms'] == pytest.approx(r['spearman_variables_solve_ms']) and r['instances'] == 5


def test_correlations_constant_column_is_null(con):
    # A constant timing has no correlation; DuckDB's NaN becomes NULL (JSON null).
    correlation_table(con, [('test', 'd', 1, 1, 5.0), ('test', 'd', 1, 2, 5.0), ('test', 'd', 1, 3, 5.0)])
    r = ilp.correlations(con, 't', 'split')[0]
    assert r['spearman_variables_solve_ms'] is None and r['pearson_variables_build_ms'] is None


def test_within_strata_weights_and_filters(con):
    # Stratum (d, 1): rho 1 over 3; (d, 2): rho -1 over 4; (d, 4): 1 instance, below the minimum;
    # (e, 1): constant timing, undefined correlation, counted but left out of the mean.
    rows = ([('test', 'd', 1, v, float(v)) for v in (1, 2, 3)] + [('test', 'd', 2, v, float(5 - v)) for v in (1, 2, 3, 4)]
            + [('test', 'd', 4, 1, 1.0)] + [('test', 'e', 1, v, 5.0) for v in (1, 2)])
    correlation_table(con, rows)
    r = ilp.within_strata(con, 't', 'split, dataset, sample_rate', 2)[0]
    assert (r['strata'], r['instances']) == (3, 9)
    assert r['spearman_variables_solve_ms_weighted_mean'] == pytest.approx((3 * 1 - 4 * 1) / 7)
    assert (r['spearman_variables_solve_ms_min'], r['spearman_variables_solve_ms_max']) == (pytest.approx(-1), pytest.approx(1))
    # A minimum above every stratum leaves nothing.
    assert ilp.within_strata(con, 't', 'split, dataset, sample_rate', 100) == []


def test_binned_timings_bins_and_budget(con, tmp_path):
    load_synthetic(con, tmp_path)
    bins = ilp.binned_timings(con, 'ilp', 3)
    # Three instances in three bins, ordered by variables, with each bin's mix.
    assert [(b['bin'], b['variables_min'], b['solve_at_budget_share']) for b in bins] == [(1, 10, 0.0), (2, 30, 1.0), (3, 50, 0.0)]
    assert bins[2]['instances_by_sample_rate'] == {4: 1} and bins[2]['instances_by_dataset'] == {'B3D1': 1}


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_binned_timings_breaks_ties_deterministically(tmp_path, order):
    # Three instances tied on variables and constraints: bins follow the video order whatever the insertion order.
    solves = [10.0, 100.0, 50.0]
    rows = [instance_row('jnc2', 'test', f'te0{i}.mp4', A, 10, 5.0, solves[i], tmp_path) for i in order]
    connection = duckdb.connect()
    load(connection, rows)
    bins = ilp.binned_timings(connection, 'ilp', 3)
    assert [b['solve_at_budget_share'] for b in bins] == [0.0, 1.0, 0.0]
    connection.close()


def test_budget_sensitivity_uses_each_runs_budget(con, tmp_path):
    # Default budget 0.1 s: 96, 100, 104, 111 ms; a 0.2 s time limit: 150 ms is below it.
    rows = [instance_row('jnc7', 'test', f'te0{i}.mp4', A, 10, 5.0, s, tmp_path) for i, s in enumerate([96.0, 100.0, 104.0, 111.0])]
    rows += [instance_row('jnc7', 'valid', 'va00.mp4', A, 10, 5.0, 150.0, tmp_path, time_limit=0.2)]
    load(con, rows)
    got = {r['split']: r for r in ilp.budget_sensitivity(con)}
    test = got['test']
    assert (test['at_95pct'], test['at_100pct'], test['at_105pct'], test['at_110pct']) == (4, 3, 1, 1)
    assert got['valid']['at_100pct'] == 0 and got['valid']['at_95pct'] == 0
    # The pooled row (split NULL) sums both splits.
    assert got[None]['instances'] == 5 and got[None]['at_100pct'] == 3 and test['share_at_100pct'] == 0.75


def test_config_summary_and_solution_checks(con, tmp_path):
    load_synthetic(con, tmp_path)
    configs = {r['config']: r for r in ilp.config_summary(con)}
    assert configs[A]['instances'] == 2 and configs[A]['validated'] and configs[A]['variables_median'] == 20
    assert configs[A]['solve_at_budget_share'] == 0.5 and configs[B]['kept_share_median'] == 0.0
    checks = {r['dataset']: r for r in ilp.solution_checks(con)}
    # The B3D clip kept nothing although variables are forced: no solution, still consistent.
    assert checks['jnc0']['no_solution'] == 1 and checks['jnc0']['consistent'] == 1
    assert checks['jnc0']['with_forced_dropped'] == 0
    assert checks['caldot1-y05']['consistent'] == 2 and checks['caldot1-y05']['components_differ_from_kept'] == 0


def test_solution_checks_flags_violations(con, tmp_path):
    # A clip that kept something yet dropped a forced variable, one with a split polyomino, and one whose saved
    # grid has more components than kept polyominoes (and than variables).
    rows = [instance_row('jnc2', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path, kept=3, saved_components=3,
                         forced_dropped=1),
            instance_row('jnc2', 'test', 'te01.mp4', A, 10, 5.0, 10.0, tmp_path, split_polyominoes=2),
            instance_row('jnc2', 'test', 'te02.mp4', A, 10, 5.0, 10.0, tmp_path, saved_components=11)]
    load(con, rows)
    r = ilp.solution_checks(con)[0]
    assert (r['with_forced_dropped'], r['with_split_polyominoes'], r['components_differ_from_kept'],
            r['components_above_variables'], r['consistent']) == (1, 1, 1, 1, 1)


def test_input_freshness_counts_newer_inputs(con, tmp_path):
    # te01's scores and te02's table were rewritten after the output; te00 is clean.
    rows = [instance_row('ams-y05', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path),
            instance_row('ams-y05', 'test', 'te01.mp4', A, 10, 5.0, 10.0, tmp_path, scores_mtime=3.0),
            instance_row('ams-y05', 'test', 'te02.mp4', A, 10, 5.0, 10.0, tmp_path, table_mtime=2.5)]
    load(con, rows)
    r = ilp.input_freshness(con)[0]
    assert (r['instances'], r['scores_newer_than_output'], r['table_newer_than_output']) == (3, 1, 1)


def log_run(video, build, solve, **extra):
    # One logged run of jnc0 test configuration A.
    return dict(dict(dataset='jnc0', split='test', video=video, config=A, time_limit='None', log='w',
                     logged_polyominoes=10, logged_frames=10, logged_selected=5, logged_build_ms=build,
                     logged_solve_ms=solve), **extra)


def test_log_comparison_tolerance(con, tmp_path):
    # Cached timings; p022 prints them with one decimal.
    cached = {'te00.mp4': (5.0, 10.0), 'te01.mp4': (5.05, 10.25), 'te02.mp4': (5.0, 10.0), 'te03.mp4': (5.0, 10.0)}
    load(con, [instance_row('jnc0', 'test', v, A, 10, b, s, tmp_path) for v, (b, s) in cached.items()])
    runs = [
        # Exact.
        log_run('te00.mp4', 5.0, 10.0),
        # Half-unit boundaries: Python prints 5.05 as 5.0 and 10.25 as 10.2, which DuckDB's round() would not match.
        log_run('te01.mp4', float(f'{5.05:.1f}'), float(f'{10.25:.1f}')),
        # Only the build differs: stale.
        log_run('te02.mp4', 5.2, 10.0),
        # Only the solve differs: stale.
        log_run('te03.mp4', 5.0, 10.2),
    ]
    out = ilp.log_comparison(con, runs)
    current = {r['video']: r['current'] for r in out['rows']}
    assert current == {'te00.mp4': True, 'te01.mp4': True, 'te02.mp4': False, 'te03.mp4': False}
    # The scope names the covered instances' configuration and rule.
    assert out['scope'] == [dict(dataset='jnc0', split='test', config=A, rule='current', written='2026-04-19', runs=2,
                                 variables_min=10, variables_max=10)]


def test_log_comparison_counts_matches(con, tmp_path):
    load_synthetic(con, tmp_path)
    runs = [
        # Current: timings equal the cached 5.0 / 10.0, counts match.
        dict(dataset='caldot1-y05', split='test', video='te00.mp4', config=A, time_limit='None', log='w',
             logged_polyominoes=10, logged_frames=10, logged_selected=5, logged_build_ms=5.0, logged_solve_ms=10.0),
        # Stale: the cache was rewritten after this run.
        dict(dataset='caldot1-y05', split='test', video='te01.mp4', config=A, time_limit='None', log='w',
             logged_polyominoes=99, logged_frames=10, logged_selected=1, logged_build_ms=1.0, logged_solve_ms=10.0),
        # Not in the cache at all.
        dict(dataset='jnc0', split='valid', video='va00.mp4', config=Z, time_limit='None', log='w',
             logged_polyominoes=1, logged_frames=1, logged_selected=1, logged_build_ms=1.0, logged_solve_ms=1.0),
    ]
    out = ilp.log_comparison(con, runs)
    assert (out['runs'], out['current'], out['polyominoes_match'], out['frames_match'], out['selected_match']) == (3, 1, 1, 1, 1)
    # No log: nothing compared.
    assert ilp.log_comparison(con, [])['runs'] == 0


def test_feedback_sample_selects_first_six_sorted(con, tmp_path):
    # Seven matching outputs on one dataset and one non-matching configuration.
    rows = [instance_row('ams-y05', 'test', f'te0{i // 2}.mp4', f'S_60_1_0{4 + i % 2}0_r050_sortcython', 10 + i, 5.0,
                         10.0, tmp_path) for i in range(7)]
    rows += [instance_row('ams-y05', 'test', 'te00.mp4', 'S_60_1_040_r025_sortcython', 1, 5.0, 10.0, tmp_path),
             instance_row('ams-y05', 'valid', 'va00.mp4', 'S_60_1_040_r050_sortcython', 1, 5.0, 10.0, tmp_path)]
    load(con, rows)
    sample = ilp.feedback_sample(con)
    # Sorted by video then configuration, the first six test outputs at r = 0.5 and sample rate 1.
    assert [(r['video'], r['variables']) for r in sample] == [
        ('te00.mp4', 10), ('te00.mp4', 11), ('te01.mp4', 12), ('te01.mp4', 13), ('te02.mp4', 14), ('te02.mp4', 15)]
    # The feedback-style count is the saved grid's components.
    assert sample[0]['feedback_style_components'] == 5


def rule_rows(tmp_path, video, boundary, current_ok, floor_ok, current_variables=10, floor_variables=12,
              written='2026-04-19'):
    # The two rule rows of one instance; an inconsistent one has a kept tile outside its input.
    return [instance_row('jnc6', 'test', video, A, current_variables, 5.0, 10.0, tmp_path, rule='current',
                         written=written, boundary_tiles=boundary, tiles_outside_input=0 if current_ok else 1),
            instance_row('jnc6', 'test', video, A, floor_variables, 5.0, 10.0, tmp_path, rule='floor',
                         written=written, boundary_tiles=boundary, tiles_outside_input=0 if floor_ok else 1)]


def test_load_tables_chooses_the_consistent_rule(con, tmp_path):
    rows = (rule_rows(tmp_path, 'te00.mp4', 0, True, True)       # no boundary tile: identical, reported as current
            + rule_rows(tmp_path, 'te01.mp4', 3, False, True)    # only floor explains the saved grid
            + rule_rows(tmp_path, 'te02.mp4', 3, False, True)    # floor again, so floor is the majority
            + rule_rows(tmp_path, 'te03.mp4', 3, True, False)    # only current explains it
            + rule_rows(tmp_path, 'te04.mp4', 3, True, True)     # both consistent: ambiguous, majority (floor)
            + rule_rows(tmp_path, 'te05.mp4', 3, False, False)   # neither: majority (floor), still inconsistent
            + rule_rows(tmp_path, 'te06.mp4', 0, False, False))  # no boundary tile but inconsistent: neither
    load(con, rows)
    got = con.execute('SELECT video, rule_match, rule, consistent FROM instances ORDER BY video').fetchall()
    assert got == [('te00.mp4', 'identical', 'current', True), ('te01.mp4', 'floor', 'floor', True),
                   ('te02.mp4', 'floor', 'floor', True), ('te03.mp4', 'current', 'current', True),
                   ('te04.mp4', 'ambiguous', 'floor', True), ('te05.mp4', 'neither', 'floor', False),
                   ('te06.mp4', 'neither', 'floor', False)]
    # Every (instance, rule) pair stays available for the rule comparison.
    assert con.execute('SELECT count(*) FROM instances_all').fetchone()[0] == 14
    checks = ilp.solution_checks(con)[0]
    assert (checks['rule_identical'], checks['rule_floor'], checks['rule_current'], checks['rule_ambiguous'],
            checks['rule_neither'], checks['consistent']) == (1, 2, 1, 1, 2, 5)


def test_load_tables_breaks_ties_by_write_day(con, tmp_path):
    # April outputs decide floor; a May rerun decides current, and its ambiguous clip follows May, not April.
    rows = (rule_rows(tmp_path, 'te00.mp4', 3, False, True) + rule_rows(tmp_path, 'te01.mp4', 3, False, True)
            + rule_rows(tmp_path, 'te02.mp4', 3, True, False, written='2026-05-29')
            + rule_rows(tmp_path, 'te03.mp4', 3, True, True, written='2026-05-29')
            + rule_rows(tmp_path, 'te04.mp4', 3, True, True, written='2026-06-01'))
    load(con, rows)
    got = dict(con.execute('SELECT video, rule FROM instances').fetchall())
    # te04's day has no decided clip, so it takes the split majority (floor, 2 of 3).
    assert got == {'te00.mp4': 'floor', 'te01.mp4': 'floor', 'te02.mp4': 'current', 'te03.mp4': 'current',
                   'te04.mp4': 'floor'}
    days = {r['written']: r for r in ilp.rule_by_day(con)}
    assert days['2026-05-29']['current'] == 1 and days['2026-05-29']['ambiguous'] == 1
    assert days['2026-05-29']['reported_rules'] == ['current'] and days['2026-04-19']['floor'] == 2
    assert days['2026-05-29']['configs'] == 1 and days['2026-05-29']['configs_first5'] == [A]


def test_load_tables_tied_day_falls_back_to_the_split(con, tmp_path):
    # Day D1 is tied (one floor, one current), so its ambiguous clip takes the split majority (floor, 2 of 3).
    rows = (rule_rows(tmp_path, 'te00.mp4', 3, False, True, written='2026-04-19')
            + rule_rows(tmp_path, 'te01.mp4', 3, True, False, written='2026-04-19')
            + rule_rows(tmp_path, 'te02.mp4', 3, True, True, written='2026-04-19')
            + rule_rows(tmp_path, 'te03.mp4', 3, False, True, written='2026-04-20'))
    load(con, rows)
    assert dict(con.execute('SELECT video, rule FROM instances').fetchall())['te02.mp4'] == 'floor'


def test_load_tables_tied_split_defaults_to_current(con, tmp_path):
    # A tie on the day and on the split leaves no majority, so the current rule is reported.
    rows = (rule_rows(tmp_path, 'te00.mp4', 3, False, True) + rule_rows(tmp_path, 'te01.mp4', 3, True, False)
            + rule_rows(tmp_path, 'te02.mp4', 3, True, True))
    load(con, rows)
    assert dict(con.execute('SELECT video, rule FROM instances').fetchall())['te02.mp4'] == 'current'


def test_load_tables_defaults_to_current_without_a_decided_rule(con, tmp_path):
    # Only an ambiguous instance: no majority, so the current rule is reported.
    load(con, rule_rows(tmp_path, 'te00.mp4', 2, True, True), validated={'jnc6': set()})
    assert con.execute('SELECT rule_match, rule FROM instances').fetchall() == [('ambiguous', 'current')]


def test_rule_effect_relative_change(con, tmp_path):
    # The current rule drops boundary tiles: 10 vs 12 and 9 vs 10 variables.
    rows = (rule_rows(tmp_path, 'te00.mp4', 3, False, True, 10, 12)
            + rule_rows(tmp_path, 'te01.mp4', 3, False, True, 9, 10))
    load(con, rows)
    r = ilp.rule_effect(con)[0]
    assert r['instances'] == 2 and r['variables_change_min'] == pytest.approx(10 / 12 - 1)
    assert r['variables_change_median'] == pytest.approx(((10 / 12 - 1) + (9 / 10 - 1)) / 2)


def tradeoff(variants):
    # A load_test_tradeoff pair for one dataset: Tetris rows (variant, fps, HOTA, threshold) and a reference
    # at 10 fps and HOTA 0.8.
    tetris = pd.DataFrame([dict(variant_id=v, throughput_fps=f, HOTA_HOTA=h, time=100.0 / f,
                                tracking_accuracy_threshold=t) for v, f, h, t in variants])
    return tetris, pd.Series(dict(throughput_fps=10.0, HOTA_HOTA=0.8, time=10.0))


def test_operating_point_sizes(con, tmp_path):
    # The fastest within 5% (bound 0.76) does not prune; the fastest pruning one has two packing variants.
    tradeoffs = {'jnc0': tradeoff([('C_60_1_r050_none_s100_sortcython', 50.0, 0.79, np.nan),
                                   ('C_60_1_040_r050_none_s100_sortcython', 40.0, 0.78, 0.4),
                                   ('C_60_1_040_r050_tr_s100_sortcython', 30.0, 0.78, 0.4),
                                   ('C_60_4_080_r025_none_s100_bytetrackcython', 100.0, 0.5, 0.8)])}
    load(con, [instance_row('jnc0', 'test', v, A, n, 5.0, 10.0, tmp_path) for v, n in (('te00.mp4', 10), ('te01.mp4', 30))]
         + [instance_row('jnc0', 'test', v, B, 5, 5.0, 10.0, tmp_path) for v in ('te00.mp4', 'te01.mp4')])
    out = ilp.operating_point_sizes(con, tradeoffs, 0.05)['B3D1']
    # The operating point has no ILP instance.
    op = out['operating_point']
    assert op['meets_bound'] and not op['prunes'] and op['instances'] is None
    # The pruning point maps to its p022 configuration and covers both test videos.
    fp = out['fastest_pruning_point']
    assert (fp['variant'], fp['ilp_config'], fp['instances'], fp['videos'], fp['variables_median']) == (
        'C_60_1_040_r050_none_s100_sortcython', A, 2, 2, 20.0)
    assert out['hota_bound'] == pytest.approx(0.76)


def test_operating_point_sizes_unmet_and_incomplete(con, tmp_path):
    # Nothing meets the bound: both points report the best HOTA instead.
    load(con, [instance_row('jnc0', 'test', 'te00.mp4', A, 10, 5.0, 10.0, tmp_path),
               instance_row('jnc0', 'test', 'te01.mp4', B, 10, 5.0, 10.0, tmp_path)])
    low = {'jnc0': tradeoff([('C_60_1_040_r050_none_s100_sortcython', 40.0, 0.5, 0.4)])}
    out = ilp.operating_point_sizes(con, low, 0.05)['B3D1']
    assert out['operating_point'] == dict(meets_bound=False, best_hota=0.5)
    assert out['fastest_pruning_point'] == dict(meets_bound=False, best_hota=0.5)
    # A pruning point with an instance on only one of the two test videos is an error.
    ok = {'jnc0': tradeoff([('C_60_1_040_r050_none_s100_sortcython', 40.0, 0.79, 0.4)])}
    with pytest.raises(AssertionError):
        ilp.operating_point_sizes(con, ok, 0.05)


def test_real_clip_matches_p022_log():
    # caldot2-y05 va00 at ShuffleNet05_60_16_040_r050_sortcython: p022's log reports 79 polyominoes over 58
    # frames and 79 selected (LOGS/p022_exec_prune_polyominoes/worker_0.stdout), and its runtime.jsonl matches.
    param = 'ShuffleNet05_60_16_040_r050_sortcython'
    runtime = ilp.cache.exec('caldot2-y05', 'pruned-polyominoes', 'va00.mp4', param, 'score', 'runtime.jsonl')
    if not runtime.exists():
        pytest.skip('cache not mounted')
    rows, _ = ilp.measure_video(('caldot2-y05', 'valid', 'va00.mp4', [param]))
    # This run (2026-05-29) used the current rule, whose instance matches the log and the saved grid.
    current = next(r for r in rows if r['rule'] == 'current')
    assert (current['variables'], current['frames'], current['kept'], current['saved_components']) == (79, 58, 79, 79)
    assert current['tiles_outside_input'] == 0 and current['forced_dropped'] == 0 and current['window_violations'] == 0
    # Its inputs predate the output.
    assert current['scores_mtime'] <= current['output_mtime'] and current['table_mtime'] <= current['output_mtime']


def test_real_april_clip_matches_floor_rule():
    # An April 2026 test output keeps tiles scoring exactly 127 at r = 0.5: only the floor rule explains it.
    param = 'ShuffleNet05_60_1_040_r050_sortcython'
    runtime = ilp.cache.exec('jnc0', 'pruned-polyominoes', 'te01.mp4', param, 'score', 'runtime.jsonl')
    if not runtime.exists():
        pytest.skip('cache not mounted')
    rows, _ = ilp.measure_video(('jnc0', 'test', 'te01.mp4', [param]))
    by_rule = {r['rule']: r for r in rows}
    assert by_rule['current']['tiles_outside_input'] > 0
    floor = by_rule['floor']
    assert (floor['tiles_outside_input'], floor['split_polyominoes'], floor['forced_dropped'],
            floor['window_violations']) == (0, 0, 0, 0)
    assert floor['saved_components'] == floor['kept']


def test_real_largest_instance_distinct_rows():
    # The largest test instance (jnc2 te04): 56,984 rows as added, 17,845 distinct (1,325 forced + 16,520 windows),
    # recounted independently in review.
    param = 'ShuffleNet05_60_1_080_r025_bytetrackcython'
    runtime = ilp.cache.exec('jnc2', 'pruned-polyominoes', 'te04.mp4', param, 'score', 'runtime.jsonl')
    if not runtime.exists():
        pytest.skip('cache not mounted')
    rows, _ = ilp.measure_video(('jnc2', 'test', 'te04.mp4', [param]))
    floor = next(r for r in rows if r['rule'] == 'floor')
    assert (floor['constraints'], floor['distinct_constraints'], floor['forced_variables'],
            floor['distinct_window_constraints']) == (56984, 17845, 1325, 16520)
