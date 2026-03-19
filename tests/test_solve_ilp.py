import numpy as np


from polyis.sample.ilp.gurobi import solve_ilp


def test_solve_ilp_selects_first_and_last_frame():
    # 3 frames, 1x1 grid, one polyomino per frame; max distance covers all frames.
    tile_to_polyomino_id = np.zeros((3, 1, 1), dtype=np.int16)
    polyomino_lengths = [[1], [1], [1]]
    max_sampling_distance = np.array([[10.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    # First frame (0, polyomino 0) and last frame (2, polyomino 0) must always be selected.
    assert (0, 0) in selected
    assert (2, 0) in selected


def test_solve_ilp_returns_set_of_tuples():
    # Verify return type is a set of (int, int) tuples.
    tile_to_polyomino_id = np.zeros((2, 1, 1), dtype=np.int16)
    polyomino_lengths = [[1], [1]]
    max_sampling_distance = np.array([[5.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    assert isinstance(selected, set)
    for item in selected:
        assert isinstance(item, tuple) and len(item) == 2


def test_solve_ilp_mandatory_bridge_forces_both_endpoints():
    # Frames 0 and 2 cover tile (0,0), frame 1 does not; gap=2 exceeds max_distance=1 → both forced.
    tile_to_polyomino_id = np.full((3, 1, 1), -1, dtype=np.int16)
    tile_to_polyomino_id[0, 0, 0] = 0
    tile_to_polyomino_id[2, 0, 0] = 0
    polyomino_lengths = [[1], [], [1]]
    max_sampling_distance = np.array([[1.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    # Both endpoints of the bridge must be in the solution.
    assert (0, 0) in selected
    assert (2, 0) in selected


def test_solve_ilp_window_constraint_satisfied():
    # 4 frames, 1x1 grid, max_distance=2; frame 0 selected → at least one of frames 1-2 must be selected.
    tile_to_polyomino_id = np.zeros((4, 1, 1), dtype=np.int16)
    polyomino_lengths = [[2], [1], [1], [3]]  # frames 1,2 are cheaper than frame 3
    max_sampling_distance = np.array([[2.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    # Frame 0 is forced (first). Frame 3 is forced (last). The window covers frames 1-2; at least one must be in.
    assert (0, 0) in selected
    assert (3, 0) in selected
    assert (1, 0) in selected or (2, 0) in selected


def test_solve_ilp_minimizes_total_cells():
    # 3 frames, 1x1 grid; frame 1 polyomino has 100 cells (expensive), frame 0 and 2 are cheap.
    # With max_distance=2 the window constraint is satisfied by frame 2 alone without including frame 1.
    tile_to_polyomino_id = np.zeros((3, 1, 1), dtype=np.int16)
    polyomino_lengths = [[1], [100], [1]]
    max_sampling_distance = np.array([[2.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    # Optimal solution is frames 0 and 2 only (total 2 cells), avoiding the expensive frame 1.
    assert (0, 0) in selected
    assert (2, 0) in selected
    assert (1, 0) not in selected


def test_solve_ilp_uncovered_tiles_ignored():
    # 2x1 grid; tile (0,0) covered in all frames, tile (1,0) never covered → no constraints for (1,0).
    tile_to_polyomino_id = np.full((2, 2, 1), -1, dtype=np.int16)
    tile_to_polyomino_id[0, 0, 0] = 0
    tile_to_polyomino_id[1, 0, 0] = 0
    polyomino_lengths = [[1], [1]]
    max_sampling_distance = np.array([[5.0], [5.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 2, 1).selected

    # Only the polyomino covering (0,0) appears in both frames; (1,0) is never covered.
    assert (0, 0) in selected
    assert (1, 0) in selected


def test_solve_ilp_single_frame_no_constraints():
    # Single frame: tile appears in only one frame so len(pos)==1, which skips all constraints.
    # The minimizer has no obligation to select it, so the result is empty.
    tile_to_polyomino_id = np.zeros((1, 1, 1), dtype=np.int16)
    polyomino_lengths = [[3]]
    max_sampling_distance = np.array([[5.0]])

    selected = solve_ilp(tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1).selected

    # No constraints → minimizer selects nothing (objective drives all vars to 0).
    assert selected == set()


def test_solve_ilp_default_time_limit():
    # Verify the default time limit equals the module-level constant.
    import inspect
    sig = inspect.signature(solve_ilp)
    default = sig.parameters["time_limit_seconds"].default
    assert default == 0.5


def test_solve_ilp_explicit_time_limit_produces_valid_result():
    # Passing an explicit time_limit_seconds should still yield a correct solution.
    tile_to_polyomino_id = np.zeros((3, 1, 1), dtype=np.int16)
    polyomino_lengths = [[1], [100], [1]]
    max_sampling_distance = np.array([[2.0]])

    # Use a generous limit so the optimal solution is still found.
    result = solve_ilp(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1,
        time_limit_seconds=30.0,
    )

    # Optimal: frames 0 and 2 only (total 2 cells).
    assert (0, 0) in result.selected
    assert (2, 0) in result.selected
    assert (1, 0) not in result.selected


def test_solve_ilp_very_short_time_limit_returns_result():
    # Even with an extremely short time limit the function must return without error.
    # The solution may be suboptimal or empty, but no exception should be raised.
    tile_to_polyomino_id = np.zeros((3, 1, 1), dtype=np.int16)
    polyomino_lengths = [[1], [1], [1]]
    max_sampling_distance = np.array([[10.0]])

    result = solve_ilp(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1,
        time_limit_seconds=1e-6,
    )

    # Result must be a set (possibly empty if the solver found nothing in time).
    assert isinstance(result.selected, set)
