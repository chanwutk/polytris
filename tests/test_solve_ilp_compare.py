"""
Comparison tests: verify that all three solve_ilp implementations agree on
objective value and selected set:

  * polyis.sample.ilp.pulp   — PuLP + Gurobi backend  (reference)
  * polyis.sample.ilp.gurobi — gurobipy Python API
  * polyis.sample.ilp.c      — Cython / Gurobi C API
"""
import random
from time import monotonic

import numpy as np
import pytest

import gurobipy  # noqa: F401

from polyis.sample.ilp.gurobi import solve_ilp as solve_ilp_gurobi
from polyis.sample.ilp.pulp import solve_ilp as solve_ilp_pulp
from polyis.sample.ilp.c.gurobi import solve_ilp as solve_ilp_c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_objective(selected: set[tuple[int, int]], polyomino_lengths: list[list[int]]) -> int:
    """Sum of cells for the selected (frame, polyomino_id) pairs."""
    return sum(polyomino_lengths[b][k] for (b, k) in selected)


def make_random_instance(
    num_frames: int,
    grid_h: int,
    grid_w: int,
    coverage_prob: float,
    max_polyominoes_per_frame: int,
    max_cell_count: int,
    max_distance: float,
    rng: random.Random,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Generate a random ILP instance."""
    # Assign each tile in each frame a polyomino ID (or -1 if uncovered).
    tile_to_polyomino_id = np.full((num_frames, grid_h, grid_w), -1, dtype=np.int16)
    polyomino_lengths: list[list[int]] = []

    for b in range(num_frames):
        num_polys = rng.randint(1, max_polyominoes_per_frame)
        lengths = [rng.randint(1, max_cell_count) for _ in range(num_polys)]
        polyomino_lengths.append(lengths)
        for n in range(grid_h):
            for m in range(grid_w):
                if rng.random() < coverage_prob:
                    tile_to_polyomino_id[b, n, m] = rng.randint(0, num_polys - 1)

    max_sampling_distance = np.full((grid_h, grid_w), max_distance)
    return tile_to_polyomino_id, polyomino_lengths, max_sampling_distance


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_compare_objective_small(seed: int):
    """On small random instances both solvers must achieve the same minimum objective."""
    rng = random.Random(seed)
    tile_to_polyomino_id, polyomino_lengths, max_sampling_distance = make_random_instance(
        num_frames=5,
        grid_h=3,
        grid_w=3,
        coverage_prob=0.6,
        max_polyominoes_per_frame=4,
        max_cell_count=10,
        max_distance=2.0,
        rng=rng,
    )

    selected_ref = solve_ilp_pulp(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 3, 3
    )
    selected_py = solve_ilp_gurobi(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 3, 3
    ).selected
    selected_cy = solve_ilp_c(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 3, 3
    ).selected

    obj_ref = compute_objective(selected_ref, polyomino_lengths)
    obj_py  = compute_objective(selected_py,  polyomino_lengths)
    obj_cy  = compute_objective(selected_cy,  polyomino_lengths)

    assert obj_py == obj_ref, (
        f"seed={seed}: gurobipy objective {obj_py} != pulp objective {obj_ref}\n"
        f"  gurobipy selected: {sorted(selected_py)}\n"
        f"  pulp selected:     {sorted(selected_ref)}"
    )
    assert obj_cy == obj_ref, (
        f"seed={seed}: cython objective {obj_cy} != pulp objective {obj_ref}\n"
        f"  cython selected: {sorted(selected_cy)}\n"
        f"  pulp selected:   {sorted(selected_ref)}"
    )

    assert selected_py == selected_ref, (
        f"seed={seed}: gurobipy selected {sorted(selected_py)} != pulp selected {sorted(selected_ref)}"
    )
    assert selected_cy == selected_ref, (
        f"seed={seed}: cython selected {sorted(selected_cy)} != pulp selected {sorted(selected_ref)}"
    )


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_compare_objective_medium(seed: int):
    """On medium random instances both solvers must achieve the same minimum objective."""
    rng = random.Random(seed)
    tile_to_polyomino_id, polyomino_lengths, max_sampling_distance = make_random_instance(
        num_frames=128,
        grid_h=16,
        grid_w=16,
        coverage_prob=0.5,
        max_polyominoes_per_frame=6,
        max_cell_count=20,
        max_distance=3.0,
        rng=rng,
    )

    selected_ref = solve_ilp_pulp(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 5, 5
    )
    selected_py = solve_ilp_gurobi(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 5, 5
    ).selected
    selected_cy = solve_ilp_c(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 5, 5
    ).selected

    obj_ref = compute_objective(selected_ref, polyomino_lengths)
    obj_py  = compute_objective(selected_py,  polyomino_lengths)
    obj_cy  = compute_objective(selected_cy,  polyomino_lengths)

    assert obj_py == obj_ref, (
        f"seed={seed}: gurobipy objective {obj_py} != pulp objective {obj_ref}"
    )
    assert obj_cy == obj_ref, (
        f"seed={seed}: cython objective {obj_cy} != pulp objective {obj_ref}"
    )
    assert selected_py == selected_ref, (
        f"seed={seed}: gurobipy selected {sorted(selected_py)} != pulp selected {sorted(selected_ref)}"
    )
    assert selected_cy == selected_ref, (
        f"seed={seed}: cython selected {sorted(selected_cy)} != pulp selected {sorted(selected_ref)}"
    )


def test_compare_mandatory_bridge_instance():
    """Deterministic instance with a forced mandatory bridge — both solvers must agree."""
    # 4 frames, 1x1 grid. Gap between frames 1 and 3 (gap=2) exceeds max_distance=1.
    # Frame 2 is skipped (tile is uncovered), so frames 1 and 3 are forced.
    tile_to_polyomino_id = np.full((4, 1, 1), -1, dtype=np.int16)
    tile_to_polyomino_id[0, 0, 0] = 0  # frame 0, polyomino 0
    tile_to_polyomino_id[1, 0, 0] = 0  # frame 1, polyomino 0
    tile_to_polyomino_id[3, 0, 0] = 0  # frame 3, polyomino 0 (gap of 2 from frame 1)
    polyomino_lengths = [[5], [3], [], [7]]
    max_sampling_distance = np.array([[1.0]])

    selected_ref = solve_ilp_pulp(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1
    )
    selected_py = solve_ilp_gurobi(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1
    ).selected
    selected_cy = solve_ilp_c(
        tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, 1, 1
    ).selected

    obj_ref = compute_objective(selected_ref, polyomino_lengths)
    obj_py  = compute_objective(selected_py,  polyomino_lengths)
    obj_cy  = compute_objective(selected_cy,  polyomino_lengths)

    assert obj_py == obj_ref
    assert obj_cy == obj_ref
    assert selected_py == selected_ref
    assert selected_cy == selected_ref


@pytest.mark.parametrize("seed,num_frames,grid_h,grid_w,max_distance", [
    (0,   5,   3,  3, 2.0),   # small
    (10, 128, 16, 16, 3.0),   # medium
    (10, 512, 16, 16, 16.0),   # large
])
def test_runtime_comparison(seed, num_frames, grid_h, grid_w, max_distance, capsys):
    """Print build/solve timing for all three implementations on the same instance."""
    rng = random.Random(seed)
    tile_to_polyomino_id, polyomino_lengths, max_sampling_distance = make_random_instance(
        num_frames=num_frames,
        grid_h=grid_h,
        grid_w=grid_w,
        coverage_prob=0.6,
        max_polyominoes_per_frame=4,
        max_cell_count=10,
        max_distance=max_distance,
        rng=rng,
    )
    args = (tile_to_polyomino_id, polyomino_lengths, max_sampling_distance, grid_h, grid_w)

    # PuLP: no internal timing — measure total wall time.
    # t0 = monotonic()
    # solve_ilp_pulp(*args)
    # pulp_total_ms = (monotonic() - t0) * 1000

    # gurobipy Python API: build_ms + solve_ms from ILPResult.
    res_py = solve_ilp_gurobi(*args)

    # Cython C API: build_ms + solve_ms from ILPResult.
    res_cy = solve_ilp_c(*args)

    # Print a three-row comparison table.
    header = f"\n{'':20s} {'build_ms':>10s} {'solve_ms':>10s} {'total_ms':>10s}"
    sep    = "-" * len(header)
    rows = [
        # f"{'pulp (gurobi)':20s} {'—':>10s} {'—':>10s} {pulp_total_ms:>10.1f}",
        f"{'gurobipy':20s} {res_py.build_ms:>10.1f} {res_py.solve_ms:>10.1f} {res_py.build_ms + res_py.solve_ms:>10.1f}",
        f"{'cython c api':20s} {res_cy.build_ms:>10.1f} {res_cy.solve_ms:>10.1f} {res_cy.build_ms + res_cy.solve_ms:>10.1f}",
    ]
    instance_info = f"seed={seed}  frames={num_frames}  grid={grid_h}x{grid_w}"
    with capsys.disabled():
        print(f"\n{instance_info}")
        print(header)
        print(sep)
        for row in rows:
            print(row)
