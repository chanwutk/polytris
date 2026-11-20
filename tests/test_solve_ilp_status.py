"""The status, solution count, and lower bound that both ILP solvers report alongside the selection."""
import random

import numpy as np
import pytest

from polyis.sample.ilp.c.gurobi import solve_ilp as solve_ilp_c
from polyis.sample.ilp.gurobi import ILPResult, solve_ilp as solve_ilp_gurobi
from tests.test_solve_ilp_compare import compute_objective, make_random_instance


SOLVERS = {'gurobipy': solve_ilp_gurobi, 'c': solve_ilp_c}
OPTIMAL, TIME_LIMIT = 2, 9


def medium_instance(seed: int):
    # A 40-frame, 6x6 instance large enough that a zero time limit stops Gurobi before it finishes.
    return make_random_instance(40, 6, 6, 0.5, 4, 8, 4.0, random.Random(seed))


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('seed', range(5))
def test_optimal_runs_report_status_and_a_bound_equal_to_the_objective(solver, seed):
    tile_ids, lengths, distance = make_random_instance(8, 3, 3, 0.6, 3, 5, 3.0, random.Random(seed))
    result = SOLVERS[solver](tile_ids, lengths, distance, 3, 3, time_limit_seconds=60)
    objective = compute_objective(result.selected, lengths)
    assert result.status == OPTIMAL
    assert result.sol_count >= 1
    # Gurobi's default MIP gap is relative 1e-4, far below one tile on these instances.
    assert result.obj_bound == pytest.approx(objective, abs=0.5)


@pytest.mark.parametrize('solver', SOLVERS)
def test_time_limited_run_reports_time_limit_and_a_bound_below_the_objective(solver):
    tile_ids, lengths, distance = medium_instance(0)
    result = SOLVERS[solver](tile_ids, lengths, distance, 6, 6, time_limit_seconds=0)
    assert result.status == TIME_LIMIT
    # A bound, when Gurobi has one, never exceeds the objective of a solution it found.
    if result.obj_bound is not None and result.sol_count:
        assert result.obj_bound <= compute_objective(result.selected, lengths) + 1e-6
    # Without a solution there is no selection.
    if not result.sol_count:
        assert result.selected == set()


@pytest.mark.parametrize('solver', SOLVERS)
def test_instance_without_polyominoes_is_optimal_with_zero_bound(solver):
    tile_ids = np.full((3, 2, 2), -1, dtype=np.int16)
    result = SOLVERS[solver](tile_ids, [[], [], []], np.full((2, 2), 2.0), 2, 2)
    assert result.selected == set()
    assert result.status == OPTIMAL
    assert result.obj_bound == pytest.approx(0)


@pytest.mark.parametrize('seed', range(5))
def test_solvers_agree_on_status_and_bound(seed):
    tile_ids, lengths, distance = make_random_instance(10, 3, 3, 0.6, 3, 5, 3.0, random.Random(seed))
    a = solve_ilp_gurobi(tile_ids, lengths, distance, 3, 3, time_limit_seconds=60)
    b = solve_ilp_c(tile_ids, lengths, distance, 3, 3, time_limit_seconds=60)
    assert (a.status, a.sol_count > 0) == (b.status, b.sol_count > 0)
    assert a.obj_bound == pytest.approx(b.obj_bound, abs=0.5)


def test_result_fields_default_to_none_for_existing_callers():
    result = ILPResult(selected=set(), build_ms=1.0, solve_ms=2.0)
    assert (result.status, result.sol_count, result.obj_bound) == (None, None, None)
    selected, build_ms, solve_ms = result[:3]
    assert (selected, build_ms, solve_ms) == (set(), 1.0, 2.0)
