"""
Search-space reduction for pruning (the author's proposal, profiling/DECISIONS.md A1-A3).

Two tolerances give the same pruning ILP, hence the same result, when their per-tile gaps
max(table[:, :, tol] // s, 1) agree on every tile the never-relevant mask allows (other tiles always score
0 and are in no polyomino).  So, per (tracker, sample rate), tolerances are grouped by those gaps and only one
representative per group runs its ILP.

All-1s gaps do NOT mean every polyomino is kept: the ILP constrains only tiles that appear in at least two
sampled frames, so polyominoes made only of single-appearance tiles are still dropped (DECISIONS.md A4).
Hence every group, all-1s or not, still solves once.  Never merged with the no-pruning branch either (p022
binarizes with >= floor(T_r*255), p030 with > T_r*255).
"""

from typing import NamedTuple

import numpy as np

from execution.config import MISTRACK_TOLERANCES


class ToleranceGroup(NamedTuple):
    tolerances: tuple[float, ...]   # in sweep order; the first is the representative that runs
    all_ones: bool                  # every relevant tile's gap is 1 (informational; the ILP still runs)

    @property
    def representative(self) -> float:
        return self.tolerances[0]


def gaps(table: np.ndarray, tolerance: float, sample_rate: int) -> np.ndarray:
    """p022's per-tile max sampling distance (execution2.operators.max_sampling_distance, without its checks)."""
    distance = table[:, :, MISTRACK_TOLERANCES.index(tolerance)].copy()
    distance //= sample_rate
    return np.maximum(distance, 1)


def group_tolerances(table: np.ndarray, mask: np.ndarray, tolerances: list[float], sample_rate: int,
                     reduce: bool = True) -> list[ToleranceGroup]:
    """Tolerances grouped by identical gaps on mask tiles (each alone when reduce is False)."""
    if not reduce:
        return [ToleranceGroup((t,), False) for t in tolerances]
    groups: list[tuple[list[float], np.ndarray]] = []
    for t in tolerances:
        g = gaps(table, t, sample_rate)[mask]
        for members, ref in groups:
            if np.array_equal(ref, g):
                members.append(t)
                break
        else:
            groups.append(([t], g))
    return [ToleranceGroup(tuple(members), bool((ref == 1).all())) for members, ref in groups]


def sweep_groups(cfg) -> dict[tuple[str, int], list[ToleranceGroup]]:
    """Groups for every (tracker, sample rate) of a sweep config; empty when no tolerance is swept."""
    tolerances = [t for t in cfg.params.mistrack_tolerances if t is not None]
    if not tolerances:
        return {}
    mask = np.load(cfg.artifacts.never_relevant_bitmap).astype(bool)
    out = {}
    for tracker in cfg.params.trackers:
        table = np.load(cfg.artifacts.max_rate_tables[tracker])
        if table.shape[:2] != mask.shape:
            raise ValueError(f'max_rate_table for {tracker} has grid {table.shape[:2]}, never-relevant bitmap {mask.shape}')
        for s in cfg.params.sample_rates:
            out[(tracker, s)] = group_tolerances(table, mask, tolerances, s, cfg.params.reduce_search_space)
    return out

