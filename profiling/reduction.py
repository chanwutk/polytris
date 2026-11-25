"""
Configurations with identical pruning (the profiling sweep's dedup; first proposed in
experiments/fanout_profiling/DECISIONS.md A1-A3).

Two tolerances give the same pruning ILP, hence the same pruned grids and the same results downstream, when their
per-tile gaps max(table[:, :, tol] // s, 1) agree on every tile the never-relevant mask allows (other tiles always
score 0 and are in no polyomino).  So, per (tracker, sample rate), tolerances are grouped by those gaps and only the
first tolerance of each group needs to run.

All-1s gaps do NOT mean every polyomino is kept: the ILP constrains only tiles that appear in at least two
sampled frames, so polyominoes made only of single-appearance tiles are still dropped (DECISIONS.md A4).
Hence every group, all-1s or not, still runs once.  Never merged with the no-pruning configuration either (p022
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
    """p022's per-tile max sampling distance (execution.operators.max_sampling_distance, without its checks)."""
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
        # Join the first group whose gaps match exactly; otherwise start a new group.
        for members, ref in groups:
            if np.array_equal(ref, g):
                members.append(t)
                break
        else:
            groups.append(([t], g))
    return [ToleranceGroup(tuple(members), bool((ref == 1).all())) for members, ref in groups]


def duplicates(cfg, configs: list) -> dict[str, str]:
    """Each duplicate configuration's id -> the id of the configuration that runs for it.

    A configuration is a duplicate when its tolerance shares an identical-ILP group (for its tracker and sample
    rate) with an earlier tolerance; it equals the group representative's configuration in every other parameter.
    Empty when dedup_pruning is off or no tolerance is swept.  cfg is a profiling.config.SweepConfig.
    """
    tolerances = [t for t in cfg.params.mistrack_tolerances if t is not None]
    if not cfg.params.dedup_pruning or not tolerances:
        return {}
    mask = np.load(cfg.artifacts.never_relevant_bitmap).astype(bool)
    # tolerance -> its group's representative, per (tracker, sample rate).
    representative: dict[tuple[str, int, float], float] = {}
    for tracker in cfg.params.trackers:
        table = np.load(cfg.artifacts.max_rate_tables[tracker])
        if table.shape[:2] != mask.shape:
            raise ValueError(f'max_rate_table for {tracker} has grid {table.shape[:2]}, never-relevant bitmap {mask.shape}')
        for s in cfg.params.sample_rates:
            for group in group_tolerances(table, mask, tolerances, s):
                for t in group.tolerances:
                    representative[(tracker, s, t)] = group.representative
    by_params = {_key(c.params): c.id for c in configs}
    out = {}
    for c in configs:
        tol = c.params['mistrack_tolerance']
        if tol is None:
            continue
        rep = representative[(c.params['tracker'], c.params['sample_rate'], tol)]
        if rep != tol:
            # The representative's configuration: the same parameters with the representative tolerance.
            out[c.id] = by_params[_key({**c.params, 'mistrack_tolerance': rep})]
    return out


def _key(params: dict) -> tuple:
    """A hashable form of a configuration's parameters."""
    return tuple(sorted(params.items()))
