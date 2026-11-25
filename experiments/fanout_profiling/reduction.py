"""
The fan-out sweep's tolerance groups per (tracker, sample rate).  The grouping itself is profiling/reduction.py's
(one copy, shared with the profiling sweep's dedup).
"""

import numpy as np

# Re-exported: the fan-out sweep's tests import these names from here.
from profiling.reduction import ToleranceGroup, gaps, group_tolerances  # noqa: F401


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
