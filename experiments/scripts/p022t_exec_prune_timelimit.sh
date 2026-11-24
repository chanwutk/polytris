#!/bin/bash
# Run p022_exec_prune_polyominoes.py for each time limit value in the experiment.
# Results for each limit go into the separate '022t_pruned_polyominoes_timelimit' stage,
# keyed by 'tl{value}' subdirectory, so runs are independent and never overwrite each other.
#
# Usage (from repo root, inside the container):
#   bash scripts/p022t_exec_prune_timelimit.sh [--test] [--valid]
#
# By default (no flags) the valid videoset is used.

set -euo pipefail

VIDEOSET_FLAGS="${@:---valid}"

TIME_LIMITS=(0.01 0.05 0.1 0.5 1)

for tl in "${TIME_LIMITS[@]}"; do
    echo ""
    echo "========================================================"
    echo ">>> Time limit: ${tl}s"
    echo "========================================================"
    python scripts/p022_exec_prune_polyominoes.py ${VIDEOSET_FLAGS} --time-limit "${tl}"
done

echo ""
echo "All time-limit variants complete."
