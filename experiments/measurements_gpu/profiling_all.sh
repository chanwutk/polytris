#!/usr/bin/env bash
# Profile every dataset in turn, unattended: for each sweep config, profiling.main (the timed runs), then
# profiling.evaluate and profiling.aggregate (never overlapping a timed run).  Run on ace's host inside tmux, so it
# outlives the SSH session:
#
#   tmux new -d -s profiling '<checkout>/experiments/measurements_gpu/profiling_all.sh <checkout>'
#
# Each step runs in a throwaway polyis-polyis container on GPU 0 with the checkout at /polyis, the data and cache
# read-only, and only <data>/polyis-data/profiling (the sweeps' output_root) and a scratch log directory writable.
# Every sweep resumes (finished configurations are skipped), so rerunning this script continues where it stopped.
# A failed step is logged and the next one still runs.  Progress: <data>/polyis-data/profiling/logs/.
set -uo pipefail

checkout=${1:?usage: profiling_all.sh <checkout path> [datasets...]}
shift
# caldot1-y05 first: its first configurations give the measured per-run time and startup overhead.
if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(caldot1-y05 caldot2-y05 ams-y05 jnc0 jnc2 jnc6 jnc7)
fi
data=${DATA_ROOT:-/data/cwkt/data}
out=$data/polyis-data/profiling
logs=$out/logs
mkdir -p "$logs" "$data/profiling-script-logs"

# One step in a throwaway container; tetris's script loggers write under /polyis-cache/LOGS, so that is scratch.
step() {
  docker run --rm --gpus device=0 --shm-size 64g --network host -w /polyis \
    -e POLYIS_DATA=/polyis-data -e POLYIS_CACHE=/polyis-cache \
    -v "$checkout:/polyis" -v "$data/polyis-data:/polyis-data:ro" -v "$out:/polyis-data/profiling" \
    -v "$data/polyis-cache:/polyis-cache:ro" -v "$data/profiling-script-logs:/polyis-cache/LOGS" \
    -v "$HOME/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro" polyis-polyis "$@"
}

for d in "${datasets[@]}"; do
  config=experiments/configs/profiling/$d.yaml
  for cmd in "profiling.main --config $config" "profiling.evaluate --config $config --procs 16" \
             "profiling.aggregate --config $config"; do
    echo "== $(date -Is) $d: python -m $cmd" | tee -a "$logs/driver.log"
    step python -m $cmd >> "$logs/$d.log" 2>&1
    echo "   $(date -Is) exit $?" | tee -a "$logs/driver.log"
  done
done
echo "== $(date -Is) all datasets done" | tee -a "$logs/driver.log"
