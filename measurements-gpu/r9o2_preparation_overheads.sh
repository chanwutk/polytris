#!/bin/bash
# Inside bombe_container.sh: bash measurements-gpu/r9o2_preparation_overheads.sh <dataset...>
# Hashed saved inputs live in /scratch/inputs/<dataset>/; results and compiler caches stay in /scratch/overheads/.
set -euo pipefail
[ "$#" -gt 0 ] || { echo "usage: $0 <dataset...>" >&2; exit 2; }
mkdir -p /scratch/logs
for dataset in "$@"; do
  for step in train reference sweep persistent classifier; do
    [ ! -f /scratch/overheads/$dataset/$step/result.json ] || continue
    log=/scratch/logs/${dataset}_${step}.log
    python measurements-gpu/r9o2_preparation_overheads.py --dataset "$dataset" --step "$step" \
      --inputs /scratch/inputs --scratch /scratch/overheads > "$log" 2>&1
    tail -1 "$log"
  done
done
for dataset in "$@"; do
  log=/scratch/logs/${dataset}_tables.log
  python measurements-gpu/r9o2_preparation_overheads.py --dataset "$dataset" --step tables \
    --inputs /scratch/inputs --scratch /scratch/overheads > "$log" 2>&1
  tail -1 "$log"
done
